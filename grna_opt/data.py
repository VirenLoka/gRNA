"""Seed-guide selection and target-DNA handling.

Two inputs feed the engine:

* a **seed guide** — a low-efficacy 23-mer that the generator learns to improve;
* a **target DNA region** — the context the cross-attention attends over,
  supplied directly in ``config.yaml`` (there is no genome FASTA in this repo, so
  flanking sequence cannot be recovered from the datasets' chr/start/end columns).

Three ways to obtain the seed are supported, matching ``seed_guide.mode``:

``dataset``
    Take a guide from the bottom percentile of measured efficacy in one of the
    DeepCRISPR paper datasets.  This is the default: it gives a genuine, measured
    low-efficacy starting point.
``target_scan``
    Enumerate every NGG site in the supplied target region, score them all with
    the frozen scorer, and start from the worst.  Use this when the guide must
    genuinely bind *your* target.
``examples``
    Take a negative straight from ``DeepCRISPR/examples/``.  Regression files
    sort ascending by measured efficacy; classification files keep label-0 rows.
    The off-target formats additionally carry a *paired* sgRNA target site, which
    ``target.source: paired`` can adopt as the target DNA.
``explicit``
    Use a 23-mer you name outright.

A naming trap worth repeating: in the on-target files the column the DeepCRISPR
README labels "Target Seq" is the 23nt guide+PAM itself, not a separate target
region.  Only the off-target files pair a guide with a distinct DNA sequence.
"""

from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .encoding import GUIDE_LENGTH, hamming_distance, reverse_complement
from .logging_utils import get_logger

# 23nt window whose last three bases are a canonical NGG PAM.
_PAM_SITE = re.compile(r"(?=([ACGT]{21}GG))")
_VALID_DNA = re.compile(r"^[ACGT]+$")


@dataclass(frozen=True)
class ExampleFormat:
    """Column layout of one DeepCRISPR ``examples/`` file type.

    Note the naming trap: in the on-target files the column the README calls
    "Target Seq" *is* the 23nt guide+PAM, not a separate target region.  Only the
    off-target files hold two distinct sequences per row — the sgRNA's intended
    site and a mismatched genomic site — so they are the only source here that
    yields a genuine (guide, target) pair.
    """

    name: str
    task: str
    n_cols: int
    guide_col: int
    label_col: int
    target_col: int | None = None   # paired DNA, off-target files only
    locus_cols: tuple[int, ...] = ()
    id_col: int | None = None

    @property
    def is_paired(self) -> bool:
        return self.target_col is not None


# Keyed by file extension, matching DeepCRISPR/examples/README.md.
EXAMPLE_FORMATS: dict[str, ExampleFormat] = {
    ".rsgt": ExampleFormat("on_target_seq", "regression", 6,
                           guide_col=4, label_col=5, locus_cols=(0, 1, 2, 3)),
    ".episgt": ExampleFormat("on_target_cls", "classification", 10,
                             guide_col=4, label_col=9, locus_cols=(0, 1, 2, 3)),
    ".repisgt": ExampleFormat("on_target_reg", "regression", 10,
                              guide_col=4, label_col=9, locus_cols=(0, 1, 2, 3)),
    # guide_col points at the *off-target* site (the sequence that fails to cut,
    # i.e. the negative); target_col points at the sgRNA's intended target site.
    ".epiotrt": ExampleFormat("off_target_cls", "classification", 12,
                              guide_col=6, label_col=11, target_col=1, id_col=0),
    ".repiotrt": ExampleFormat("off_target_reg", "regression", 12,
                               guide_col=6, label_col=11, target_col=1, id_col=0),
}


@dataclass
class SeedGuide:
    """A starting guide plus whatever provenance we have for it."""

    sequence: str
    source: str
    measured_efficacy: float | None = None
    predicted_efficacy: float | None = None
    locus: str | None = None
    strand: str | None = None
    target_offset: int | None = None  # index in the target region, if found
    paired_target: str | None = None  # sgRNA site from an off-target example row
    # Raw label from the source file. For on-target files this is measured
    # on-target efficacy; for off-target files it is cleavage at the *off-target*
    # locus, which is a different quantity — hence kept separate from
    # `measured_efficacy` so the two are never conflated in reports.
    file_label: float | None = None
    file_label_meaning: str | None = None
    guide_id: str | None = None       # sgRNA id from the off-target files, e.g. "sg2"

    @property
    def spacer(self) -> str:
        return self.sequence[:20]

    @property
    def pam(self) -> str:
        return self.sequence[20:]

    def describe(self) -> str:
        bits = [f"{self.sequence} (spacer={self.spacer} PAM={self.pam})",
                f"source={self.source}"]
        if self.measured_efficacy is not None:
            bits.append(f"measured_ontarget={self.measured_efficacy:.4f}")
        if self.predicted_efficacy is not None:
            bits.append(f"predicted={self.predicted_efficacy:.4f}")
        if self.file_label is not None and self.measured_efficacy is None:
            bits.append(f"file_label={self.file_label:g} ({self.file_label_meaning})")
        if self.locus:
            bits.append(f"locus={self.locus}{self.strand or ''}")
        if self.target_offset is not None:
            bits.append(f"target_offset={self.target_offset}")
        return "  ".join(bits)


def clean_dna(sequence: str, label: str = "sequence") -> str:
    """Strip whitespace/FASTA headers and validate the alphabet."""
    lines = [ln.strip() for ln in sequence.strip().splitlines()]
    lines = [ln for ln in lines if ln and not ln.startswith(">")]
    cleaned = "".join(lines).upper().replace(" ", "")
    if not cleaned:
        raise ValueError(f"{label} is empty")
    if not _VALID_DNA.match(cleaned):
        bad = sorted(set(cleaned) - set("ACGT"))
        raise ValueError(f"{label} contains non-ACGT character(s): {bad}")
    return cleaned


def ensure_paper_data(dataset_path: str | Path,
                      repo_dir: str | Path = "DeepCRISPR") -> Path:
    """Resolve a paper-data path, untarring the bundled archives on first use."""
    dataset_path = Path(dataset_path)
    if dataset_path.exists():
        return dataset_path

    repo_dir = Path(repo_dir)
    # paper_data2/* lives in the regression archive, paper_data/* in classification.
    archives = {
        "paper_data2": repo_dir / "paper_data-regression.tar.gz",
        "paper_data": repo_dir / "paper_data-classification.tar.gz",
    }
    parts = dataset_path.parts
    root = next((p for p in parts if p in archives), None)
    if root is None:
        raise FileNotFoundError(f"dataset not found and not extractable: {dataset_path}")

    archive = archives[root]
    if not archive.exists():
        raise FileNotFoundError(f"{dataset_path} missing and {archive} not present")

    extract_to = Path(*parts[: parts.index(root)]) if parts.index(root) else Path(".")
    get_logger().info("extracting %s -> %s", archive, extract_to)
    with tarfile.open(archive) as tar:
        tar.extractall(extract_to)

    if not dataset_path.exists():
        raise FileNotFoundError(f"{archive} did not contain {dataset_path}")
    return dataset_path


def load_ontar_dataset(path: str | Path, sequence_column: int = 4,
                       label_column: int = 9) -> pd.DataFrame:
    """Read a ``.repisgt``/``.episgt``/``.rsgt`` table into (sequence, label, locus)."""
    path = ensure_paper_data(path)
    raw = pd.read_csv(path, sep="\t", header=None)

    if sequence_column >= raw.shape[1] or label_column >= raw.shape[1]:
        raise ValueError(
            f"{path} has {raw.shape[1]} columns; requested sequence_column="
            f"{sequence_column}, label_column={label_column}"
        )

    frame = pd.DataFrame({
        "sequence": raw.iloc[:, sequence_column].astype(str).str.upper(),
        "label": pd.to_numeric(raw.iloc[:, label_column], errors="coerce"),
    })
    if raw.shape[1] >= 4:
        frame["locus"] = (raw.iloc[:, 0].astype(str) + ":" + raw.iloc[:, 1].astype(str)
                          + "-" + raw.iloc[:, 2].astype(str))
        frame["strand"] = raw.iloc[:, 3].astype(str)
    else:
        frame["locus"] = None
        frame["strand"] = None

    valid = frame["sequence"].str.len().eq(GUIDE_LENGTH) & frame["label"].notna()
    dropped = (~valid).sum()
    if dropped:
        get_logger().warning("dropped %d malformed row(s) from %s", dropped, path)
    return frame[valid].reset_index(drop=True)


def select_low_efficacy_guide(dataset_path: str | Path, sequence_column: int = 4,
                              label_column: int = 9, percentile: float = 5.0,
                              index: int = 0) -> SeedGuide:
    """Pick a guide from the bottom ``percentile`` of measured efficacy."""
    frame = load_ontar_dataset(dataset_path, sequence_column, label_column)
    cutoff = float(np.percentile(frame["label"], percentile))
    pool = frame[frame["label"] <= cutoff].sort_values("label").reset_index(drop=True)

    if pool.empty:
        raise ValueError(f"no guides at or below the {percentile}th percentile in {dataset_path}")
    if index >= len(pool):
        raise IndexError(
            f"seed_guide.dataset.index={index} but only {len(pool)} guides sit below "
            f"the {percentile}th percentile (cutoff={cutoff:.4f})"
        )

    row = pool.iloc[index]
    get_logger().info(
        "low-efficacy pool: %d guides <= %.4f (%.1fth pct) out of %d; taking index %d",
        len(pool), cutoff, percentile, len(frame), index,
    )
    return SeedGuide(
        sequence=row["sequence"],
        source=f"{Path(dataset_path).name}[{percentile}th pct, idx {index}]",
        measured_efficacy=float(row["label"]),
        locus=row.get("locus"),
        strand=row.get("strand"),
    )


def detect_example_format(path: str | Path) -> ExampleFormat:
    """Resolve a ``DeepCRISPR/examples/`` file to its column layout."""
    path = Path(path)
    fmt = EXAMPLE_FORMATS.get(path.suffix)
    if fmt is None:
        raise ValueError(
            f"unrecognised example file type {path.suffix!r} for {path}; "
            f"expected one of {sorted(EXAMPLE_FORMATS)}"
        )
    return fmt


def load_examples(path: str | Path) -> pd.DataFrame:
    """Load an ``examples/`` file into (guide, label, target, locus, id).

    ``target`` is populated only for the off-target formats, which are the only
    ones carrying a DNA sequence distinct from the guide.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"example file not found: {path}")
    fmt = detect_example_format(path)

    raw = pd.read_csv(path, sep="\t", header=None)
    if raw.shape[1] != fmt.n_cols:
        raise ValueError(
            f"{path} has {raw.shape[1]} columns but format '{fmt.name}' expects "
            f"{fmt.n_cols}"
        )

    frame = pd.DataFrame({
        "guide": raw.iloc[:, fmt.guide_col].astype(str).str.upper().str.strip(),
        "label": pd.to_numeric(raw.iloc[:, fmt.label_col], errors="coerce"),
    })
    frame["target"] = (raw.iloc[:, fmt.target_col].astype(str).str.upper().str.strip()
                       if fmt.is_paired else None)
    frame["locus"] = (
        raw.iloc[:, fmt.locus_cols[0]].astype(str) + ":"
        + raw.iloc[:, fmt.locus_cols[1]].astype(str) + "-"
        + raw.iloc[:, fmt.locus_cols[2]].astype(str)
        if fmt.locus_cols else None
    )
    frame["strand"] = (raw.iloc[:, fmt.locus_cols[3]].astype(str)
                       if fmt.locus_cols else None)
    frame["id"] = raw.iloc[:, fmt.id_col].astype(str) if fmt.id_col is not None else None
    frame.attrs["format"] = fmt

    valid = frame["guide"].str.len().eq(GUIDE_LENGTH) & frame["label"].notna()
    if (~valid).any():
        get_logger().warning("dropped %d malformed row(s) from %s", int((~valid).sum()), path)
    out = frame[valid].reset_index(drop=True)
    out.attrs["format"] = fmt
    return out


_LABEL_MEANING = {
    "on_target_seq": "measured on-target efficacy",
    "on_target_reg": "measured on-target efficacy",
    "on_target_cls": "on-target class",
    "off_target_reg": "cleavage at the off-target locus",
    "off_target_cls": "off-target class",
}


def select_negative_from_examples(path: str | Path, index: int = 0,
                                  scorer=None, rank_by: str = "predicted") -> SeedGuide:
    """Take the ``index``-th weakest guide from an ``examples/`` file.

    Args:
        rank_by: ``"predicted"`` ranks every row by the frozen on-target model's
            predicted efficacy, ascending.  This is the default because the
            off-target files' own labels are all 0.0 — they score cleavage at the
            off-target locus, not on-target efficacy, so they cannot order
            candidates for this task at all.  ``"label"`` uses the file's label
            instead, which is only meaningful for the on-target formats.
    """
    frame = load_examples(path)
    fmt: ExampleFormat = frame.attrs["format"]
    logger = get_logger()

    if rank_by == "predicted":
        if scorer is None:
            raise ValueError("rank_by='predicted' needs a scorer")
        scores = scorer.score_sequences(frame["guide"].tolist()).cpu().numpy()
        pool = frame.assign(predicted=scores).sort_values("predicted").reset_index(drop=True)
        logger.info(
            "%s [%s]: %d row(s); predicted on-target efficacy %.4f-%.4f, ranked ascending",
            Path(path).name, fmt.name, len(pool),
            float(pool["predicted"].min()), float(pool["predicted"].max()),
        )
        if fmt.name.startswith("off_target"):
            logger.info(
                "file labels ignored (they score cleavage at the off-target locus, "
                "not on-target efficacy) — ranking comes from the frozen scorer"
            )
    elif rank_by == "label":
        if fmt.task == "classification":
            pool = frame[frame["label"] == 0].reset_index(drop=True)
            if pool.empty:
                raise ValueError(f"{path} contains no label-0 (negative) rows")
        else:
            pool = frame.sort_values("label").reset_index(drop=True)
            if pool["label"].nunique() == 1:
                logger.warning(
                    "every label in %s is %g, so rank_by='label' cannot order "
                    "candidates — falling back to file order (use rank_by='predicted')",
                    Path(path).name, float(pool["label"].iloc[0]),
                )
        pool = pool.assign(predicted=np.nan)
        logger.info("%s [%s]: %d candidate(s) ranked by file label",
                    Path(path).name, fmt.name, len(pool))
    else:
        raise ValueError(f"unknown rank_by={rank_by!r}; expected predicted|label")

    if index >= len(pool):
        raise IndexError(
            f"seed_guide.examples.index={index} but only {len(pool)} candidate(s) "
            f"available in {path}"
        )

    row = pool.iloc[index]
    is_ontarget = fmt.name.startswith("on_target")
    predicted = row["predicted"]

    guide = SeedGuide(
        sequence=row["guide"],
        source=f"{Path(path).name}[{fmt.name}, {rank_by} rank {index}]",
        measured_efficacy=float(row["label"]) if is_ontarget else None,
        predicted_efficacy=None if pd.isna(predicted) else float(predicted),
        locus=row["locus"],
        strand=row["strand"],
        file_label=float(row["label"]),
        file_label_meaning=_LABEL_MEANING.get(fmt.name, "unknown"),
        guide_id=row["id"],
    )
    guide.paired_target = row["target"] if fmt.is_paired else None
    if guide.paired_target:
        logger.info(
            "paired sgRNA target site %s (%d mismatch(es) from the seed guide)",
            guide.paired_target, hamming_distance(guide.sequence, guide.paired_target),
        )
    return guide


def find_pam_sites(target: str, both_strands: bool = True) -> list[tuple[str, int, str]]:
    """Every 23-mer in ``target`` ending in NGG, as (sequence, offset, strand)."""
    sites = [(m.group(1), m.start(), "+") for m in _PAM_SITE.finditer(target)]
    if both_strands:
        rc = reverse_complement(target)
        n = len(target)
        for m in _PAM_SITE.finditer(rc):
            # Map the reverse-strand hit back to a forward-strand coordinate.
            sites.append((m.group(1), n - m.start() - GUIDE_LENGTH, "-"))
    return sites


def select_worst_site_in_target(target: str, scorer, both_strands: bool = True) -> SeedGuide:
    """Score every NGG site in the target and return the lowest-scoring one."""
    sites = find_pam_sites(target, both_strands)
    if not sites:
        raise ValueError(
            "no NGG PAM site found in the target sequence; SpCas9 requires one, "
            "and the target may be too short or lack GG dinucleotides"
        )

    sequences = [s[0] for s in sites]
    with torch.no_grad():
        scores = scorer.score_sequences(sequences).cpu().numpy()
    worst = int(np.argmin(scores))
    sequence, offset, strand = sites[worst]

    get_logger().info(
        "scanned %d NGG site(s) in target; predicted efficacy range [%.4f, %.4f]",
        len(sites), float(scores.min()), float(scores.max()),
    )
    return SeedGuide(
        sequence=sequence,
        source=f"target_scan[worst of {len(sites)}]",
        predicted_efficacy=float(scores[worst]),
        strand=strand,
        target_offset=offset,
    )


def locate_in_target(guide: str, target: str) -> tuple[int | None, str | None]:
    """Find a guide in the target on either strand -> (offset, strand)."""
    idx = target.find(guide)
    if idx >= 0:
        return idx, "+"
    idx = target.find(reverse_complement(guide))
    if idx >= 0:
        return idx, "-"
    return None, None


def resolve_seed_guide(config, scorer=None, target: str | None = None) -> SeedGuide:
    """Dispatch on ``seed_guide.mode`` and annotate with target position."""
    logger = get_logger()
    mode = config.seed_guide.mode
    if target is None:
        target = clean_dna(config.target.sequence, "target.sequence")

    if mode == "dataset":
        cfg = config.seed_guide.dataset
        guide = select_low_efficacy_guide(
            cfg.path, cfg.sequence_column, cfg.label_column, cfg.percentile, cfg.index
        )
    elif mode == "examples":
        cfg = config.seed_guide.examples
        guide = select_negative_from_examples(cfg.path, cfg.index, scorer=scorer,
                                              rank_by=cfg.rank_by)
    elif mode == "target_scan":
        if scorer is None:
            raise ValueError("seed_guide.mode='target_scan' needs a scorer")
        guide = select_worst_site_in_target(target, scorer)
    elif mode == "explicit":
        sequence = config.seed_guide.sequence
        if not sequence:
            raise ValueError("seed_guide.mode='explicit' requires seed_guide.sequence")
        sequence = clean_dna(sequence, "seed_guide.sequence")
        if len(sequence) != GUIDE_LENGTH:
            raise ValueError(
                f"seed guide must be {GUIDE_LENGTH}nt (20nt spacer + 3nt PAM), "
                f"got {len(sequence)}nt"
            )
        guide = SeedGuide(sequence=sequence, source="explicit")
    else:
        raise ValueError(
            f"unknown seed_guide.mode={mode!r}; "
            "expected dataset|examples|target_scan|explicit"
        )

    # An empty target means the caller resolves it from the seed afterwards
    # (target.source == "paired"), so there is nothing to locate against yet.
    if not target:
        return guide

    if guide.target_offset is None:
        offset, strand = locate_in_target(guide.sequence, target)
        guide.target_offset = offset
        if strand:
            guide.strand = strand

    if guide.target_offset is None:
        # Not an error: the generator still conditions on the target, and with the
        # sequence-only scorer the reward never sees the target anyway.  But the
        # optimised guide will not be a drop-in edit of a real site in this region.
        logger.warning(
            "seed guide %s was not found in the target region on either strand — "
            "cross-attention will still condition on the target, but the result is "
            "not a site-matched edit of it (set seed_guide.mode='target_scan' if you "
            "want a guide that genuinely binds this target)",
            guide.sequence,
        )

    if guide.pam[1:] != "GG":
        logger.warning("seed guide PAM %r is not NGG; SpCas9 will not cut here", guide.pam)

    return guide
