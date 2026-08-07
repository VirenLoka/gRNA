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
``explicit``
    Use a 23-mer you name outright.
"""

from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .encoding import GUIDE_LENGTH, reverse_complement
from .logging_utils import get_logger

# 23nt window whose last three bases are a canonical NGG PAM.
_PAM_SITE = re.compile(r"(?=([ACGT]{21}GG))")
_VALID_DNA = re.compile(r"^[ACGT]+$")


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
            bits.append(f"measured={self.measured_efficacy:.4f}")
        if self.predicted_efficacy is not None:
            bits.append(f"predicted={self.predicted_efficacy:.4f}")
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


def resolve_seed_guide(config, scorer=None) -> SeedGuide:
    """Dispatch on ``seed_guide.mode`` and annotate with target position."""
    logger = get_logger()
    mode = config.seed_guide.mode
    target = clean_dna(config.target.sequence, "target.sequence")

    if mode == "dataset":
        cfg = config.seed_guide.dataset
        guide = select_low_efficacy_guide(
            cfg.path, cfg.sequence_column, cfg.label_column, cfg.percentile, cfg.index
        )
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
            f"unknown seed_guide.mode={mode!r}; expected dataset|target_scan|explicit"
        )

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
