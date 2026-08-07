"""Typed configuration loaded from ``config.yaml``.

Nested dataclasses rather than raw dicts so a typo in the YAML fails loudly at
load time instead of silently taking a default halfway through a training run.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class RunConfig:
    name: str | None = None            # auto-timestamped when null
    output_dir: str = "runs"
    seed: int = 42
    device: str = "auto"               # auto | cpu | cuda | mps
    log_level: str = "INFO"


@dataclass
class ScorerConfig:
    checkpoint: str = "checkpoints/ontar_cnn_reg_seq.pt"


@dataclass
class TargetConfig:
    # config -> use `sequence` below
    # paired -> use the sgRNA target site paired with the seed guide, which only
    #           the off-target example formats provide
    source: str = "config"
    name: str = "user_target"
    sequence: str = ""


@dataclass
class DatasetSeedConfig:
    path: str = "DeepCRISPR/paper_data2/ontar/hela.repisgt"
    sequence_column: int = 4
    label_column: int = 9
    percentile: float = 5.0            # bottom N% by measured efficacy
    index: int = 0                     # which guide within that pool, ascending


@dataclass
class ExamplesSeedConfig:
    path: str = "DeepCRISPR/examples/eg_reg_off_target.repiotrt"
    index: int = 0                     # 0 = weakest candidate
    # predicted -> rank by the frozen on-target model (required for off-target
    #              files, whose labels are all 0.0 and mean something else)
    # label     -> rank by the file's own label
    rank_by: str = "predicted"


@dataclass
class SeedGuideConfig:
    mode: str = "dataset"              # dataset | examples | target_scan | explicit
    sequence: str | None = None        # used when mode == "explicit"
    dataset: DatasetSeedConfig = field(default_factory=DatasetSeedConfig)
    examples: ExamplesSeedConfig = field(default_factory=ExamplesSeedConfig)


@dataclass
class GeneratorConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    soft_prompt_length: int = 8        # 0 disables the learnable prefix
    target_max_length: int = 512
    # Biases the untrained output toward the seed guide's own bases so step 0
    # emits (approximately) the seed and training searches for selective
    # mutations away from it.  0.0 gives a fully unbiased random start.
    identity_bias: float = 3.0


@dataclass
class GumbelConfig:
    hard: bool = True                  # straight-through
    tau_start: float = 2.0
    tau_end: float = 0.5
    anneal: str = "exponential"        # linear | exponential | none
    n_samples: int = 4                 # samples per step; reward averaged


@dataclass
class ConstraintConfig:
    lock_pam: bool = True
    pam_lock_positions: list[int] = field(default_factory=lambda: [21, 22])
    # Guards left off by default per the Stage-0 decision (PAM lock only).
    # Raise any weight above 0 to switch that penalty on.
    edit_distance_weight: float = 0.0
    seed_region_weight: float = 0.0
    seed_region_start: int = 10        # PAM-proximal seed, positions 11-20
    gc_weight: float = 0.0
    gc_low: float = 0.40
    gc_high: float = 0.70
    homopolymer_weight: float = 0.0
    homopolymer_max_run: int = 4


@dataclass
class TrainingConfig:
    steps: int = 2000
    lr: float = 3.0e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    entropy_weight: float = 0.01       # keeps the distribution from collapsing early
    log_every: int = 25
    eval_every: int = 100
    checkpoint_every: int = 500


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    scorer: ScorerConfig = field(default_factory=ScorerConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    seed_guide: SeedGuideConfig = field(default_factory=SeedGuideConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    gumbel: GumbelConfig = field(default_factory=GumbelConfig)
    constraints: ConstraintConfig = field(default_factory=ConstraintConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(cls, data: Any, path: str):
    """Recursively instantiate a dataclass tree, rejecting unknown keys."""
    if not dataclasses.is_dataclass(cls):
        return data
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"config section '{path}' must be a mapping, got {type(data).__name__}")

    # `from __future__ import annotations` stores field types as strings, so
    # resolve them before testing for nested dataclasses.
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(
            f"unknown key(s) {sorted(unknown)} in config section '{path or 'root'}'; "
            f"valid keys are {sorted(known)}"
        )

    kwargs = {}
    for name, value in data.items():
        field_type = hints[name]
        child_path = f"{path}.{name}" if path else name
        if dataclasses.is_dataclass(field_type):
            kwargs[name] = _build(field_type, value, child_path)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    """Read and validate ``config.yaml``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open() as handle:
        raw = yaml.safe_load(handle) or {}
    return _build(Config, raw, "")


def resolve_device(name: str = "auto"):
    """Map ``run.device`` onto a concrete torch device."""
    import torch

    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
