"""Biological constraints on generated guides.

Per the Stage-0 decision the only constraint active by default is the **PAM
lock**.  It is enforced structurally rather than as a penalty: the logits at the
locked positions are masked to -inf everywhere except the required base, so
every sample is a valid SpCas9 site by construction and no reward is ever spent
learning to keep the PAM.

The remaining functions are soft penalties, all weighted 0.0 in the shipped
config.  They exist because optimising directly against a frozen CNN reliably
finds adversarial sequences — 23-mers that score near 1.0 and mean nothing
biologically.  Raise the corresponding weight in ``config.yaml`` to switch one
on; nothing else needs to change.
"""

from __future__ import annotations

import torch

from .encoding import NT_TO_IDX, VOCAB_SIZE

_NEG_INF = -1e9


def build_pam_mask(guide_length: int, lock_positions: list[int], base: str = "G",
                   device=None) -> torch.Tensor:
    """Additive logit mask pinning ``lock_positions`` to ``base``.

    Returns ``[guide_length, vocab_size]`` of zeros, with -inf at every base
    other than ``base`` on the locked positions.  Add it to the generator's
    logits before sampling.
    """
    mask = torch.zeros(guide_length, VOCAB_SIZE, device=device)
    keep = NT_TO_IDX[base.upper()]
    for pos in lock_positions:
        if not 0 <= pos < guide_length:
            raise ValueError(f"PAM lock position {pos} outside [0, {guide_length})")
        mask[pos] = _NEG_INF
        mask[pos, keep] = 0.0
    return mask


def edit_distance_penalty(one_hot: torch.Tensor, seed_one_hot: torch.Tensor,
                          position_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Differentiable soft Hamming distance from the seed guide.

    With a straight-through one-hot this is the exact mismatch count in the
    forward pass while staying differentiable backward.  ``position_weights``
    lets PAM-proximal seed-region mismatches cost more than PAM-distal ones.
    """
    match = (one_hot * seed_one_hot).sum(dim=-1)  # [batch, length], 1.0 where equal
    mismatch = 1.0 - match
    if position_weights is not None:
        mismatch = mismatch * position_weights.unsqueeze(0)
    return mismatch.sum(dim=-1).mean()


def seed_region_weights(guide_length: int, seed_region_start: int,
                        inside: float = 3.0, outside: float = 1.0,
                        device=None) -> torch.Tensor:
    """Position weights emphasising the PAM-proximal seed region.

    Mismatches in the ~10nt next to the PAM abolish binding far more reliably
    than PAM-distal ones, so they should be discouraged harder.
    """
    weights = torch.full((guide_length,), outside, device=device)
    weights[seed_region_start:] = inside
    return weights


def gc_content(one_hot: torch.Tensor, spacer_length: int = 20) -> torch.Tensor:
    """Differentiable GC fraction of the spacer (PAM excluded)."""
    spacer = one_hot[:, :spacer_length, :]
    gc = spacer[..., NT_TO_IDX["G"]] + spacer[..., NT_TO_IDX["C"]]
    return gc.sum(dim=-1) / spacer_length


def gc_penalty(one_hot: torch.Tensor, low: float = 0.40, high: float = 0.70,
               spacer_length: int = 20) -> torch.Tensor:
    """Hinge penalty for GC content outside the usable window.

    Guides below ~40% or above ~70% GC bind poorly or promote secondary
    structure; both ends are worth discouraging.
    """
    gc = gc_content(one_hot, spacer_length)
    return (torch.relu(low - gc) + torch.relu(gc - high)).mean()


def homopolymer_penalty(one_hot: torch.Tensor, max_run: int = 4,
                        spacer_length: int = 20) -> torch.Tensor:
    """Penalise runs of the same base longer than ``max_run``.

    A ``TTTT`` run in particular acts as a Pol III terminator and truncates
    sgRNA transcription, so it is a real failure mode rather than a style issue.
    """
    spacer = one_hot[:, :spacer_length, :]
    window = max_run + 1
    if spacer.shape[1] < window:
        return spacer.new_zeros(())

    # Product across a sliding window is 1 only where all `window` bases match.
    runs = []
    for start in range(spacer.shape[1] - window + 1):
        chunk = spacer[:, start : start + window, :]
        runs.append(chunk.prod(dim=1).sum(dim=-1))
    return torch.stack(runs, dim=-1).sum(dim=-1).mean()


class ConstraintSet:
    """Bundles the active constraints for a run."""

    def __init__(self, config, guide_length: int, seed_one_hot: torch.Tensor | None = None,
                 device=None):
        self.config = config
        self.guide_length = guide_length
        self.seed_one_hot = seed_one_hot
        self.device = device

        self.pam_mask = (
            build_pam_mask(guide_length, config.pam_lock_positions, "G", device)
            if config.lock_pam else None
        )
        self.position_weights = (
            seed_region_weights(guide_length, config.seed_region_start, device=device)
            if config.seed_region_weight > 0 else None
        )

    def apply_logit_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """Structurally enforce the PAM before sampling."""
        if self.pam_mask is None:
            return logits
        return logits + self.pam_mask.unsqueeze(0)

    def penalty(self, one_hot: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Total weighted soft penalty plus a per-term breakdown for logging."""
        cfg = self.config
        total = one_hot.new_zeros(())
        parts: dict[str, float] = {}

        if cfg.edit_distance_weight > 0 and self.seed_one_hot is not None:
            term = edit_distance_penalty(one_hot, self.seed_one_hot)
            total = total + cfg.edit_distance_weight * term
            parts["edit_distance"] = float(term)

        if cfg.seed_region_weight > 0 and self.seed_one_hot is not None:
            term = edit_distance_penalty(one_hot, self.seed_one_hot, self.position_weights)
            total = total + cfg.seed_region_weight * term
            parts["seed_region"] = float(term)

        if cfg.gc_weight > 0:
            term = gc_penalty(one_hot, cfg.gc_low, cfg.gc_high)
            total = total + cfg.gc_weight * term
            parts["gc"] = float(term)

        if cfg.homopolymer_weight > 0:
            term = homopolymer_penalty(one_hot, cfg.homopolymer_max_run)
            total = total + cfg.homopolymer_weight * term
            parts["homopolymer"] = float(term)

        return total, parts

    @property
    def any_penalty_active(self) -> bool:
        cfg = self.config
        return any(w > 0 for w in (cfg.edit_distance_weight, cfg.seed_region_weight,
                                   cfg.gc_weight, cfg.homopolymer_weight))
