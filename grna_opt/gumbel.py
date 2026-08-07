"""Straight-through Gumbel-softmax over the nucleotide vocabulary.

The engine needs a sequence that is *discrete* for the scorer (DeepCRISPR was
trained on hard one-hot 23-mers, and feeding it a blurry simplex point would put
it off-distribution) but *differentiable* for the generator.  The
straight-through estimator gives both: the forward value is an exact one-hot,
while the backward pass pretends the soft relaxation was used.

    y_soft = softmax((logits + gumbel_noise) / tau)
    y_hard = onehot(argmax(y_soft))
    y      = (y_hard - y_soft).detach() + y_soft

``y`` equals ``y_hard`` numerically and carries ``y_soft``'s gradient.

Temperature is annealed over training: high tau early keeps the relaxation smooth
and the gradient well-behaved while the generator is random; low tau late makes
the sample sharp so the surrogate gradient matches what the scorer actually saw.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_EPS = 1e-10


def sample_gumbel(shape, device, dtype=torch.float32) -> torch.Tensor:
    """Gumbel(0, 1) noise via inverse transform sampling."""
    u = torch.rand(shape, device=device, dtype=dtype)
    return -torch.log(-torch.log(u + _EPS) + _EPS)


def gumbel_softmax(logits: torch.Tensor, tau: float = 1.0, hard: bool = True,
                   noise_scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a relaxed one-hot from ``logits``.

    Args:
        logits: ``[..., vocab_size]`` unnormalised scores.
        tau: softmax temperature; lower is sharper.
        hard: apply the straight-through discretisation.  When False the raw
            soft sample is returned, which is differentiable but off-distribution
            for the scorer.
        noise_scale: multiplier on the Gumbel noise.  1.0 is true Gumbel-softmax
            sampling.  Below 1.0 the sample is pulled toward the argmax — less
            exploration, more exploitation of the currently-preferred base.  0.0
            is a deterministic softmax with no sampling at all.

    Returns:
        ``(y, y_soft)`` — the (possibly straight-through) sample, and the
        underlying soft distribution for entropy/diagnostic use.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")

    if noise_scale > 0:
        noise = sample_gumbel(logits.shape, logits.device, logits.dtype) * noise_scale
        perturbed = logits + noise
    else:
        perturbed = logits

    y_soft = F.softmax(perturbed / tau, dim=-1)
    if not hard:
        return y_soft, y_soft

    index = y_soft.argmax(dim=-1, keepdim=True)
    y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
    y = (y_hard - y_soft).detach() + y_soft
    return y, y_soft


def anneal_temperature(step: int, total_steps: int, tau_start: float, tau_end: float,
                       schedule: str = "exponential") -> float:
    """Temperature for a given training step."""
    if schedule == "none" or total_steps <= 1:
        return tau_start

    progress = min(max(step / (total_steps - 1), 0.0), 1.0)
    if schedule == "linear":
        return tau_start + (tau_end - tau_start) * progress
    if schedule == "exponential":
        if tau_start <= 0 or tau_end <= 0:
            raise ValueError("exponential annealing needs positive tau_start and tau_end")
        return tau_start * math.exp(math.log(tau_end / tau_start) * progress)
    raise ValueError(f"unknown anneal schedule {schedule!r}; expected linear|exponential|none")


def distribution_entropy(probs: torch.Tensor) -> torch.Tensor:
    """Mean per-position entropy in nats — a collapse detector.

    Falls toward 0 as the generator commits to one base per position.  The
    trainer adds this to the objective with a small weight so the distribution
    does not collapse before the scorer has taught it anything.
    """
    return -(probs * torch.log(probs + _EPS)).sum(dim=-1).mean()
