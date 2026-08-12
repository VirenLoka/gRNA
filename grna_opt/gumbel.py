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
# Finite rather than -inf: an all-masked row would otherwise softmax to NaN.
_NEG_INF = -1e9


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


def constrained_gumbel_softmax(logits: torch.Tensor, automaton, tau: float = 1.0,
                               hard: bool = True, noise_scale: float = 1.0,
                               static_mask: torch.Tensor | None = None,
                               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a sequence left to right, masking illegal bases at every position.

    Context-dependent rules ("T is banned here *if* the previous three were TTT")
    cannot be written as a precomputed ``[length, vocab]`` mask, so sampling walks
    the sequence and asks the automaton which bases remain legal given the prefix
    already committed.  The result is valid by construction.

    The generator is untouched by this: it still emits all positions in a single
    parallel forward pass.  Only sampling is sequential, and it costs one small
    masking op per position with no extra model evaluations.

    Args:
        logits: ``[batch, length, vocab]`` from the generator.
        automaton: a :class:`~grna_opt.validity.ValidityAutomaton`.
        static_mask: optional ``[length, vocab]`` additive mask applied on top
            (this is where the position-wise PAM lock enters).
        noise_scale: 0.0 makes this a deterministic constrained greedy decode.

    Returns:
        ``(y, y_soft)`` stacked to ``[batch, length, vocab]``, with the same
        straight-through gradient semantics as :func:`gumbel_softmax`.
    """
    batch, length, _ = logits.shape
    automaton.reset(batch, logits.device)

    samples, softs = [], []
    for position in range(length):
        step_logits = logits[:, position, :]
        if static_mask is not None:
            step_logits = step_logits + static_mask[position].unsqueeze(0)

        legal = automaton.allowed(position)
        step_logits = step_logits.masked_fill(~legal, _NEG_INF)

        y, y_soft = gumbel_softmax(step_logits, tau=tau, hard=hard, noise_scale=noise_scale)
        automaton.advance(position, y.argmax(dim=-1))
        samples.append(y)
        softs.append(y_soft)

    return torch.stack(samples, dim=1), torch.stack(softs, dim=1)


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
