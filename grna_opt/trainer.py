"""Training loop: soft-prompt optimisation of a guide against a frozen scorer.

Only the generator receives updates.  DeepCRISPR is frozen but stays inside the
autograd graph, so the gradient of its predicted efficacy flows back through the
straight-through Gumbel-softmax into the generator's parameters.

Per step:

1. generator emits logits ``[1, 23, 4]`` conditioned on the seed guide + target;
2. the PAM lock is applied as an additive logit mask;
3. ``n_samples`` straight-through Gumbel samples are drawn (averaging their
   reward cuts the variance the estimator introduces);
4. the frozen scorer predicts efficacy for each sample;
5. loss = -mean efficacy + constraint penalties - entropy bonus;
6. backward, clip, Adam step.

Greedy decoding (argmax of the masked logits, no noise) is evaluated separately
as the deterministic "current answer", and the best sequence ever seen — by
either route — is tracked throughout.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .constraints import ConstraintSet
from .encoding import (GUIDE_LENGTH, hamming_distance, one_hot_to_sequence,
                       sequence_to_indices)
from .generator import GuideOptimizer
from .gumbel import anneal_temperature, distribution_entropy, gumbel_softmax
from .logging_utils import MetricHistory, get_logger


@dataclass
class OptimizationResult:
    """Everything a run produces, ready for JSON serialisation."""

    seed_sequence: str
    seed_score: float
    best_sequence: str
    best_score: float
    final_greedy_sequence: str
    final_greedy_score: float
    mutations: list[dict[str, Any]] = field(default_factory=list)
    hamming_distance: int = 0
    steps_run: int = 0
    wall_seconds: float = 0.0

    @property
    def improvement(self) -> float:
        return self.best_score - self.seed_score

    def summary_lines(self) -> list[str]:
        lines = [
            f"seed      {self.seed_sequence}  efficacy={self.seed_score:.4f}",
            f"optimised {self.best_sequence}  efficacy={self.best_score:.4f}",
            f"delta     {self.improvement:+.4f}  "
            f"({self.hamming_distance} substitution(s) from seed)",
        ]
        if self.mutations:
            changes = ", ".join(
                f"{m['from']}{m['position'] + 1}{m['to']}" for m in self.mutations
            )
            lines.append(f"mutations {changes}")
        return lines


def describe_mutations(seed: str, optimised: str) -> list[dict[str, Any]]:
    """Per-position substitutions, 0-indexed, with spacer/PAM region labels."""
    changes = []
    for i, (a, b) in enumerate(zip(seed, optimised)):
        if a != b:
            region = "PAM" if i >= 20 else ("seed_region" if i >= 10 else "PAM_distal")
            changes.append({"position": i, "from": a, "to": b, "region": region})
    return changes


class GuideTrainer:
    """Optimises one (seed guide, target DNA) pair."""

    def __init__(self, config, scorer, seed_sequence: str, target_sequence: str,
                 device: torch.device, run_dir: Path):
        self.config = config
        self.scorer = scorer
        self.device = device
        self.run_dir = Path(run_dir)
        self.logger = get_logger()

        self.seed_sequence = seed_sequence
        self.target_sequence = target_sequence

        self.seed_idx = torch.from_numpy(sequence_to_indices(seed_sequence)).unsqueeze(0).to(device)
        self.target_idx = torch.from_numpy(sequence_to_indices(target_sequence)).unsqueeze(0).to(device)
        self.seed_one_hot = F.one_hot(self.seed_idx, num_classes=4).float()

        self.generator = GuideOptimizer(config.generator, guide_length=GUIDE_LENGTH).to(device)
        self.generator.apply_identity_bias(self.seed_idx, config.generator.identity_bias)
        self.generator.to(device)

        self.constraints = ConstraintSet(
            config.constraints, GUIDE_LENGTH, seed_one_hot=self.seed_one_hot, device=device
        )

        self.optimizer = torch.optim.AdamW(
            self.generator.parameters(), lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
        self.history = MetricHistory(self.run_dir)

        with torch.no_grad():
            self.seed_score = float(self.scorer(self.seed_one_hot)[0])

        self.best_sequence = seed_sequence
        self.best_score = self.seed_score

    # ----------------------------------------------------------------- #

    def _masked_logits(self) -> torch.Tensor:
        logits = self.generator(self.seed_idx, self.target_idx)
        return self.constraints.apply_logit_mask(logits)

    @torch.no_grad()
    def greedy_decode(self) -> tuple[str, float]:
        """Deterministic argmax decode and its predicted efficacy."""
        self.generator.eval()
        logits = self._masked_logits()
        one_hot = F.one_hot(logits.argmax(dim=-1), num_classes=4).float()
        score = float(self.scorer(one_hot)[0])
        self.generator.train()
        return one_hot_to_sequence(one_hot[0]), score

    def _register(self, sequence: str, score: float) -> None:
        if score > self.best_score:
            self.best_score = score
            self.best_sequence = sequence

    # ----------------------------------------------------------------- #

    def train(self) -> OptimizationResult:
        cfg = self.config.training
        gumbel_cfg = self.config.gumbel
        started = time.time()

        self.logger.info("generator: %s trainable parameters",
                         f"{self.generator.num_parameters():,}")
        self.logger.info("seed guide %s scores %.4f under %s",
                         self.seed_sequence, self.seed_score, self.scorer.source_model)
        if not self.constraints.any_penalty_active:
            self.logger.info(
                "PAM lock is the only active constraint — the optimiser is free to "
                "drift arbitrarily far from the seed, and adversarial high-scoring "
                "sequences are expected (raise constraints.*_weight to restrain it)"
            )

        self.generator.train()
        progress = tqdm(range(cfg.steps), desc="optimising", unit="step", dynamic_ncols=True)

        for step in progress:
            tau = anneal_temperature(step, cfg.steps, gumbel_cfg.tau_start,
                                     gumbel_cfg.tau_end, gumbel_cfg.anneal)

            logits = self._masked_logits()                       # [1, 23, 4]
            batched = logits.expand(gumbel_cfg.n_samples, -1, -1)  # [S, 23, 4]
            one_hot, soft = gumbel_softmax(batched, tau=tau, hard=gumbel_cfg.hard)

            scores = self.scorer(one_hot)                        # [S]
            reward = scores.mean()

            penalty, penalty_parts = self.constraints.penalty(one_hot)
            entropy = distribution_entropy(F.softmax(logits, dim=-1))

            loss = -reward + penalty - cfg.entropy_weight * entropy

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.generator.parameters(),
                                                       cfg.grad_clip)
            self.optimizer.step()

            # Any sampled sequence is a real candidate — keep the best of them.
            best_in_batch = int(scores.argmax())
            sampled_sequence = one_hot_to_sequence(one_hot[best_in_batch])
            self._register(sampled_sequence, float(scores[best_in_batch]))

            greedy_sequence, greedy_score = (None, float("nan"))
            if step % cfg.eval_every == 0 or step == cfg.steps - 1:
                greedy_sequence, greedy_score = self.greedy_decode()
                self._register(greedy_sequence, greedy_score)

            row = {
                "step": step,
                "tau": round(tau, 5),
                "loss": float(loss),
                "reward_mean": float(reward),
                "reward_max": float(scores.max()),
                "greedy_score": greedy_score,
                "best_score": self.best_score,
                "entropy": float(entropy),
                "penalty": float(penalty),
                "grad_norm": float(grad_norm),
                "hamming": hamming_distance(self.seed_sequence, self.best_sequence),
            }
            row.update({f"penalty_{k}": v for k, v in penalty_parts.items()})
            self.history.log(**row)

            progress.set_postfix({
                "eff": f"{float(reward):.3f}",
                "best": f"{self.best_score:.3f}",
                "tau": f"{tau:.2f}",
            })

            if step % cfg.log_every == 0 or step == cfg.steps - 1:
                self.logger.info(
                    "step %5d | tau %.3f | reward %.4f | best %.4f (%s, %dnt changed) | H %.3f",
                    step, tau, float(reward), self.best_score, self.best_sequence,
                    row["hamming"], float(entropy),
                )

            if cfg.checkpoint_every and step and step % cfg.checkpoint_every == 0:
                self.save_checkpoint(step)

        progress.close()
        self.history.close()

        final_sequence, final_score = self.greedy_decode()
        self._register(final_sequence, final_score)
        self.save_checkpoint(cfg.steps, name="generator_final.pt")

        return OptimizationResult(
            seed_sequence=self.seed_sequence,
            seed_score=self.seed_score,
            best_sequence=self.best_sequence,
            best_score=self.best_score,
            final_greedy_sequence=final_sequence,
            final_greedy_score=final_score,
            mutations=describe_mutations(self.seed_sequence, self.best_sequence),
            hamming_distance=hamming_distance(self.seed_sequence, self.best_sequence),
            steps_run=cfg.steps,
            wall_seconds=time.time() - started,
        )

    def save_checkpoint(self, step: int, name: str | None = None) -> Path:
        path = self.run_dir / (name or f"generator_step{step}.pt")
        torch.save(
            {
                "step": step,
                "state_dict": self.generator.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "seed_sequence": self.seed_sequence,
                "best_sequence": self.best_sequence,
                "best_score": self.best_score,
            },
            path,
        )
        return path

    @torch.no_grad()
    def cross_attention_map(self) -> np.ndarray:
        """Final-layer cross-attention, ``[guide_length, target_length]``.

        Shows which part of the target region each guide position attended to.
        """
        self.generator.eval()
        _, weights = self.generator(self.seed_idx, self.target_idx, return_attention=True)
        self.generator.train()
        return weights[0].detach().cpu().numpy()
