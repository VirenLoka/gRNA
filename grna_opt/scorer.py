"""Frozen, differentiable DeepCRISPR scorer.

This is the reward model.  Its parameters never receive updates, but gradients
*flow through it* into the generator — that is the whole point of the setup, and
the reason DeepCRISPR had to be ported out of TensorFlow.

Important calibration finding (measured, not assumed): although the TF source
names the regression output ``logits_l``, it is **not** a logit.  The head was
trained with MSE directly against the [0, 1] efficacy labels, so its raw output
is already an efficacy estimate in label space.  Measured against held-out rows
of the paper datasets (n=300 per cell line):

    cell line   spearman   MAE(raw)   MAE(sigmoid(raw))
    HeLa         +0.894      0.038          0.311
    HCT116       +0.889      0.035          0.327
    HL60         +0.844      0.026          0.332

Applying a sigmoid would compress the useful range to roughly [0.49, 0.73] and
flatten the gradient.  The scorer therefore returns the raw value throughout and
the trainer maximises it directly.

For the classification head the output is a post-softmax confidence, which *is*
squashed — see the README for why that head is not the default.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .deepcrispr_torch import DeepCRISPROnTarget
from .encoding import to_model_input


class DeepCRISPRScorer(nn.Module):
    """Wraps :class:`DeepCRISPROnTarget` as a frozen reward function.

    Args:
        checkpoint_path: a ``.pt`` produced by ``grna_opt.convert_checkpoint``.
        device: torch device to place the model on.
    """

    def __init__(self, checkpoint_path: str | Path, device: str | torch.device = "cpu"):
        super().__init__()
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"{checkpoint_path} not found — run "
                f"`python -m grna_opt.convert_checkpoint --model ontar_cnn_reg_seq` first"
            )
        meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        self.in_channels: int = meta["in_channels"]
        self.head: str = meta["head"]
        self.source_model: str = meta.get("source_model", checkpoint_path.stem)

        self.model = DeepCRISPROnTarget(in_channels=self.in_channels, head=self.head)
        self.model.load_state_dict(meta["state_dict"])
        self.model.to(device)
        self.freeze()

    def freeze(self) -> None:
        """Disable updates and lock inference behaviour."""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def train(self, mode: bool = True):
        """Stay in eval mode regardless of what the trainer does to the tree."""
        super().train(mode)
        self.model.eval()
        return self

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def forward(self, one_hot: torch.Tensor,
                epigenetic: torch.Tensor | None = None) -> torch.Tensor:
        """Score relaxed or hard one-hot guides.

        Args:
            one_hot: ``[batch, length, 4]``.  Straight-through Gumbel-softmax
                output is fine — it is one-hot in value and differentiable in
                gradient.
            epigenetic: ``[batch, length, 4]`` binary channels, required only by
                the 8-channel checkpoints.

        Returns:
            ``[batch]`` predicted efficacy (regression head, already in label
            space) or positive-class confidence (classification head).
        """
        x = to_model_input(one_hot)  # [batch, 4, 1, length]

        if self.in_channels == 8:
            if epigenetic is None:
                raise ValueError(
                    f"{self.source_model} is an 8-channel model and needs epigenetic "
                    "channels (CTCF, DNase, H3K4me3, RRBS); use ontar_cnn_reg_seq "
                    "for sequence-only scoring"
                )
            x = torch.cat([x, to_model_input(epigenetic)], dim=1)

        return self.model(x)

    def efficacy(self, one_hot: torch.Tensor,
                 epigenetic: torch.Tensor | None = None) -> torch.Tensor:
        """Efficacy for reporting, clamped to the [0, 1] range of the labels.

        The regression head occasionally lands slightly outside [0, 1] (observed
        range roughly -0.06 to +0.99); clamping only affects display, never the
        value the trainer optimises.
        """
        score = self.forward(one_hot, epigenetic)
        return score.clamp(0.0, 1.0) if self.head == "regression" else score

    @torch.no_grad()
    def score_sequences(self, sequences: list[str]) -> torch.Tensor:
        """Convenience path for evaluating plain strings (no gradients)."""
        from .encoding import sequence_to_one_hot
        import numpy as np

        batch = np.stack([sequence_to_one_hot(s) for s in sequences])
        one_hot = torch.from_numpy(batch).to(self.device)
        return self.forward(one_hot)
