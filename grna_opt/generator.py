"""Cross-attention transformer that mutates a guide against a target DNA region.

Data flow:

    seed guide (23nt) ─embed─┐
                             ├─> N x [ self-attn | cross-attn -> target | FFN ] ─> logits [23, 4]
    target DNA ─embed─self-attn─┘ (memory)

The guide stream is the query; the encoded target is the key/value memory, so
every guide position can look at every target position when deciding what base
to place.  Output is a per-position distribution over {A, C, G, T}, which the
straight-through Gumbel-softmax turns into a hard one-hot for the frozen scorer.

A learnable **soft prompt** — a short block of free parameters prepended to the
guide stream — carries task-level state that is not tied to any single position.
It participates in attention but is dropped before the output projection.

The output bias is initialised to favour the seed guide's own bases
(``identity_bias``), so step 0 emits approximately the seed and training is
genuinely a search for *selective mutations* away from it rather than generation
from scratch.  Set it to 0.0 for an unbiased random start.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoding import GUIDE_LENGTH, VOCAB_SIZE


class MultiHeadBlock(nn.Module):
    """Pre-LN block: masked-free self-attention, cross-attention, feed-forward."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                               batch_first=True)
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                memory_key_padding_mask: torch.Tensor | None = None):
        h = self.norm_self(x)
        attn, _ = self.self_attn(h, h, h, need_weights=False)
        x = x + self.dropout(attn)

        h = self.norm_cross(x)
        attn, weights = self.cross_attn(h, memory, memory,
                                        key_padding_mask=memory_key_padding_mask,
                                        need_weights=True, average_attn_weights=True)
        x = x + self.dropout(attn)

        x = x + self.dropout(self.ff(self.norm_ff(x)))
        return x, weights


class TargetEncoder(nn.Module):
    """Contextualises the target DNA into cross-attention memory."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float,
                 n_layers: int, max_length: int):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.position = nn.Embedding(max_length, d_model)
        self.max_length = max_length
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        # norm_first=True is incompatible with the nested-tensor fast path, which
        # torch would otherwise warn about on every construction.
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, target_idx: torch.Tensor) -> torch.Tensor:
        length = target_idx.shape[1]
        if length > self.max_length:
            raise ValueError(
                f"target of {length}nt exceeds generator.target_max_length="
                f"{self.max_length}; raise it or shorten the region"
            )
        positions = torch.arange(length, device=target_idx.device)
        h = self.embed(target_idx) + self.position(positions).unsqueeze(0)
        return self.norm(self.encoder(h))


class GuideOptimizer(nn.Module):
    """Generates an optimised guide conditioned on a seed guide and a target.

    Args:
        config: a :class:`~grna_opt.config.GeneratorConfig`.
        guide_length: length of the guide to emit (23 for SpCas9).
    """

    def __init__(self, config, guide_length: int = GUIDE_LENGTH):
        super().__init__()
        d_model = config.d_model
        self.guide_length = guide_length
        self.soft_prompt_length = config.soft_prompt_length

        self.guide_embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.guide_position = nn.Embedding(guide_length, d_model)

        self.target_encoder = TargetEncoder(
            d_model, config.n_heads, config.d_ff, config.dropout,
            n_layers=max(1, config.n_layers // 2), max_length=config.target_max_length,
        )

        if self.soft_prompt_length > 0:
            self.soft_prompt = nn.Parameter(torch.randn(self.soft_prompt_length, d_model) * 0.02)
        else:
            self.register_parameter("soft_prompt", None)

        self.blocks = nn.ModuleList([
            MultiHeadBlock(d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, VOCAB_SIZE)

        self.dropout = nn.Dropout(config.dropout)
        self._identity_bias_applied = False

    def apply_identity_bias(self, seed_idx: torch.Tensor, strength: float) -> None:
        """Bias the initial output toward the seed guide's own bases.

        Registers a fixed per-position additive bias of ``strength`` on the seed's
        base.  With the default strength the untrained model emits the seed with
        high probability, so optimisation starts *at* the negative guide and the
        trajectory shows which positions the model chooses to mutate.
        """
        if strength == 0:
            self.register_buffer("identity_bias", None)
            return
        bias = torch.zeros(self.guide_length, VOCAB_SIZE)
        bias[torch.arange(self.guide_length), seed_idx.reshape(-1).cpu()] = strength
        self.register_buffer("identity_bias", bias)
        self._identity_bias_applied = True

    def forward(self, seed_idx: torch.Tensor, target_idx: torch.Tensor,
                return_attention: bool = False):
        """Emit per-position logits over the nucleotide vocabulary.

        Args:
            seed_idx: ``[batch, guide_length]`` int64 indices of the seed guide.
            target_idx: ``[batch, target_length]`` int64 indices of the target DNA.

        Returns:
            ``[batch, guide_length, 4]`` logits, or ``(logits, attention)`` when
            ``return_attention`` — attention is ``[batch, guide_length, target_length]``
            from the final cross-attention layer, useful for seeing which part of
            the target drove each mutation.
        """
        batch = seed_idx.shape[0]
        memory = self.target_encoder(target_idx)

        positions = torch.arange(self.guide_length, device=seed_idx.device)
        x = self.guide_embed(seed_idx) + self.guide_position(positions).unsqueeze(0)
        x = self.dropout(x)

        if self.soft_prompt is not None:
            prompt = self.soft_prompt.unsqueeze(0).expand(batch, -1, -1)
            x = torch.cat([prompt, x], dim=1)

        weights = None
        for block in self.blocks:
            x, weights = block(x, memory)

        if self.soft_prompt is not None:
            x = x[:, self.soft_prompt_length :, :]
            if weights is not None:
                weights = weights[:, self.soft_prompt_length :, :]

        logits = self.output(self.norm_out(x))

        bias = getattr(self, "identity_bias", None)
        if bias is not None:
            logits = logits + bias.unsqueeze(0)

        return (logits, weights) if return_attention else logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
