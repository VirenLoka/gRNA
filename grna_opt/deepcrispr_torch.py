"""Differentiable PyTorch port of the DeepCRISPR on-target CNN.

The original model (``DeepCRISPR/deepcrispr/deepcrispr_src.py``) is TensorFlow 1.3
plus Sonnet 1.9.  Gradients cannot cross from a TF1 graph into a PyTorch module —
there is no shared autograd tape — so the optimisation engine needs the scorer
living inside PyTorch.  This module reproduces ``build_ontar_model`` and
``build_ontar_reg_model`` layer for layer and loads the converted TF weights.

Fidelity notes, all verified against the checkpoints:

* Every BatchNorm in the original is invoked as ``bn(x, False, test_local_stats=False)``
  — i.e. permanently in inference mode using the stored moving statistics.  The
  port therefore has no training-mode BN path at all; it is deterministic.
* The encoder BNs (``ebn_1u``..``ebn_5u``) were built with ``offset=False`` and
  Sonnet's default ``scale=False``, so they carry neither beta nor gamma.  A
  separate learned ``beta_i`` bias is added *after* normalisation and before the
  ReLU.  The head BNs (``ebn_6l``..``ebn_8l``) keep Sonnet's default
  ``offset=True``, so they have beta but still no gamma.
* ``snt.Conv2D`` defaults to ``SAME`` padding, which for the stride-2 layers at
  widths 12->6 and 6->3 pads *asymmetrically* (0 left, 1 right).  Using
  ``padding=1`` in PyTorch would silently shift the feature map, so TF's padding
  rule is reimplemented explicitly.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Sonnet 1.9 BatchNorm default.  The stored moving variances here are O(1)-O(10^3),
# so the exact value is numerically irrelevant, but keep it faithful anyway.
SONNET_BN_EPS = 1e-5

ENCODER_CHANNELS = [32, 64, 64, 256, 256]
ENCODER_STRIDES = [1, 2, 1, 2, 1]
HEAD_CHANNELS = [512, 512, 1024]
HEAD_STRIDES = [2, 1, 1]


def _tf_same_padding(width: int, kernel: int, stride: int) -> tuple[int, int]:
    """TensorFlow's SAME padding split, which favours the right edge."""
    out_width = -(-width // stride)  # ceil division
    total = max((out_width - 1) * stride + kernel - width, 0)
    left = total // 2
    return left, total - left


class SameConv1xK(nn.Module):
    """Conv2D with kernel [1, k] and TensorFlow-compatible SAME/VALID padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3,
                 stride: int = 1, padding: str = "SAME"):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel),
                              stride=(1, stride), padding=0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.padding == "SAME":
            left, right = _tf_same_padding(x.shape[-1], self.kernel, self.stride)
            if left or right:
                x = F.pad(x, (left, right))
        return self.conv(x)


class InferenceBatchNorm(nn.Module):
    """Sonnet BatchNorm frozen in inference mode: normalise by moving statistics.

    ``offset`` mirrors Sonnet's flag — when False no beta is learned.  Sonnet's
    default ``scale=False`` means gamma never exists, matching the checkpoints.
    """

    def __init__(self, num_features: int, offset: bool = True, eps: float = SONNET_BN_EPS):
        super().__init__()
        self.eps = eps
        self.register_buffer("moving_mean", torch.zeros(num_features))
        self.register_buffer("moving_variance", torch.ones(num_features))
        if offset:
            self.register_buffer("beta", torch.zeros(num_features))
        else:
            self.beta = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = (1, -1, 1, 1)
        mean = self.moving_mean.view(shape)
        var = self.moving_variance.view(shape)
        out = (x - mean) * torch.rsqrt(var + self.eps)
        if self.beta is not None:
            out = out + self.beta.view(shape)
        return out


class DeepCRISPROnTarget(nn.Module):
    """On-target efficacy CNN.

    Args:
        in_channels: 4 for the sequence-only model, 8 when epigenetic channels
            are concatenated.
        head: ``"regression"`` returns the raw scalar logit exactly as
            ``build_ontar_reg_model`` does (no activation — this is the signal
            with the cleanest gradients).  ``"classification"`` reproduces
            ``build_ontar_model``: softmax over 2 logits, taking class 1.

    Input is ``[batch, in_channels, 1, 23]`` (NCHW), the PyTorch transpose of the
    ``[batch, 1, 23, channels]`` NHWC tensor the TF placeholder expects.
    """

    def __init__(self, in_channels: int = 4, head: str = "regression"):
        super().__init__()
        if head not in ("regression", "classification"):
            raise ValueError(f"head must be 'regression' or 'classification', got {head!r}")
        self.in_channels = in_channels
        self.head = head

        encoder = []
        encoder_bn = []
        betas = []
        channels = in_channels
        for out_channels, stride in zip(ENCODER_CHANNELS, ENCODER_STRIDES):
            encoder.append(SameConv1xK(channels, out_channels, 3, stride, "SAME"))
            encoder_bn.append(InferenceBatchNorm(out_channels, offset=False))
            betas.append(nn.Parameter(torch.zeros(out_channels)))
            channels = out_channels
        self.encoder = nn.ModuleList(encoder)
        self.encoder_bn = nn.ModuleList(encoder_bn)
        self.betas = nn.ParameterList(betas)

        head_layers = []
        head_bn = []
        for i, (out_channels, stride) in enumerate(zip(HEAD_CHANNELS, HEAD_STRIDES)):
            # e_8 is the only VALID-padded layer; it collapses width 3 -> 1.
            padding = "VALID" if i == len(HEAD_CHANNELS) - 1 else "SAME"
            head_layers.append(SameConv1xK(channels, out_channels, 3, stride, padding))
            head_bn.append(InferenceBatchNorm(out_channels, offset=True))
            channels = out_channels
        self.head_layers = nn.ModuleList(head_layers)
        self.head_bn = nn.ModuleList(head_bn)

        out_dim = 1 if head == "regression" else 2
        self.output = SameConv1xK(channels, out_dim, kernel=1, stride=1, padding="SAME")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a batch of one-hot encoded 23-mers.

        Returns a ``[batch]`` tensor: the raw regression logit, or the
        classification confidence for the positive class.
        """
        if x.dim() != 4:
            raise ValueError(f"expected [batch, channels, 1, length], got {tuple(x.shape)}")

        for conv, bn, beta in zip(self.encoder, self.encoder_bn, self.betas):
            x = F.relu(bn(conv(x)) + beta.view(1, -1, 1, 1))

        for conv, bn in zip(self.head_layers, self.head_bn):
            x = F.relu(bn(conv(x)))

        x = self.output(x)  # [batch, out_dim, 1, 1]
        x = x.squeeze(-1).squeeze(-1)  # [batch, out_dim]

        if self.head == "regression":
            return x.squeeze(-1)
        return F.softmax(x, dim=-1)[:, 1]


# --------------------------------------------------------------------------- #
# TensorFlow -> PyTorch weight transfer
# --------------------------------------------------------------------------- #

def _tf_conv_to_torch(w):
    """TF conv kernel [kh, kw, in, out] -> PyTorch [out, in, kh, kw]."""
    return torch.from_numpy(w).permute(3, 2, 0, 1).contiguous()


def load_tf_weights(model: DeepCRISPROnTarget, tensors: dict) -> DeepCRISPROnTarget:
    """Copy weights from a parsed DeepCRISPR checkpoint into the port.

    ``DCModelOntar`` restores with ``{v.op.name[6:]: v}``, stripping the ``ontar/``
    scope, so checkpoint keys are bare: ``e_1/w``, ``ebn_1u/moving_mean``, ``beta_1``.
    """
    missing = []

    def take(key: str):
        if key not in tensors:
            missing.append(key)
            return None
        return tensors[key]

    with torch.no_grad():
        # Encoder: e_1..e_5 / ebn_1u..ebn_5u / beta_1..beta_5
        for i in range(len(ENCODER_CHANNELS)):
            n = i + 1
            w, b = take(f"e_{n}/w"), take(f"e_{n}/b")
            if w is not None:
                model.encoder[i].conv.weight.copy_(_tf_conv_to_torch(w))
            if b is not None:
                model.encoder[i].conv.bias.copy_(torch.from_numpy(b))
            mean, var = take(f"ebn_{n}u/moving_mean"), take(f"ebn_{n}u/moving_variance")
            if mean is not None:
                model.encoder_bn[i].moving_mean.copy_(torch.from_numpy(mean).reshape(-1))
            if var is not None:
                model.encoder_bn[i].moving_variance.copy_(torch.from_numpy(var).reshape(-1))
            beta = take(f"beta_{n}")
            if beta is not None:
                model.betas[i].copy_(torch.from_numpy(beta))

        # Head: e_6..e_8 / ebn_6l..ebn_8l
        for i in range(len(HEAD_CHANNELS)):
            n = i + 6
            w, b = take(f"e_{n}/w"), take(f"e_{n}/b")
            if w is not None:
                model.head_layers[i].conv.weight.copy_(_tf_conv_to_torch(w))
            if b is not None:
                model.head_layers[i].conv.bias.copy_(torch.from_numpy(b))
            mean, var = take(f"ebn_{n}l/moving_mean"), take(f"ebn_{n}l/moving_variance")
            beta = take(f"ebn_{n}l/beta")
            if mean is not None:
                model.head_bn[i].moving_mean.copy_(torch.from_numpy(mean).reshape(-1))
            if var is not None:
                model.head_bn[i].moving_variance.copy_(torch.from_numpy(var).reshape(-1))
            if beta is not None:
                model.head_bn[i].beta.copy_(torch.from_numpy(beta).reshape(-1))

        # Output projection e_9
        w, b = take("e_9/w"), take("e_9/b")
        if w is not None:
            model.output.conv.weight.copy_(_tf_conv_to_torch(w))
        if b is not None:
            model.output.conv.bias.copy_(torch.from_numpy(b))

    if missing:
        raise KeyError(f"checkpoint is missing expected variables: {missing}")
    return model


def infer_architecture(tensors: dict) -> tuple[int, str]:
    """Read input channel count and head type straight off the checkpoint shapes."""
    in_channels = int(tensors["e_1/w"].shape[2])
    out_dim = int(tensors["e_9/w"].shape[3])
    head = "regression" if out_dim == 1 else "classification"
    return in_channels, head


def build_from_checkpoint(tensors: dict) -> DeepCRISPROnTarget:
    """Instantiate the port matching a checkpoint and load its weights."""
    in_channels, head = infer_architecture(tensors)
    model = DeepCRISPROnTarget(in_channels=in_channels, head=head)
    return load_tf_weights(model, tensors)
