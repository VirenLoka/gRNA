"""Nucleotide encoding shared by the scorer, the generator and the data layer.

The channel order matches ``ntmap`` in ``DeepCRISPR/deepcrispr/utils.py`` exactly
(A, C, G, T).  Getting this wrong would silently permute the scorer's input, so
every one-hot in the pipeline goes through here.
"""

from __future__ import annotations

import numpy as np
import torch

VOCAB = "ACGT"
VOCAB_SIZE = len(VOCAB)
NT_TO_IDX = {nt: i for i, nt in enumerate(VOCAB)}
IDX_TO_NT = {i: nt for nt, i in NT_TO_IDX.items()}

# DeepCRISPR is trained on 23nt = 20nt spacer + 3nt NGG PAM.
GUIDE_LENGTH = 23
SPACER_LENGTH = 20
PAM_SLICE = slice(20, 23)


def sequence_to_indices(seq: str) -> np.ndarray:
    """'ACGT' -> int64 index array. Raises on non-ACGT characters."""
    seq = seq.strip().upper()
    try:
        return np.array([NT_TO_IDX[c] for c in seq], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(
            f"sequence contains non-ACGT character {exc.args[0]!r}: {seq!r}"
        ) from None


def sequence_to_one_hot(seq: str, dtype=np.float32) -> np.ndarray:
    """'ACGT' -> [length, 4] one-hot array."""
    idx = sequence_to_indices(seq)
    out = np.zeros((len(idx), VOCAB_SIZE), dtype=dtype)
    out[np.arange(len(idx)), idx] = 1.0
    return out


def indices_to_sequence(idx) -> str:
    """Index array/tensor -> 'ACGT' string."""
    if isinstance(idx, torch.Tensor):
        idx = idx.detach().cpu().numpy()
    return "".join(IDX_TO_NT[int(i)] for i in np.asarray(idx).ravel())


def one_hot_to_sequence(one_hot) -> str:
    """[length, 4] one-hot (or soft distribution) -> 'ACGT' via argmax."""
    if isinstance(one_hot, torch.Tensor):
        one_hot = one_hot.detach().cpu().numpy()
    return indices_to_sequence(np.asarray(one_hot).argmax(axis=-1))


def to_model_input(one_hot: torch.Tensor) -> torch.Tensor:
    """[batch, length, 4] -> [batch, 4, 1, length] for :class:`DeepCRISPROnTarget`.

    Differentiable: this is only a transpose plus an unsqueeze, so gradients from
    the scorer flow straight back to the generator's relaxed one-hot.
    """
    if one_hot.dim() != 3:
        raise ValueError(f"expected [batch, length, vocab], got {tuple(one_hot.shape)}")
    return one_hot.permute(0, 2, 1).unsqueeze(2)


def hamming_distance(a: str, b: str) -> int:
    """Positional mismatch count between two equal-length sequences."""
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    return sum(x != y for x, y in zip(a, b))


def reverse_complement(seq: str) -> str:
    return seq.upper().translate(str.maketrans("ACGT", "TGCA"))[::-1]
