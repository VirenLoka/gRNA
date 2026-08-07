"""Pure-python reader for TensorFlow v2 checkpoint bundles (`.index` + `.data-*`).

DeepCRISPR ships TF 1.3 / Sonnet 1.9 checkpoints, which cannot be loaded by any
TensorFlow that installs on a modern Python.  Rather than pin an ancient
interpreter, this module parses the checkpoint format directly:

  * `.index` is a ``tensorflow::table`` (an SSTable / LevelDB-style block table)
    mapping tensor name -> serialised ``BundleEntryProto``.
  * `.data-00000-of-00001` holds the raw little-endian tensor payloads at the
    byte offsets named by those entries.

Only the subset of the format the DeepCRISPR bundles actually use is handled:
single shard, uncompressed blocks, dense (unsliced) tensors.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# TF DataType enum -> numpy dtype, restricted to what these checkpoints contain.
_TF_DTYPES = {
    1: np.float32,
    2: np.float64,
    3: np.int32,
    9: np.int64,
    10: np.bool_,
}

_FOOTER_LEN = 48
_MAGIC = bytes.fromhex("57fb808b247547db")  # tensorflow::table magic, little-endian


def _read_varint(buf: bytes, pos: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _read_block_handle(buf: bytes, pos: int) -> tuple[int, int, int]:
    offset, pos = _read_varint(buf, pos)
    size, pos = _read_varint(buf, pos)
    return offset, size, pos


def _iter_block_entries(block: bytes):
    """Yield (key, value) from an SSTable data block with prefix-compressed keys."""
    num_restarts = struct.unpack("<I", block[-4:])[0]
    end = len(block) - 4 - 4 * num_restarts
    pos = 0
    key = b""
    while pos < end:
        shared, pos = _read_varint(block, pos)
        non_shared, pos = _read_varint(block, pos)
        value_len, pos = _read_varint(block, pos)
        key = key[:shared] + block[pos : pos + non_shared]
        pos += non_shared
        value = block[pos : pos + value_len]
        pos += value_len
        yield key, value


def _parse_bundle_entry(buf: bytes) -> tuple[int | None, list[int], int, int]:
    """Decode BundleEntryProto: 1=dtype, 2=shape, 3=shard_id, 4=offset, 5=size."""
    pos = 0
    dtype: int | None = None
    shape: list[int] = []
    offset = 0
    size = 0
    while pos < len(buf):
        tag, pos = _read_varint(buf, pos)
        field, wire = tag >> 3, tag & 7
        if wire == 0:  # varint
            value, pos = _read_varint(buf, pos)
            if field == 1:
                dtype = value
            elif field == 4:
                offset = value
            elif field == 5:
                size = value
        elif wire == 2:  # length-delimited
            length, pos = _read_varint(buf, pos)
            payload = buf[pos : pos + length]
            pos += length
            if field == 2:
                shape = _parse_tensor_shape(payload)
        elif wire == 5:  # fixed32 (crc32c)
            pos += 4
        elif wire == 1:  # fixed64
            pos += 8
        else:
            break
    return dtype, shape, offset, size


def _parse_tensor_shape(buf: bytes) -> list[int]:
    """TensorShapeProto { repeated Dim dim = 2 { int64 size = 1 } }."""
    dims: list[int] = []
    pos = 0
    while pos < len(buf):
        tag, pos = _read_varint(buf, pos)
        if (tag >> 3) != 2 or (tag & 7) != 2:
            break
        length, pos = _read_varint(buf, pos)
        dim = buf[pos : pos + length]
        pos += length
        inner = 0
        while inner < len(dim):
            dim_tag, inner = _read_varint(dim, inner)
            if (dim_tag >> 3) == 1 and (dim_tag & 7) == 0:
                size, inner = _read_varint(dim, inner)
                dims.append(size)
            else:
                break
    return dims


def read_checkpoint(prefix: str | Path) -> dict[str, np.ndarray]:
    """Load every dense tensor from a TF v2 checkpoint.

    Args:
        prefix: checkpoint path without the ``.index``/``.data-*`` suffix,
            e.g. ``trained_models/ontar_cnn_reg_seq/model.ckpt-seq``.

    Returns:
        Mapping of variable name to numpy array.
    """
    prefix = Path(prefix)
    index_bytes = (prefix.parent / f"{prefix.name}.index").read_bytes()

    footer = index_bytes[-_FOOTER_LEN:]
    if footer[-8:] != _MAGIC:
        raise ValueError(f"{prefix}.index is not a tensorflow::table (bad magic)")

    pos = 0
    _, _, pos = _read_block_handle(footer, pos)  # metaindex, unused
    index_off, index_size, pos = _read_block_handle(footer, pos)

    entries: dict[str, tuple] = {}
    index_block = index_bytes[index_off : index_off + index_size]
    for _, handle in _iter_block_entries(index_block):
        data_off, data_size, _ = _read_block_handle(handle, 0)
        data_block = index_bytes[data_off : data_off + data_size]
        for key, value in _iter_block_entries(data_block):
            if key:  # the empty key holds BundleHeaderProto, not a tensor
                entries[key.decode()] = _parse_bundle_entry(value)

    payload = (prefix.parent / f"{prefix.name}.data-00000-of-00001").read_bytes()

    tensors: dict[str, np.ndarray] = {}
    for name, (dtype, shape, offset, size) in entries.items():
        np_dtype = _TF_DTYPES.get(dtype)
        if np_dtype is None:
            continue  # strings / unsupported types: not present in these bundles
        count = size // np.dtype(np_dtype).itemsize
        flat = np.frombuffer(payload, dtype=np_dtype, count=count, offset=offset)
        tensors[name] = flat.reshape(shape).copy() if shape else flat.reshape(()).copy()
    return tensors


def find_checkpoint_prefix(model_dir: str | Path) -> Path:
    """Resolve the checkpoint prefix inside a model directory.

    Prefers the name recorded in the ``checkpoint`` file, falling back to the
    single ``*.index`` present.
    """
    model_dir = Path(model_dir)
    marker = model_dir / "checkpoint"
    if marker.exists():
        for line in marker.read_text().splitlines():
            if line.startswith("model_checkpoint_path:"):
                name = line.split(":", 1)[1].strip().strip('"')
                candidate = model_dir / Path(name).name
                if (model_dir / f"{candidate.name}.index").exists():
                    return candidate
    indexes = sorted(model_dir.glob("*.index"))
    if len(indexes) != 1:
        raise FileNotFoundError(
            f"expected exactly one *.index in {model_dir}, found {len(indexes)}"
        )
    return indexes[0].with_suffix("")
