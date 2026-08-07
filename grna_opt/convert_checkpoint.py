"""One-off conversion of DeepCRISPR TF1 checkpoints into PyTorch ``.pt`` files.

Run this once on any machine; the resulting ``.pt`` is what the training box
loads, so the GPU machine never needs TensorFlow either.

    python -m grna_opt.convert_checkpoint --model ontar_cnn_reg_seq
    python -m grna_opt.convert_checkpoint --all
"""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

import torch

from .deepcrispr_torch import build_from_checkpoint, infer_architecture
from .tf_checkpoint import find_checkpoint_prefix, read_checkpoint

ONTAR_MODELS = ["ontar_cnn_reg_seq", "ontar_pt_cnn_reg", "ontar_ptaug_cnn"]


def ensure_extracted(model_name: str, trained_models_dir: Path) -> Path:
    """Untar ``<model_name>.tar.gz`` into ``trained_models_dir`` if needed."""
    model_dir = trained_models_dir / model_name
    if model_dir.is_dir():
        return model_dir
    archive = trained_models_dir / f"{model_name}.tar.gz"
    if not archive.exists():
        raise FileNotFoundError(f"neither {model_dir} nor {archive} exists")
    with tarfile.open(archive) as tar:
        tar.extractall(trained_models_dir)
    if not model_dir.is_dir():
        raise RuntimeError(f"{archive} did not contain a {model_name}/ directory")
    return model_dir


def convert(model_name: str, trained_models_dir: Path, output_dir: Path) -> Path:
    model_dir = ensure_extracted(model_name, trained_models_dir)
    prefix = find_checkpoint_prefix(model_dir)
    tensors = read_checkpoint(prefix)

    in_channels, head = infer_architecture(tensors)
    model = build_from_checkpoint(tensors)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{model_name}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "in_channels": in_channels,
            "head": head,
            "source_model": model_name,
            "source_checkpoint": prefix.name,
        },
        out_path,
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=ONTAR_MODELS, help="single model to convert")
    parser.add_argument("--all", action="store_true", help="convert all on-target models")
    parser.add_argument("--trained-models-dir", type=Path,
                        default=Path("DeepCRISPR/trained_models"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints"))
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("pass --model NAME or --all")

    targets = ONTAR_MODELS if args.all else [args.model]
    for name in targets:
        path = convert(name, args.trained_models_dir, args.output_dir)
        meta = torch.load(path, map_location="cpu", weights_only=False)
        print(f"{name:20s} -> {path}  "
              f"(in_channels={meta['in_channels']}, head={meta['head']}, "
              f"{sum(t.numel() for t in meta['state_dict'].values()):,} params)")


if __name__ == "__main__":
    main()
