#!/usr/bin/env python3
"""Run the guide RNA optimisation engine end to end.

Trains a randomly-initialised cross-attention transformer to mutate one
low-efficacy guide into a high-efficacy one, using a frozen DeepCRISPR on-target
CNN as the reward model.

    python run.py                                  # defaults from config.yaml
    python run.py --config config.yaml
    python run.py --target-file my_region.fa       # override the target DNA
    python run.py --seed-guide TGGTTCTATACTCAGGAGCCAGG
    python run.py --steps 500 --device cpu
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

from grna_opt.config import load_config, resolve_device
from grna_opt.data import clean_dna, resolve_seed_guide
from grna_opt.encoding import GUIDE_LENGTH
from grna_opt.logging_utils import (create_run_dir, save_config_snapshot,
                                    save_summary, setup_logging)
from grna_opt.scorer import DeepCRISPRScorer
from grna_opt.trainer import GuideTrainer

PLACEHOLDER_MARKER = "AAAGGCTGAGCACGCCGGTGGTCCATCTCTACATCCTTTATCTCC"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--target-file", type=Path,
                        help="file holding the target DNA (plain or FASTA)")
    parser.add_argument("--target", type=str, help="target DNA inline")
    parser.add_argument("--seed-guide", type=str,
                        help=f"explicit {GUIDE_LENGTH}nt seed guide (spacer + PAM)")
    parser.add_argument("--steps", type=int, help="override training.steps")
    parser.add_argument("--device", type=str, help="override run.device")
    parser.add_argument("--run-name", type=str, help="override run.name")
    return parser.parse_args()


def apply_overrides(config, args: argparse.Namespace) -> None:
    if args.target_file:
        config.target.sequence = args.target_file.read_text()
        config.target.name = args.target_file.stem
    if args.target:
        config.target.sequence = args.target
        config.target.name = "cli_target"
    if args.seed_guide:
        config.seed_guide.mode = "explicit"
        config.seed_guide.sequence = args.seed_guide
    if args.steps is not None:
        config.training.steps = args.steps
    if args.device:
        config.run.device = args.device
    if args.run_name:
        config.run.name = args.run_name


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)

    run_dir = create_run_dir(config.run.output_dir, config.run.name)
    logger = setup_logging(run_dir, config.run.log_level)
    save_config_snapshot(run_dir, config.to_dict())

    set_seed(config.run.seed)
    device = resolve_device(config.run.device)

    logger.info("=" * 78)
    logger.info("guide RNA optimisation engine")
    logger.info("run directory: %s", run_dir)
    logger.info("device: %s | seed: %d", device, config.run.seed)
    logger.info("=" * 78)

    # --- frozen reward model ------------------------------------------------
    scorer = DeepCRISPRScorer(config.scorer.checkpoint, device=device)
    logger.info("scorer: %s (%d-channel, %s head) — frozen",
                scorer.source_model, scorer.in_channels, scorer.head)

    # --- target DNA ---------------------------------------------------------
    target = clean_dna(config.target.sequence, "target.sequence")
    if PLACEHOLDER_MARKER in target:
        logger.warning(
            "using the placeholder demo target from config.yaml — replace "
            "target.sequence (or pass --target-file) with your real region"
        )
    logger.info("target '%s': %dnt", config.target.name, len(target))
    if len(target) > config.generator.target_max_length:
        logger.error(
            "target is %dnt but generator.target_max_length is %d; raise it in "
            "config.yaml or trim the region",
            len(target), config.generator.target_max_length,
        )
        return 1

    # --- seed guide ---------------------------------------------------------
    seed_guide = resolve_seed_guide(config, scorer=scorer)
    logger.info("seed guide: %s", seed_guide.describe())

    # --- optimise -----------------------------------------------------------
    trainer = GuideTrainer(
        config=config, scorer=scorer, seed_sequence=seed_guide.sequence,
        target_sequence=target, device=device, run_dir=run_dir,
    )
    result = trainer.train()

    # --- report -------------------------------------------------------------
    logger.info("=" * 78)
    logger.info("optimisation complete in %.1fs (%d steps)",
                result.wall_seconds, result.steps_run)
    for line in result.summary_lines():
        logger.info(line)
    logger.info("=" * 78)

    if result.improvement <= 0:
        logger.warning(
            "no improvement over the seed — try more steps, a higher lr, or a "
            "lower gumbel.tau_end"
        )

    summary = {
        "run_dir": str(run_dir),
        "device": str(device),
        "scorer": scorer.source_model,
        "target_name": config.target.name,
        "target_length": len(target),
        "seed_source": seed_guide.source,
        "seed_measured_efficacy": seed_guide.measured_efficacy,
        "seed_locus": seed_guide.locus,
        "seed_sequence": result.seed_sequence,
        "seed_predicted_efficacy": result.seed_score,
        "best_sequence": result.best_sequence,
        "best_predicted_efficacy": result.best_score,
        "final_greedy_sequence": result.final_greedy_sequence,
        "final_greedy_efficacy": result.final_greedy_score,
        "improvement": result.improvement,
        "hamming_distance": result.hamming_distance,
        "mutations": result.mutations,
        "steps": result.steps_run,
        "wall_seconds": result.wall_seconds,
    }
    save_summary(run_dir, summary)
    logger.info("wrote summary.json, metrics.csv, run.log and checkpoints to %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
