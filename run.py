#!/usr/bin/env python3
"""Run the guide RNA optimisation engine end to end.

Trains a randomly-initialised cross-attention transformer to mutate one
low-efficacy guide into a high-efficacy one, using a frozen DeepCRISPR on-target
CNN as the reward model.

With the shipped config, each run draws a fresh seed and a different target
site from DeepCRISPR/examples/eg_reg_off_target.repiotrt (13 unique targets) —
pass --seed to reproduce a specific run, or --seed-guide/--target for a fixed
pair.

    python run.py                                  # defaults from config.yaml
    python run.py --config config.yaml
    python run.py --seed 12345                     # reproduce a prior run
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

from grna_opt.config import load_config, resolve_device, resolve_seed
from grna_opt.data import clean_dna, resolve_seed_guide
from grna_opt.encoding import GUIDE_LENGTH, hamming_distance
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
    parser.add_argument("--seed", type=int,
                        help="pin run.seed (default: null, a fresh seed drawn each run)")
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
    if args.seed is not None:
        config.run.seed = args.seed


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

    # Resolved (not just applied) before the config snapshot is written, so a
    # null seed's *actual* drawn value ends up in config.resolved.yaml and this
    # run stays reproducible even though the default varies run to run.
    seed_was_null = config.run.seed is None
    config.run.seed = resolve_seed(config.run.seed)
    set_seed(config.run.seed)
    save_config_snapshot(run_dir, config.to_dict())
    device = resolve_device(config.run.device)

    logger.info("=" * 78)
    logger.info("guide RNA optimisation engine")
    logger.info("run directory: %s", run_dir)
    logger.info("device: %s | seed: %d%s", device, config.run.seed,
                " (auto-drawn — pass --seed or set run.seed to reproduce this run)"
                if seed_was_null else "")
    logger.info("=" * 78)

    # --- frozen reward model ------------------------------------------------
    scorer = DeepCRISPRScorer(config.scorer.checkpoint, device=device)
    logger.info("scorer: %s (%d-channel, %s head) — frozen",
                scorer.source_model, scorer.in_channels, scorer.head)

    # --- target DNA and seed guide ------------------------------------------
    # With target.source == "paired" the target comes from the seed's own row in
    # an off-target example file, so the seed has to be resolved first.
    paired = config.target.source == "paired"

    if paired:
        seed_guide = resolve_seed_guide(config, scorer=scorer, target="")
        if not seed_guide.paired_target:
            logger.error(
                "target.source='paired' needs a seed from an off-target example "
                "file (.epiotrt/.repiotrt), which is the only format pairing a "
                "guide with a distinct target site; %s provides none",
                seed_guide.source,
            )
            return 1
        target = clean_dna(seed_guide.paired_target, "paired target site")
        config.target.name = f"paired:{seed_guide.guide_id or 'site'}"
        logger.info(
            "target is the sgRNA's intended site, %d mismatch(es) from the seed",
            hamming_distance(seed_guide.sequence, target),
        )
    else:
        target = clean_dna(config.target.sequence, "target.sequence")
        if PLACEHOLDER_MARKER in target:
            logger.warning(
                "using the placeholder demo target from config.yaml — replace "
                "target.sequence (or pass --target-file) with your real region"
            )
        seed_guide = resolve_seed_guide(config, scorer=scorer, target=target)

    logger.info("target '%s': %dnt", config.target.name, len(target))
    if len(target) > config.generator.target_max_length:
        logger.error(
            "target is %dnt but generator.target_max_length is %d; raise it in "
            "config.yaml or trim the region",
            len(target), config.generator.target_max_length,
        )
        return 1
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
        "seed": config.run.seed,
        "scorer": scorer.source_model,
        "target_source": config.target.source,
        "target_name": config.target.name,
        "target_sequence": target if len(target) <= 128 else None,
        "target_length": len(target),
        "seed_source": seed_guide.source,
        "seed_guide_id": seed_guide.guide_id,
        "seed_measured_ontarget_efficacy": seed_guide.measured_efficacy,
        "seed_file_label": seed_guide.file_label,
        "seed_file_label_meaning": seed_guide.file_label_meaning,
        "seed_locus": seed_guide.locus,
        "seed_target_mismatches": (
            hamming_distance(seed_guide.sequence, target) if len(target) == GUIDE_LENGTH else None
        ),
        "seed_sequence": result.seed_sequence,
        "seed_predicted_efficacy": result.seed_score,
        "best_sequence": result.best_sequence,
        "best_predicted_efficacy": result.best_score,
        "final_greedy_sequence": result.final_greedy_sequence,
        "final_greedy_efficacy": result.final_greedy_score,
        "improvement": result.improvement,
        "hamming_distance": result.hamming_distance,
        "mutations": result.mutations,
        "seed_validity_violations": result.seed_violations,
        "best_validity_violations": result.best_violations,
        "validity_gates": config.constraints.validity.__dict__,
        "steps": result.steps_run,
        "wall_seconds": result.wall_seconds,
    }
    save_summary(run_dir, summary)
    logger.info("wrote summary.json, metrics.csv, run.log and checkpoints to %s", run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
