"""Run directory setup, logging, and metric history.

Every run gets its own directory holding the resolved config, a full log, and a
CSV of per-step metrics, so a run on the GPU box is reproducible and auditable
after the fact.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from tqdm.auto import tqdm

LOGGER_NAME = "grna_opt"


class TqdmLoggingHandler(logging.Handler):
    """Console handler that writes via ``tqdm.write``.

    A plain StreamHandler would print straight to stderr and tear the progress
    bar apart on every log line; routing through tqdm keeps the bar pinned to the
    bottom while log records scroll above it.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stdout)
        except Exception:  # pragma: no cover - never let logging kill a run
            self.handleError(record)


def create_run_dir(output_dir: str | Path, name: str | None = None) -> Path:
    """Make ``<output_dir>/<name>``, timestamping the name when not supplied."""
    if not name:
        name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def setup_logging(run_dir: Path, level: str = "INFO") -> logging.Logger:
    """Configure the package logger to write to both stdout and ``run.log``."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    console = TqdmLoggingHandler()
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s",
                                           datefmt="%H:%M:%S"))
    logger.addHandler(console)

    file_handler = logging.FileHandler(run_dir / "run.log", mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    logger.addHandler(file_handler)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def save_config_snapshot(run_dir: Path, config_dict: dict[str, Any]) -> None:
    """Persist the fully resolved config next to the results."""
    with (run_dir / "config.resolved.yaml").open("w") as handle:
        yaml.safe_dump(config_dict, handle, sort_keys=False, default_flow_style=False)


class MetricHistory:
    """Append-only per-step metrics, streamed to CSV and kept in memory."""

    def __init__(self, run_dir: Path, filename: str = "metrics.csv"):
        self.path = Path(run_dir) / filename
        self.rows: list[dict[str, Any]] = []
        self._writer: csv.DictWriter | None = None
        self._handle = None

    def log(self, **metrics: Any) -> None:
        self.rows.append(metrics)
        if self._writer is None:
            self._handle = self.path.open("w", newline="")
            self._writer = csv.DictWriter(self._handle, fieldnames=list(metrics))
            self._writer.writeheader()
        self._writer.writerow(metrics)
        self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None


def save_summary(run_dir: str | Path, summary: dict[str, Any]) -> Path:
    """Write the run's final result to ``summary.json``."""
    path = Path(run_dir) / "summary.json"
    with path.open("w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    return path
