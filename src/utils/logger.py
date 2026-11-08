"""
logger.py — Structured JSON logging for the MIRAGE-DoH artifact.

Provides:
- Human-readable output on stdout (INFO level by default).
- JSON output in outputs/logs/<run_id>/run.json for automated diffing.
- A run_metadata.json file capturing git commit, config hash, seed, and
  timestamp so a reviewer can trace exactly what produced a given output.

Thesis reference: configs/default_mirage_config.yaml §9 (Logging section).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Formats log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Attach any structured extras (e.g., logger.info("msg", extra={"epoch": 5}))
        if hasattr(record, "extras"):
            log_obj.update(record.extras)
        # Attach exception info if present.
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def _get_git_commit() -> str:
    """Return the current git commit hash, or 'unknown' if not a git repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent,
        ).decode("ascii").strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _get_config_hash(config_path: Optional[Path]) -> str:
    """Return the SHA-256 of the config file, or 'unknown'."""
    if config_path is None or not config_path.is_file():
        return "unknown"
    h = hashlib.sha256()
    h.update(config_path.read_bytes())
    return h.hexdigest()


def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    config_path: Optional[Path] = None,
    level: int = logging.INFO,
    run_id: Optional[str] = None,
) -> logging.Logger:
    """Set up dual human+JSON logging.

    Args:
        name: Logger name (usually the script's __name__).
        log_dir: Directory to write the JSON log. Defaults to outputs/logs/.
        config_path: Path to the loaded config file, for metadata hashing.
        level: Logging level (default INFO).
        run_id: Unique identifier for this run. If None, a timestamp is used.

    Returns:
        Configured logging.Logger instance.
    """
    if run_id is None:
        run_id = time.strftime("%Y%m%d_%H%M%S")

    if log_dir is None:
        log_dir = Path("outputs/logs")
    log_dir = Path(log_dir)
    run_dir = log_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()  # avoid duplicate handlers if called twice

    # 1. Human handler (stdout)
    c_handler = logging.StreamHandler(sys.stdout)
    c_format = logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s")
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)

    # 2. JSON handler (file)
    json_path = run_dir / "run.json"
    f_handler = logging.FileHandler(json_path, encoding="utf-8")
    f_handler.setFormatter(JSONFormatter())
    logger.addHandler(f_handler)

    # 3. Write run_metadata.json for provenance.
    metadata = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _get_git_commit(),
        "config_path": str(config_path) if config_path else None,
        "config_hash": _get_config_hash(config_path),
        "python_version": sys.version,
        "cwd": os.getcwd(),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    logger.info("Logging initialized (run_id=%s)", run_id)
    return logger


def log_metrics(logger: logging.Logger, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """Log a metrics dictionary in a structured way.

    Args:
        logger: The logger instance.
        metrics: Dictionary of metric name -> value.
        step: Optional step number (epoch, iteration) for time-series metrics.
    """
    extras = {"metrics": metrics}
    if step is not None:
        extras["step"] = step
    # Log as a structured extra so JSONFormatter picks it up.
    logger.info("Metrics recorded", extra={"extras": extras})
