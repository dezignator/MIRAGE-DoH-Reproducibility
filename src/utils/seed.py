"""
seed.py — Deterministic seeding for the MIRAGE-DoH artifact.

Every script in the artifact calls `seed_everything(42)` as its first
executable line. This guarantees that two runs with the same config and
the same hardware produce byte-identical results (see
REPRODUCIBILITY_CHECKLIST.md §7).

Thesis reference: Chapter 3, Section 3.11.1 ("splits were stratified
according to class and for the temporal evaluation capture date") and
configs/default_mirage_config.yaml §2 (seed: 42).
"""

from __future__ import annotations

import os
import random
import sys
from typing import Literal

import numpy as np


def _check_python_version() -> None:
    """Ensure the runtime matches the versions the artifact was tested on."""
    if sys.version_info < (3, 9):
        raise RuntimeError(
            f"Python >= 3.9 is required; you are running {sys.version_info.major}."
            f"{sys.version_info.minor}. See requirements.txt."
        )
    if sys.version_info >= (3, 12):
        sys.stderr.write(
            "WARNING: Python 3.12+ is not officially tested with this artifact "
            "(pycryptodome has not stabilised on it). Proceed at your own risk.\n"
        )


def seed_everything(
    seed: int = 42,
    determinism: Literal["strict", "warn", "off"] = "strict",
) -> None:
    """Seed all random number generators.

    Args:
        seed: The random seed. Default 42 (configs/default_mirage_config.yaml §2).
        determinism: PyTorch determinism mode.
            "strict": torch.use_deterministic_algorithms(True). Fails if a
                      non-deterministic op is used.
            "warn":   torch.use_deterministic_algorithms(True, warn_only=True).
            "off":    Do not call use_deterministic_algorithms.
    """
    _check_python_version()

    # 1. Python stdlib
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch (optional import so this module works without torch)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Configure CuDNN for determinism.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch 1.8+ deterministic algorithms.
        if determinism == "strict":
            # warn_only=False raises RuntimeError on non-deterministic ops.
            torch.use_deterministic_algorithms(True, warn_only=False)
        elif determinism == "warn":
            torch.use_deterministic_algorithms(True, warn_only=True)
        elif determinism == "off":
            pass
        else:
            raise ValueError(
                f"Unknown determinism mode: {determinism!r}. "
                f"Choose from 'strict', 'warn', 'off'."
            )

    except ImportError:
        sys.stderr.write(
            "WARNING: PyTorch is not installed. PyTorch seeding skipped. "
            "Install with: pip install torch==2.3.1\n"
        )
