"""
timing_probe.py — ECDSA timing probe generator.

Thesis reference: Table 6 (row 2), Table 7 (row 2).
Attack: Side-channel timing analysis for lattice basis extraction.
Vulnerability exploited: Fixed KEM parameters.
MIRAGE-DoH response: MCS escalates to Pi_2 + lattice re-randomization.

Substitution note: Emulates the timing signature of a TLS-Attacker
ECDSA timing probe by injecting microsecond-level variance into the
inter-arrival times.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mirage.utils.math_utils import FEATURE_NAMES

logger = logging.getLogger(__name__)
ALL_COLUMNS = ["label"] + FEATURE_NAMES


class TimingProbeGenerator:
    """Generates microsecond-level timing variance probes."""

    def __init__(self, seed: int = 42, variance_us: float = 0.5):
        self.rng = np.random.default_rng(seed)
        self.variance_us = variance_us

    def generate(self, n_samples: int = 1000) -> pd.DataFrame:
        logger.info("Generating Timing Probe attack data (%d samples)...", n_samples)
        rows = []
        for _ in range(n_samples):
            row = {col: 0.0 for col in FEATURE_NAMES}
            # Timing: precise, repeating delays with microsecond variance.
            base_delay = 10.0
            jitter = self.rng.normal(0, self.variance_us / 1000.0)  # us to ms
            row["inter_arrival_mean"] = base_delay + jitter
            row["inter_arrival_std"] = self.variance_us / 1000.0
            row["jitter_mean_ms"] = jitter
            row["jitter_std_ms"] = self.variance_us / 1000.0
            # Sizes remain normal to avoid size-based detection.
            row["query_size_mean"] = self.rng.normal(60.0, 10.0)
            row["packets_per_second"] = self.rng.normal(100.0, 5.0)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.insert(0, "label", 1)
        return df[ALL_COLUMNS]
