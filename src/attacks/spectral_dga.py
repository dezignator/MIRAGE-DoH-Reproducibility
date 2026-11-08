"""
spectral_dga.py — Low-frequency Domain Generation Algorithm generator.

Thesis reference: Table 6 (row 5), Table 7 (row 4).
Attack: Low-frequency domain generation aligned with HTTP/2 frame timing.
Vulnerability exploited: Frequency-blind detection systems.
MIRAGE-DoH response: SGI decomposes traffic via graph Laplacian eigenprojections.

Substitution note: Emulates Foreshadow by generating traffic with a
low-frequency periodic component (0.1-0.3 Hz) in the inter-arrival times.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mirage.utils.math_utils import FEATURE_NAMES

logger = logging.getLogger(__name__)
ALL_COLUMNS = ["label"] + FEATURE_NAMES


class SpectralDGAGenerator:
    """Generates traffic with low-frequency spectral signatures."""

    def __init__(self, seed: int = 42, frequency_hz: float = 0.2):
        self.rng = np.random.default_rng(seed)
        self.frequency_hz = frequency_hz

    def generate(self, n_samples: int = 1000) -> pd.DataFrame:
        logger.info("Generating Spectral DGA attack data (%d samples)...", n_samples)
        rows = []
        # Generate a slow oscillation.
        period_ms = (1.0 / self.frequency_hz) * 1000.0  # 5000 ms for 0.2 Hz
        for i in range(n_samples):
            # Sinusoidal timing modulation + noise.
            phase = (i / n_samples) * 2 * np.pi * (n_samples / 100.0)  # slow wave
            oscillation = 500.0 * np.sin(phase)
            
            row = {col: 0.0 for col in FEATURE_NAMES}
            row["inter_arrival_mean"] = 100.0 + oscillation + self.rng.normal(0, 10)
            row["inter_arrival_std"] = 20.0
            row["query_size_mean"] = self.rng.normal(70.0, 15.0)
            row["frame_count"] = self.rng.poisson(15)
            row["packets_per_second"] = self.rng.normal(20.0, 5.0)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.insert(0, "label", 1)
        return df[ALL_COLUMNS]
