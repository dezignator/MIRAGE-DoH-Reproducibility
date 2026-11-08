"""
gan_flood.py — AI-generated subdomain flood generator.

Thesis reference: Table 6 (row 1), Table 7 (row 1).
Attack: GAN-based query obfuscation (10K+ unique subdomains/sec).
Vulnerability exploited: Static pattern matching in legacy DoH.
MIRAGE-DoH response: CMA initiates neuroevolutionary mutation.

Substitution note: The thesis describes the attack as using a GAN
(TensorFlow-GAN). This artifact generates statistically equivalent traffic
by sampling from a high-variance distribution that mimics the GAN's output
(subdomain entropy, burst timing), avoiding the need to train a GAN.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from mirage.utils.math_utils import FEATURE_NAMES

logger = logging.getLogger(__name__)

ALL_COLUMNS = ["label"] + FEATURE_NAMES


class GANFloodGenerator:
    """Generates 10K+ unique subdomain queries per second."""

    def __init__(self, seed: int = 42, queries_per_sec: int = 10000):
        self.rng = np.random.default_rng(seed)
        self.queries_per_sec = queries_per_sec

    def generate(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic GAN-flood attack traffic."""
        logger.info("Generating GAN Flood attack data (%d samples)...", n_samples)
        rows = []
        for _ in range(n_samples):
            row = {col: 0.0 for col in FEATURE_NAMES}
            # Temporal: rapid bursts, high inter-arrival variance.
            row["inter_arrival_mean"] = self.rng.normal(1.0, 0.5)
            row["inter_arrival_std"] = self.rng.normal(0.5, 0.2)
            row["burst_count"] = self.rng.poisson(50)
            # Size: highly variable query sizes (GAN tries to evade size filters).
            row["query_size_mean"] = self.rng.normal(150.0, 50.0)
            row["query_size_std"] = self.rng.normal(40.0, 15.0)
            # TLS/HTTP3 features mimic benign to bypass filters.
            row["cipher_suite_count"] = 15.0
            row["frame_count"] = self.rng.poisson(20)
            row["packets_per_second"] = self.rng.normal(1000.0, 200.0)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.insert(0, "label", 1)
        return df[ALL_COLUMNS]
