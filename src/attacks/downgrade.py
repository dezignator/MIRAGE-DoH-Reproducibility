"""
downgrade.py — Protocol downgrade attack generator.

Thesis reference: Table 6 (row 3).
Attack: TLS 1.3 -> TLS 1.2 injection via ECS manipulation.
Vulnerability exploited: Lack of causal protocol hardening.
MIRAGE-DoH response: CCE replays attack, updates protocol logic.

Substitution note: Emulates the resulting traffic after a downgrade by
setting TLS-specific features to their TLS 1.2 equivalents.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mirage.utils.math_utils import FEATURE_NAMES

logger = logging.getLogger(__name__)
ALL_COLUMNS = ["label"] + FEATURE_NAMES


class DowngradeGenerator:
    """Generates traffic with protocol downgrade signatures."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate(self, n_samples: int = 1000) -> pd.DataFrame:
        logger.info("Generating Protocol Downgrade attack data (%d samples)...", n_samples)
        rows = []
        for _ in range(n_samples):
            row = {col: 0.0 for col in FEATURE_NAMES}
            # TLS features: TLS 1.2 code (771), fewer extensions, different ciphers.
            row["tls_version_code"] = 771.0  # TLS 1.2
            row["cipher_suite_count"] = self.rng.normal(20, 3)
            row["extension_count"] = self.rng.normal(8, 2)
            # Temporal: slightly slower handshake (extra round trips).
            row["inter_arrival_mean"] = self.rng.normal(25.0, 5.0)
            row["flow_duration_ms"] = self.rng.normal(200.0, 50.0)
            # Size: slightly larger due to older cipher overhead.
            row["query_size_mean"] = self.rng.normal(80.0, 15.0)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.insert(0, "label", 1)
        return df[ALL_COLUMNS]
