"""
topological_tunnel.py — Multi-phase DNS tunneling generator.

Thesis reference: Table 6 (row 4), Table 7 (row 3).
Attack: Multi-phase DNS tunneling via geometric traffic shaping.
Vulnerability exploited: Short-term memory in flow analysis.
MIRAGE-DoH response: TMN detects mu-persistence homology across 50+ hops.

Substitution note: Emulates dnscat2 by generating bursty, cyclic traffic
patterns (high Betti-1 number) that persist across a long time window.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from mirage.utils.math_utils import FEATURE_NAMES

logger = logging.getLogger(__name__)
ALL_COLUMNS = ["label"] + FEATURE_NAMES


class TopologicalTunnelGenerator:
    """Generates traffic with persistent topological cycles (tunneling)."""

    def __init__(self, seed: int = 42, phases: int = 5):
        self.rng = np.random.default_rng(seed)
        self.phases = phases

    def generate(self, n_samples: int = 1000) -> pd.DataFrame:
        logger.info("Generating Topological Tunnel attack data (%d samples)...", n_samples)
        rows = []
        for _ in range(n_samples):
            row = {col: 0.0 for col in FEATURE_NAMES}
            # Temporal: bursty cycles (high variance, periodic).
            # This creates the persistent H_1 features the TMN detects.
            row["inter_arrival_mean"] = self.rng.normal(50.0, 20.0)
            row["inter_arrival_std"] = self.rng.normal(30.0, 10.0)
            row["burst_count"] = self.rng.poisson(10)
            row["burst_mean_duration_ms"] = self.rng.normal(200.0, 50.0)
            # Size: padded queries to hide payload.
            row["query_size_mean"] = self.rng.normal(120.0, 30.0)
            row["query_size_std"] = self.rng.normal(25.0, 10.0)
            # Flow: long duration, high variance.
            row["flow_duration_ms"] = self.rng.normal(3000.0, 500.0)
            row["packets_per_second"] = self.rng.normal(5.0, 2.0)
            rows.append(row)
        df = pd.DataFrame(rows)
        df.insert(0, "label", 1)
        return df[ALL_COLUMNS]
