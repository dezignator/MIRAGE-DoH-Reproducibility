"""
mcs.py — Metamorphic Cryptographic Shards (MCS).

Thesis reference: Chapter 3, Section 3.7.1 and 3.7.2, Equations 11-13,
Table 5, and Section 3.15.1.

Implements:
- The KEM combiner (Eq. 11).
- The three-tier shard configuration Π_0, Π_1, Π_2 (Eq. 13).
- The threat-severity scoring and morphing protocol.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np

from mirage.utils.math_utils import (
    cipher_morphing_policy,
    kem_combiner,
    MCS_MORPH_TARGET_MS,
    PQ_SECURITY_BITS,
    threat_severity_score,
)

logger = logging.getLogger(__name__)


class MetamorphicCryptographicShard:
    """A single metamorphic shard.

    Manages the current configuration (Π_0, Π_1, or Π_2), the KEM combiner
    (Eq. 11), and the morphing protocol (Section 3.15.1).
    """

    def __init__(self, initial_level: str = "pi_0"):
        self.level = initial_level
        self.morph_history: list = []
        # Mock KEM state. In production, this would be backed by a real
        # KEM implementation (ML-KEM, X25519, Classic McEliece). The
        # artifact ships with mock seeds so the morph-time can be measured
        # without requiring a full NIST-compliant KEM stack.
        self.kem_state = {
            "ss_1": os.urandom(32),
            "ss_2": os.urandom(32),
            "ct_1": os.urandom(32),
            "ct_2": os.urandom(32),
            "ctx": b"MIRAGE-DoH-MCS-v1",
        }

    def get_current_config(self) -> Dict[str, str]:
        """Return the current shard configuration."""
        configs = {
            "pi_0": {"lattice": "ml_kem_768", "second": "x25519", "category": "3"},
            "pi_1": {"lattice": "ml_kem_1024", "second": "x25519", "category": "5"},
            "pi_2": {"lattice": "ml_kem_1024", "second": "classic_mceliece", "category": "5"},
        }
        return configs[self.level]

    def derive_key(self) -> bytes:
        """Derive the control-plane key K_ctrl using Eq. 11."""
        return kem_combiner(
            ss_1=self.kem_state["ss_1"],
            ss_2=self.kem_state["ss_2"],
            ct_1=self.kem_state["ct_1"],
            ct_2=self.kem_state["ct_2"],
            ctx=self.kem_state["ctx"],
        )

    def morph(self, threat_score: float) -> Dict[str, float]:
        """Execute the morphing protocol (Section 3.15.1).

        Args:
            threat_score: the rho_t value from Eq. 13a.

        Returns:
            Dict with keys: old_level, new_level, morph_time_ms.
        """
        old_level = self.level
        new_level = cipher_morphing_policy(threat_score)

        # Enforce policy floor (Section 3.7.2): no downgrade below Pi_0.
        level_order = ["pi_0", "pi_1", "pi_2"]
        if level_order.index(new_level) < level_order.index(old_level):
            new_level = old_level  # block downgrade

        start = time.perf_counter()
        if new_level != old_level:
            # Simulate the morph: new KEM state.
            self.kem_state = {
                "ss_1": os.urandom(32),
                "ss_2": os.urandom(32),
                "ct_1": os.urandom(32),
                "ct_2": os.urandom(32),
                "ctx": b"MIRAGE-DoH-MCS-v1",
            }
            self.level = new_level
            self.morph_history.append({"from": old_level, "to": new_level, "threat_score": threat_score})
            logger.info("Morphed %s -> %s (threat_score=%.3f)", old_level, new_level, threat_score)
        end = time.perf_counter()

        return {
            "old_level": old_level,
            "new_level": self.level,
            "morph_time_ms": (end - start) * 1000.0,
        }

    def effective_pq_security(self) -> int:
        """Return the effective post-quantum security bits (Eq. 12)."""
        return PQ_SECURITY_BITS[self.level]


class ThreatSeverityScorer:
    """Eq. 13a: rho_t = sigma(W_c z_t + b_c).

    Tiny linear + sigmoid layer. The weights are initialized xavier-uniform
    (configs/mcs_crypto_params.yaml §1). They are learned jointly with the
    CMA during training.
    """

    def __init__(self, input_dim: int = 46):
        rng = np.random.default_rng(42)
        bound = np.sqrt(6.0 / (input_dim + 1))  # xavier-uniform
        self.W_c = rng.uniform(-bound, bound, size=(1, input_dim))
        self.b_c = np.zeros(1)

    def __call__(self, z_t: np.ndarray) -> float:
        return threat_severity_score(z_t, self.W_c, self.b_c)
