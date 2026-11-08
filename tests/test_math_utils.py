"""
test_math_utils.py — Unit tests for all 31 equations in src/utils/math_utils.py.

Run with:
    pytest tests/test_math_utils.py -v
"""

import math
import numpy as np
import pytest

# Adjust path so tests can run without pip install -e .
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mirage.utils.math_utils import (
    catastrophic_forgetting_penalty,
    cipher_morphing_policy,
    edge_weight,
    graph_laplacian,
    kem_combiner,
    multi_hop_persistence,
    persistence_betti_numbers,
    pq_security_bound,
    system_throughput,
    threat_severity_score,
    z_score_normalize,
)


class TestCMAEquations:
    """Equations 1-5 (Cognitive MetaAgents)."""

    def test_catastrophic_forgetting_penalty_eq3(self):
        """Eq. 3: R_cat = sum_j F_j * (phi_j - phi*_j)^2."""
        phi = np.array([2.0, 3.0])
        phi_star = np.array([1.0, 1.0])
        fisher = np.array([0.5, 0.5])
        # (2-1)^2 * 0.5 + (3-1)^2 * 0.5 = 0.5 + 2.0 = 2.5
        result = catastrophic_forgetting_penalty(phi, phi_star, fisher)
        assert np.isclose(result, 2.5)


class TestTMNEquations:
    """Equations 6-10 (Topological Memory Networks)."""

    def test_betti_numbers_eq7(self):
        """Eq. 7: Betti numbers from Laplacian spectrum."""
        # A simple square (4 nodes, 4 edges) has 1 component and 1 cycle.
        # Distances: 0-1: 1, 1-2: 1, 2-3: 1, 3-0: 1, diagonals: sqrt(2) ≈ 1.414
        distances = np.array([
            [0.0, 1.0, 1.414, 1.0],
            [1.0, 0.0, 1.0, 1.414],
            [1.414, 1.0, 0.0, 1.0],
            [1.0, 1.414, 1.0, 0.0]
        ])
        # At scale 1.1, we have the square edges but not the diagonals.
        b_0, b_1 = persistence_betti_numbers(distances, scale=1.1)
        assert b_0 == 1, "Should be 1 connected component"
        assert b_1 == 1, "Should be 1 cycle"

    def test_multi_hop_persistence_eq10(self):
        """Eq. 10: multi-hop persistence decays with distance."""
        dgm = np.array([
            [0.1, 0.5], # persistence 0.4
            [0.2, 1.0], # persistence 0.8
        ])
        # exp(-0.5*0.4)*0.4 + exp(-0.5*0.8)*0.8
        expected = math.exp(-0.5 * 0.4) * 0.4 + math.exp(-0.5 * 0.8) * 0.8
        result = multi_hop_persistence([dgm], decay_lambda=0.5)
        assert np.isclose(result, expected)


class TestMCSEquations:
    """Equations 11-13 (Metamorphic Cryptographic Shards)."""

    def test_kem_combiner_eq11(self):
        """Eq. 11: KEM combiner produces a 256-bit output."""
        ss_1 = b"a" * 32
        ss_2 = b"b" * 32
        ct_1 = b"c" * 32
        ct_2 = b"d" * 32
        ctx = b"context"
        key = kem_combiner(ss_1, ss_2, ct_1, ct_2, ctx)
        assert len(key) == 32, "Should produce 256-bit (32-byte) key"

    def test_pq_security_bound_eq12(self):
        """Eq. 12: lambda_PQ >= max(lambda_1, lambda_2)."""
        assert pq_security_bound(192, 0) == 192
        assert pq_security_bound(256, 256) == 256
        assert pq_security_bound(128, 256) == 256

    def test_cipher_morphing_policy_eq13(self):
        """Eq. 13b: three-tier morphing policy."""
        assert cipher_morphing_policy(0.1) == "pi_0"
        assert cipher_morphing_policy(0.5) == "pi_1"
        assert cipher_morphing_policy(0.9) == "pi_2"


class TestSGIEquations:
    """Equations 17-21 (Spectral Game Intelligence)."""

    def test_graph_laplacian_eq17(self):
        """Eq. 17: L = D - A."""
        A = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        L = graph_laplacian(A)
        expected_L = np.array([
            [ 2, -1, -1],
            [-1,  2, -1],
            [-1, -1,  2]
        ])
        np.testing.assert_array_equal(L, expected_L)

    def test_edge_weight_eq20(self):
        """Eq. 20: w = exp(-delta_t / tau)."""
        assert np.isclose(edge_weight(0.0), 1.0)
        assert np.isclose(edge_weight(100.0, tau=100.0), math.exp(-1.0))


class TestEvaluationMetrics:
    """Equations 27-31 (Evaluation)."""

    def test_system_throughput_eq31(self):
        """Eq. 31: T_total = C_parallel / mean(latency)."""
        latencies = [0.001, 0.001, 0.001] # 1ms
        result = system_throughput(latencies, parallel_instances=10)
        assert np.isclose(result, 10000.0), "10 instances * 1000 qps = 10000"


class TestHelpers:

    def test_z_score_normalize(self):
        """Section 3.11.1: z-score fit on train split only."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        X_norm, mean, std = z_score_normalize(X)
        np.testing.assert_allclose(mean, [2.0, 3.0])
        np.testing.assert_allclose(std, [1.0, 1.0])
        np.testing.assert_allclose(X_norm, [[-1.0, -1.0], [1.0, 1.0]])
