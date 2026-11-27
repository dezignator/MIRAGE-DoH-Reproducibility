#!/usr/bin/env python3
"""
extended_experimentation/tests/test_attack_scenarios.py

Tests for the attack-catalogue and staged-tunnelling generators.

We wrote these tests against the modules that produce the campaign
data the training pipeline consumes. Their purpose is to catch the
three failure modes that would silently corrupt a campaign:

  1. An HMM whose transition rows did not sum to one, producing a
     sampler that drifts toward a single state.
  2. A shaper that returned arrays of the wrong length, producing a
     flow whose packet count silently disagreed with its metadata.
  3. An orchestrator that completed without producing the flows the
     report would cite.

None of the tests requires torch. Every generator below runs on CPU
with only numpy as a dependency.
"""

from __future__ import annotations

import numpy as np
import pytest

from extended_experimentation.attack_scenarios.polymorphic_shaping_simulator import (
    AttackCampaignSimulator,
    AttackFamily,
    DistributionSpec,
    HMMSpec,
    Label,
    PolymorphicShaper,
    Protocol,
    ShaperConfig,
    SimulationConfig,
)
from extended_experimentation.attack_scenarios.staged_tunneling_orchestrator import (
    CampaignOutcome,
    FailureMode,
    RecordingObserver,
    RetryPolicy,
    ScenarioSpec,
    StageSpec,
    StagedTunnelingOrchestrator,
    default_scenario_matrix,
)


# --------------------------------------------------------------------------- #
# HMM specification
# --------------------------------------------------------------------------- #


def _valid_hmm() -> HMMSpec:
    return HMMSpec(
        n_states=2,
        initial=np.array([0.5, 0.5]),
        transition=np.array([[0.5, 0.5], [0.5, 0.5]]),
        size_dist=[
            DistributionSpec("const", {"value": 100.0}),
            DistributionSpec("const", {"value": 200.0}),
        ],
        iat_dist=[
            DistributionSpec("const", {"value": 0.01}),
            DistributionSpec("const", {"value": 0.02}),
        ],
    )


def test_hmm_validate_accepts_a_well_formed_specification() -> None:
    _valid_hmm().validate()  # must not raise


def test_hmm_validate_rejects_rows_that_do_not_sum_to_one() -> None:
    bad = _valid_hmm()
    bad.transition = np.array([[0.9, 0.5], [0.5, 0.5]])
    with pytest.raises(ValueError):
        bad.validate()


def test_hmm_validate_rejects_an_initial_vector_that_does_not_sum_to_one() -> None:
    bad = _valid_hmm()
    bad.initial = np.array([0.3, 0.3])
    with pytest.raises(ValueError):
        bad.validate()


def test_hmm_sample_states_returns_one_state_per_step() -> None:
    hmm = _valid_hmm()
    rng = np.random.default_rng(20231130)
    states = hmm.sample_states(rng, 100)
    assert states.shape == (100,)
    assert states.min() >= 0
    assert states.max() < hmm.n_states


# --------------------------------------------------------------------------- #
# Shaper
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "family",
    [
        AttackFamily.SIZE_MORPH,
        AttackFamily.IAT_JITTER,
        AttackFamily.BIMODAL_BURST,
        AttackFamily.STAGED_TUNNEL,
    ],
)
def test_shaper_preserves_array_shapes(family: AttackFamily) -> None:
    shaper = PolymorphicShaper(
        family=family,
        rng=np.random.default_rng(0),
        config=ShaperConfig(),
        mimic_pool=None,
    )
    sizes = np.full(64, 500.0)
    iats = np.full(64, 0.02)
    directions = np.zeros(64, dtype=int)
    new_sizes, new_iats, new_dirs = shaper.apply(sizes, iats, directions)
    assert new_sizes.shape == sizes.shape
    assert new_iats.shape == iats.shape
    assert new_dirs.shape == directions.shape


def test_shaper_size_morph_produces_variation() -> None:
    """A size-morph shaper must actually change some sizes."""
    shaper = PolymorphicShaper(
        family=AttackFamily.SIZE_MORPH,
        rng=np.random.default_rng(0),
        config=ShaperConfig(size_morph_amplitude=0.5),
        mimic_pool=None,
    )
    sizes = np.full(100, 500.0)
    iats = np.full(100, 0.02)
    directions = np.zeros(100, dtype=int)
    new_sizes, _, _ = shaper.apply(sizes, iats, directions)
    # At least one size must differ from 500 by more than 1 %.
    deviations = np.abs(new_sizes - 500.0) / 500.0
    assert deviations.max() > 0.01


# --------------------------------------------------------------------------- #
# Campaign simulator
# --------------------------------------------------------------------------- #


def test_campaign_simulator_produces_the_expected_flow_count() -> None:
    config = SimulationConfig(
        flows_per_benign_profile=3,
        flows_per_attack_family=3,
        max_flow_packets=200,
    )
    simulator = AttackCampaignSimulator(config)
    flows = list(simulator.generate())
    # Five benign profiles x 3 + five attack families x 3 = 30.
    assert len(flows) == 30


def test_campaign_simulator_interleaves_labels() -> None:
    config = SimulationConfig(
        flows_per_benign_profile=3,
        flows_per_attack_family=3,
        max_flow_packets=200,
    )
    simulator = AttackCampaignSimulator(config)
    flows = list(simulator.generate())
    # The simulator's interleave guarantees that adjacent pairs alternate
    # benign, attack, benign, attack for the first 30 flows.
    for i in range(0, 30, 2):
        assert flows[i].label == Label.BENIGN
        assert flows[i + 1].label == Label.ATTACK


def test_campaign_simulator_flow_records_are_self_consistent() -> None:
    config = SimulationConfig(
        flows_per_benign_profile=1,
        flows_per_attack_family=1,
        max_flow_packets=100,
    )
    simulator = AttackCampaignSimulator(config)
    flows = list(simulator.generate())
    for flow in flows:
        assert flow.n_packets == len(flow.sizes)
        assert flow.n_packets == len(flow.iats)
        assert flow.n_packets == len(flow.directions)
        assert flow.duration_s >= 0.0
        summary = flow.summary()
        assert summary["n_packets"] == flow.n_packets


# --------------------------------------------------------------------------- #
# Staged tunnelling orchestrator
# --------------------------------------------------------------------------- #


def _small_scenario() -> ScenarioSpec:
    stage = StageSpec(
        index=0,
        name="test-stage",
        payload_bytes_target=500.0,   # yields 4 packets at 380 bytes/packet
        beacon_interval_s=0.1,
        beacon_jitter_ratio=0.05,
        preferred_protocol=Protocol.DOH,
        shaper_config=ShaperConfig(),
        growth_factor=1.0,
    )
    return ScenarioSpec(
        name="test-scenario",
        stages=(stage,),
        retry=RetryPolicy(failure_mode=FailureMode.NONE),
        seed_offset=0,
        description="single-stage scenario used by the test suite",
    )


def test_orchestrator_completes_a_small_scenario() -> None:
    observer = RecordingObserver()
    orchestrator = StagedTunnelingOrchestrator(
        scenario=_small_scenario(),
        seed=20231130,
        observers=[observer],
    )
    result = orchestrator.run()
    assert result.outcome is CampaignOutcome.COMPLETED
    assert len(result.flows) == 1
    assert result.flows[0].n_packets == 4
    assert result.flows[0].label == Label.ATTACK
    assert result.delivered_bytes > 0.0


def test_orchestrator_emits_the_expected_event_sequence() -> None:
    orchestrator = StagedTunnelingOrchestrator(
        scenario=_small_scenario(),
        seed=20231130,
    )
    result = orchestrator.run()
    kinds = [event.kind for event in result.events]
    assert kinds[0] == "campaign_start"
    assert kinds[-1] == "campaign_end"
    assert kinds.count("stage_enter") == 1
    assert kinds.count("stage_exit") == 1
    assert kinds.count("beacon") == 4
    # No retries: the retry policy is FailureMode.NONE.
    assert kinds.count("retry") == 0


def test_default_scenario_matrix_has_the_expected_scenarios() -> None:
    matrix = default_scenario_matrix()
    expected_names = {
        "dns_exfil_baseline",
        "bursty_exfil",
        "slow_exfil",
        "multi_resolver_handoff",
        "adaptive_mimicry_tunnel",
    }
    assert set(matrix.keys()) == expected_names
    for name, spec in matrix.items():
        assert spec.stages, f"{name} has no stages"
        assert spec.retry.max_attempts > 0
