#!/usr/bin/env python3
"""


Multi-stage DNS tunneling campaign orchestrator for the MIRAGE-DoH
extended experimentation suite.

We implemented this module to generate controlled multi-stage tunneling
campaigns that mirror the "staged tunneling" and "adaptive mimicry" families
described in Chapter 3. It reuses the hidden-Markov emission model and the
per-transport framing profiles from ``polymorphic_shaping_simulator`` and
lays on top of them a stage-aware lifecycle manager. The lifecycle models:

  * multi-stage payload progression with per-stage growth,
  * command-and-control beacon scheduling with configurable jitter,
  * adaptive retry behaviour under simulated resolver failure,
  * resolver handoff choreography across DoH, DoT and DoQ endpoints.

The orchestrator exposes a small observer protocol so that the
topological-memory and spectral-analysis modules can inspect intermediate
campaign state without depending on the orchestrator internals. We used
that protocol during the campaign archive build so that a run could be
replayed event-for-event from the emitted event log.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol as TypingProtocol,
    Sequence,
    Tuple,
)

import numpy as np

from .polymorphic_shaping_simulator import (
    THESIS_REFERENCE,
    AttackFamily,
    BenignProfile,
    FlowRecord,
    FRAMING,
    HMMSpec,
    Label,
    Protocol,
    ShaperConfig,
    _benign_profiles,
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

LOGGER = logging.getLogger("mirage.extended.staged_tunneling")

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class CampaignOutcome(str, Enum):
    """Final disposition of a tunnel campaign."""

    COMPLETED = "completed"
    ABORTED_MAX_RETRIES = "aborted_max_retries"
    ABORTED_STAGE_BUDGET = "aborted_stage_budget"
    ABORTED_OBSERVER = "aborted_observer"


class FailureMode(str, Enum):
    """Failure modes injected during simulated retry testing."""

    NONE = "none"
    TRANSIENT = "transient"
    PERSISTENT = "persistent"


# --------------------------------------------------------------------------- #
# Endpoints and resolver handoff
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Endpoint:
    """A resolver endpoint participating in the campaign."""

    resolver_id: str
    protocol: Protocol
    rtt_ms_median: float
    rtt_ms_std: float
    capacity_qps: float
    failure_bias: float = 0.02

    def simulate_rtt_ms(self, rng: np.random.Generator) -> float:
        return float(max(0.5, rng.normal(self.rtt_ms_median, self.rtt_ms_std)))


@dataclass
class ResolverPool:
    """A pool of endpoints with weighted handoff.

    We kept the pool intentionally simple: weighted sampling by capacity with
    a sticky-affinity mechanism that biases toward the currently active
    resolver. That matches the handoff behaviour we observed in real DoH
    resolver pools during the Chapter 3 dataset collection.
    """

    endpoints: List[Endpoint]
    affinity: float = 0.72

    def __post_init__(self) -> None:
        if not self.endpoints:
            raise ValueError("resolver pool must contain at least one endpoint")
        self._capacities = np.array(
            [e.capacity_qps for e in self.endpoints], dtype=float
        )
        self._capacities /= self._capacities.sum()
        self._current: int = 0

    def select(self, rng: np.random.Generator, allow_switch: bool) -> Endpoint:
        if not allow_switch or rng.random() < self.affinity:
            return self.endpoints[self._current]
        self._current = int(rng.choice(len(self.endpoints), p=self._capacities))
        return self.endpoints[self._current]

    def by_protocol(self, protocol: Protocol) -> List[Endpoint]:
        return [e for e in self.endpoints if e.protocol is protocol]


def _default_resolver_pool() -> ResolverPool:
    """We authored a nine-endpoint pool covering all three transports."""

    endpoints = [
        Endpoint("doh-a.example", Protocol.DOH, 28.0, 4.0, 12000.0, 0.010),
        Endpoint("doh-b.example", Protocol.DOH, 34.0, 6.5, 8000.0, 0.018),
        Endpoint("doh-c.example", Protocol.DOH, 41.0, 8.0, 6000.0, 0.025),
        Endpoint("dot-a.example", Protocol.DOT, 22.0, 3.5, 15000.0, 0.008),
        Endpoint("dot-b.example", Protocol.DOT, 31.0, 5.0, 9000.0, 0.014),
        Endpoint("dot-c.example", Protocol.DOT, 47.0, 9.5, 4500.0, 0.030),
        Endpoint("doq-a.example", Protocol.DOQ, 18.0, 3.0, 18000.0, 0.006),
        Endpoint("doq-b.example", Protocol.DOQ, 26.0, 4.5, 13000.0, 0.012),
        Endpoint("doq-c.example", Protocol.DOQ, 39.0, 7.0, 7000.0, 0.020),
    ]
    return ResolverPool(endpoints=endpoints, affinity=0.72)


# --------------------------------------------------------------------------- #
# Stage specification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StageSpec:
    """A single stage in a multi-stage tunnel campaign.

    A stage describes an intended payload volume, a target beacon cadence,
    and the shaper configuration that was applied to emissions during the
    stage. The orchestrator used these fields to drive both the emission
    schedule and the per-stage observer callbacks.
    """

    index: int
    name: str
    payload_bytes_target: float
    beacon_interval_s: float
    beacon_jitter_ratio: float
    preferred_protocol: Optional[Protocol]
    shaper_config: ShaperConfig
    growth_factor: float = 1.0

    def jittered_interval(self, rng: np.random.Generator) -> float:
        lo = self.beacon_interval_s * (1.0 - self.beacon_jitter_ratio)
        hi = self.beacon_interval_s * (1.0 + self.beacon_jitter_ratio)
        return float(max(1e-3, rng.uniform(lo, hi)))


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RetryPolicy:
    """Deterministic-with-backoff retry policy used during stage execution."""

    max_attempts: int = 5
    base_backoff_s: float = 0.25
    max_backoff_s: float = 6.0
    backoff_multiplier: float = 1.8
    failure_mode: FailureMode = FailureMode.TRANSIENT

    def backoff(self, attempt: int) -> float:
        raw = self.base_backoff_s * (self.backoff_multiplier ** attempt)
        return float(min(self.max_backoff_s, raw))

    def attempt_fails(
        self, rng: np.random.Generator, endpoint: Endpoint, attempt: int
    ) -> bool:
        if self.failure_mode is FailureMode.NONE:
            return False
        if self.failure_mode is FailureMode.PERSISTENT:
            return True
        # TRANSIENT: failure probability decreases with retries but is never
        # zero; we scaled the endpoint failure bias by an attempt decay.
        p = endpoint.failure_bias * math.exp(-0.6 * attempt)
        return bool(rng.random() < p)


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #


@dataclass
class CampaignEvent:
    """A single observable event during a campaign.

    We kept event payloads flat and JSON-serialisable so that archives can
    be diffed and compared across runs without special tooling.
    """

    t: float
    kind: str
    campaign_id: str
    stage_index: int
    payload: Dict[str, object]

    def to_json(self) -> Dict[str, object]:
        return {
            "t": self.t,
            "kind": self.kind,
            "campaign_id": self.campaign_id,
            "stage_index": self.stage_index,
            "payload": self.payload,
        }


# --------------------------------------------------------------------------- #
# Observer protocol
# --------------------------------------------------------------------------- #


class OrchestratorObserver(TypingProtocol):
    """Observer interface for downstream analysis modules.

    Both the topological-memory trainer and the spectral predictor implemented
    this protocol in the extended suite. The protocol was intentionally
    read-only and side-effect-free from the orchestrator's point of view; an
    observer that raised was treated as a campaign abort so the failure was
    never silently swallowed.
    """

    def on_campaign_start(
        self, campaign_id: str, stages: Sequence[StageSpec], t: float
    ) -> None: ...

    def on_stage_enter(
        self, campaign_id: str, stage_index: int, stage: StageSpec, t: float
    ) -> None: ...

    def on_beacon(
        self,
        campaign_id: str,
        stage_index: int,
        t: float,
        payload_bytes: float,
        endpoint: Endpoint,
    ) -> None: ...

    def on_retry(
        self,
        campaign_id: str,
        stage_index: int,
        attempt: int,
        t: float,
        endpoint: Endpoint,
    ) -> None: ...

    def on_stage_exit(
        self, campaign_id: str, stage_index: int, t: float, delivered: float
    ) -> None: ...

    def on_campaign_end(
        self, campaign_id: str, outcome: CampaignOutcome, t: float
    ) -> None: ...


class RecordingObserver:
    """A default observer that records every event in memory."""

    def __init__(self) -> None:
        self.events: List[CampaignEvent] = []

    def on_campaign_start(self, campaign_id, stages, t):
        self.events.append(
            CampaignEvent(
                t=t,
                kind="campaign_start",
                campaign_id=campaign_id,
                stage_index=-1,
                payload={"n_stages": len(stages)},
            )
        )

    def on_stage_enter(self, campaign_id, stage_index, stage, t):
        self.events.append(
            CampaignEvent(
                t=t,
                kind="stage_enter",
                campaign_id=campaign_id,
                stage_index=stage_index,
                payload={
                    "name": stage.name,
                    "payload_target": stage.payload_bytes_target,
                    "protocol": (
                        stage.preferred_protocol.value
                        if stage.preferred_protocol
                        else None
                    ),
                },
            )
        )

    def on_beacon(self, campaign_id, stage_index, t, payload_bytes, endpoint):
        self.events.append(
            CampaignEvent(
                t=t,
                kind="beacon",
                campaign_id=campaign_id,
                stage_index=stage_index,
                payload={
                    "bytes": payload_bytes,
                    "resolver": endpoint.resolver_id,
                    "protocol": endpoint.protocol.value,
                },
            )
        )

    def on_retry(self, campaign_id, stage_index, attempt, t, endpoint):
        self.events.append(
            CampaignEvent(
                t=t,
                kind="retry",
                campaign_id=campaign_id,
                stage_index=stage_index,
                payload={"attempt": attempt, "resolver": endpoint.resolver_id},
            )
        )

    def on_stage_exit(self, campaign_id, stage_index, t, delivered):
        self.events.append(
            CampaignEvent(
                t=t,
                kind="stage_exit",
                campaign_id=campaign_id,
                stage_index=stage_index,
                payload={"delivered": delivered},
            )
        )

    def on_campaign_end(self, campaign_id, outcome, t):
        self.events.append(
            CampaignEvent(
                t=t,
                kind="campaign_end",
                campaign_id=campaign_id,
                stage_index=-1,
                payload={"outcome": outcome.value},
            )
        )


# --------------------------------------------------------------------------- #
# Scenario matrix
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScenarioSpec:
    """A named scenario that fully determines a campaign's behaviour."""

    name: str
    stages: Tuple[StageSpec, ...]
    retry: RetryPolicy
    seed_offset: int
    description: str


def _stage(
    index: int,
    name: str,
    target: float,
    interval: float,
    jitter: float,
    protocol: Optional[Protocol],
    amplitude: float = 0.35,
) -> StageSpec:
    return StageSpec(
        index=index,
        name=name,
        payload_bytes_target=target,
        beacon_interval_s=interval,
        beacon_jitter_ratio=jitter,
        preferred_protocol=protocol,
        shaper_config=ShaperConfig(
            size_morph_amplitude=amplitude,
            stage_count=4,
            stage_gain=1.6,
        ),
        growth_factor=1.35,
    )


def default_scenario_matrix() -> Dict[str, ScenarioSpec]:
    """We authored five scenarios covering the campaign classes in Chapter 3."""

    baseline = ScenarioSpec(
        name="dns_exfil_baseline",
        description="Steady multi-stage exfiltration over DoH.",
        stages=(
            _stage(0, "handshake", 8_000.0, 6.0, 0.15, Protocol.DOH),
            _stage(1, "fingerprint", 24_000.0, 4.0, 0.18, Protocol.DOH),
            _stage(2, "bulk", 96_000.0, 2.0, 0.22, Protocol.DOH, amplitude=0.55),
            _stage(3, "drain", 160_000.0, 1.5, 0.30, Protocol.DOH, amplitude=0.62),
        ),
        retry=RetryPolicy(),
        seed_offset=0,
    )

    bursty = ScenarioSpec(
        name="bursty_exfil",
        description="Burst-emission exfiltration with high instantaneous rate.",
        stages=(
            _stage(0, "burst-1", 30_000.0, 0.8, 0.40, Protocol.DOT),
            _stage(1, "burst-2", 120_000.0, 0.5, 0.55, Protocol.DOT, amplitude=0.70),
            _stage(2, "burst-3", 240_000.0, 0.4, 0.60, Protocol.DOT, amplitude=0.78),
        ),
        retry=RetryPolicy(max_attempts=3, base_backoff_s=0.15),
        seed_offset=101,
    )

    slow = ScenarioSpec(
        name="slow_exfil",
        description="Low-rate, high-jitter exfiltration designed to evade timing detectors.",
        stages=(
            _stage(0, "warmup", 4_000.0, 22.0, 0.35, Protocol.DOQ),
            _stage(1, "steady", 12_000.0, 18.0, 0.45, Protocol.DOQ),
            _stage(2, "steady-long", 36_000.0, 15.0, 0.50, Protocol.DOQ, amplitude=0.28),
            _stage(3, "final", 60_000.0, 12.0, 0.55, Protocol.DOQ, amplitude=0.42),
        ),
        retry=RetryPolicy(max_attempts=7, base_backoff_s=0.60),
        seed_offset=202,
    )

    handoff = ScenarioSpec(
        name="multi_resolver_handoff",
        description="Campaign that rotates across DoH, DoT and DoQ resolvers.",
        stages=(
            _stage(0, "doh-stage", 18_000.0, 5.0, 0.20, Protocol.DOH),
            _stage(1, "dot-stage", 42_000.0, 4.0, 0.25, Protocol.DOT),
            _stage(2, "doq-stage", 78_000.0, 3.0, 0.30, Protocol.DOQ, amplitude=0.55),
            _stage(3, "return-doh", 120_000.0, 2.5, 0.35, Protocol.DOH, amplitude=0.60),
        ),
        retry=RetryPolicy(),
        seed_offset=303,
    )

    mimicry = ScenarioSpec(
        name="adaptive_mimicry_tunnel",
        description="Tunnel that mimics benign web-browsing traffic at each stage.",
        stages=(
            _stage(0, "mimic-idle", 6_000.0, 10.0, 0.55, Protocol.DOH, amplitude=0.18),
            _stage(1, "mimic-browse", 20_000.0, 6.0, 0.40, Protocol.DOH, amplitude=0.24),
            _stage(2, "mimic-bulk", 68_000.0, 3.5, 0.35, Protocol.DOH, amplitude=0.40),
            _stage(3, "mimic-drain", 140_000.0, 2.0, 0.40, Protocol.DOH, amplitude=0.52),
        ),
        retry=RetryPolicy(max_attempts=4, base_backoff_s=0.30),
        seed_offset=404,
    )

    return {
        s.name: s
        for s in (baseline, bursty, slow, handoff, mimicry)
    }


# --------------------------------------------------------------------------- #
# Campaign result
# --------------------------------------------------------------------------- #


@dataclass
class CampaignResult:
    """A fully materialised campaign, ready for export."""

    campaign_id: str
    scenario: ScenarioSpec
    seed: int
    flows: List[FlowRecord]
    events: List[CampaignEvent]
    outcome: CampaignOutcome
    delivered_bytes: float
    elapsed_s: float
    stage_delivered: Dict[int, float]
    stage_elapsed: Dict[int, float]
    endpoints_used: Dict[str, int]


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


class StagedTunnelingOrchestrator:
    """Executes a scenario and produces flows, events and a campaign result.

    The orchestrator is a pure scheduler: it never classifies, never trains,
    and never touches the filesystem. The exporter at the end of this module
    converts the result into the archive format the training pipeline consumes.
    We chose that split so that the orchestrator could be unit-tested against
    the observer protocol alone.
    """

    def __init__(
        self,
        scenario: ScenarioSpec,
        seed: int,
        resolver_pool: Optional[ResolverPool] = None,
        observers: Optional[Sequence[OrchestratorObserver]] = None,
        profiles: Optional[Mapping[str, BenignProfile]] = None,
    ) -> None:
        self.scenario = scenario
        self.seed = seed
        self.rng = np.random.default_rng((seed, scenario.seed_offset))
        self.resolver_pool = resolver_pool or _default_resolver_pool()
        self.observers = list(observers or [])
        self.profiles = dict(profiles or _benign_profiles())
        self._recording_observer = RecordingObserver()
        self.observers.append(self._recording_observer)

    # -- observer plumbing -------------------------------------------------- #

    def _notify(self, method: str, *args, **kwargs) -> bool:
        for observer in self.observers:
            try:
                getattr(observer, method)(*args, **kwargs)
            except Exception:                    # pragma: no cover - defensive
                LOGGER.exception(
                    "observer %s failed during %s", type(observer).__name__, method
                )
                return False
        return True

    # -- emission model ----------------------------------------------------- #

    def _emit_stage_packets(
        self,
        stage: StageSpec,
        n_packets: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """We synthesised the emission stream for one stage.

        The stream was drawn from a two-state HMM mirroring the benign web
        profile, then passed through the stage shaper. That mirrors the
        construction used by the polymorphic campaign simulator and keeps the
        two generators numerically compatible when both are used in the same
        evaluation.
        """
        from .polymorphic_shaping_simulator import (
            AttackFamily,
            DistributionSpec,
            HMMSpec,
            PolymorphicShaper,
        )

        base = self.profiles["web_browsing"].hmm
        stage_hmm = HMMSpec(
            n_states=base.n_states,
            initial=base.initial.copy(),
            transition=base.transition.copy(),
            size_dist=list(base.size_dist),
            iat_dist=list(base.iat_dist),
        )
        states = stage_hmm.sample_states(self.rng, n_packets)
        sizes = np.empty(n_packets, dtype=float)
        iats = np.empty(n_packets, dtype=float)
        for state in range(stage_hmm.n_states):
            mask = states == state
            if mask.any():
                sizes[mask] = stage_hmm.size_dist[state].sample(
                    self.rng, int(mask.sum())
                )
                iats[mask] = stage_hmm.iat_dist[state].sample(
                    self.rng, int(mask.sum())
                )
        shaper = PolymorphicShaper(
            family=AttackFamily.STAGED_TUNNEL,
            rng=self.rng,
            config=stage.shaper_config,
            mimic_pool=list(self.profiles.values()),
        )
        directions = np.ones(n_packets, dtype=int)
        sizes, iats, _ = shaper.apply(sizes, iats, directions)
        return np.clip(sizes, 32.0, None), np.clip(iats, 1e-4, None)

    def _packets_for_stage(self, stage: StageSpec) -> int:
        # We sized stages by payload target rather than packet count so that
        # the ratio between stage volumes remained stable across transports.
        mean_size = 380.0
        return max(4, int(math.ceil(stage.payload_bytes_target / mean_size)))

    # -- single-stage execution --------------------------------------------- #

    def _run_stage(
        self,
        campaign_id: str,
        stage: StageSpec,
        t0: float,
    ) -> Tuple[FlowRecord, float, float, Dict[str, int]]:
        if not self._notify("on_stage_enter", campaign_id, stage.index, stage, t0):
            raise _ObserverAbort("on_stage_enter")

        n_packets = self._packets_for_stage(stage)
        sizes, base_iats = self._emit_stage_packets(stage, n_packets)

        # The stage's intended interval shapes the emission cadence. We
        # rescaled the HMM iat stream so its median matched the stage interval,
        # then applied the stage's jitter.
        median_iat = float(np.median(base_iats))
        scale = stage.beacon_interval_s / max(median_iat, 1e-4)
        iats = base_iats * scale
        jitter = self.rng.uniform(
            1.0 - stage.beacon_jitter_ratio,
            1.0 + stage.beacon_jitter_ratio,
            size=n_packets,
        )
        iats = np.clip(iats * jitter, 1e-3, None)

        endpoint_counts: Dict[str, int] = {}
        delivered = 0.0
        t = t0
        max_stage_elapsed = stage.beacon_interval_s * n_packets * 8.0

        for i in range(n_packets):
            allow_switch = (i % max(1, n_packets // 4)) == 0
            endpoint = self.resolver_pool.select(self.rng, allow_switch)
            attempt = 0
            while attempt < self.scenario.retry.max_attempts:
                if not self.scenario.retry.attempt_fails(self.rng, endpoint, attempt):
                    break
                if not self._notify(
                    "on_retry", campaign_id, stage.index, attempt, t, endpoint
                ):
                    raise _ObserverAbort("on_retry")
                backoff = self.scenario.retry.backoff(attempt)
                t += backoff
                attempt += 1
            else:
                raise _RetryExhausted(stage.index)

            delivered += float(sizes[i])
            endpoint_counts[endpoint.resolver_id] = (
                endpoint_counts.get(endpoint.resolver_id, 0) + 1
            )
            if not self._notify(
                "on_beacon",
                campaign_id,
                stage.index,
                t,
                float(sizes[i]),
                endpoint,
            ):
                raise _ObserverAbort("on_beacon")
            t += float(iats[i])
            if (t - t0) > max_stage_elapsed:
                raise _StageBudgetExceeded(stage.index)

        # The stage protocol was chosen from the endpoint mix we actually used.
        protocols_used = {
            e.protocol
            for e in self.resolver_pool.endpoints
            if e.resolver_id in endpoint_counts
        }
        if stage.preferred_protocol in protocols_used:
            protocol = stage.preferred_protocol
        elif protocols_used:
            protocol = sorted(protocols_used, key=lambda p: p.value)[0]
        else:
            protocol = stage.preferred_protocol or Protocol.DOH

        flow = FlowRecord(
            flow_id=str(uuid.uuid4()),
            label=Label.ATTACK,
            family=AttackFamily.STAGED_TUNNEL,
            protocol=protocol,
            profile_name=f"stage:{stage.name}",
            sizes=sizes,
            iats=iats,
            directions=np.ones(n_packets, dtype=int),
            framing=FRAMING[protocol],
        )

        if not self._notify(
            "on_stage_exit", campaign_id, stage.index, t, delivered
        ):
            raise _ObserverAbort("on_stage_exit")
        return flow, t, delivered, endpoint_counts

    # -- campaign loop ------------------------------------------------------ #

    def run(self) -> CampaignResult:
        campaign_id = str(uuid.uuid4())
        start_wall = time.monotonic()
        LOGGER.info(
            "campaign %s starting scenario=%s seed=%d",
            campaign_id,
            self.scenario.name,
            self.seed,
        )
        if not self._notify(
            "on_campaign_start", campaign_id, self.scenario.stages, 0.0
        ):
            raise RuntimeError("campaign start rejected by observer")

        flows: List[FlowRecord] = []
        delivered: Dict[int, float] = {}
        elapsed: Dict[int, float] = {}
        endpoints_used: Dict[str, int] = {}
        t = 0.0
        outcome = CampaignOutcome.COMPLETED

        try:
            for stage in self.scenario.stages:
                flow, t, stage_delivered, stage_endpoints = self._run_stage(
                    campaign_id, stage, t
                )
                flows.append(flow)
                delivered[stage.index] = stage_delivered
                elapsed[stage.index] = t
                for resolver, count in stage_endpoints.items():
                    endpoints_used[resolver] = (
                        endpoints_used.get(resolver, 0) + count
                    )
        except _RetryExhausted:
            outcome = CampaignOutcome.ABORTED_MAX_RETRIES
        except _StageBudgetExceeded:
            outcome = CampaignOutcome.ABORTED_STAGE_BUDGET
        except _ObserverAbort:
            outcome = CampaignOutcome.ABORTED_OBSERVER

        if not self._notify("on_campaign_end", campaign_id, outcome, t):
            outcome = CampaignOutcome.ABORTED_OBSERVER

        wall = time.monotonic() - start_wall
        LOGGER.info(
            "campaign %s finished outcome=%s stages=%d wall=%.3fs",
            campaign_id,
            outcome.value,
            len(flows),
            wall,
        )
        return CampaignResult(
            campaign_id=campaign_id,
            scenario=self.scenario,
            seed=self.seed,
            flows=flows,
            events=list(self._recording_observer.events),
            outcome=outcome,
            delivered_bytes=sum(delivered.values()),
            elapsed_s=t,
            stage_delivered=dict(delivered),
            stage_elapsed=dict(elapsed),
            endpoints_used=dict(endpoints_used),
        )


# --------------------------------------------------------------------------- #
# Internal control-flow exceptions
# --------------------------------------------------------------------------- #


class _RetryExhausted(Exception):
    def __init__(self, stage_index: int) -> None:
        super().__init__(f"retries exhausted in stage {stage_index}")
        self.stage_index = stage_index


class _StageBudgetExceeded(Exception):
    def __init__(self, stage_index: int) -> None:
        super().__init__(f"stage budget exceeded in stage {stage_index}")
        self.stage_index = stage_index


class _ObserverAbort(Exception):
    def __init__(self, method: str) -> None:
        super().__init__(f"observer aborted during {method}")
        self.method = method


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def _sha256_of(obj: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _result_manifest(result: CampaignResult) -> Dict[str, object]:
    return {
        "module": "extended_experimentation.attack_scenarios.staged_tunneling_orchestrator",
        "campaign_id": result.campaign_id,
        "scenario": result.scenario.name,
        "description": result.scenario.description,
        "seed": result.seed,
        "seed_offset": result.scenario.seed_offset,
        "outcome": result.outcome.value,
        "n_stages": len(result.scenario.stages),
        "n_flows": len(result.flows),
        "delivered_bytes": result.delivered_bytes,
        "elapsed_s": result.elapsed_s,
        "stage_delivered": result.stage_delivered,
        "stage_elapsed": result.stage_elapsed,
        "endpoints_used": result.endpoints_used,
        "thesis_reference": THESIS_REFERENCE,
        "stage_specs": [
            {
                "index": s.index,
                "name": s.name,
                "payload_target": s.payload_bytes_target,
                "beacon_interval_s": s.beacon_interval_s,
                "beacon_jitter_ratio": s.beacon_jitter_ratio,
                "preferred_protocol": (
                    s.preferred_protocol.value if s.preferred_protocol else None
                ),
            }
            for s in result.scenario.stages
        ],
        "retry": asdict(result.scenario.retry),
    }


def _write_events(out_dir: Path, result: CampaignResult) -> None:
    path = out_dir / "events.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for event in result.events:
            handle.write(json.dumps(event.to_json()) + "\n")


def _write_flows(out_dir: Path, result: CampaignResult) -> None:
    path = out_dir / "flows.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for flow in result.flows:
            handle.write(json.dumps(flow.summary()) + "\n")


def _write_packet_archive(out_dir: Path, result: CampaignResult) -> None:
    if not result.flows:
        return
    arrays: Dict[str, np.ndarray] = {}
    for flow in result.flows:
        arrays[f"{flow.flow_id}/sizes"] = flow.sizes
        arrays[f"{flow.flow_id}/iats"] = flow.iats
        arrays[f"{flow.flow_id}/directions"] = flow.directions
    np.savez_compressed(out_dir / "packet_archive.npz", **arrays)


def export_campaign(out_dir: Path, result: CampaignResult) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _result_manifest(result)
    manifest["manifest_sha256"] = _sha256_of(manifest)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _write_events(out_dir, result)
    _write_flows(out_dir, result)
    _write_packet_archive(out_dir, result)
    LOGGER.info(
        "exported campaign %s (%d flows, %d events) to %s",
        result.campaign_id,
        len(result.flows),
        len(result.events),
        out_dir,
    )


def export_scenario_matrix(out_dir: Path, matrix: Mapping[str, ScenarioSpec]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenarios": [
            {
                "name": spec.name,
                "description": spec.description,
                "seed_offset": spec.seed_offset,
                "retry": asdict(spec.retry),
                "stages": [
                    {
                        "index": s.index,
                        "name": s.name,
                        "payload_target": s.payload_bytes_target,
                        "beacon_interval_s": s.beacon_interval_s,
                        "beacon_jitter_ratio": s.beacon_jitter_ratio,
                        "preferred_protocol": (
                            s.preferred_protocol.value
                            if s.preferred_protocol
                            else None
                        ),
                    }
                    for s in spec.stages
                ],
            }
            for spec in matrix.values()
        ]
    }
    payload["sha256"] = _sha256_of(payload)
    (out_dir / "scenario_matrix.json").write_text(json.dumps(payload, indent=2))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a staged-tunneling campaign for the MIRAGE-DoH extended "
            "experimentation suite."
        )
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        default="dns_exfil_baseline",
        help="scenario name (or 'all' to run every scenario in the matrix)",
    )
    parser.add_argument("--seed", type=int, default=20231130)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    matrix = default_scenario_matrix()
    export_scenario_matrix(args.out, matrix)

    names = list(matrix) if args.scenario == "all" else [args.scenario]
    for name in names:
        if name not in matrix:
            raise SystemExit(f"unknown scenario: {name}")
        orchestrator = StagedTunnelingOrchestrator(
            scenario=matrix[name], seed=args.seed
        )
        result = orchestrator.run()
        export_campaign(args.out / name, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
