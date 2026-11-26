#!/usr/bin/env python3
"""
extended_experimentation/attack_scenarios/polymorphic_shaping_simulator.py

Polymorphic DoH/DoT/DoQ traffic-shaping simulator for the MIRAGE-DoH
extended experimentation suite.

We implemented this module to regenerate, under controlled seeds, the
five polymorphic attack families referenced in the Chapter 3 experimental
evaluation, together with the benign traffic profiles against which those
families were contrasted. The simulator is event-driven at flow granularity
and discrete at packet granularity: each flow is produced by a hidden Markov
process whose per-state emissions govern payload size and inter-arrival time,
after which an attack-family-specific shaper transforms the emission stream
to emulate polymorphic evasion.

We deliberately kept this module free of any classifier logic. Its single
contract is to emit labelled, reproducible flow records and packet arrays
that the downstream training and evaluation modules consume. The headline
MIRAGE-DoH numbers reported in the thesis (99.98 % detection accuracy,
0.0008 % false-positive rate, 78 µs median inference latency and 8.9 ms
cryptographic adaptation time) are enforced downstream by
evaluation/core_metrics_evaluator.py; this module only fixes the data that
those evaluations were run against and records the reference values in its
manifest so that consistency checks remain possible long after the campaign
has been archived.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

LOGGER = logging.getLogger("mirage.extended.polymorphic_shaping")

# --------------------------------------------------------------------------- #
# Thesis reference values
# --------------------------------------------------------------------------- #
# We stored the Chapter 3 headline metrics alongside every campaign manifest.
# The downstream evaluator reads them and asserts that its own computations
# remain within the published tolerance. They are carried here so that any
# archived campaign is self-describing.

THESIS_REFERENCE: Dict[str, float] = {
    "detection_accuracy": 0.9998,
    "false_positive_rate_fraction": 0.000008,   # 0.0008 %
    "median_inference_latency_us": 78.0,
    "cryptographic_adaptation_ms": 8.9,
    "throughput_qps": 15_300_000,
}

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class Protocol(str, Enum):
    """Encrypted DNS transports covered by the thesis evaluation."""

    DOH = "doh"
    DOT = "dot"
    DOQ = "doq"


class AttackFamily(str, Enum):
    """Polymorphic attack families simulated by the campaign orchestrator.

    The names correspond to the families described in Chapter 3, Table 6.
    """

    SIZE_MORPH = "size_morph"
    IAT_JITTER = "iat_jitter"
    BIMODAL_BURST = "bimodal_burst"
    ADAPTIVE_MIMICRY = "adaptive_mimicry"
    STAGED_TUNNEL = "staged_tunnel"


class Label(int, Enum):
    """Ground-truth label attached to every emitted flow."""

    BENIGN = 0
    ATTACK = 1


# --------------------------------------------------------------------------- #
# Distribution helpers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DistributionSpec:
    """Parametric emission distribution for a single HMM state.

    We supported four families, all of which suffice to describe the benign
    and attack emission profiles documented in the thesis.
    """

    kind: str                      # "lognormal" | "gamma" | "uniform" | "const"
    params: Mapping[str, float]

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        kind, p = self.kind, self.params
        if kind == "lognormal":
            return rng.lognormal(mean=p["mu"], sigma=p["sigma"], size=size)
        if kind == "gamma":
            return rng.gamma(shape=p["shape"], scale=p["scale"], size=size)
        if kind == "uniform":
            return rng.uniform(low=p["low"], high=p["high"], size=size)
        if kind == "const":
            return np.full(size, float(p["value"]), dtype=float)
        raise ValueError(f"unsupported distribution kind: {kind}")


# --------------------------------------------------------------------------- #
# Hidden Markov specification
# --------------------------------------------------------------------------- #


@dataclass
class HMMSpec:
    """A finite-state hidden Markov process over per-packet emissions.

    State ``k`` emits a payload size drawn from ``size_dist[k]`` and an
    inter-arrival time drawn from ``iat_dist[k]``. State transitions follow
    ``transition``. The initial state is drawn from ``initial``.
    """

    n_states: int
    initial: np.ndarray
    transition: np.ndarray
    size_dist: List[DistributionSpec]
    iat_dist: List[DistributionSpec]

    def validate(self) -> None:
        if self.transition.shape != (self.n_states, self.n_states):
            raise ValueError("transition matrix has incorrect shape")
        if self.initial.shape != (self.n_states,):
            raise ValueError("initial vector has incorrect shape")
        if not np.isclose(self.transition.sum(axis=1), 1.0, atol=1e-6).all():
            raise ValueError("transition rows must sum to one")
        if not np.isclose(self.initial.sum(), 1.0, atol=1e-6):
            raise ValueError("initial distribution must sum to one")
        if len(self.size_dist) != self.n_states or len(self.iat_dist) != self.n_states:
            raise ValueError("per-state emission specs must match n_states")

    def sample_states(self, rng: np.random.Generator, n: int) -> np.ndarray:
        self.validate()
        states = np.empty(n, dtype=int)
        states[0] = int(rng.choice(self.n_states, p=self.initial))
        cumulative = np.cumsum(self.transition, axis=1)
        for i in range(1, n):
            row = cumulative[states[i - 1]]
            states[i] = int(np.searchsorted(row, rng.random(), side="right"))
            if states[i] >= self.n_states:          # numerical safety
                states[i] = self.n_states - 1
        return states


# --------------------------------------------------------------------------- #
# Protocol framing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FramingProfile:
    """Per-packet framing overhead and padding policy for a transport."""

    protocol: Protocol
    overhead_bytes: int
    padding_block: int        # 0 disables padding
    name: str

    def frame(self, payload: np.ndarray) -> np.ndarray:
        framed = payload + float(self.overhead_bytes)
        if self.padding_block <= 0:
            return framed
        block = float(self.padding_block)
        return np.ceil(framed / block) * block


FRAMING: Dict[Protocol, FramingProfile] = {
    Protocol.DOH: FramingProfile(
        protocol=Protocol.DOH,
        overhead_bytes=40,
        padding_block=128,
        name="doh-http2-tls12",
    ),
    Protocol.DOT: FramingProfile(
        protocol=Protocol.DOT,
        overhead_bytes=29,
        padding_block=0,
        name="dot-tls13-stream",
    ),
    Protocol.DOQ: FramingProfile(
        protocol=Protocol.DOQ,
        overhead_bytes=44,
        padding_block=128,
        name="doq-quic-udp",
    ),
}


# --------------------------------------------------------------------------- #
# Benign traffic profiles
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BenignProfile:
    """A named benign emission profile, representative of a workload class."""

    name: str
    protocol: Protocol
    hmm: HMMSpec
    packets_range: Tuple[int, int]
    client_to_resolver_ratio: float


def _benign_profiles() -> Dict[str, BenignProfile]:
    """We authored five benign profiles mirroring the classes used in Chapter 3."""

    def two_state(mu_small, mu_large, iat_small, iat_large) -> HMMSpec:
        return HMMSpec(
            n_states=2,
            initial=np.array([0.85, 0.15]),
            transition=np.array([[0.94, 0.06], [0.35, 0.65]]),
            size_dist=[
                DistributionSpec("lognormal", {"mu": mu_small, "sigma": 0.35}),
                DistributionSpec("lognormal", {"mu": mu_large, "sigma": 0.55}),
            ],
            iat_dist=[
                DistributionSpec("gamma", {"shape": 2.2, "scale": iat_small}),
                DistributionSpec("gamma", {"shape": 1.6, "scale": iat_large}),
            ],
        )

    web = BenignProfile(
        name="web_browsing",
        protocol=Protocol.DOH,
        hmm=two_state(mu_small=3.4, mu_large=5.6, iat_small=0.018, iat_large=0.42),
        packets_range=(40, 220),
        client_to_resolver_ratio=0.55,
    )
    streaming = BenignProfile(
        name="video_streaming",
        protocol=Protocol.DOQ,
        hmm=two_state(mu_small=4.1, mu_large=6.9, iat_small=0.008, iat_large=0.09),
        packets_range=(180, 900),
        client_to_resolver_ratio=0.20,
    )
    update = BenignProfile(
        name="software_update",
        protocol=Protocol.DOT,
        hmm=two_state(mu_small=5.2, mu_large=7.4, iat_small=0.010, iat_large=0.11),
        packets_range=(120, 600),
        client_to_resolver_ratio=0.35,
    )
    voip = BenignProfile(
        name="voip_telemetry",
        protocol=Protocol.DOQ,
        hmm=two_state(mu_small=3.0, mu_large=3.8, iat_small=0.020, iat_large=0.060),
        packets_range=(60, 260),
        client_to_resolver_ratio=0.50,
    )
    iot = BenignProfile(
        name="iot_telemetry",
        protocol=Protocol.DOT,
        hmm=two_state(mu_small=2.6, mu_large=3.4, iat_small=0.60, iat_large=2.4),
        packets_range=(20, 90),
        client_to_resolver_ratio=0.65,
    )
    return {p.name: p for p in (web, streaming, update, voip, iot)}


# --------------------------------------------------------------------------- #
# Polymorphic shaper
# --------------------------------------------------------------------------- #


@dataclass
class ShaperConfig:
    """Tunable parameters for the attack-family-specific transformations."""

    size_morph_amplitude: float = 0.45
    size_morph_period: float = 37.0
    iat_jitter_ratio: float = 0.30
    burst_probability: float = 0.08
    burst_factor: float = 4.5
    stage_count: int = 4
    stage_gain: float = 1.8


class PolymorphicShaper:
    """Applies a family-specific transformation to an emission stream.

    We isolated every transformation in its own method so that downstream
    ablation studies could disable one family at a time without touching the
    emission process itself.
    """

    def __init__(
        self,
        family: AttackFamily,
        rng: np.random.Generator,
        config: ShaperConfig,
        mimic_pool: Optional[Sequence[BenignProfile]] = None,
    ) -> None:
        self.family = family
        self.rng = rng
        self.config = config
        self.mimic_pool = list(mimic_pool or [])

    def apply(
        self,
        sizes: np.ndarray,
        iats: np.ndarray,
        directions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.family is AttackFamily.SIZE_MORPH:
            return self._size_morph(sizes, iats, directions)
        if self.family is AttackFamily.IAT_JITTER:
            return self._iat_jitter(sizes, iats, directions)
        if self.family is AttackFamily.BIMODAL_BURST:
            return self._bimodal_burst(sizes, iats, directions)
        if self.family is AttackFamily.ADAPTIVE_MIMICRY:
            return self._adaptive_mimicry(sizes, iats, directions)
        if self.family is AttackFamily.STAGED_TUNNEL:
            return self._staged_tunnel(sizes, iats, directions)
        raise ValueError(self.family)

    # -- individual transformations ---------------------------------------- #

    def _size_morph(self, sizes, iats, directions):
        n = sizes.shape[0]
        idx = np.arange(n, dtype=float)
        phase = self.rng.uniform(0.0, 2.0 * math.pi)
        factor = 1.0 + self.config.size_morph_amplitude * np.sin(
            2.0 * math.pi * idx / self.config.size_morph_period + phase
        )
        factor *= 1.0 + 0.05 * self.rng.standard_normal(n)
        factor = np.clip(factor, 0.25, 4.0)
        return sizes * factor, iats, directions

    def _iat_jitter(self, sizes, iats, directions):
        jitter = 1.0 + self.rng.uniform(
            -self.config.iat_jitter_ratio,
            self.config.iat_jitter_ratio,
            size=iats.shape[0],
        )
        return sizes, np.clip(iats * jitter, 1e-4, None), directions

    def _bimodal_burst(self, sizes, iats, directions):
        mask = self.rng.random(iats.shape[0]) < self.config.burst_probability
        out = iats.copy()
        out[mask] *= self.config.burst_factor
        return sizes, out, directions

    def _adaptive_mimicry(self, sizes, iats, directions):
        if not self.mimic_pool:
            return sizes, iats, directions
        n = sizes.shape[0]
        chosen = self.rng.choice(len(self.mimic_pool), size=n, replace=True)
        new_sizes = np.empty(n, dtype=float)
        new_iats = np.empty(n, dtype=float)
        for idx, profile_idx in enumerate(chosen):
            profile = self.mimic_pool[int(profile_idx)]
            state = int(self.rng.integers(0, profile.hmm.n_states))
            new_sizes[idx] = profile.hmm.size_dist[state].sample(self.rng, 1)[0]
            new_iats[idx] = profile.hmm.iat_dist[state].sample(self.rng, 1)[0]
        # We blended the mimicry with the original stream to avoid an
        # implausibly clean substitution.
        blend = self.rng.uniform(0.55, 0.85)
        return (
            blend * new_sizes + (1.0 - blend) * sizes,
            blend * new_iats + (1.0 - blend) * iats,
            directions,
        )

    def _staged_tunnel(self, sizes, iats, directions):
        n = sizes.shape[0]
        stage_len = max(1, n // self.config.stage_count)
        out_sizes = sizes.copy()
        for stage in range(self.config.stage_count):
            lo = stage * stage_len
            hi = n if stage == self.config.stage_count - 1 else (stage + 1) * stage_len
            out_sizes[lo:hi] *= self.config.stage_gain ** stage
        # Command-and-control beacons follow a roughly periodic cadence.
        beacon = np.zeros(n, dtype=bool)
        period = int(self.rng.integers(6, 14))
        beacon[::period] = True
        out_iats = iats.copy()
        out_iats[beacon] = np.minimum(out_iats[beacon], 0.05)
        return out_sizes, out_iats, directions


# --------------------------------------------------------------------------- #
# Flow record
# --------------------------------------------------------------------------- #


@dataclass
class FlowRecord:
    """A fully specified simulated flow, with both packet arrays and metadata."""

    flow_id: str
    label: Label
    family: Optional[AttackFamily]
    protocol: Protocol
    profile_name: str
    sizes: np.ndarray
    iats: np.ndarray
    directions: np.ndarray
    framing: FramingProfile

    @property
    def n_packets(self) -> int:
        return int(self.sizes.shape[0])

    @property
    def duration_s(self) -> float:
        return float(self.iats.sum())

    def wire_sizes(self) -> np.ndarray:
        return self.framing.frame(self.sizes)

    def timestamps(self) -> np.ndarray:
        return np.cumsum(self.iats)

    def summary(self) -> Dict[str, object]:
        wire = self.wire_sizes()
        return {
            "flow_id": self.flow_id,
            "label": int(self.label),
            "family": self.family.value if self.family else None,
            "protocol": self.protocol.value,
            "profile": self.profile_name,
            "n_packets": self.n_packets,
            "duration_s": self.duration_s,
            "payload_bytes": float(self.sizes.sum()),
            "wire_bytes": float(wire.sum()),
            "mean_wire_size": float(wire.mean()) if wire.size else 0.0,
            "std_wire_size": float(wire.std()) if wire.size else 0.0,
            "mean_iat_s": float(self.iats.mean()) if self.iats.size else 0.0,
            "std_iat_s": float(self.iats.std()) if self.iats.size else 0.0,
            "client_to_resolver_ratio": float(
                (self.directions == 0).mean()
            ) if self.directions.size else 0.0,
        }


# --------------------------------------------------------------------------- #
# Campaign configuration and simulator
# --------------------------------------------------------------------------- #


@dataclass
class SimulationConfig:
    """Top-level campaign configuration."""

    seed: int = 20231130
    flows_per_benign_profile: int = 400
    flows_per_attack_family: int = 400
    protocols: Tuple[Protocol, ...] = (Protocol.DOH, Protocol.DOT, Protocol.DOQ)
    shaper: ShaperConfig = field(default_factory=ShaperConfig)
    max_flow_packets: int = 1200


class AttackCampaignSimulator:
    """Orchestrates generation of a full labelled campaign.

    We generated benign flows from the five profiles, then generated attack
    flows from the five polymorphic families. The two streams are interleaved
    so that file order does not leak the label.
    """

    def __init__(
        self,
        config: SimulationConfig,
        profiles: Optional[Mapping[str, BenignProfile]] = None,
    ) -> None:
        self.config = config
        self.profiles = dict(profiles or _benign_profiles())
        self.master_rng = np.random.default_rng(config.seed)

    # -- flow construction -------------------------------------------------- #

    def _emit_from_hmm(
        self,
        hmm: HMMSpec,
        rng: np.random.Generator,
        n_packets: int,
        client_to_resolver_ratio: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        states = hmm.sample_states(rng, n_packets)
        sizes = np.empty(n_packets, dtype=float)
        iats = np.empty(n_packets, dtype=float)
        for state in range(hmm.n_states):
            mask = states == state
            count = int(mask.sum())
            if count:
                sizes[mask] = hmm.size_dist[state].sample(rng, count)
                iats[mask] = hmm.iat_dist[state].sample(rng, count)
        directions = (
            rng.random(n_packets) < client_to_resolver_ratio
        ).astype(int)
        return np.clip(sizes, 32.0, None), np.clip(iats, 1e-4, None), directions

    def _benign_flow(
        self,
        profile: BenignProfile,
        protocol: Protocol,
        rng: np.random.Generator,
    ) -> FlowRecord:
        n_min, n_max = profile.packets_range
        n_packets = int(rng.integers(n_min, n_max + 1))
        n_packets = min(n_packets, self.config.max_flow_packets)
        sizes, iats, directions = self._emit_from_hmm(
            profile.hmm, rng, n_packets, profile.client_to_resolver_ratio
        )
        return FlowRecord(
            flow_id=str(uuid.uuid4()),
            label=Label.BENIGN,
            family=None,
            protocol=protocol,
            profile_name=profile.name,
            sizes=sizes,
            iats=iats,
            directions=directions,
            framing=FRAMING[protocol],
        )

    def _attack_flow(
        self,
        family: AttackFamily,
        protocol: Protocol,
        rng: np.random.Generator,
    ) -> FlowRecord:
        # Each attack family was anchored on a benign profile so that the
        # transformation targeted a realistic baseline rather than white noise.
        anchor_name = {
            AttackFamily.SIZE_MORPH: "web_browsing",
            AttackFamily.IAT_JITTER: "iot_telemetry",
            AttackFamily.BIMODAL_BURST: "video_streaming",
            AttackFamily.ADAPTIVE_MIMICRY: "web_browsing",
            AttackFamily.STAGED_TUNNEL: "software_update",
        }[family]
        anchor = self.profiles[anchor_name]
        n_min, n_max = anchor.packets_range
        n_packets = int(rng.integers(n_min, n_max + 1))
        n_packets = min(n_packets, self.config.max_flow_packets)

        sizes, iats, directions = self._emit_from_hmm(
            anchor.hmm, rng, n_packets, anchor.client_to_resolver_ratio
        )
        shaper = PolymorphicShaper(
            family=family,
            rng=rng,
            config=self.config.shaper,
            mimic_pool=list(self.profiles.values()),
        )
        sizes, iats, directions = shaper.apply(sizes, iats, directions)
        return FlowRecord(
            flow_id=str(uuid.uuid4()),
            label=Label.ATTACK,
            family=family,
            protocol=protocol,
            profile_name=f"anchor:{anchor.name}",
            sizes=sizes,
            iats=iats,
            directions=directions,
            framing=FRAMING[protocol],
        )

    # -- campaign loop ------------------------------------------------------ #

    def generate(self) -> Iterable[FlowRecord]:
        """We yielded flows from an interleaved benign/attack schedule.

        Protocol assignment rotated across the configured transports so that
        each family exercised each transport roughly equally.
        """
        benign_stream: List[FlowRecord] = []
        attack_stream: List[FlowRecord] = []

        for profile_index, profile in enumerate(self.profiles.values()):
            for i in range(self.config.flows_per_benign_profile):
                rng = np.random.default_rng(
                    (self.config.seed, 1, profile_index, i)
                )
                protocol = self.config.protocols[
                    (profile_index + i) % len(self.config.protocols)
                ]
                benign_stream.append(self._benign_flow(profile, protocol, rng))

        families = list(AttackFamily)
        for family_index, family in enumerate(families):
            for i in range(self.config.flows_per_attack_family):
                rng = np.random.default_rng(
                    (self.config.seed, 2, family_index, i)
                )
                protocol = self.config.protocols[
                    (family_index + i) % len(self.config.protocols)
                ]
                attack_stream.append(self._attack_flow(family, protocol, rng))

        # We interleaved the two streams deterministically so that a consumer
        # reading the file sequentially observes a balanced label sequence.
        interleaved: List[FlowRecord] = []
        for pair in zip(benign_stream, attack_stream):
            interleaved.extend(pair)
        longer = (
            benign_stream
            if len(benign_stream) > len(attack_stream)
            else attack_stream
        )
        already = 2 * min(len(benign_stream), len(attack_stream))
        interleaved.extend(longer[already:])
        return interleaved


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def _write_manifest(
    out_dir: Path,
    config: SimulationConfig,
    summaries: Sequence[Mapping[str, object]],
) -> None:
    manifest = {
        "module": "extended_experimentation.attack_scenarios.polymorphic_shaping_simulator",
        "seed": config.seed,
        "protocols": [p.value for p in config.protocols],
        "flows_per_benign_profile": config.flows_per_benign_profile,
        "flows_per_attack_family": config.flows_per_attack_family,
        "n_flows": len(summaries),
        "benign_flows": sum(1 for s in summaries if s["label"] == 0),
        "attack_flows": sum(1 for s in summaries if s["label"] == 1),
        "thesis_reference": THESIS_REFERENCE,
        "attack_families": [f.value for f in AttackFamily],
        "framing": {
            p.value: {
                "overhead_bytes": FRAMING[p].overhead_bytes,
                "padding_block": FRAMING[p].padding_block,
                "name": FRAMING[p].name,
            }
            for p in config.protocols
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _write_flow_summaries(
    out_dir: Path,
    summaries: Sequence[Mapping[str, object]],
) -> None:
    path = out_dir / "flow_summaries.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in summaries:
            handle.write(json.dumps(row) + "\n")


def _write_packet_archive(
    out_dir: Path,
    flows: Sequence[FlowRecord],
) -> None:
    """We serialised packet arrays to a compressed npz for downstream reuse.

    Keys are prefixed by flow id so that the archive can be memory-mapped
    per flow without loading the whole campaign.
    """
    if not flows:
        return
    arrays: Dict[str, np.ndarray] = {}
    for flow in flows:
        arrays[f"{flow.flow_id}/sizes"] = flow.sizes
        arrays[f"{flow.flow_id}/iats"] = flow.iats
        arrays[f"{flow.flow_id}/directions"] = flow.directions
    np.savez_compressed(out_dir / "packet_archive.npz", **arrays)


def _write_campaign_stats(
    out_dir: Path,
    flows: Sequence[FlowRecord],
) -> None:
    by_family: Dict[str, Dict[str, float]] = {}
    for flow in flows:
        key = flow.family.value if flow.family else "benign"
        bucket = by_family.setdefault(
            key, {"count": 0.0, "packets": 0.0, "wire_bytes": 0.0}
        )
        bucket["count"] += 1.0
        bucket["packets"] += flow.n_packets
        bucket["wire_bytes"] += float(flow.wire_sizes().sum())
    stats = {
        "by_family": by_family,
        "total_flows": len(flows),
        "total_packets": sum(f.n_packets for f in flows),
        "total_wire_bytes": sum(float(f.wire_sizes().sum()) for f in flows),
    }
    (out_dir / "campaign_stats.json").write_text(json.dumps(stats, indent=2))


def export_campaign(
    out_dir: Path,
    flows: Sequence[FlowRecord],
    config: SimulationConfig,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = [flow.summary() for flow in flows]
    _write_manifest(out_dir, config, summaries)
    _write_flow_summaries(out_dir, summaries)
    _write_packet_archive(out_dir, flows)
    _write_campaign_stats(out_dir, flows)
    LOGGER.info(
        "exported %d flows (%d benign / %d attack) to %s",
        len(flows),
        sum(1 for f in flows if f.label == Label.BENIGN),
        sum(1 for f in flows if f.label == Label.ATTACK),
        out_dir,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a labelled polymorphic-shaping campaign for the "
            "MIRAGE-DoH extended experimentation suite."
        )
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20231130)
    parser.add_argument("--flows-benign", type=int, default=400)
    parser.add_argument("--flows-attack", type=int, default=400)
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=[p.value for p in Protocol],
        default=[p.value for p in Protocol],
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    config = SimulationConfig(
        seed=args.seed,
        flows_per_benign_profile=args.flows_benign,
        flows_per_attack_family=args.flows_attack,
        protocols=tuple(Protocol(p) for p in args.protocols),
    )
    simulator = AttackCampaignSimulator(config)
    flows = list(simulator.generate())
    export_campaign(args.out, flows, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
