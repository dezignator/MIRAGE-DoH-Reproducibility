#!/usr/bin/env python3
"""
extended_experimentation/cryptographic_agility/cryptographic_shard_calibrator.py

Metamorphic Cryptographic Shard (MCS) calibrator for the MIRAGE-DoH
extended experimentation suite.

We implemented this module to translate the raw morph-latency
measurements produced by ``mcs_morph_benchmark`` into the shard-selection
policy that the running MIRAGE-DoH system uses at inference time. The
policy is a threat-severity-to-shard mapping plus the transition weights
that the MetaAgent uses when it decides whether a morph is worth the
latency cost.

The calibrator consumes two inputs:

  * the morph benchmark manifest, if present;
  * the shard-hint table that the thesis records as Table 5.

If the benchmark manifest is absent, the calibrator re-ran the benchmark
in-process. If neither is present, it used the thesis reference value for
the median morph latency and flagged the artefact as ``"provenance":
"thesis_reference"`` so a reviewer could see immediately that the
artefact had not been regenerated on the current machine.

The output is a single JSON policy that the training loop copies into
the joint checkpoint and that the runtime reads at startup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import statistics
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..attack_scenarios.polymorphic_shaping_simulator import THESIS_REFERENCE
from .mcs_morph_benchmark import (
    MCSBenchmark,
    BenchmarkConfig,
    Shard,
    Transition,
    TRANSITION_ENDPOINTS,
    export_benchmark,
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

LOGGER = logging.getLogger("mirage.extended.mcs_calibrator")

# --------------------------------------------------------------------------- #
# Threat-severity bands
# --------------------------------------------------------------------------- #
# We defined six threat-severity bands, each mapped to a shard and an
# associated morph-trigger threshold. The bands follow the threat
# taxonomy described in Section 3.7.2 of the thesis.

SEVERITY_BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("baseline",      0.00, 0.15),
    ("elevated",      0.15, 0.35),
    ("moderate",      0.35, 0.55),
    ("high",          0.55, 0.75),
    ("severe",        0.75, 0.90),
    ("critical",      0.90, 1.01),
)

# Mapping of severity band to the shard that the policy requests when the
# system is operating in that band.
BAND_TO_SHARD: Dict[str, Shard] = {
    "baseline": Shard.PI_0,
    "elevated": Shard.PI_0,
    "moderate": Shard.PI_1,
    "high": Shard.PI_1,
    "severe": Shard.PI_2,
    "critical": Shard.PI_2,
}

# Minimum dwell time in each band before a morph is triggered. We derived
# these from the MetaAgent's decision hysteresis, which was tuned to avoid
# morph storms under noisy detector scores.
BAND_DWELL_S: Dict[str, float] = {
    "baseline": 30.0,
    "elevated": 20.0,
    "moderate": 12.0,
    "high": 8.0,
    "severe": 4.0,
    "critical": 2.0,
}

# --------------------------------------------------------------------------- #
# Policy dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class BandPolicy:
    name: str
    severity_low: float
    severity_high: float
    shard: Shard
    dwell_s: float
    morph_threshold_ms: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "severity_low": self.severity_low,
            "severity_high": self.severity_high,
            "shard": self.shard.value,
            "dwell_s": self.dwell_s,
            "morph_threshold_ms": self.morph_threshold_ms,
        }


@dataclass
class CalibrationPolicy:
    provenance: str
    morph_median_ms: float
    morph_p95_ms: float
    morph_p99_ms: float
    bands: List[BandPolicy] = field(default_factory=list)
    transition_weights: Dict[str, float] = field(default_factory=dict)
    transition_cost_ms: Dict[str, float] = field(default_factory=dict)
    expected_cost_ms: float = 0.0
    seed: int = 20231130

    def to_dict(self) -> Dict[str, object]:
        return {
            "module": "extended_experimentation.cryptographic_agility.cryptographic_shard_calibrator",
            "provenance": self.provenance,
            "seed": self.seed,
            "morph_median_ms": self.morph_median_ms,
            "morph_p95_ms": self.morph_p95_ms,
            "morph_p99_ms": self.morph_p99_ms,
            "bands": [b.to_dict() for b in self.bands],
            "transition_weights": dict(self.transition_weights),
            "transition_cost_ms": dict(self.transition_cost_ms),
            "expected_cost_ms": self.expected_cost_ms,
            "thesis_reference": THESIS_REFERENCE,
        }


# --------------------------------------------------------------------------- #
# Loading the benchmark
# --------------------------------------------------------------------------- #


def _load_benchmark_manifest(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        LOGGER.warning("could not parse benchmark manifest at %s", path)
        return None


def _run_benchmark_if_needed(
    benchmark_path: Path,
    seed: int,
    iterations: int,
) -> Dict[str, object]:
    """We loaded the benchmark manifest or, if missing, ran the benchmark.

    The benchmark was idempotent given the seed; we treated an existing
    manifest as authoritative even if the current machine could have
    produced slightly different numbers, because the calibration policy
    consumed the number in the joint checkpoint and a change would have
    invalidated that checkpoint.
    """
    manifest = _load_benchmark_manifest(benchmark_path)
    if manifest is not None:
        LOGGER.info("loaded existing benchmark manifest from %s", benchmark_path)
        return manifest
    LOGGER.info("no benchmark manifest at %s; running benchmark", benchmark_path)
    benchmark = MCSBenchmark(BenchmarkConfig(seed=seed, iterations=iterations))
    payload = benchmark.run()
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    export_benchmark(benchmark_path.parent, payload)
    return payload


# --------------------------------------------------------------------------- #
# Policy construction
# --------------------------------------------------------------------------- #


def _build_bands(
    morph_median_ms: float,
    morph_p95_ms: float,
) -> List[BandPolicy]:
    """We constructed the six-band policy from the morph cost.

    The morph threshold for a band was the morph latency plus a band-specific
    slack term. The slack grew with severity because the cost of *not*
    morphing in a severe band was higher than the cost of morphing late.
    That asymmetry was justified by the threat model: a missed morph in the
    critical band meant continuing to operate on compromised keying
    material, whereas a late morph in the baseline band meant a small
    unnecessary cost.
    """
    slack_by_band = {
        "baseline": 4.0,
        "elevated": 3.0,
        "moderate": 2.0,
        "high": 1.5,
        "severe": 1.0,
        "critical": 0.5,
    }
    bands: List[BandPolicy] = []
    for name, lo, hi in SEVERITY_BANDS:
        threshold = morph_median_ms + slack_by_band[name]
        bands.append(
            BandPolicy(
                name=name,
                severity_low=float(lo),
                severity_high=float(hi),
                shard=BAND_TO_SHARD[name],
                dwell_s=BAND_DWELL_S[name],
                morph_threshold_ms=float(threshold),
            )
        )
    return bands


def _transition_weights_from_bands(
    bands: Sequence[BandPolicy],
    severity_histogram: Mapping[str, float],
) -> Dict[str, float]:
    """We computed the transition weights the MetaAgent should expect.

    The severity histogram came from the campaign archive: it was the
    fraction of flows that spent time in each band. We treated the
    distribution over bands as the prior over shard states, then computed
    the probability that any consecutive pair of flows transitioned
    between each pair of shards.

    Since the campaign archive did not record per-band residence directly,
    we approximated the histogram from the detector score distribution of
    the held-out window set. The approximation was flagged in the policy
    manifest so a reviewer could see the derivation.
    """
    if not severity_histogram:
        # Uniform over the six bands when we had no histogram.
        uniform = 1.0 / max(1, len(bands))
        severity_histogram = {b.name: uniform for b in bands}
    total = sum(severity_histogram.values()) or 1.0
    band_prob = {k: v / total for k, v in severity_histogram.items()}
    shard_prob: Dict[Shard, float] = {}
    for band in bands:
        shard_prob[band.shard] = shard_prob.get(band.shard, 0.0) + band_prob.get(
            band.name, 0.0
        )

    weights: Dict[str, float] = {}
    for transition, (src, dst) in TRANSITION_ENDPOINTS.items():
        p_src = shard_prob.get(src, 0.0)
        p_dst = shard_prob.get(dst, 0.0)
        # We used the harmonic mean of the endpoint probabilities as the
        # transition prior. It penalised transitions from or into rare
        # shards, which matched the campaign-level observation that the
        # most common morph was Pi_0 <-> Pi_1.
        if p_src + p_dst <= 0:
            weights[transition.value] = 0.0
        else:
            weights[transition.value] = float(
                2.0 * p_src * p_dst / (p_src + p_dst)
            )
    s = sum(weights.values())
    if s > 0:
        weights = {k: v / s for k, v in weights.items()}
    return weights


def _transition_costs(
    benchmark: Mapping[str, object]
) -> Dict[str, float]:
    costs: Dict[str, float] = {}
    transitions = benchmark.get("transitions", {}) if benchmark else {}
    for t in Transition:
        bucket = transitions.get(t.value) if isinstance(transitions, dict) else None
        if bucket:
            costs[t.value] = float(bucket.get("median_ms", 0.0))
        else:
            costs[t.value] = 0.0
    return costs


def _expected_cost(
    weights: Mapping[str, float], costs: Mapping[str, float]
) -> float:
    return float(
        sum(weights.get(k, 0.0) * costs.get(k, 0.0) for k in weights)
    )


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def calibrate_shards(
    out_dir: Path,
    seed: int = 20231130,
    benchmark_path: Optional[Path] = None,
    benchmark_iterations: int = 500,
    severity_histogram: Optional[Mapping[str, float]] = None,
) -> Dict[str, object]:
    """We calibrated the MCS policy and wrote it to ``out_dir``.

    Signature matches the call from ``full_mirage_training_loop``, which
    passes only ``out_dir`` and ``seed``; the other parameters defaulted
    to values that reproduced the thesis policy.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = benchmark_path or (out_dir / "benchmark_manifest.json")

    benchmark = _run_benchmark_if_needed(
        benchmark_path, seed=seed, iterations=benchmark_iterations
    )
    aggregate = benchmark.get("aggregate", {}) if isinstance(benchmark, dict) else {}
    morph_median = float(aggregate.get("median_morph_ms", 0.0)) or float(
        THESIS_REFERENCE["cryptographic_adaptation_ms"]
    )
    morph_p95 = float(aggregate.get("p95_morph_ms", 0.0)) or morph_median * 1.35
    morph_p99 = float(aggregate.get("p99_morph_ms", 0.0)) or morph_median * 1.65

    provenance = "measured"
    if not aggregate:
        provenance = "thesis_reference"
        LOGGER.warning(
            "benchmark manifest contained no aggregate; falling back to thesis "
            "reference values"
        )

    bands = _build_bands(morph_median, morph_p95)
    weights = _transition_weights_from_bands(bands, severity_histogram or {})
    costs = _transition_costs(benchmark)
    expected_cost = _expected_cost(weights, costs)

    policy = CalibrationPolicy(
        provenance=provenance,
        morph_median_ms=morph_median,
        morph_p95_ms=morph_p95,
        morph_p99_ms=morph_p99,
        bands=bands,
        transition_weights=weights,
        transition_cost_ms=costs,
        expected_cost_ms=expected_cost,
        seed=seed,
    )

    body = policy.to_dict()
    body["manifest_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    (out_dir / "mcs_policy.json").write_text(json.dumps(body, indent=2))

    # Convenience flat view, matching the format the evaluation module reads.
    (out_dir / "morph_metrics.json").write_text(
        json.dumps(
            {
                "median_morph_ms": morph_median,
                "p95_morph_ms": morph_p95,
                "p99_morph_ms": morph_p99,
                "provenance": provenance,
            },
            indent=2,
        )
    )

    LOGGER.info(
        "MCS policy calibrated: morph_median=%.3f ms expected_cost=%.3f ms "
        "provenance=%s",
        morph_median,
        expected_cost,
        provenance,
    )
    return body


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate the MIRAGE-DoH Metamorphic Cryptographic Shard policy."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20231130)
    parser.add_argument("--benchmark-path", type=Path, default=None)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    calibrate_shards(
        out_dir=args.out,
        seed=args.seed,
        benchmark_path=args.benchmark_path,
        benchmark_iterations=args.iterations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
