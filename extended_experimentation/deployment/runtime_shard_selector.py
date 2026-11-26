#!/usr/bin/env python3
"""
extended_experimentation/deployment/runtime_shard_selector.py

Runtime shard selector for the MIRAGE-DoH extended experimentation suite.

We implemented this module to close the loop between the calibrated MCS
policy and the running system. The calibrator produced a policy; the
selector consumed it at runtime, tracked the current shard state and the
current severity band, and decided whether a morph was warranted for each
inference call.

The selector was deliberately stateful and single-tenant. Each resolver
process was expected to instantiate one selector and hold it across
inference calls. Sharing a selector across tenants would have required
per-tenant state, which in turn would have required a shared store; that
was out of scope for the thesis and would have changed the latency
profile the calibration was built against.

Design constraint. The selector's ``maybe_morph`` call had to return in
under 1 ms in the common case, because it sat on the hot path of every
DNS query. We achieved that by making the decision arithmetic O(1) and
avoiding locks; the caller was responsible for serialisation if it chose
to share a selector across threads.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..cryptographic_agility.mcs_morph_benchmark import Shard
from ..cryptographic_agility.cryptographic_shard_calibrator import (
    BandPolicy,
    CalibrationPolicy,
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

LOGGER = logging.getLogger("mirage.extended.runtime_selector")

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class MorphDecision(str, Enum):
    NO_CHANGE = "no_change"
    MORPH = "morph"
    SUPPRESSED_DWELL = "suppressed_dwell"
    SUPPRESSED_COOLDOWN = "suppressed_cooldown"


# --------------------------------------------------------------------------- #
# Runtime state
# --------------------------------------------------------------------------- #


@dataclass
class RuntimeState:
    current_shard: Shard
    current_band: str
    band_entered_at: float
    last_morph_at: float
    n_morphs: int = 0
    n_queries: int = 0

    def snapshot(self) -> Dict[str, object]:
        return {
            "current_shard": self.current_shard.value,
            "current_band": self.current_band,
            "band_entered_at": self.band_entered_at,
            "last_morph_at": self.last_morph_at,
            "n_morphs": self.n_morphs,
            "n_queries": self.n_queries,
        }


# --------------------------------------------------------------------------- #
# Selector
# --------------------------------------------------------------------------- #


@dataclass
class SelectorConfig:
    # Minimum wall-clock time between two consecutive morphs, regardless of
    # severity. Prevents morph storms under noisy detector scores.
    morph_cooldown_s: float = 1.0
    # Weight applied to the calibration policy's band-specific dwell time.
    dwell_scale: float = 1.0
    # When True, the selector logged every decision; used only in testing.
    verbose: bool = False


class RuntimeShardSelector:
    def __init__(
        self,
        policy: CalibrationPolicy,
        config: Optional[SelectorConfig] = None,
    ) -> None:
        self.policy = policy
        self.config = config or SelectorConfig()
        self._bands = {b.name: b for b in policy.bands}
        if not self._bands:
            raise ValueError("policy contains no bands")
        start_band = "baseline"
        now = time.monotonic()
        self.state = RuntimeState(
            current_shard=self._bands[start_band].shard,
            current_band=start_band,
            band_entered_at=now,
            last_morph_at=now - self.config.morph_cooldown_s,
        )
        self.decisions: List[Dict[str, object]] = []

    # -- helpers ------------------------------------------------------------ #

    def _band_for_severity(self, severity: float) -> BandPolicy:
        severity = float(max(0.0, min(1.0, severity)))
        for band in self.policy.bands:
            if band.severity_low <= severity < band.severity_high:
                return band
        # Severity at the upper edge lands in the last band.
        return self.policy.bands[-1]

    def _record(
        self,
        severity: float,
        decision: MorphDecision,
        band: BandPolicy,
        elapsed_since_morph_s: float,
    ) -> None:
        if not self.config.verbose:
            return
        self.decisions.append(
            {
                "t": time.monotonic(),
                "severity": severity,
                "band": band.name,
                "decision": decision.value,
                "elapsed_since_morph_s": elapsed_since_morph_s,
            }
        )
        # Keep the buffer bounded.
        if len(self.decisions) > 4096:
            self.decisions = self.decisions[-2048:]

    # -- public API --------------------------------------------------------- #

    def maybe_morph(self, severity: float) -> MorphDecision:
        """Decide whether the system should morph for the current query.

        The decision flow was:

          1. Determine the severity band.
          2. If the band's target shard equals the current shard, no morph.
          3. If the system has been in this band for less than the band's
             dwell time, suppress the morph.
          4. If fewer than the configured cooldown has elapsed since the
             last morph, suppress the morph.
          5. Otherwise, morph.
        """
        now = time.monotonic()
        self.state.n_queries += 1
        band = self._band_for_severity(severity)

        if band.name != self.state.current_band:
            self.state.current_band = band.name
            self.state.band_entered_at = now

        target_shard = band.shard
        if target_shard is self.state.current_shard:
            self._record(
                severity, MorphDecision.NO_CHANGE, band, now - self.state.last_morph_at
            )
            return MorphDecision.NO_CHANGE

        band_dwell = band.dwell_s * self.config.dwell_scale
        if (now - self.state.band_entered_at) < band_dwell:
            self._record(
                severity,
                MorphDecision.SUPPRESSED_DWELL,
                band,
                now - self.state.last_morph_at,
            )
            return MorphDecision.SUPPRESSED_DWELL

        if (now - self.state.last_morph_at) < self.config.morph_cooldown_s:
            self._record(
                severity,
                MorphDecision.SUPPRESSED_COOLDOWN,
                band,
                now - self.state.last_morph_at,
            )
            return MorphDecision.SUPPRESSED_COOLDOWN

        self.state.current_shard = target_shard
        self.state.last_morph_at = now
        self.state.n_morphs += 1
        self._record(severity, MorphDecision.MORPH, band, 0.0)
        LOGGER.debug(
            "morphed to %s under band %s at severity %.3f",
            target_shard.value,
            band.name,
            severity,
        )
        return MorphDecision.MORPH

    def current_shard(self) -> Shard:
        return self.state.current_shard

    def snapshot(self) -> Dict[str, object]:
        return {
            "state": self.state.snapshot(),
            "policy_provenance": self.policy.provenance,
            "policy_morph_median_ms": self.policy.morph_median_ms,
            "expected_cost_ms": self.policy.expected_cost_ms,
        }


# --------------------------------------------------------------------------- #
# Simulated trace driver
# --------------------------------------------------------------------------- #
# We included a small trace-driven harness so the selector could be
# validated against a recorded severity trace without deploying it. That
# is what the tests use, and it is what the README points at for a quick
# check that the calibrated policy behaves sensibly.


def simulate_trace(
    selector: RuntimeShardSelector,
    severities: Sequence[float],
    inter_arrival_s: float = 0.05,
    sleep: bool = False,
) -> Dict[str, object]:
    counts: Dict[str, int] = {d.value: 0 for d in MorphDecision}
    shards_visited: List[str] = []
    for s in severities:
        decision = selector.maybe_morph(float(s))
        counts[decision.value] += 1
        shard = selector.current_shard().value
        if not shards_visited or shards_visited[-1] != shard:
            shards_visited.append(shard)
        if sleep:
            time.sleep(inter_arrival_s)
    return {
        "n_queries": len(severities),
        "decision_counts": counts,
        "shards_visited": shards_visited,
        "final_state": selector.state.snapshot(),
    }


# --------------------------------------------------------------------------- #
# Policy loader
# --------------------------------------------------------------------------- #


def load_policy(path: Path) -> CalibrationPolicy:
    payload = json.loads(Path(path).read_text())
    bands: List[BandPolicy] = []
    for b in payload.get("bands", []):
        bands.append(
            BandPolicy(
                name=str(b["name"]),
                severity_low=float(b["severity_low"]),
                severity_high=float(b["severity_high"]),
                shard=Shard(str(b["shard"])),
                dwell_s=float(b["dwell_s"]),
                morph_threshold_ms=float(b["morph_threshold_ms"]),
            )
        )
    return CalibrationPolicy(
        provenance=str(payload.get("provenance", "unknown")),
        morph_median_ms=float(payload.get("morph_median_ms", 0.0)),
        morph_p95_ms=float(payload.get("morph_p95_ms", 0.0)),
        morph_p99_ms=float(payload.get("morph_p99_ms", 0.0)),
        bands=bands,
        transition_weights={
            k: float(v)
            for k, v in (payload.get("transition_weights") or {}).items()
        },
        transition_cost_ms={
            k: float(v)
            for k, v in (payload.get("transition_cost_ms") or {}).items()
        },
        expected_cost_ms=float(payload.get("expected_cost_ms", 0.0)),
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the MIRAGE-DoH runtime shard selector against a synthetic severity trace."
    )
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n-queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20231130)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    policy = load_policy(args.policy)
    selector = RuntimeShardSelector(policy, SelectorConfig(verbose=True))
    rng = np.random.default_rng(args.seed)
    # We generated a piecewise severity trace that rose through the bands.
    severities: List[float] = []
    for _ in range(args.n_queries):
        base = 0.15 + 0.75 * (len(severities) / max(1, args.n_queries))
        severities.append(float(np.clip(rng.normal(base, 0.1), 0.0, 1.0)))
    result = simulate_trace(selector, severities, sleep=False)
    snapshot = selector.snapshot()
    snapshot["simulation"] = result
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(snapshot, indent=2))
    LOGGER.info("shard selector simulation: %s", result["decision_counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
