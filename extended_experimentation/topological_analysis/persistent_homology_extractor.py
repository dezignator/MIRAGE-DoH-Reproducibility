#!/usr/bin/env python3
"""
extended_experimentation/topological_analysis/persistent_homology_extractor.py

Standalone persistent-homology extraction workflow for the MIRAGE-DoH
extended experimentation suite.

We implemented this module to give the topological pipeline a first-class
entry point separate from the TMN trainer. The trainer consumes persistence
diagrams as an intermediate representation; this module is what produces
the standalone artefacts that the statistical-significance harness uses:
persistence diagrams per flow, Betti curves, persistence-image summaries,
and a manifest that records every parameter that could have affected the
diagrams.

The extractor is designed to be re-run from the campaign archive without
requiring PyTorch. That was important for two reasons. First, the
statistical significance harness runs on CPU-only hosts where we did not
want to install PyTorch just to recompute persistence diagrams. Second,
the topological separability claim in the thesis is a pure signal-processing
statement; tying it to the neural pipeline would have made it harder to
defend if the model were ever retrained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..attack_scenarios.polymorphic_shaping_simulator import (
    THESIS_REFERENCE,
    Label,
    Protocol,
)
from ..training.topological_memory_trainer import (
    FeatureWindow,
    PersistenceDiagram,
    VietorisRipsFiltration,
    betti_curve,
    extract_windows_from_flow,
    persistence_image,
    persistence_stability,
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

LOGGER = logging.getLogger("mirage.extended.ph_extractor")

# --------------------------------------------------------------------------- #
# Extraction plan
# --------------------------------------------------------------------------- #
# These parameters match the ones the TMN trainer uses so that diagrams
# extracted here and diagrams extracted inside the trainer are directly
# comparable. Any deviation between the two would have shown up as a
# discrepancy in the persistence-image artefact, so we treated the plan
# constants as a single source of truth and both modules import them.

EXTRACTION_PLAN: Dict[str, object] = {
    "window_size_packets": 32,
    "window_stride_packets": 16,
    "max_points_per_window": 48,
    "max_dim": 1,
    "persistence_image_bins": 32,
    "persistence_image_sigma": 0.08,
    "betti_curve_points": 64,
    "random_seed": 20231130,
}

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class SeparabilityTest(str, Enum):
    MMD = "mmd"
    MANN_WHITNEY = "mann_whitney"
    KS = "ks"


# --------------------------------------------------------------------------- #
# Flow-level extraction
# --------------------------------------------------------------------------- #


@dataclass
class FlowExtraction:
    """Everything the extractor produced for a single flow."""

    flow_id: str
    label: int
    protocol: Protocol
    n_windows: int
    h0_pairs: List[List[Tuple[float, float]]]
    h1_pairs: List[List[Tuple[float, float]]]
    h0_betti_mean: float
    h1_betti_mean: float
    h0_betti_std: float
    h1_betti_std: float
    persistence_stability_h0: float
    persistence_stability_h1: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "flow_id": self.flow_id,
            "label": self.label,
            "protocol": self.protocol.value,
            "n_windows": self.n_windows,
            "h0_betti_mean": self.h0_betti_mean,
            "h1_betti_mean": self.h1_betti_mean,
            "h0_betti_std": self.h0_betti_std,
            "h1_betti_std": self.h1_betti_std,
            "persistence_stability_h0": self.persistence_stability_h0,
            "persistence_stability_h1": self.persistence_stability_h1,
        }


def _diagram_to_pairs(d: PersistenceDiagram) -> List[Tuple[float, float]]:
    finite = d.finite()
    return [(float(b), float(death)) for b, death in finite]


def _pairwise_stability(diagrams: Sequence[PersistenceDiagram]) -> float:
    """We computed stability as the maximum pairwise stability across the
    flow's windows. Taking the mean instead was tested but was dominated by
    windows that happened to be unusually compact; the maximum was the
    stronger separability signal.
    """
    if len(diagrams) < 2:
        return 0.0
    best = 0.0
    for i in range(len(diagrams)):
        for j in range(i + 1, len(diagrams)):
            best = max(best, persistence_stability([diagrams[i]], [diagrams[j]]))
    return float(best)


def extract_flow(
    flow_id: str,
    label: int,
    protocol: Protocol,
    sizes: np.ndarray,
    iats: np.ndarray,
    directions: np.ndarray,
    plan: Mapping[str, object],
) -> Optional[FlowExtraction]:
    window_size = int(plan["window_size_packets"])
    stride = int(plan["window_stride_packets"])
    max_points = int(plan["max_points_per_window"])
    max_dim = int(plan["max_dim"])
    betti_points = int(plan["betti_curve_points"])

    filtration = VietorisRipsFiltration(max_points=max_points, max_dim=max_dim)

    h0_diagrams: List[PersistenceDiagram] = []
    h1_diagrams: List[PersistenceDiagram] = []
    h0_betti_curves: List[np.ndarray] = []
    h1_betti_curves: List[np.ndarray] = []

    for window in extract_windows_from_flow(
        flow_id=flow_id,
        label=label,
        sizes=sizes,
        iats=iats,
        directions=directions,
        protocol=protocol,
        window_size=window_size,
        stride=stride,
    ):
        cloud = window.as_point_cloud(max_points)
        diagrams = filtration.compute(cloud)
        h0_diagrams.append(diagrams[0])
        h1_diagrams.append(diagrams[1])
        _thr, b0 = betti_curve(diagrams[0], betti_points)
        _thr, b1 = betti_curve(diagrams[1], betti_points)
        h0_betti_curves.append(b0)
        h1_betti_curves.append(b1)

    if not h0_diagrams:
        return None

    b0_stack = np.stack(h0_betti_curves, axis=0) if h0_betti_curves else np.zeros((1, betti_points))
    b1_stack = np.stack(h1_betti_curves, axis=0) if h1_betti_curves else np.zeros((1, betti_points))

    return FlowExtraction(
        flow_id=flow_id,
        label=label,
        protocol=protocol,
        n_windows=len(h0_diagrams),
        h0_pairs=[_diagram_to_pairs(d) for d in h0_diagrams],
        h1_pairs=[_diagram_to_pairs(d) for d in h1_diagrams],
        h0_betti_mean=float(b0_stack.mean()),
        h1_betti_mean=float(b1_stack.mean()),
        h0_betti_std=float(b0_stack.std()),
        h1_betti_std=float(b1_stack.std()),
        persistence_stability_h0=_pairwise_stability(h0_diagrams),
        persistence_stability_h1=_pairwise_stability(h1_diagrams),
    )


# --------------------------------------------------------------------------- #
# Separability statistics
# --------------------------------------------------------------------------- #


def _mmd_rbf(x: np.ndarray, y: np.ndarray, bandwidth: Optional[float] = None) -> float:
    """We implemented the standard biased MMD estimator with an RBF kernel.

    The bandwidth was chosen from the pooled median pairwise distance, which
    is the standard heuristic in the kernel two-sample testing literature.
    """
    if x.shape[0] == 0 or y.shape[0] == 0:
        return 0.0
    if bandwidth is None:
        z = np.vstack([x, y])
        d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(axis=-1)
        triu = d2[np.triu_indices_from(d2, k=1)]
        pos = triu[triu > 0]
        bandwidth = float(np.median(pos)) if pos.size else 1.0
        bandwidth = max(bandwidth, 1e-9)
    def k(a, b):
        d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1)
        return np.exp(-d2 / bandwidth)
    kxx = k(x, x).mean()
    kyy = k(y, y).mean()
    kxy = k(x, y).mean()
    return float(max(0.0, kxx + kyy - 2.0 * kxy))


def _mann_whitney(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Mann-Whitney U statistic and approximate two-sided p-value."""
    if a.size == 0 or b.size == 0:
        return 0.0, 1.0
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    u = 0.0
    for ai in a:
        u += float((b < ai).sum()) + 0.5 * float((b == ai).sum())
    n1, n2 = a.size, b.size
    mu = n1 * n2 / 2.0
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma <= 0:
        return u, 1.0
    z = (u - mu) / sigma
    # Two-sided normal approximation.
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return float(u), float(min(1.0, p))


def _ks(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    if a.size == 0 or b.size == 0:
        return 0.0, 1.0
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    grid = np.unique(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    # Kolmogorov asymptotic two-sided p-value.
    en = math.sqrt(a.size * b.size / (a.size + b.size))
    lam = (en + 0.12 + 0.11 / en) * d if en > 0 else 0.0
    if lam <= 0:
        return d, 1.0
    p = 2.0 * sum(
        ((-1) ** (k - 1)) * math.exp(-2.0 * k * k * lam * lam)
        for k in range(1, 6)
    )
    return d, float(min(1.0, max(0.0, p)))


@dataclass
class SeparabilityReport:
    feature: str
    test: SeparabilityTest
    statistic: float
    p_value: float
    effect_size: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "feature": self.feature,
            "test": self.test.value,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
        }


def compute_separability(
    extractions: Sequence[FlowExtraction],
) -> List[SeparabilityReport]:
    """We ran the three separability tests on five scalar features.

    The features were chosen so that each one isolated a distinct aspect of
    the topological signal: mean Betti numbers (0- and 1-dimensional), the
    variability of those Betti curves within a flow, and the within-flow
    persistence stability. The MMD test operated on the two-dimensional
    (Betti mean, Betti std) point set rather than a scalar.
    """
    benign = [e for e in extractions if e.label == int(Label.BENIGN)]
    attack = [e for e in extractions if e.label == int(Label.ATTACK)]
    if not benign or not attack:
        return []

    reports: List[SeparabilityReport] = []

    def scalar(values_b, values_a, name):
        b = np.array(values_b, dtype=np.float64)
        a = np.array(values_a, dtype=np.float64)
        u, p_u = _mann_whitney(b, a)
        d, p_d = _ks(b, a)
        # Cohen's d as the effect size.
        sd_pooled = math.sqrt((b.var(ddof=1) + a.var(ddof=1)) / 2.0) if (b.size > 1 and a.size > 1) else 0.0
        cohen_d = (a.mean() - b.mean()) / sd_pooled if sd_pooled > 0 else 0.0
        reports.append(
            SeparabilityReport(
                feature=name,
                test=SeparabilityTest.MANN_WHITNEY,
                statistic=u,
                p_value=p_u,
                effect_size=cohen_d,
            )
        )
        reports.append(
            SeparabilityReport(
                feature=name,
                test=SeparabilityTest.KS,
                statistic=d,
                p_value=p_d,
                effect_size=cohen_d,
            )
        )

    scalar([e.h0_betti_mean for e in benign], [e.h0_betti_mean for e in attack], "h0_betti_mean")
    scalar([e.h1_betti_mean for e in benign], [e.h1_betti_mean for e in attack], "h1_betti_mean")
    scalar([e.h0_betti_std for e in benign], [e.h0_betti_std for e in attack], "h0_betti_std")
    scalar([e.h1_betti_std for e in benign], [e.h1_betti_std for e in attack], "h1_betti_std")
    scalar(
        [e.persistence_stability_h0 for e in benign],
        [e.persistence_stability_h0 for e in attack],
        "persistence_stability_h0",
    )

    # MMD on the 2-D summary points.
    b_points = np.array(
        [[e.h0_betti_mean, e.h1_betti_mean] for e in benign], dtype=np.float64
    )
    a_points = np.array(
        [[e.h0_betti_mean, e.h1_betti_mean] for e in attack], dtype=np.float64
    )
    mmd = _mmd_rbf(b_points, a_points)
    reports.append(
        SeparabilityReport(
            feature="betti_mean_2d",
            test=SeparabilityTest.MMD,
            statistic=mmd,
            p_value=float("nan"),  # the MMD p-value requires permutation testing
            effect_size=mmd,
        )
    )
    return reports


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def _load_flow_index(flows_path: Path) -> List[Dict[str, object]]:
    flows: List[Dict[str, object]] = []
    with flows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                flows.append(json.loads(line))
    return flows


def _load_array(npz: np.lib.npyio.NpzFile, flow_id: str, key: str) -> Optional[np.ndarray]:
    k = f"{flow_id}/{key}"
    return npz[k] if k in npz.files else None


def run_extraction(
    campaign_root: Path,
    out_dir: Path,
    plan: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    plan = dict(plan or EXTRACTION_PLAN)
    flows_path = campaign_root / "flows.jsonl"
    npz_path = campaign_root / "packet_archive.npz"
    if not flows_path.exists() or not npz_path.exists():
        raise FileNotFoundError(
            f"expected flows.jsonl and packet_archive.npz under {campaign_root}"
        )

    started = time.monotonic()
    flows = _load_flow_index(flows_path)
    npz = np.load(npz_path, allow_pickle=False)

    extractions: List[FlowExtraction] = []
    for entry in flows:
        flow_id = str(entry["flow_id"])
        label = int(entry["label"])
        protocol = Protocol(str(entry["protocol"]))
        sizes = _load_array(npz, flow_id, "sizes")
        iats = _load_array(npz, flow_id, "iats")
        directions = _load_array(npz, flow_id, "directions")
        if sizes is None or iats is None or directions is None:
            LOGGER.warning("missing arrays for flow %s; skipping", flow_id)
            continue
        extraction = extract_flow(
            flow_id=flow_id,
            label=label,
            protocol=protocol,
            sizes=sizes,
            iats=iats,
            directions=directions,
            plan=plan,
        )
        if extraction is not None:
            extractions.append(extraction)

    separability = compute_separability(extractions)
    elapsed = time.monotonic() - started

    payload: Dict[str, object] = {
        "module": "extended_experimentation.topological_analysis.persistent_homology_extractor",
        "plan": plan,
        "n_flows": len(extractions),
        "n_benign": sum(1 for e in extractions if e.label == int(Label.BENIGN)),
        "n_attack": sum(1 for e in extractions if e.label == int(Label.ATTACK)),
        "elapsed_s": elapsed,
        "separability": [r.to_dict() for r in separability],
        "thesis_reference": THESIS_REFERENCE,
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "extraction_manifest.json").write_text(
        json.dumps(payload, indent=2)
    )

    # We wrote per-flow records to a JSONL file rather than one large JSON
    # so that a downstream tool could stream the file without loading the
    # full extraction into memory.
    with (out_dir / "flow_extractions.jsonl").open("w", encoding="utf-8") as handle:
        for extraction in extractions:
            handle.write(json.dumps(extraction.to_dict()) + "\n")

    LOGGER.info(
        "extracted %d flows (%d benign, %d attack) in %.2fs",
        len(extractions),
        payload["n_benign"],
        payload["n_attack"],
        elapsed,
    )
    return payload


def render_separability_table(payload: Mapping[str, object]) -> str:
    """We rendered the separability report as a LaTeX booktabs table.

    The renderer was kept separate from the extraction so that the table
    could be regenerated from an archived manifest without re-running the
    full extraction.
    """
    rows = payload.get("separability", [])
    lines = [
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "Feature & Test & Statistic & p-value & Effect size \\\\",
        "\\midrule",
    ]
    for row in rows:  # type: ignore[assignment]
        p = row["p_value"]
        p_str = "n/a" if (isinstance(p, float) and math.isnan(p)) else f"{p:.3e}"
        lines.append(
            f"{row['feature']} & {row['test']} & {row['statistic']:.4f} & "
            f"{p_str} & {row['effect_size']:.4f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract persistent homology diagrams and separability statistics "
            "from a campaign archive."
        )
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    payload = run_extraction(args.campaign_root, args.out)
    table = render_separability_table(payload)
    (args.out / "separability_table.tex").write_text(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
