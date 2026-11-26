#!/usr/bin/env python3
"""
extended_experimentation/reporting/thesis_figure_reproducer.py

Figure reproducer for the MIRAGE-DoH extended experimentation suite.

We implemented this module to regenerate the four data figures that
accompany the extended experiments. The figures are:

  Figure S1  Detection accuracy vs. attack family.
  Figure S2  Inference latency distribution by transport.
  Figure S3  MCS morph latency by transition.
  Figure S4  Topological separability (Betti mean and stability).

Matplotlib is optional. If it was unavailable, the module still wrote the
underlying data (in JSON and NumPy ``.npz``) so that the figures could be
re-rendered on another machine. We chose this two-tier approach because
the thesis repository's reproducibility claim rested on the data being
reproducible, not on the figure being rendered on the specific host the
examiner used.

The reproducer did not generate the schematic architecture figure (Figure
1 in Chapter 3). That figure was drawn in TikZ and is not data-driven;
regenerating it from scratch would have required re-deriving a diagram
whose purpose was pedagogical rather than empirical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..attack_scenarios.polymorphic_shaping_simulator import THESIS_REFERENCE

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

LOGGER = logging.getLogger("mirage.extended.figure_reproducer")

# --------------------------------------------------------------------------- #
# Optional matplotlib
# --------------------------------------------------------------------------- #

try:  # pragma: no cover - import gate
    import matplotlib

    matplotlib.use("Agg")  # noqa: E402
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except Exception:  # pragma: no cover - defensive
    plt = None  # type: ignore[assignment]
    _HAS_MPL = False

# --------------------------------------------------------------------------- #
# Figure specs
# --------------------------------------------------------------------------- #


@dataclass
class FigureSpec:
    name: str
    title: str
    description: str
    width_in: float = 6.0
    height_in: float = 4.0
    dpi: int = 150


FIGURES: Dict[str, FigureSpec] = {
    "S1": FigureSpec(
        name="S1",
        title="Detection accuracy by attack family",
        description="Accuracy per simulated attack family, with reference line.",
    ),
    "S2": FigureSpec(
        name="S2",
        title="Inference latency distribution by transport",
        description="Boxplots of per-sample latency across DoH, DoT and DoQ.",
    ),
    "S3": FigureSpec(
        name="S3",
        title="MCS morph latency by transition",
        description="Median and P95 morph latency across the six transitions.",
    ),
    "S4": FigureSpec(
        name="S4",
        title="Topological separability",
        description="Betti mean and within-flow persistence stability by label.",
    ),
}

# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def _load_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        LOGGER.warning("missing artefact %s", path)
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        LOGGER.exception("failed to parse %s", path)
        return None


# --------------------------------------------------------------------------- #
# Figure data extractors
# --------------------------------------------------------------------------- #


def _s1_data(tables_manifest: Optional[Mapping[str, object]]) -> Dict[str, List[float]]:
    if not tables_manifest:
        return {"families": [], "accuracies": []}
    rows = tables_manifest.get("s1_rows", [])
    families: List[str] = []
    accuracies: List[float] = []
    for r in rows:  # type: ignore[assignment]
        families.append(str(r["family"]))
        accuracies.append(float(r["accuracy"]))
    return {"families": families, "accuracies": accuracies}


def _s2_data(core_metrics: Optional[Mapping[str, object]]) -> Dict[str, Dict[str, float]]:
    """We returned per-protocol latency summaries. The full per-sample
    distribution was not preserved by the core metrics evaluator, so the
    figure showed a summary derived from the recorded percentiles rather
    than the raw samples. That was documented in the figure caption.
    """
    if not core_metrics:
        return {}
    by_protocol = core_metrics.get("by_protocol", {})
    out: Dict[str, Dict[str, float]] = {}
    for protocol, bucket in (by_protocol or {}).items():
        if not isinstance(bucket, dict):
            continue
        latency = bucket.get("latency", {})
        if not isinstance(latency, dict):
            continue
        out[str(protocol)] = {
            "median": float(latency.get("median_us", 0.0)),
            "p95": float(latency.get("p95_us", 0.0)),
            "p99": float(latency.get("p99_us", 0.0)),
        }
    return out


def _s3_data(mcs_manifest: Optional[Mapping[str, object]]) -> Dict[str, Dict[str, float]]:
    if not mcs_manifest:
        return {}
    transitions = mcs_manifest.get("transitions", {})
    out: Dict[str, Dict[str, float]] = {}
    for name, bucket in (transitions or {}).items():
        if not isinstance(bucket, dict):
            continue
        out[name] = {
            "median": float(bucket.get("median_ms", 0.0)),
            "p95": float(bucket.get("p95_ms", 0.0)),
        }
    return out


def _s4_data(extraction_manifest: Optional[Mapping[str, object]]) -> Dict[str, List[float]]:
    """We projected the separability report into per-label statistics.

    The extraction manifest contained p-values and effect sizes, not the
    raw per-flow Betti means; the per-flow data lived in the JSONL
    artefact. The figure therefore showed the effect sizes, which is what
    the thesis caption described. Extracting per-flow distributions would
    have required re-running the extraction, which the reproducer
    deliberately avoided.
    """
    if not extraction_manifest:
        return {"features": [], "effect_sizes": [], "p_values": []}
    rows = extraction_manifest.get("separability", [])
    features: List[str] = []
    effect_sizes: List[float] = []
    p_values: List[float] = []
    for r in rows:  # type: ignore[assignment]
        features.append(f"{r['feature']} ({r['test']})")
        effect_sizes.append(float(r["effect_size"]))
        p = r["p_value"]
        p_values.append(float(p) if isinstance(p, (int, float)) and not math.isnan(p) else 1.0)
    return {
        "features": features,
        "effect_sizes": effect_sizes,
        "p_values": p_values,
    }


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #


def _render_s1(data: Dict[str, List[float]], spec: FigureSpec, out_dir: Path) -> None:
    if not _HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(spec.width_in, spec.height_in))
    families = data["families"]
    accs = data["accuracies"]
    x = np.arange(len(families))
    ax.bar(x, accs, color="#3a7ca5")
    ax.axhline(
        THESIS_REFERENCE["detection_accuracy"],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="thesis reference",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=25, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.995, 1.0005)
    ax.set_title(spec.title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"figure_{spec.name}.png", dpi=spec.dpi)
    plt.close(fig)


def _render_s2(data: Dict[str, Dict[str, float]], spec: FigureSpec, out_dir: Path) -> None:
    if not _HAS_MPL or not data:
        return
    fig, ax = plt.subplots(figsize=(spec.width_in, spec.height_in))
    protocols = sorted(data.keys())
    positions = np.arange(len(protocols))
    medians = [data[p]["median"] for p in protocols]
    p95s = [data[p]["p95"] for p in protocols]
    p99s = [data[p]["p99"] for p in protocols]

    # We plotted the recorded percentiles as a bar with error bars rather
    # than a boxplot, because only the percentiles survived the artefact.
    ax.bar(positions, medians, color="#3a7ca5", label="median")
    ax.errorbar(
        positions,
        medians,
        yerr=[
            [m - m for m in medians],
            [p - m for m, p in zip(medians, p95s)],
        ],
        fmt="none",
        ecolor="black",
        capsize=4,
        label="95%",
    )
    ax.scatter(positions, p99s, color="red", marker="x", label="P99")
    ax.set_xticks(positions)
    ax.set_xticklabels([p.upper() for p in protocols])
    ax.set_ylabel("Latency (\\textmu s)")
    ax.set_title(spec.title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"figure_{spec.name}.png", dpi=spec.dpi)
    plt.close(fig)


def _render_s3(data: Dict[str, Dict[str, float]], spec: FigureSpec, out_dir: Path) -> None:
    if not _HAS_MPL or not data:
        return
    fig, ax = plt.subplots(figsize=(spec.width_in, spec.height_in))
    names = sorted(data.keys())
    positions = np.arange(len(names))
    medians = [data[n]["median"] for n in names]
    p95s = [data[n]["p95"] for n in names]

    ax.bar(positions, medians, color="#2e6f40", label="median")
    ax.scatter(positions, p95s, color="red", marker="x", label="P95")
    ax.axhline(
        THESIS_REFERENCE["cryptographic_adaptation_ms"],
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="thesis reference (median)",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Morph latency (ms)")
    ax.set_title(spec.title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"figure_{spec.name}.png", dpi=spec.dpi)
    plt.close(fig)


def _render_s4(data: Dict[str, List[float]], spec: FigureSpec, out_dir: Path) -> None:
    if not _HAS_MPL or not data["features"]:
        return
    fig, ax = plt.subplots(figsize=(spec.width_in * 1.1, spec.height_in))
    features = data["features"]
    effects = data["effect_sizes"]
    positions = np.arange(len(features))
    colors = ["#c0392b" if p < 0.05 else "#7f8c8d" for p in data["p_values"]]
    ax.barh(positions, effects, color=colors)
    ax.set_yticks(positions)
    ax.set_yticklabels(features, fontsize=8)
    ax.set_xlabel("Effect size (Cohen's $d$ or MMD)")
    ax.set_title(spec.title)
    fig.tight_layout()
    fig.savefig(out_dir / f"figure_{spec.name}.png", dpi=spec.dpi)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def reproduce_figures(
    core_metrics_path: Path,
    tables_manifest_path: Path,
    mcs_manifest_path: Path,
    extraction_manifest_path: Path,
    out_dir: Path,
) -> Dict[str, object]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    core_metrics = _load_json(core_metrics_path)
    tables_manifest = _load_json(tables_manifest_path)
    mcs_manifest = _load_json(mcs_manifest_path)
    extraction_manifest = _load_json(extraction_manifest_path)

    s1 = _s1_data(tables_manifest)
    s2 = _s2_data(core_metrics)
    s3 = _s3_data(mcs_manifest)
    s4 = _s4_data(extraction_manifest)

    _render_s1(s1, FIGURES["S1"], out_dir)
    _render_s2(s2, FIGURES["S2"], out_dir)
    _render_s3(s3, FIGURES["S3"], out_dir)
    _render_s4(s4, FIGURES["S4"], out_dir)

    # We wrote the raw data alongside the figures so that re-rendering on
    # another host did not require re-running the upstream pipeline.
    np.savez_compressed(
        out_dir / "figure_data.npz",
        s1_accuracies=np.asarray(s1["accuracies"], dtype=np.float64),
        s2_protocols=np.asarray(sorted(s2.keys()) if s2 else [], dtype=object),
        s2_medians=np.asarray(
            [s2[p]["median"] for p in sorted(s2.keys())] if s2 else [],
            dtype=np.float64,
        ),
        s3_transitions=np.asarray(sorted(s3.keys()) if s3 else [], dtype=object),
        s3_medians=np.asarray(
            [s3[t]["median"] for t in sorted(s3.keys())] if s3 else [],
            dtype=np.float64,
        ),
        s4_effects=np.asarray(s4["effect_sizes"], dtype=np.float64),
    )

    manifest: Dict[str, object] = {
        "module": "extended_experimentation.reporting.thesis_figure_reproducer",
        "matplotlib_available": _HAS_MPL,
        "figures": {
            name: {
                "title": spec.title,
                "description": spec.description,
                "png_file": f"figure_{name}.png" if _HAS_MPL else None,
            }
            for name, spec in FIGURES.items()
        },
        "s1_families": s1["families"],
        "s1_accuracies": s1["accuracies"],
        "s2_protocols": list(s2.keys()),
        "s3_transitions": list(s3.keys()),
        "s4_features": s4["features"],
        "thesis_reference": THESIS_REFERENCE,
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    (out_dir / "figures_manifest.json").write_text(json.dumps(manifest, indent=2))

    if not _HAS_MPL:
        LOGGER.warning(
            "matplotlib unavailable; wrote figure data only to %s",
            out_dir / "figure_data.npz",
        )
    else:
        LOGGER.info("wrote figures S1-S4 to %s", out_dir)
    return manifest


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the S1-S4 MIRAGE-DoH data figures."
    )
    parser.add_argument("--core-metrics", type=Path, required=True)
    parser.add_argument("--tables-manifest", type=Path, required=True)
    parser.add_argument("--mcs-manifest", type=Path, required=True)
    parser.add_argument("--extraction-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    reproduce_figures(
        core_metrics_path=args.core_metrics,
        tables_manifest_path=args.tables_manifest,
        mcs_manifest_path=args.mcs_manifest,
        extraction_manifest_path=args.extraction_manifest,
        out_dir=args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
