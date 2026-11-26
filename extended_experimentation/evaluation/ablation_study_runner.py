#!/usr/bin/env python3
"""
extended_experimentation/evaluation/ablation_study_runner.py

Partial ablation study runner for the MIRAGE-DoH extended experimentation
suite.

We implemented this module to run the *component-level* ablation that
Section 7.3.1 of the thesis describes as future work. We did not perform
a full factorial ablation, and this module did not pretend otherwise: it
ran four single-module-disabled variants plus the full model, each against
its own evaluation checkpoint, and reported the delta against the full
model.

The module does not retrain. It consumes checkpoints that were already
produced by ``full_mirage_training_loop`` with the ``--skip`` flag set
appropriately, plus the corresponding evaluation artefacts. That was a
deliberate architectural choice: the ablation table needed to be
reproducible from the checkpoint archive alone, without re-running four
training jobs.

Four variants were supported:

  no_tmn        TMN branch disabled; the MetaAgent was trained on the raw
                sequence features alone.
  no_mcs        MCS policy set to the baseline Pi_0; no morphing.
  no_spectral   Spectral predictor detached; MetaAgent used the pooled
                feature only.
  no_cce        Causal counterfactual environment disabled; MetaAgent
                used the observed attack distribution only.

The full-model baseline is the fifth column. The output table contains
accuracy, false-positive rate, median latency, and delta against baseline
for each of the four variants.
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

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

LOGGER = logging.getLogger("mirage.extended.ablation")

# --------------------------------------------------------------------------- #
# Variants
# --------------------------------------------------------------------------- #


class Variant(str, Enum):
    FULL = "full"
    NO_TMN = "no_tmn"
    NO_MCS = "no_mcs"
    NO_SPECTRAL = "no_spectral"
    NO_CCE = "no_cce"


VARIANT_DESCRIPTIONS: Dict[Variant, str] = {
    Variant.FULL: "Full MIRAGE-DoH reference",
    Variant.NO_TMN: "TMN branch disabled",
    Variant.NO_MCS: "MCS policy pinned to Pi_0 baseline",
    Variant.NO_SPECTRAL: "Spectral predictor detached",
    Variant.NO_CCE: "Causal counterfactual environment disabled",
}

# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #
# We declared what "a material regression" looked like *before* running the
# ablation, so that we could not be tempted to relax the threshold when a
# variant turned out worse than expected. A variant counted as a material
# regression on accuracy if the delta was below -0.002 (0.2 percentage
# points). That threshold was chosen to match the tightest tolerance band
# on the core metrics evaluator.

MATERIAL_REGRESSION_ACCURACY = -0.002
MATERIAL_REGRESSION_FPR = 0.000005
MATERIAL_REGRESSION_LATENCY_US = 5.0

# --------------------------------------------------------------------------- #
# Variant result
# --------------------------------------------------------------------------- #


@dataclass
class VariantMetrics:
    variant: Variant
    description: str
    detection_accuracy: float
    false_positive_rate_fraction: float
    median_latency_us: float
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    provenance: str = "measured"

    def to_dict(self) -> Dict[str, object]:
        return {
            "variant": self.variant.value,
            "description": self.description,
            "detection_accuracy": self.detection_accuracy,
            "false_positive_rate_fraction": self.false_positive_rate_fraction,
            "median_latency_us": self.median_latency_us,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "provenance": self.provenance,
        }


@dataclass
class AblationDelta:
    variant: Variant
    accuracy_delta: float
    fpr_delta: float
    latency_delta_us: float
    accuracy_regression: bool
    fpr_regression: bool
    latency_regression: bool
    verdict: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def _load_metrics(path: Path) -> Dict[str, float]:
    """We loaded a metrics artefact and flattened the numbers we needed.

    The core metrics evaluator writes ``core_metrics.json`` with the
    metrics under ``metrics``. We accepted that exact shape; other artefacts
    with a ``metrics`` map were also accepted to keep the module reusable
    for evaluation runs that produced a slightly different envelope.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "metrics" in payload and isinstance(payload["metrics"], dict):
        return {k: float(v) for k, v in payload["metrics"].items() if _is_number(v)}
    return {k: float(v) for k, v in payload.items() if _is_number(v)}


def _is_number(x: object) -> bool:
    try:
        float(x)  # type: ignore[arg-type]
        return True
    except (TypeError, ValueError):
        return False


def _metrics_for_variant(
    variant: Variant,
    variant_root: Path,
    fallback: Optional[VariantMetrics] = None,
) -> VariantMetrics:
    """We located and loaded a variant's metrics, allowing a fallback.

    A missing variant directory was treated as a "not run" state. Rather
    than fabricate numbers, the loader returned the fallback if supplied
    and otherwise raised. The caller decided whether missing variants were
    acceptable (they were for the internal dry-run, not for the thesis
    table).
    """
    core = variant_root / "core_metrics.json"
    if core.exists():
        metrics = _load_metrics(core)
        return VariantMetrics(
            variant=variant,
            description=VARIANT_DESCRIPTIONS[variant],
            detection_accuracy=metrics.get("detection_accuracy", 0.0),
            false_positive_rate_fraction=metrics.get(
                "false_positive_rate_fraction", 0.0
            ),
            median_latency_us=metrics.get("median_inference_latency_us", 0.0),
            precision=metrics.get("precision", 0.0),
            recall=metrics.get("recall", 0.0),
            f1=metrics.get("f1", 0.0),
            provenance="measured",
        )
    if fallback is not None:
        return fallback
    raise FileNotFoundError(
        f"no metrics artefact for variant {variant.value} under {variant_root}"
    )


# --------------------------------------------------------------------------- #
# Delta computation
# --------------------------------------------------------------------------- #


def _compute_delta(
    baseline: VariantMetrics, variant: VariantMetrics
) -> AblationDelta:
    accuracy_delta = variant.detection_accuracy - baseline.detection_accuracy
    fpr_delta = (
        variant.false_positive_rate_fraction
        - baseline.false_positive_rate_fraction
    )
    latency_delta = variant.median_latency_us - baseline.median_latency_us
    acc_reg = accuracy_delta < MATERIAL_REGRESSION_ACCURACY
    fpr_reg = fpr_delta > MATERIAL_REGRESSION_FPR
    lat_reg = latency_delta > MATERIAL_REGRESSION_LATENCY_US

    if not (acc_reg or fpr_reg or lat_reg):
        verdict = "no_material_change"
    elif acc_reg and fpr_reg:
        verdict = "material_regression_accuracy_and_fpr"
    elif acc_reg:
        verdict = "material_regression_accuracy"
    elif fpr_reg:
        verdict = "material_regression_fpr"
    else:
        verdict = "material_regression_latency"

    return AblationDelta(
        variant=variant.variant,
        accuracy_delta=accuracy_delta,
        fpr_delta=fpr_delta,
        latency_delta_us=latency_delta,
        accuracy_regression=acc_reg,
        fpr_regression=fpr_reg,
        latency_regression=lat_reg,
        verdict=verdict,
    )


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #


@dataclass
class AblationConfig:
    seed: int = 20231130
    variants: Tuple[Variant, ...] = (
        Variant.FULL,
        Variant.NO_TMN,
        Variant.NO_MCS,
        Variant.NO_SPECTRAL,
        Variant.NO_CCE,
    )


class AblationStudyRunner:
    def __init__(self, config: Optional[AblationConfig] = None) -> None:
        self.config = config or AblationConfig()

    def run(
        self,
        checkpoint_root: Path,
        strict: bool = False,
    ) -> Dict[str, object]:
        """We ran the ablation against the checkpoint root.

        The checkpoint root was expected to contain a subdirectory per
        variant, each with a ``core_metrics.json``. The full variant
        subdirectory was required; the four ablated variants were optional
        and the manifest recorded whether each was present.
        """
        checkpoint_root = Path(checkpoint_root)
        full_metrics = _metrics_for_variant(
            Variant.FULL, checkpoint_root / Variant.FULL.value
        )
        variants: List[VariantMetrics] = [full_metrics]
        deltas: List[AblationDelta] = []
        present: Dict[str, bool] = {Variant.FULL.value: True}

        for variant in self.config.variants:
            if variant is Variant.FULL:
                continue
            variant_root = checkpoint_root / variant.value
            try:
                vm = _metrics_for_variant(variant, variant_root)
                present[variant.value] = True
            except FileNotFoundError:
                LOGGER.warning(
                    "ablation variant %s has no metrics artefact under %s; "
                    "recording as absent",
                    variant.value,
                    variant_root,
                )
                present[variant.value] = False
                continue
            variants.append(vm)
            deltas.append(_compute_delta(full_metrics, vm))

        # We recorded the mean material-regression rate across variants as a
        # single summary. It is not a p-value and is not reported in the
        # thesis; it existed purely so that the run's output could be
        # summarised in a single line.
        n_material = sum(1 for d in deltas if d.accuracy_regression or d.fpr_regression)
        material_fraction = n_material / max(1, len(deltas))

        payload: Dict[str, object] = {
            "module": "extended_experimentation.evaluation.ablation_study_runner",
            "config": asdict(self.config),
            "variants_present": present,
            "variants": [v.to_dict() for v in variants],
            "deltas": [d.to_dict() for d in deltas],
            "material_regression_fraction": material_fraction,
            "thresholds": {
                "accuracy": MATERIAL_REGRESSION_ACCURACY,
                "fpr": MATERIAL_REGRESSION_FPR,
                "latency_us": MATERIAL_REGRESSION_LATENCY_US,
            },
            "thesis_reference": THESIS_REFERENCE,
            "notes": (
                "Partial ablation only. A full factorial design was not "
                "performed; see Section 7.3.1 of the thesis."
            ),
        }
        payload["manifest_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

        if strict and not present.get(Variant.NO_TMN.value, False):
            raise RuntimeError(
                "strict ablation requires the no_tmn variant checkpoint"
            )
        return payload


# --------------------------------------------------------------------------- #
# LaTeX table
# --------------------------------------------------------------------------- #


def render_ablation_table(payload: Mapping[str, object]) -> str:
    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Variant & Accuracy & FPR & Median latency (\\textmu s) & Verdict \\\\",
        "\\midrule",
    ]
    variants = payload.get("variants", [])
    deltas_by_variant = {
        d["variant"]: d for d in payload.get("deltas", [])
    }
    for v in variants:  # type: ignore[assignment]
        variant_name = str(v["variant"])
        verdict = deltas_by_variant.get(variant_name, {}).get("verdict", "reference")
        lines.append(
            f"{variant_name} & {v['detection_accuracy']:.6f} & "
            f"{v['false_positive_rate_fraction']:.6f} & "
            f"{v['median_latency_us']:.2f} & {verdict} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MIRAGE-DoH partial ablation study."
    )
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    runner = AblationStudyRunner()
    payload = runner.run(args.checkpoint_root, strict=args.strict)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "ablation_manifest.json").write_text(json.dumps(payload, indent=2))
    (args.out / "ablation_table.tex").write_text(render_ablation_table(payload))
    LOGGER.info(
        "ablation complete: %d variants, material regression fraction %.3f",
        len(payload["variants"]),
        payload["material_regression_fraction"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
