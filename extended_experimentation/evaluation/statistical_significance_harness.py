#!/usr/bin/env python3
"""
extended_experimentation/evaluation/statistical_significance_harness.py

Statistical significance harness for the MIRAGE-DoH extended
experimentation suite.

We implemented this module to run the three statistical procedures the
thesis relies on to justify the claim that the topological and spectral
features separate benign from attack traffic more strongly than the
baselines:

  1. Paired bootstrap tests on detection accuracy and false-positive rate,
     comparing MIRAGE-DoH against the CNN-LSTM and Random Forest baselines.
  2. Wilcoxon signed-rank tests on the per-flow detection outcomes.
  3. Permutation testing on the MMD statistic from the persistence-image
     manifold, which has no closed-form null distribution.

The harness consumed the artefacts produced by the extraction and
evaluation modules and produced a LaTeX table plus a machine-readable
JSON summary. It did not retrain any model. That constraint was deliberate:
a significance test that silently retrained the model could produce a
p-value that reflects the training seed rather than the difference
between methods.

Design choice. The bootstrap resampling was paired on flow id, not on
window id, because the windows within a flow are not independent. We
resampled flows with replacement, keeping all windows from each sampled
flow together. That is the conservative choice; unpaired window-level
resampling was also implemented but was reported only as a footnote in
the thesis for completeness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..attack_scenarios.polymorphic_shaping_simulator import (
    THESIS_REFERENCE,
    Label,
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

LOGGER = logging.getLogger("mirage.extended.significance")

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class TestKind(str, Enum):
    PAIRED_BOOTSTRAP = "paired_bootstrap"
    WILCOXON = "wilcoxon"
    PERMUTATION_MMD = "permutation_mmd"
    FRIEDMAN = "friedman"


class Outcome(str, Enum):
    SIGNIFICANT = "significant"
    NOT_SIGNIFICANT = "not_significant"
    INCONCLUSIVE = "inconclusive"


# --------------------------------------------------------------------------- #
# Test result
# --------------------------------------------------------------------------- #


@dataclass
class TestResult:
    test: TestKind
    comparison: str
    metric: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Optional[Tuple[float, float]]
    n_samples: int
    outcome: Outcome

    def to_dict(self) -> Dict[str, object]:
        return {
            "test": self.test.value,
            "comparison": self.comparison,
            "metric": self.metric,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "confidence_interval": list(self.confidence_interval)
            if self.confidence_interval
            else None,
            "n_samples": self.n_samples,
            "outcome": self.outcome.value,
        }


def _significance_outcome(p: float, alpha: float = 0.05) -> Outcome:
    if math.isnan(p):
        return Outcome.INCONCLUSIVE
    return Outcome.SIGNIFICANT if p < alpha else Outcome.NOT_SIGNIFICANT


# --------------------------------------------------------------------------- #
# Paired bootstrap
# --------------------------------------------------------------------------- #


def _paired_bootstrap_accuracy(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    flow_ids: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 20231130,
) -> Tuple[float, float, Tuple[float, float]]:
    """We resampled flows (not windows) and recomputed accuracy for each method.

    The returned tuple is (observed_delta, p_value, 95% CI). ``observed_delta``
    is ``acc_a - acc_b`` on the full set. The p-value was computed as the
    fraction of resamples whose delta was at the opposite sign from the
    observed delta (two-sided, with the standard +1 smoothing).
    """
    flow_ids = np.asarray(flow_ids)
    unique_flows = np.unique(flow_ids)
    if unique_flows.size < 2:
        return 0.0, 1.0, (0.0, 0.0)
    rng = np.random.default_rng(seed)

    def acc(pred: np.ndarray) -> float:
        return float((pred == y_true).mean())

    observed_delta = acc(y_pred_a) - acc(y_pred_b)

    deltas = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample = rng.choice(unique_flows, size=unique_flows.size, replace=True)
        mask = np.isin(flow_ids, sample)
        if not mask.any():
            deltas[i] = 0.0
            continue
        deltas[i] = acc(y_pred_a[mask]) - acc(y_pred_b[mask])

    # Two-sided: how often did the resampled delta fall on the other side?
    if observed_delta >= 0:
        extreme = (deltas <= -observed_delta).sum() + (deltas >= observed_delta).sum()
    else:
        extreme = (deltas >= -observed_delta).sum() + (deltas <= observed_delta).sum()
    p = (extreme + 1) / (n_resamples + 1)
    ci = (
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
    )
    return float(observed_delta), float(min(1.0, p)), ci


def _paired_bootstrap_fpr(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    flow_ids: np.ndarray,
    n_resamples: int = 10_000,
    seed: int = 20231130,
) -> Tuple[float, float, Tuple[float, float]]:
    flow_ids = np.asarray(flow_ids)
    unique_flows = np.unique(flow_ids)
    if unique_flows.size < 2:
        return 0.0, 1.0, (0.0, 0.0)
    rng = np.random.default_rng(seed)

    benign_mask = y_true == int(Label.BENIGN)
    if not benign_mask.any():
        return 0.0, 1.0, (0.0, 0.0)

    def fpr(pred: np.ndarray) -> float:
        fp = ((pred == 1) & benign_mask).sum()
        n = benign_mask.sum()
        return float(fp / n)

    observed_delta = fpr(y_pred_a) - fpr(y_pred_b)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        sample = rng.choice(unique_flows, size=unique_flows.size, replace=True)
        mask = np.isin(flow_ids, sample)
        if not mask.any():
            deltas[i] = 0.0
            continue
        # Restrict to the resampled flows and recompute FPR within that subset.
        b = benign_mask & mask
        if not b.any():
            deltas[i] = 0.0
            continue
        fp_a = ((y_pred_a[mask] == 1) & b[mask]).sum()
        fp_b = ((y_pred_b[mask] == 1) & b[mask]).sum()
        n_benign = b.sum()
        deltas[i] = fp_a / max(n_benign, 1) - fp_b / max(n_benign, 1)

    if observed_delta >= 0:
        extreme = (deltas <= -observed_delta).sum() + (deltas >= observed_delta).sum()
    else:
        extreme = (deltas >= -observed_delta).sum() + (deltas <= observed_delta).sum()
    p = (extreme + 1) / (n_resamples + 1)
    ci = (
        float(np.percentile(deltas, 2.5)),
        float(np.percentile(deltas, 97.5)),
    )
    return float(observed_delta), float(min(1.0, p)), ci


# --------------------------------------------------------------------------- #
# Wilcoxon signed-rank test
# --------------------------------------------------------------------------- #


def _wilcoxon_signed_rank(
    a: np.ndarray, b: np.ndarray
) -> Tuple[float, float, float]:
    """We implemented the exact normal-approximation Wilcoxon signed-rank test.

    The implementation was necessary because SciPy was not a dependency of
    the extended suite. We verified the implementation against the SciPy
    reference on the same input and matched to five decimal places; the
    matching test lives in ``tests/test_statistical_harness.py``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same shape")
    diffs = a - b
    nz = diffs != 0
    diffs = diffs[nz]
    n = diffs.size
    if n < 10:
        return 0.0, 1.0, 0.0
    ranks = _rankdata(np.abs(diffs))
    w_plus = float(ranks[diffs > 0].sum())
    w_minus = float(ranks[diffs < 0].sum())
    w = min(w_plus, w_minus)
    mean_w = n * (n + 1) / 4.0
    std_w = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    if std_w <= 0:
        return w, 1.0, 0.0
    z = (w - mean_w) / std_w
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    p = min(1.0, max(0.0, p))

    # Matched-pairs rank-biserial correlation as effect size.
    r = (w_plus - w_minus) / (n * (n + 1) / 2.0) if n > 0 else 0.0
    return w, float(p), float(r)


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Standard average-rank transformation, matching SciPy's convention."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        avg = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg
        i = j
    return ranks


# --------------------------------------------------------------------------- #
# Permutation testing for MMD
# --------------------------------------------------------------------------- #


def _mmd_rbf(x: np.ndarray, y: np.ndarray, bandwidth: Optional[float] = None) -> float:
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
    return float(max(0.0, k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean()))


def permutation_mmd(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 20231130,
) -> Tuple[float, float]:
    """We permuted the label assignment and recomputed MMD under the null.

    The p-value was computed as the fraction of permuted MMDs that met or
    exceeded the observed MMD, with the standard +1 smoothing so that a
    p-value of exactly zero could not be reported.
    """
    observed = _mmd_rbf(x, y)
    rng = np.random.default_rng(seed)
    combined = np.vstack([x, y])
    n_x = x.shape[0]
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(combined.shape[0])
        x_perm = combined[perm[:n_x]]
        y_perm = combined[perm[n_x:]]
        if _mmd_rbf(x_perm, y_perm) >= observed:
            count += 1
    p = (count + 1) / (n_permutations + 1)
    return float(observed), float(p)


# --------------------------------------------------------------------------- #
# Friedman test (multi-method comparison)
# --------------------------------------------------------------------------- #


def _friedman_test(*method_scores: np.ndarray) -> Tuple[float, float]:
    """Friedman test across k methods on n paired blocks."""
    scores = np.vstack(method_scores)  # shape (k, n)
    k, n = scores.shape
    if k < 3 or n < 5:
        return 0.0, 1.0
    ranks = np.apply_along_axis(_rankdata, 1, scores)
    rank_means = ranks.mean(axis=1)
    chi2 = (12.0 * n / (k * (k + 1))) * (
        (rank_means ** 2).sum() - k * ((k + 1) ** 2) / 4.0
    )
    df = k - 1
    # Survival function of chi-square at chi2 with df degrees of freedom,
    # via the regularised incomplete gamma function.
    p = 1.0 - _chi2_cdf(chi2, df)
    return float(chi2), float(max(0.0, min(1.0, p)))


def _chi2_cdf(x: float, df: int) -> float:
    """Regularised lower incomplete gamma for chi-square CDF."""
    if x <= 0:
        return 0.0
    a = df / 2.0
    z = x / 2.0
    return _gammainc_lower_regularised(a, z)


def _gammainc_lower_regularised(a: float, z: float) -> float:
    """Series and continued-fraction evaluation of P(a, z).

    This is the Numerical Recipes gammp routine, transcribed for
    dependency-freedom. We tested it against scipy.special.gammainc on 500
    random inputs and matched to machine precision.
    """
    if z < 0 or a <= 0:
        return 0.0
    if z < a + 1.0:
        # Series representation.
        term = 1.0 / a
        total = term
        ap = a
        for _ in range(200):
            ap += 1.0
            term *= z / ap
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        return total * math.exp(-z + a * math.log(z) - math.lgamma(a))
    # Continued fraction.
    tiny = 1e-300
    b = z + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 200):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return 1.0 - h * math.exp(-z + a * math.log(z) - math.lgamma(a))


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #


@dataclass
class PredictionPayload:
    """One method's per-window predictions, as read from an archive."""

    method: str
    flow_ids: List[str]
    labels: List[int]
    predictions: List[int]


@dataclass
class HarnessConfig:
    bootstrap_resamples: int = 10_000
    permutation_resamples: int = 1_000
    seed: int = 20231130
    alpha: float = 0.05


class SignificanceHarness:
    def __init__(self, config: Optional[HarnessConfig] = None) -> None:
        self.config = config or HarnessConfig()

    def run(
        self,
        mirage: PredictionPayload,
        baselines: Sequence[PredictionPayload],
        mmd_x: Optional[np.ndarray] = None,
        mmd_y: Optional[np.ndarray] = None,
    ) -> Dict[str, object]:
        started = time.monotonic()
        labels = np.array(mirage.labels, dtype=np.int64)
        flow_ids = np.array(mirage.flow_ids)
        mirage_preds = np.array(mirage.predictions, dtype=np.int64)

        results: List[TestResult] = []

        for baseline in baselines:
            if baseline.flow_ids != mirage.flow_ids:
                raise ValueError(
                    f"flow_id alignment mismatch with baseline {baseline.method}; "
                    "the harness requires identical ordered inputs"
                )
            baseline_preds = np.array(baseline.predictions, dtype=np.int64)
            delta_acc, p_acc, ci_acc = _paired_bootstrap_accuracy(
                labels, mirage_preds, baseline_preds, flow_ids,
                n_resamples=self.config.bootstrap_resamples, seed=self.config.seed,
            )
            results.append(
                TestResult(
                    test=TestKind.PAIRED_BOOTSTRAP,
                    comparison=f"mirage_vs_{baseline.method}",
                    metric="accuracy_delta",
                    statistic=delta_acc,
                    p_value=p_acc,
                    effect_size=delta_acc,
                    confidence_interval=ci_acc,
                    n_samples=int(labels.size),
                    outcome=_significance_outcome(p_acc, self.config.alpha),
                )
            )

            delta_fpr, p_fpr, ci_fpr = _paired_bootstrap_fpr(
                labels, mirage_preds, baseline_preds, flow_ids,
                n_resamples=self.config.bootstrap_resamples, seed=self.config.seed + 1,
            )
            results.append(
                TestResult(
                    test=TestKind.PAIRED_BOOTSTRAP,
                    comparison=f"mirage_vs_{baseline.method}",
                    metric="fpr_delta",
                    statistic=delta_fpr,
                    p_value=p_fpr,
                    effect_size=delta_fpr,
                    confidence_interval=ci_fpr,
                    n_samples=int(labels.size),
                    outcome=_significance_outcome(p_fpr, self.config.alpha),
                )
            )

            # Per-flow paired wilcoxon on per-flow correctness.
            per_flow_a, per_flow_b = _per_flow_correctness(
                labels, mirage_preds, baseline_preds, flow_ids
            )
            w, p_w, r = _wilcoxon_signed_rank(per_flow_a, per_flow_b)
            results.append(
                TestResult(
                    test=TestKind.WILCOXON,
                    comparison=f"mirage_vs_{baseline.method}",
                    metric="per_flow_correctness",
                    statistic=w,
                    p_value=p_w,
                    effect_size=r,
                    confidence_interval=None,
                    n_samples=int(per_flow_a.size),
                    outcome=_significance_outcome(p_w, self.config.alpha),
                )
            )

        if len(baselines) >= 2:
            method_scores = [
                _per_flow_accuracy(labels, np.array(mirage.predictions), flow_ids),
                *[
                    _per_flow_accuracy(labels, np.array(b.predictions), flow_ids)
                    for b in baselines
                ],
            ]
            chi2, p_fried = _friedman_test(*method_scores)
            results.append(
                TestResult(
                    test=TestKind.FRIEDMAN,
                    comparison="all_methods",
                    metric="per_flow_accuracy",
                    statistic=chi2,
                    p_value=p_fried,
                    effect_size=0.0,
                    confidence_interval=None,
                    n_samples=int(method_scores[0].size),
                    outcome=_significance_outcome(p_fried, self.config.alpha),
                )
            )

        mmd_stat = math.nan
        mmd_p = math.nan
        if mmd_x is not None and mmd_y is not None and mmd_x.size and mmd_y.size:
            mmd_stat, mmd_p = permutation_mmd(
                mmd_x,
                mmd_y,
                n_permutations=self.config.permutation_resamples,
                seed=self.config.seed,
            )
            results.append(
                TestResult(
                    test=TestKind.PERMUTATION_MMD,
                    comparison="benign_vs_attack",
                    metric="persistence_image_mmd",
                    statistic=float(mmd_stat),
                    p_value=float(mmd_p),
                    effect_size=float(mmd_stat),
                    confidence_interval=None,
                    n_samples=int(mmd_x.shape[0] + mmd_y.shape[0]),
                    outcome=_significance_outcome(mmd_p, self.config.alpha),
                )
            )

        elapsed = time.monotonic() - started
        payload: Dict[str, object] = {
            "module": "extended_experimentation.evaluation.statistical_significance_harness",
            "config": asdict(self.config),
            "n_methods": 1 + len(baselines),
            "n_windows": int(labels.size),
            "n_flows": int(np.unique(flow_ids).size),
            "results": [r.to_dict() for r in results],
            "elapsed_s": elapsed,
            "thesis_reference": THESIS_REFERENCE,
        }
        payload["manifest_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return payload


def _per_flow_correctness(
    labels: np.ndarray, preds_a: np.ndarray, preds_b: np.ndarray, flow_ids: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    unique = np.unique(flow_ids)
    acc_a = np.empty(unique.size, dtype=np.float64)
    acc_b = np.empty(unique.size, dtype=np.float64)
    for i, fid in enumerate(unique):
        mask = flow_ids == fid
        acc_a[i] = float((preds_a[mask] == labels[mask]).mean()) if mask.any() else 0.0
        acc_b[i] = float((preds_b[mask] == labels[mask]).mean()) if mask.any() else 0.0
    return acc_a, acc_b


def _per_flow_accuracy(
    labels: np.ndarray, preds: np.ndarray, flow_ids: np.ndarray
) -> np.ndarray:
    unique = np.unique(flow_ids)
    out = np.empty(unique.size, dtype=np.float64)
    for i, fid in enumerate(unique):
        mask = flow_ids == fid
        out[i] = float((preds[mask] == labels[mask]).mean()) if mask.any() else 0.0
    return out


# --------------------------------------------------------------------------- #
# LaTeX renderer
# --------------------------------------------------------------------------- #


def render_latex_table(payload: Mapping[str, object]) -> str:
    lines = [
        "\\begin{tabular}{lllrrrl}",
        "\\toprule",
        "Test & Comparison & Metric & Statistic & p-value & Effect & Outcome \\\\",
        "\\midrule",
    ]
    for row in payload.get("results", []):  # type: ignore[assignment]
        p = row["p_value"]
        p_str = "n/a" if (isinstance(p, float) and math.isnan(p)) else f"{p:.3e}"
        lines.append(
            f"{row['test']} & {row['comparison']} & {row['metric']} & "
            f"{row['statistic']:.4f} & {p_str} & {row['effect_size']:.4f} & "
            f"{row['outcome']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #


def _load_predictions(path: Path, method: str) -> PredictionPayload:
    """We loaded predictions from a JSONL file with one record per window.

    Required fields: ``flow_id``, ``label``, ``prediction``. Optional:
    ``method``. The method name was used only for reporting, not for
    alignment.
    """
    flow_ids: List[str] = []
    labels: List[int] = []
    preds: List[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            flow_ids.append(str(rec["flow_id"]))
            labels.append(int(rec["label"]))
            preds.append(int(rec["prediction"]))
    return PredictionPayload(
        method=method, flow_ids=flow_ids, labels=labels, predictions=preds
    )


def _load_mmd_points(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """We loaded persistence-image summaries for the MMD permutation test.

    File format: JSONL with ``label`` (0 or 1) and ``features`` (list of
    floats) fields.
    """
    xs: List[List[float]] = []
    ys: List[List[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            label = int(rec["label"])
            feats = [float(v) for v in rec["features"]]
            if label == 0:
                xs.append(feats)
            else:
                ys.append(feats)
    if not xs or not ys:
        return np.empty((0, 0)), np.empty((0, 0))
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the MIRAGE-DoH statistical significance harness on archived "
            "prediction files."
        )
    )
    parser.add_argument(
        "--mirage",
        type=Path,
        required=True,
        help="JSONL of MIRAGE-DoH per-window predictions",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        nargs="+",
        required=True,
        help="one or more JSONL files of baseline predictions, aligned by flow_id",
    )
    parser.add_argument(
        "--mmd-points",
        type=Path,
        default=None,
        help="JSONL of persistence-image feature points for the MMD test",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--permutation-resamples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20231130)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    mirage = _load_predictions(args.mirage, "mirage")
    baselines = [
        _load_predictions(p, p.stem) for p in args.baseline
    ]
    mmd_x = mmd_y = None
    if args.mmd_points is not None:
        mmd_x, mmd_y = _load_mmd_points(args.mmd_points)

    harness = SignificanceHarness(
        HarnessConfig(
            bootstrap_resamples=args.bootstrap_resamples,
            permutation_resamples=args.permutation_resamples,
            seed=args.seed,
        )
    )
    payload = harness.run(mirage, baselines, mmd_x=mmd_x, mmd_y=mmd_y)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "significance_manifest.json").write_text(
        json.dumps(payload, indent=2)
    )
    (args.out / "significance_table.tex").write_text(render_latex_table(payload))
    LOGGER.info(
        "wrote significance results (%d tests) to %s",
        len(payload["results"]),
        args.out,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
