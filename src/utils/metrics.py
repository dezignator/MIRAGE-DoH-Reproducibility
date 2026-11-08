"""
metrics.py — Evaluation metrics for the MIRAGE-DoH artifact.

Implements the metrics reported in Tables 8, 9, 10, 11, and 12 of the
thesis. Every function is pure: it takes true labels and predictions
(or latency distributions), and returns a dictionary matching the format
expected by verification_script.py.

Thesis reference: Chapter 3, Section 3.12 (Performance metrics) and
Section 3.13 (Detection accuracy, latency, and FPR analysis).
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> Dict[str, float]:
    """Compute standard classification metrics.

    Matches the columns of Table 8: accuracy, FPR, precision, recall, F1,
    AUC-ROC.

    Args:
        y_true: (n_samples,) ground-truth labels (0 or 1).
        y_pred: (n_samples,) predicted labels (0 or 1).
        y_prob: (n_samples,) predicted probabilities for class 1 (optional).
                If provided, AUC-ROC is computed.

    Returns:
        Dict with keys: accuracy, fpr, precision, recall, f1, auc_roc.
        Values are in [0, 1] (not percentages).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy = accuracy_score(y_true, y_pred)
    # FPR = FP / (FP + TN)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    result = {
        "accuracy": float(accuracy),
        "fpr": float(fpr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc_roc": float("nan"),  # filled below if y_prob is given
    }

    if y_prob is not None:
        try:
            result["auc_roc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            # Raised if y_true contains only one class.
            result["auc_roc"] = float("nan")

    return result


def compute_latency_metrics(
    latencies_ms: Sequence[float],
) -> Dict[str, float]:
    """Compute latency percentiles.

    Matches the latency reporting in Table 8 (median latency in µs) and
    the P50/P99 latency in Tables 26 and 33 of later chapters.

    Args:
        latencies_ms: List of per-query latencies in milliseconds.

    Returns:
        Dict with keys: p50_ms, p99_ms, mean_ms, std_ms, min_ms, max_ms.
        Also includes p50_us and p99_us (microseconds) for Table 8's
        microseconds reporting.
    """
    arr = np.asarray(latencies_ms, dtype=np.float64)
    if arr.size == 0:
        return {
            "p50_ms": 0.0, "p99_ms": 0.0,
            "mean_ms": 0.0, "std_ms": 0.0,
            "min_ms": 0.0, "max_ms": 0.0,
            "p50_us": 0.0, "p99_us": 0.0,
        }

    p50 = float(np.percentile(arr, 50))
    p99 = float(np.percentile(arr, 99))
    return {
        "p50_ms": p50,
        "p99_ms": p99,
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=0)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "p50_us": p50 * 1000.0,
        "p99_us": p99 * 1000.0,
    }


def compute_throughput(
    num_queries: int,
    elapsed_seconds: float,
) -> Dict[str, float]:
    """Compute throughput in queries per second.

    Matches Table 8's throughput column (in millions of queries per
    second, M QPS).

    Args:
        num_queries: Number of queries processed.
        elapsed_seconds: Wall-clock time in seconds.

    Returns:
        Dict with keys: qps, m_qps.
    """
    if elapsed_seconds <= 0:
        return {"qps": 0.0, "m_qps": 0.0}
    qps = num_queries / elapsed_seconds
    return {
        "qps": float(qps),
        "m_qps": float(qps / 1_000_000),
    }


def compute_morph_time_metrics(
    morph_times_ms: Sequence[float],
) -> Dict[str, float]:
    """Compute morph-time statistics.

    Matches Table 9's Crypto Morph Time row and Table 8's 8.9 ms ± 0.3 ms.

    Args:
        morph_times_ms: List of per-morph durations in milliseconds.

    Returns:
        Dict with keys: median_ms, ci_95_ms, min_ms, max_ms.
    """
    arr = np.asarray(morph_times_ms, dtype=np.float64)
    if arr.size == 0:
        return {"median_ms": 0.0, "ci_95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    median = float(np.median(arr))
    # 95% CI via bootstrap (matches Table 8's "95% CI" footnote).
    n = len(arr)
    if n >= 30:
        bootstrap_medians = [
            float(np.median(np.random.choice(arr, size=n, replace=True)))
            for _ in range(1000)
        ]
        ci_low = float(np.percentile(bootstrap_medians, 2.5))
        ci_high = float(np.percentile(bootstrap_medians, 97.5))
        ci_95 = (ci_high - ci_low) / 2.0
    else:
        # Small samples: report std as proxy.
        ci_95 = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {
        "median_ms": median,
        "ci_95_ms": float(ci_95),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def compute_all_table_8_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    latencies_ms: Sequence[float],
    num_queries: int,
    elapsed_seconds: float,
    morph_times_ms: Sequence[float] | None = None,
) -> Dict[str, Any]:
    """Compute every metric that appears in Table 8 for a single model.

    Args:
        y_true, y_pred, y_prob: standard classification inputs.
        latencies_ms: per-query latencies.
        num_queries: for throughput.
        elapsed_seconds: for throughput.
        morph_times_ms: per-morph durations (only for MIRAGE-DoH).

    Returns:
        Dict with all Table 8 metrics, formatted exactly as
        expected_outputs/table_8_expected.json.
    """
    cls = compute_classification_metrics(y_true, y_pred, y_prob)
    lat = compute_latency_metrics(latencies_ms)
    tput = compute_throughput(num_queries, elapsed_seconds)

    result = {
        "accuracy_pct": round(cls["accuracy"] * 100.0, 2),
        "fpr_pct": round(cls["fpr"] * 100.0, 4),
        "precision_pct": round(cls["precision"] * 100.0, 2),
        "recall_pct": round(cls["recall"] * 100.0, 2),
        "f1_pct": round(cls["f1"] * 100.0, 2),
        "auc_roc": round(cls["auc_roc"], 4),
        "latency_median_us": round(lat["p50_us"], 0),
        "latency_p99_us": round(lat["p99_us"], 0),
        "throughput_m_qps": round(tput["m_qps"], 1),
    }

    if morph_times_ms is not None:
        morph = compute_morph_time_metrics(morph_times_ms)
        result["morph_time_ms"] = round(morph["median_ms"], 1)
        result["morph_time_ci_95_ms"] = round(morph["ci_95_ms"], 1)

    return result
