#!/usr/bin/env python3
"""
extended_experimentation/tests/test_metric_consistency.py

Consistency tests for the MIRAGE-DoH metric envelopes.

We wrote these tests to protect against a class of bug that would be
invisible until a reviewer opened a result table: a tolerance band in
one module that did not match the corresponding band in another, or a
thesis reference value that had drifted outside the band it was
supposed to fall inside. Both are silent failures — the pipeline runs,
the artefact is produced, and the inconsistency only shows up when a
human reads two files side by side.

The tests cover four things:

  1. The thesis reference values are identical across every module
     that carries its own copy.
  2. Every reference value lies inside its tolerance band.
  3. Every tolerance band is well-formed (low < high).
  4. The confusion-matrix and latency-stat primitives produce the
     arithmetic the metrics rely on.

None of these tests requires torch. They only import the dataclasses
and module-level constants from the extended suite, all of which are
imported cleanly whether or not the deep-learning dependencies are
present.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from extended_experimentation.attack_scenarios.polymorphic_shaping_simulator import (
    THESIS_REFERENCE as POLYMORPHIC_REF,
)
from extended_experimentation.evaluation.core_metrics_evaluator import (
    TOLERANCES as EVAL_TOLERANCES,
    AcceptanceStatus,
    ConfusionMatrix,
    LatencyStats,
)
from extended_experimentation.reporting.extended_result_table_generator import (
    _TOLERANCES as REPORT_TOLERANCES,
)


# --------------------------------------------------------------------------- #
# Thesis reference consistency
# --------------------------------------------------------------------------- #


def test_thesis_reference_is_present_in_the_attack_module() -> None:
    """The polymorphic_shaping_simulator module is the canonical source."""
    assert POLYMORPHIC_REF["detection_accuracy"] == pytest.approx(0.9998, abs=1e-12)
    assert POLYMORPHIC_REF["false_positive_rate_fraction"] == pytest.approx(0.000008, abs=1e-12)
    assert POLYMORPHIC_REF["median_inference_latency_us"] == pytest.approx(78.0, abs=1e-12)
    assert POLYMORPHIC_REF["cryptographic_adaptation_ms"] == pytest.approx(8.9, abs=1e-12)
    assert POLYMORPHIC_REF["throughput_qps"] == pytest.approx(15_300_000.0, abs=1.0)


def test_thesis_reference_carried_by_the_evaluator_matches_the_attack_module() -> None:
    """Every module that carries its own THESIS_REFERENCE must agree with the canonical one."""
    # The evaluator imports THESIS_REFERENCE from the attack module and
    # re-exports it, so this is effectively a same-object check. We still
    # do the field-by-field comparison because a future refactor that
    # copied the dict instead of importing it would silently pass an
    # identity check.
    from extended_experimentation.evaluation import core_metrics_evaluator

    for key, value in POLYMORPHIC_REF.items():
        assert core_metrics_evaluator.THESIS_REFERENCE[key] == value, key


# --------------------------------------------------------------------------- #
# Tolerance band structure
# --------------------------------------------------------------------------- #


def test_tolerance_bands_are_well_formed() -> None:
    for metric, band in EVAL_TOLERANCES.items():
        assert isinstance(band, tuple), metric
        assert len(band) == 2, metric
        low, high = band
        assert low < high, f"{metric}: low={low} is not < high={high}"


def test_tolerance_bands_are_shared_with_the_reporting_module() -> None:
    """The reporting module must not have drifted from the evaluator."""
    assert set(EVAL_TOLERANCES) == set(REPORT_TOLERANCES)
    for metric in EVAL_TOLERANCES:
        assert EVAL_TOLERANCES[metric] == REPORT_TOLERANCES[metric], metric


def test_thesis_reference_values_lie_inside_their_tolerance_bands() -> None:
    """Every reference value with a matching tolerance must be inside that band."""
    for metric, reference_value in POLYMORPHIC_REF.items():
        if metric not in EVAL_TOLERANCES:
            continue
        low, high = EVAL_TOLERANCES[metric]
        assert low <= reference_value <= high, (
            f"{metric}: reference {reference_value} not inside tolerance "
            f"[{low}, {high}]"
        )


# --------------------------------------------------------------------------- #
# Confusion matrix
# --------------------------------------------------------------------------- #


def test_confusion_matrix_matches_thesis_arithmetic() -> None:
    """A confusion matrix at the thesis reference must produce the thesis accuracy and FPR."""
    # We constructed a matrix whose total is one million and whose
    # accuracy and FPR are exactly the thesis values.
    cm = ConfusionMatrix(tp=499_804, fp=4, tn=499_996, fn=196)
    assert cm.n == 1_000_000
    assert abs(cm.accuracy - 0.9998) < 1e-12
    assert abs(cm.fpr - 0.000008) < 1e-12


def test_confusion_matrix_properties() -> None:
    cm = ConfusionMatrix(tp=50, fp=5, tn=45, fn=10)
    assert cm.n == 110
    assert cm.accuracy == pytest.approx(95 / 110)
    assert cm.fpr == pytest.approx(5 / 50)
    assert cm.fnr == pytest.approx(10 / 60)
    assert cm.precision == pytest.approx(50 / 55)
    assert cm.recall == pytest.approx(50 / 60)
    expected_f1 = 2 * (50 / 55) * (50 / 60) / ((50 / 55) + (50 / 60))
    assert cm.f1 == pytest.approx(expected_f1)


def test_confusion_matrix_update_accumulates_counts() -> None:
    cm = ConfusionMatrix()
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    cm.update(y_true, y_pred)
    assert cm.tp == 1
    assert cm.fp == 1
    assert cm.tn == 1
    assert cm.fn == 1
    assert cm.n == 4


def test_confusion_matrix_to_dict_round_trips() -> None:
    cm = ConfusionMatrix(tp=7, fp=2, tn=5, fn=3)
    payload = cm.to_dict()
    assert payload["tp"] == 7
    assert payload["fp"] == 2
    assert payload["tn"] == 5
    assert payload["fn"] == 3
    assert payload["n"] == 17


# --------------------------------------------------------------------------- #
# Latency statistics
# --------------------------------------------------------------------------- #


def test_latency_stats_percentiles_on_known_sequence() -> None:
    stats = LatencyStats()
    for value in (10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0):
        stats.add(value)
    assert stats.n == 10
    assert stats.median == pytest.approx(55.0)
    assert stats.mean == pytest.approx(55.0)
    # numpy percentile with linear interpolation on this sequence:
    # p95 = 95.5, p99 = 99.1.
    assert stats.p95 == pytest.approx(95.5, abs=1e-6)
    assert stats.p99 == pytest.approx(99.1, abs=1e-6)


def test_latency_stats_handles_empty_input() -> None:
    stats = LatencyStats()
    assert stats.n == 0
    assert stats.median == 0.0
    assert stats.mean == 0.0
    payload = stats.to_dict()
    assert payload["min_us"] == 0.0
    assert payload["max_us"] == 0.0


# --------------------------------------------------------------------------- #
# Acceptance status enum
# --------------------------------------------------------------------------- #


def test_acceptance_status_values_are_stable() -> None:
    """The status values are consumed by downstream tools by string, so they must not drift."""
    assert AcceptanceStatus.PASSED.value == "passed"
    assert AcceptanceStatus.FAILED.value == "failed"
    assert AcceptanceStatus.SKIPPED.value == "skipped"
