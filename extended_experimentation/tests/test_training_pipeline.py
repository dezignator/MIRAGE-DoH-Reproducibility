#!/usr/bin/env python3
"""
extended_experimentation/tests/test_training_pipeline.py

Tests for the persistence-pipeline primitives that the TMN trainer
consumes.

We wrote these tests against the parts of the training pipeline that
do not require torch. That distinction was deliberate: the TMN
trainer itself is a torch module, and running it in a CPU-only test
environment would have either required a full torch install or a
heavy mock. The persistence extraction, the image rasterisation, the
birth/persistence range computation and the train/val/test split are
all torch-independent, and they are where the majority of the
correctness risk actually lives — a bug in the split would contaminate
every result in the thesis, and a bug in the persistence computation
would silently corrupt the topological feature.

The tests cover five components:

  1. VietorisRipsFiltration behaviour on small, oversized and empty
     point clouds.
  2. The birth/persistence range computation.
  3. The persistence-image rasteriser.
  4. The train/validation/test split.
  5. The window extractor.
"""

from __future__ import annotations

import numpy as np
import pytest

from extended_experimentation.attack_scenarios.polymorphic_shaping_simulator import (
    Protocol,
)
from extended_experimentation.training.topological_memory_trainer import (
    PersistenceDiagram,
    VietorisRipsFiltration,
    _birth_persistence_ranges,
    _split_indices,
    betti_curve,
    extract_windows_from_flow,
    persistence_image,
    persistence_stability,
)


# --------------------------------------------------------------------------- #
# Vietoris-Rips filtration
# --------------------------------------------------------------------------- #


def test_filtration_rejects_oversized_point_clouds() -> None:
    filtration = VietorisRipsFiltration(max_points=10, max_dim=1)
    points = np.random.default_rng(0).normal(size=(20, 2))
    with pytest.raises(ValueError):
        filtration.compute(points)


def test_filtration_returns_empty_diagrams_for_a_single_point() -> None:
    filtration = VietorisRipsFiltration(max_points=10, max_dim=1)
    diagrams = filtration.compute(np.array([[1.0, 2.0]]))
    assert 0 in diagrams
    assert 1 in diagrams
    assert diagrams[0].n_pairs == 0
    assert diagrams[1].n_pairs == 0


def test_filtration_produces_one_h0_component_per_point() -> None:
    """For n points in general position the H0 diagram has n pairs: n-1 finite, 1 essential."""
    filtration = VietorisRipsFiltration(max_points=10, max_dim=1)
    rng = np.random.default_rng(20231130)
    points = rng.normal(size=(5, 2))
    diagrams = filtration.compute(points)
    assert diagrams[0].n_pairs == 5
    assert diagrams[0].finite().shape[0] == 4


# --------------------------------------------------------------------------- #
# Birth / persistence range
# --------------------------------------------------------------------------- #


def test_birth_persistence_ranges_handles_the_empty_case() -> None:
    birth_range, persistence_range = _birth_persistence_ranges([])
    assert birth_range == (0.0, 1.0)
    assert persistence_range == (0.0, 1.0)


def test_birth_persistence_ranges_covers_finite_pairs() -> None:
    diagrams = [
        PersistenceDiagram(dim=0, pairs=np.array([[0.0, 1.0], [0.5, 1.5]])),
        PersistenceDiagram(dim=1, pairs=np.array([[0.1, 0.5]])),
    ]
    birth_range, persistence_range = _birth_persistence_ranges(diagrams)
    assert birth_range[0] == pytest.approx(0.0)
    assert birth_range[1] >= 0.5
    assert persistence_range[0] == pytest.approx(0.0)
    assert persistence_range[1] >= 1.0


# --------------------------------------------------------------------------- #
# Persistence image
# --------------------------------------------------------------------------- #


def test_persistence_image_returns_the_configured_shape() -> None:
    diagram = PersistenceDiagram(
        dim=0,
        pairs=np.array([[0.0, 1.0], [0.5, 1.5], [0.0, 2.0]]),
    )
    img = persistence_image(
        diagram,
        bins=16,
        sigma=0.05,
        birth_range=(0.0, 2.0),
        persistence_range=(0.0, 2.0),
    )
    assert img.shape == (16, 16)
    assert img.sum() > 0.0


def test_persistence_image_of_an_empty_diagram_is_zero() -> None:
    diagram = PersistenceDiagram(dim=0, pairs=np.empty((0, 2)))
    img = persistence_image(
        diagram,
        bins=8,
        sigma=0.05,
        birth_range=(0.0, 1.0),
        persistence_range=(0.0, 1.0),
    )
    assert img.shape == (8, 8)
    assert img.sum() == 0.0


# --------------------------------------------------------------------------- #
# Train / validation / test split
# --------------------------------------------------------------------------- #


def test_split_indices_are_disjoint_and_exhaustive() -> None:
    train, val, test = _split_indices(100, 20231130)
    combined = np.concatenate([train, val, test])
    assert combined.shape == (100,)
    assert len(set(combined.tolist())) == 100
    assert train.shape == (70,)
    assert val.shape == (15,)
    assert test.shape == (15,)


def test_split_indices_handles_the_empty_case() -> None:
    train, val, test = _split_indices(0, 20231130)
    assert train.shape == (0,)
    assert val.shape == (0,)
    assert test.shape == (0,)


# --------------------------------------------------------------------------- #
# Window extraction
# --------------------------------------------------------------------------- #


def test_window_extraction_produces_the_expected_number_of_windows() -> None:
    sizes = np.linspace(100.0, 500.0, 100)
    iats = np.linspace(0.01, 0.1, 100)
    directions = (np.arange(100) % 2).astype(int)
    windows = list(
        extract_windows_from_flow(
            flow_id="test-flow",
            label=0,
            sizes=sizes,
            iats=iats,
            directions=directions,
            protocol=Protocol.DOH,
            window_size=32,
            stride=16,
        )
    )
    # For n=100, window=32, stride=16: start positions are 0, 16, 32, 48, 64.
    assert len(windows) == 5
    for window in windows:
        assert window.features.shape == (6, 32)
        assert window.flow_id == "test-flow"
        assert window.label == 0


def test_window_extraction_returns_nothing_for_short_flows() -> None:
    sizes = np.array([100.0] * 10)
    iats = np.array([0.01] * 10)
    directions = np.zeros(10, dtype=int)
    windows = list(
        extract_windows_from_flow(
            flow_id="short",
            label=0,
            sizes=sizes,
            iats=iats,
            directions=directions,
            protocol=Protocol.DOH,
            window_size=32,
            stride=16,
        )
    )
    assert windows == []


# --------------------------------------------------------------------------- #
# Persistence-stability and Betti-curve helpers
# --------------------------------------------------------------------------- #


def test_persistence_stability_is_zero_for_two_empty_diagrams() -> None:
    empty_a = PersistenceDiagram(dim=0, pairs=np.empty((0, 2)))
    empty_b = PersistenceDiagram(dim=0, pairs=np.empty((0, 2)))
    assert persistence_stability([empty_a], [empty_b]) == 0.0


def test_persistence_stability_is_bounded() -> None:
    dgm_a = PersistenceDiagram(dim=0, pairs=np.array([[0.0, 1.0], [0.5, 1.5]]))
    dgm_b = PersistenceDiagram(dim=0, pairs=np.array([[0.0, 1.1], [0.5, 1.6]]))
    stability = persistence_stability([dgm_a], [dgm_b])
    assert 0.0 <= stability <= 1.0


def test_betti_curve_has_the_requested_length() -> None:
    diagram = PersistenceDiagram(
        dim=0,
        pairs=np.array([[0.0, 1.0], [0.5, 1.5], [0.0, 2.0]]),
    )
    thresholds, curve = betti_curve(diagram, n_points=64)
    assert thresholds.shape == (64,)
    assert curve.shape == (64,)
    # The Betti number must never be negative.
    assert (curve >= 0).all()
