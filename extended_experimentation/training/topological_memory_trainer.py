#!/usr/bin/env python3
"""


Topological Memory Network (TMN) trainer for the MIRAGE-DoH extended
experimentation suite.

We implemented this module to train the topological branch of MIRAGE-DoH
described in Section 3.6. It consumes flow records emitted by the campaign
simulators, constructs flow-level feature windows, builds Vietoris-Rips
filtrations on the resulting point clouds, and trains an LSTM-attention
hybrid over the resulting persistence images. We additionally exported the
persistence diagrams and Betti curves so that the statistical-significance
harness in ``evaluation/`` could run the topological separability tests
described in the thesis without recomputing them.

The trainer is written against PyTorch. We kept the persistence backend
dependency-free (pure NumPy) so that the artifacts remain reproducible on
machines that do not have GUDHI or ripser installed. On point clouds of
up to sixty points the NumPy implementation matched the reference
persistence outputs within numerical tolerance, which was sufficient for
the sliding-window sizes used in the thesis evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - import gate
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    _TORCH_AVAILABLE = True
except Exception:                        # pragma: no cover - defensive
    torch = None                          # type: ignore[assignment]
    nn = None                             # type: ignore[assignment]
    DataLoader = None                     # type: ignore[assignment]
    Dataset = object                     # type: ignore[assignment,misc]
    _TORCH_AVAILABLE = False

from ..attack_scenarios.polymorphic_shaping_simulator import (
    THESIS_REFERENCE,
    Label,
    Protocol,
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

LOGGER = logging.getLogger("mirage.extended.tmn_trainer")

# --------------------------------------------------------------------------- #
# Hyperparameter registry
# --------------------------------------------------------------------------- #
# We captured the exact hyperparameters that produced the topological-recall
# and persistence-stability figures reported in Chapter 3. Downstream tooling
# read this registry verbatim so that any trained artifact carries the recipe
# that produced it.

TMN_HYPERPARAMETERS: Dict[str, object] = {
    "window_size_packets": 32,
    "window_stride_packets": 16,
    "max_points_per_window": 48,
    "feature_dim": 6,
    "embedding_dim": 64,
    "lstm_hidden": 96,
    "lstm_layers": 2,
    "attention_heads": 4,
    "dropout": 0.15,
    "learning_rate": 8e-4,
    "weight_decay": 1e-5,
    "batch_size": 32,
    "epochs": 24,
    "gradient_clip_norm": 1.0,
    "early_stop_patience": 4,
    "persistence_dims": [0, 1],
    "persistence_image_bins": 32,
    "persistence_image_sigma": 0.08,
    "random_seed": 20231130,
}

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class FiltrationKind(str, Enum):
    VIETORIS_RIPS = "vietoris_rips"


# --------------------------------------------------------------------------- #
# Feature window
# --------------------------------------------------------------------------- #


@dataclass
class FeatureWindow:
    """One flow-level sliding window, before point-cloud construction.

    Each column is a per-packet feature. The columns we used were chosen to
    expose the packet-size and inter-arrival structure that the topological
    branch was designed to capture, while remaining independent of payload
    content so that the transformation remained applicable under encryption.
    """

    flow_id: str
    label: int
    start_index: int
    end_index: int
    features: np.ndarray             # shape (feature_dim, n_packets)
    protocol: Protocol

    @property
    def n_packets(self) -> int:
        return int(self.features.shape[1])

    def as_point_cloud(self, max_points: int) -> np.ndarray:
        """We subsampled the window to at most ``max_points`` before TDA.

        Subsampling was deterministic given the window: we took the columns
        with the largest inter-packet distance from the per-feature median,
        which preserved the geometric structure that drove the persistence
        diagrams.
        """
        n = self.n_packets
        if n <= max_points:
            return np.ascontiguousarray(self.features.T, dtype=np.float32)
        median = np.median(self.features, axis=1, keepdims=True)
        dist = np.linalg.norm(self.features - median, axis=0)
        keep = np.argsort(-dist)[:max_points]
        keep.sort()
        return np.ascontiguousarray(self.features[:, keep].T, dtype=np.float32)


def extract_windows_from_flow(
    flow_id: str,
    label: int,
    sizes: np.ndarray,
    iats: np.ndarray,
    directions: np.ndarray,
    protocol: Protocol,
    window_size: int,
    stride: int,
) -> Iterable[FeatureWindow]:
    """We constructed feature windows from the raw packet arrays.

    Six features were computed per packet:

      0. log10 payload size,
      1. log10 inter-arrival time,
      2. signed direction (-1 client to resolver, +1 resolver to client),
      3. rolling mean of log10 size over a five-packet window,
      4. rolling mean of log10 iat over a five-packet window,
      5. packet position normalised to [0, 1] within the flow.

    The rolling means captured local context without requiring the full
    window to be re-scanned downstream.
    """
    n = sizes.shape[0]
    if n < window_size:
        return

    log_sizes = np.log10(np.clip(sizes, 1.0, None))
    log_iats = np.log10(np.clip(iats, 1e-4, None))
    sign_dir = np.where(directions == 0, -1.0, 1.0)

    kernel = np.ones(5) / 5.0
    roll_size = np.convolve(log_sizes, kernel, mode="same")
    roll_iat = np.convolve(log_iats, kernel, mode="same")
    positions = np.linspace(0.0, 1.0, n)

    features = np.vstack(
        [log_sizes, log_iats, sign_dir, roll_size, roll_iat, positions]
    )
    for start in range(0, n - window_size + 1, stride):
        end = start + window_size
        yield FeatureWindow(
            flow_id=flow_id,
            label=label,
            start_index=start,
            end_index=end,
            features=features[:, start:end].copy(),
            protocol=protocol,
        )


# --------------------------------------------------------------------------- #
# Persistence computation
# --------------------------------------------------------------------------- #


@dataclass
class PersistenceDiagram:
    """Birth-death pairs for one dimension."""

    dim: int
    pairs: np.ndarray               # shape (n_pairs, 2), [birth, death]

    def __post_init__(self) -> None:
        if self.pairs.ndim != 2 or self.pairs.shape[1] != 2:
            raise ValueError("pairs must be a 2-column array")

    @property
    def n_pairs(self) -> int:
        return int(self.pairs.shape[0])

    def finite(self) -> np.ndarray:
        mask = np.isfinite(self.pairs[:, 1])
        return self.pairs[mask]

    def betti(self, threshold: float) -> int:
        mask = (self.pairs[:, 0] <= threshold) & (
            (self.pairs[:, 1] > threshold) | ~np.isfinite(self.pairs[:, 1])
        )
        return int(mask.sum())


class VietorisRipsFiltration:
    """Vietoris-Rips persistence for small point clouds.

    We implemented the standard boundary-matrix reduction. On point clouds
    of up to sixty points the implementation was adequate; for larger clouds
    the class degraded to an informative error rather than silently
    truncating, because a truncated persistence diagram cannot be used for
    the statistical separability tests reported in the thesis.
    """

    def __init__(self, max_points: int = 60, max_dim: int = 1) -> None:
        self.max_points = max_points
        self.max_dim = max_dim

    def compute(self, points: np.ndarray) -> Dict[int, PersistenceDiagram]:
        points = np.ascontiguousarray(points, dtype=np.float64)
        n = points.shape[0]
        if n < 2:
            return {d: PersistenceDiagram(d, np.empty((0, 2))) for d in range(self.max_dim + 1)}
        if n > self.max_points:
            raise ValueError(
                f"point cloud of size {n} exceeds max_points={self.max_points}; "
                "reduce window size or increase max_points"
            )

        dists = _pairwise_distances(points)
        simplices: List[Tuple[float, Tuple[int, ...], int]] = []
        for i in range(n):
            simplices.append((0.0, (i,), 0))

        edge_index: Dict[Tuple[int, int], int] = {}
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[(i, j)] = len(simplices)
                simplices.append((float(dists[i, j]), (i, j), 1))

        triangles: List[Tuple[float, Tuple[int, int, int]]] = []
        for i in range(n):
            for j in range(i + 1, n):
                dij = dists[i, j]
                for k in range(j + 1, n):
                    t = max(dij, dists[i, k], dists[j, k])
                    triangles.append((float(t), (i, j, k)))
        triangles.sort(key=lambda x: x[0])
        triangle_index: Dict[Tuple[int, int, int], int] = {}
        for t, tri in triangles:
            triangle_index[tri] = len(simplices)
            simplices.append((t, tri, 2))

        n_simplices = len(simplices)
        boundary: List[set] = [set() for _ in range(n_simplices)]
        for idx, (_, simplex, dim) in enumerate(simplices):
            if dim == 0:
                continue
            if dim == 1:
                i, j = simplex
                boundary[idx].add(_vertex_index(i, n))
                boundary[idx].add(_vertex_index(j, n))
            elif dim == 2:
                i, j, k = simplex
                boundary[idx].add(edge_index[(i, j)])
                boundary[idx].add(edge_index[(i, k)])
                boundary[idx].add(edge_index[(j, k)])

        reduced: List[set] = [set(b) for b in boundary]
        pivot_owner: Dict[int, int] = {}
        pairs: List[Tuple[int, int]] = []
        essential_births: List[int] = []

        for j in range(n_simplices):
            while reduced[j]:
                low = max(reduced[j])
                if low in pivot_owner:
                    reduced[j] ^= reduced[pivot_owner[low]]
                else:
                    pivot_owner[low] = j
                    pairs.append((low, j))
                    break
            if not reduced[j]:
                essential_births.append(j)

        paired_birth = {p[0] for p in pairs}
        diagrams: Dict[int, List[Tuple[float, float]]] = {
            d: [] for d in range(self.max_dim + 1)
        }
        for birth_idx, death_idx in pairs:
            birth_val, _, birth_dim = simplices[birth_idx]
            death_val, _, death_dim = simplices[death_idx]
            if death_dim != birth_dim + 1:
                continue
            if birth_dim in diagrams:
                diagrams[birth_dim].append((birth_val, death_val))
        for idx in essential_births:
            if idx in paired_birth:
                continue
            val, _, dim = simplices[idx]
            if dim in diagrams:
                diagrams[dim].append((val, math.inf))

        return {
            d: PersistenceDiagram(
                dim=d,
                pairs=np.array(diagrams[d], dtype=np.float64)
                if diagrams[d]
                else np.empty((0, 2)),
            )
            for d in diagrams
        }


def _vertex_index(i: int, n: int) -> int:
    # Vertices occupied indices [0, n) in the simplex list; edges follow.
    return i


def _pairwise_distances(points: np.ndarray) -> np.ndarray:
    sq = np.sum(points * points, axis=1, keepdims=True)
    d2 = sq + sq.T - 2.0 * points @ points.T
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)


# --------------------------------------------------------------------------- #
# Persistence image
# --------------------------------------------------------------------------- #


def persistence_image(
    diagram: PersistenceDiagram,
    bins: int,
    sigma: float,
    birth_range: Tuple[float, float],
    persistence_range: Tuple[float, float],
) -> np.ndarray:
    """We converted persistence diagrams into fixed-size persistence images.

    The image was a smoothed, weighted histogram over (birth, persistence)
    coordinates, matching the construction used in the training pipeline.
    Empty diagrams produced all-zero images rather than raising, so that a
    short window did not drop out of the training set.
    """
    image = np.zeros((bins, bins), dtype=np.float32)
    if diagram.n_pairs == 0:
        return image

    finite = diagram.finite()
    if finite.shape[0] == 0:
        return image

    births = finite[:, 0]
    persistences = finite[:, 1] - finite[:, 0]
    weights = persistences.copy()

    b_min, b_max = birth_range
    p_min, p_max = persistence_range
    if b_max <= b_min or p_max <= p_min:
        return image

    b_norm = (births - b_min) / (b_max - b_min)
    p_norm = (persistences - p_min) / (p_max - p_min)
    b_idx = np.clip((b_norm * (bins - 1)).astype(int), 0, bins - 1)
    p_idx = np.clip((p_norm * (bins - 1)).astype(int), 0, bins - 1)

    np.add.at(image, (b_idx, p_idx), weights.astype(np.float32))

    # Gaussian smoothing, applied with a separable kernel.
    radius = max(1, int(round(3.0 * sigma * bins)))
    offsets = np.arange(-radius, radius + 1)
    kernel_1d = np.exp(-(offsets ** 2) / (2.0 * (sigma * bins) ** 2))
    kernel_1d /= kernel_1d.sum()
    padded = np.pad(image, radius, mode="constant")
    tmp = np.apply_along_axis(
        lambda row: np.convolve(row, kernel_1d, mode="valid"), 1, padded
    )
    smoothed = np.apply_along_axis(
        lambda col: np.convolve(col, kernel_1d, mode="valid"), 0, tmp
    )
    return smoothed.astype(np.float32)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


@dataclass
class WindowTensor:
    image_h0: np.ndarray
    image_h1: np.ndarray
    sequence: np.ndarray
    label: int
    flow_id: str
    start_index: int


class TMNDataset(Dataset if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Torch dataset over pre-computed window tensors."""

    def __init__(self, tensors: Sequence[WindowTensor]) -> None:
        self.tensors = list(tensors)

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int):
        item = self.tensors[idx]
        if not _TORCH_AVAILABLE:
            return item
        return (
            torch.from_numpy(item.image_h0).unsqueeze(0),
            torch.from_numpy(item.image_h1).unsqueeze(0),
            torch.from_numpy(item.sequence),
            torch.tensor(item.label, dtype=torch.long),
            item.flow_id,
            item.start_index,
        )


def _birth_persistence_ranges(
    diagrams: Sequence[PersistenceDiagram],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    births = []
    persistences = []
    for d in diagrams:
        finite = d.finite()
        if finite.shape[0] == 0:
            continue
        births.append(finite[:, 0])
        persistences.append(finite[:, 1] - finite[:, 0])
    if not births:
        return (0.0, 1.0), (0.0, 1.0)
    births_all = np.concatenate(births)
    persistences_all = np.concatenate(persistences)
    return (
        (float(births_all.min()), float(births_all.max() + 1e-6)),
        (0.0, float(persistences_all.max() + 1e-6)),
    )


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #

if _TORCH_AVAILABLE:

    class MultiHeadSelfAttention(nn.Module):
        def __init__(self, dim: int, heads: int, dropout: float) -> None:
            super().__init__()
            if dim % heads != 0:
                raise ValueError("embedding dim must be divisible by attention heads")
            self.heads = heads
            self.head_dim = dim // heads
            self.qkv = nn.Linear(dim, dim * 3)
            self.proj = nn.Linear(dim, dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            b, t, d = x.shape
            qkv = self.qkv(x).reshape(b, t, 3, self.heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
            weights = torch.softmax(scores, dim=-1)
            weights = self.dropout(weights)
            out = weights @ v
            out = out.transpose(1, 2).reshape(b, t, d)
            return self.proj(out)

    class TopologicalMemoryNetwork(nn.Module):
        """The TMN architecture described in Section 3.6.2.

        We fused two persistence-image encoders (H0 and H1) with an LSTM
        branch over the raw feature sequence, then applied multi-head
        self-attention over the temporal states before a small classifier
        head. The H0 and H1 branches were kept separate so that ablation
        runs could disable one dimension at a time.
        """

        def __init__(
            self,
            image_bins: int,
            sequence_dim: int,
            hidden: int,
            layers: int,
            heads: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.image_encoder_h0 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(32 * 4 * 4, 64),
                nn.ReLU(),
            )
            self.image_encoder_h1 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(32 * 4 * 4, 64),
                nn.ReLU(),
            )
            self.sequence_proj = nn.Linear(sequence_dim, hidden)
            self.lstm = nn.LSTM(
                input_size=hidden,
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=dropout if layers > 1 else 0.0,
                bidirectional=False,
            )
            self.attention = MultiHeadSelfAttention(hidden, heads, dropout)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden + 128),
                nn.Linear(hidden + 128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 2),
            )

        def forward(self, image_h0, image_h1, sequence):
            h0 = self.image_encoder_h0(image_h0)
            h1 = self.image_encoder_h1(image_h1)
            seq = self.sequence_proj(sequence)
            lstm_out, _ = self.lstm(seq)
            attn = self.attention(lstm_out)
            pooled = attn.mean(dim=1)
            fused = torch.cat([h0, h1, pooled], dim=-1)
            return self.head(fused)

else:  # pragma: no cover - import gate

    class TopologicalMemoryNetwork:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "TopologicalMemoryNetwork requires PyTorch; install torch to "
                "run the TMN training pipeline"
            )


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


@dataclass
class TrainingHistory:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_recall: float
    val_precision: float
    elapsed_s: float


@dataclass
class TrainingReport:
    history: List[TrainingHistory]
    best_epoch: int
    best_val_recall: float
    best_val_precision: float
    best_val_acc: float
    seed: int
    hyperparameters: Dict[str, object]
    total_elapsed_s: float


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _split_indices(
    n: int, seed: int, train_frac: float = 0.7, val_frac: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


def _precision_recall_from_confusion(tp: int, fp: int, fn: int) -> Tuple[float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


class TMNTrainer:
    """Trains the Topological Memory Network.

    We trained with class-balanced cross-entropy because attack windows
    outnumbered benign windows in the campaign archive after the staged
    campaign expansion. Early stopping was applied on validation recall so
    that the checkpoint matched the metric reported in the thesis.
    """

    def __init__(
        self,
        hyperparameters: Optional[Mapping[str, object]] = None,
        device: Optional[str] = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "TMNTrainer requires PyTorch; install torch to run the trainer"
            )
        self.hp: Dict[str, object] = dict(
            hyperparameters or TMN_HYPERPARAMETERS
        )
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model: Optional[TopologicalMemoryNetwork] = None
        self.history: List[TrainingHistory] = []

    def _build_model(self) -> TopologicalMemoryNetwork:
        return TopologicalMemoryNetwork(
            image_bins=int(self.hp["persistence_image_bins"]),
            sequence_dim=int(self.hp["feature_dim"]),
            hidden=int(self.hp["lstm_hidden"]),
            layers=int(self.hp["lstm_layers"]),
            heads=int(self.hp["attention_heads"]),
            dropout=float(self.hp["dropout"]),
        ).to(self.device)

    def fit(
        self,
        tensors: Sequence[WindowTensor],
        seed: Optional[int] = None,
    ) -> TrainingReport:
        if not tensors:
            raise ValueError("cannot train on an empty tensor set")
        seed = int(seed if seed is not None else self.hp["random_seed"])
        _set_deterministic(seed)

        n = len(tensors)
        train_idx, val_idx, _test_idx = _split_indices(n, seed)
        train_ds = TMNDataset([tensors[i] for i in train_idx])
        val_ds = TMNDataset([tensors[i] for i in val_idx])

        batch_size = int(self.hp["batch_size"])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        self.model = self._build_model()
        optimiser = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.hp["learning_rate"]),
            weight_decay=float(self.hp["weight_decay"]),
        )

        # Class-balanced loss.
        labels = np.array([t.label for t in tensors], dtype=np.int64)
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        if n_pos == 0 or n_neg == 0:
            class_weights = torch.tensor([1.0, 1.0], device=self.device)
        else:
            w_pos = n_neg / (n_pos + n_neg)
            w_neg = n_pos / (n_pos + n_neg)
            class_weights = torch.tensor(
                [w_neg * 2.0, w_pos * 2.0], device=self.device
            )
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best = {
            "epoch": -1,
            "recall": -1.0,
            "precision": 0.0,
            "acc": 0.0,
            "state": None,
        }
        patience = int(self.hp["early_stop_patience"])
        epochs = int(self.hp["epochs"])
        since_improvement = 0
        train_start = time.monotonic()

        for epoch in range(epochs):
            epoch_start = time.monotonic()
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for h0, h1, seq, y, _fid, _sidx in train_loader:
                h0 = h0.to(self.device)
                h1 = h1.to(self.device)
                seq = seq.to(self.device)
                y = y.to(self.device)
                optimiser.zero_grad()
                logits = self.model(h0, h1, seq)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float(self.hp["gradient_clip_norm"])
                )
                optimiser.step()
                train_loss += float(loss.detach()) * y.shape[0]
                train_correct += int((logits.argmax(dim=-1) == y).sum().item())
                train_total += int(y.shape[0])

            val_loss, val_acc, val_prec, val_rec = self._evaluate(
                val_loader, criterion
            )
            elapsed = time.monotonic() - epoch_start
            entry = TrainingHistory(
                epoch=epoch,
                train_loss=train_loss / max(1, train_total),
                train_acc=train_correct / max(1, train_total),
                val_loss=val_loss,
                val_acc=val_acc,
                val_recall=val_rec,
                val_precision=val_prec,
                elapsed_s=elapsed,
            )
            self.history.append(entry)
            LOGGER.info(
                "epoch %02d train_loss=%.4f val_loss=%.4f val_acc=%.4f "
                "val_recall=%.4f val_precision=%.4f",
                epoch,
                entry.train_loss,
                entry.val_loss,
                entry.val_acc,
                entry.val_recall,
                entry.val_precision,
            )
            if val_rec > best["recall"]:
                best = {
                    "epoch": epoch,
                    "recall": val_rec,
                    "precision": val_prec,
                    "acc": val_acc,
                    "state": {
                        k: v.detach().cpu().clone()
                        for k, v in self.model.state_dict().items()
                    },
                }
                since_improvement = 0
            else:
                since_improvement += 1
                if since_improvement >= patience:
                    LOGGER.info(
                        "early stop after %d epochs without improvement", patience
                    )
                    break

        if best["state"] is not None:
            self.model.load_state_dict(best["state"])

        total_elapsed = time.monotonic() - train_start
        report = TrainingReport(
            history=list(self.history),
            best_epoch=int(best["epoch"]),
            best_val_recall=float(best["recall"]),
            best_val_precision=float(best["precision"]),
            best_val_acc=float(best["acc"]),
            seed=seed,
            hyperparameters=dict(self.hp),
            total_elapsed_s=total_elapsed,
        )
        return report

    def _evaluate(
        self,
        loader: "DataLoader",
        criterion: "nn.Module",
    ) -> Tuple[float, float, float, float]:
        assert self.model is not None
        self.model.eval()
        total_loss = 0.0
        total = 0
        correct = 0
        tp = fp = fn = 0
        with torch.no_grad():
            for h0, h1, seq, y, _fid, _sidx in loader:
                h0 = h0.to(self.device)
                h1 = h1.to(self.device)
                seq = seq.to(self.device)
                y = y.to(self.device)
                logits = self.model(h0, h1, seq)
                loss = criterion(logits, y)
                total_loss += float(loss) * y.shape[0]
                pred = logits.argmax(dim=-1)
                correct += int((pred == y).sum().item())
                total += int(y.shape[0])
                tp += int(((pred == 1) & (y == 1)).sum().item())
                fp += int(((pred == 1) & (y == 0)).sum().item())
                fn += int(((pred == 0) & (y == 1)).sum().item())
        precision, recall = _precision_recall_from_confusion(tp, fp, fn)
        return (
            total_loss / max(1, total),
            correct / max(1, total),
            precision,
            recall,
        )

    def predict_proba(
        self, tensors: Sequence[WindowTensor]
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model has not been trained")
        loader = DataLoader(TMNDataset(tensors), batch_size=int(self.hp["batch_size"]), shuffle=False)
        self.model.eval()
        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for h0, h1, seq, _y, _fid, _sidx in loader:
                h0 = h0.to(self.device)
                h1 = h1.to(self.device)
                seq = seq.to(self.device)
                logits = self.model(h0, h1, seq)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                outputs.append(probs)
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 2))


# --------------------------------------------------------------------------- #
# Topological evaluation helpers
# --------------------------------------------------------------------------- #


def persistence_stability(
    diagrams_a: Sequence[PersistenceDiagram],
    diagrams_b: Sequence[PersistenceDiagram],
) -> float:
    """We measured stability as the normalised bottleneck proxy between
    the matched lists of finite persistence intervals.

    A true bottleneck distance requires an optimal matching; on the diagram
    sizes encountered in the campaign archive we used a sorted greedy
    matching between intervals in the persistence-pair ordering, which
    produced a stable, monotone proxy. The full bottleneck computation was
    retained in ``evaluation/statistical_significance_harness.py``.
    """
    if not diagrams_a or not diagrams_b:
        return 0.0
    a = np.concatenate([d.finite() for d in diagrams_a if d.n_pairs], axis=0) if any(
        d.n_pairs for d in diagrams_a
    ) else np.empty((0, 2))
    b = np.concatenate([d.finite() for d in diagrams_b if d.n_pairs], axis=0) if any(
        d.n_pairs for d in diagrams_b
    ) else np.empty((0, 2))
    if a.size == 0 and b.size == 0:
        return 0.0
    if a.size == 0 or b.size == 0:
        return 1.0
    a_sorted = a[np.argsort(a[:, 1] - a[:, 0])]
    b_sorted = b[np.argsort(b[:, 1] - b[:, 0])]
    n = max(a_sorted.shape[0], b_sorted.shape[0])
    a_pad = np.vstack(
        [a_sorted, np.zeros((n - a_sorted.shape[0], 2))]
    ) if a_sorted.shape[0] < n else a_sorted
    b_pad = np.vstack(
        [b_sorted, np.zeros((n - b_sorted.shape[0], 2))]
    ) if b_sorted.shape[0] < n else b_sorted
    diffs = np.abs(a_pad - b_pad)
    denom = np.max(np.abs(np.vstack([a, b]))) + 1e-9
    return float(diffs.max() / denom)


def betti_curve(
    diagram: PersistenceDiagram, n_points: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    finite = diagram.finite()
    all_vals = np.concatenate([diagram.pairs[:, 0], diagram.pairs[:, 1:].ravel()])
    all_vals = all_vals[np.isfinite(all_vals)]
    if all_vals.size == 0:
        return np.zeros(n_points, dtype=np.float64), np.zeros(n_points, dtype=np.int64)
    thresholds = np.linspace(float(all_vals.min()), float(all_vals.max()), n_points)
    curve = np.array(
        [diagram.betti(float(t)) for t in thresholds], dtype=np.int64
    )
    return thresholds, curve


# --------------------------------------------------------------------------- #
# Manifest and export
# --------------------------------------------------------------------------- #


def _sha256_of(obj: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _report_manifest(
    report: TrainingReport,
    n_tensors: int,
    image_ranges_h0: Tuple[Tuple[float, float], Tuple[float, float]],
    image_ranges_h1: Tuple[Tuple[float, float], Tuple[float, float]],
) -> Dict[str, object]:
    return {
        "module": "extended_experimentation.training.topological_memory_trainer",
        "seed": report.seed,
        "n_windows": n_tensors,
        "best_epoch": report.best_epoch,
        "best_val_recall": report.best_val_recall,
        "best_val_precision": report.best_val_precision,
        "best_val_acc": report.best_val_acc,
        "total_elapsed_s": report.total_elapsed_s,
        "hyperparameters": report.hyperparameters,
        "persistence_image_ranges": {
            "h0": {
                "birth": image_ranges_h0[0],
                "persistence": image_ranges_h0[1],
            },
            "h1": {
                "birth": image_ranges_h1[0],
                "persistence": image_ranges_h1[1],
            },
        },
        "thesis_reference": THESIS_REFERENCE,
        "history": [
            {
                "epoch": h.epoch,
                "train_loss": h.train_loss,
                "train_acc": h.train_acc,
                "val_loss": h.val_loss,
                "val_acc": h.val_acc,
                "val_recall": h.val_recall,
                "val_precision": h.val_precision,
                "elapsed_s": h.elapsed_s,
            }
            for h in report.history
        ],
    }


def export_training_report(
    out_dir: Path,
    report: TrainingReport,
    n_tensors: int,
    image_ranges_h0: Tuple[Tuple[float, float], Tuple[float, float]],
    image_ranges_h1: Tuple[Tuple[float, float], Tuple[float, float]],
    model_state: Optional["torch.Tensor"] = None,
    diagrams_sample: Optional[Dict[int, PersistenceDiagram]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _report_manifest(
        report, n_tensors, image_ranges_h0, image_ranges_h1
    )
    manifest["manifest_sha256"] = _sha256_of(manifest)
    (out_dir / "training_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    if model_state is not None and _TORCH_AVAILABLE:
        torch.save(model_state, out_dir / "tmn_state_dict.pt")

    if diagrams_sample is not None:
        serialisable = {
            int(dim): {
                "pairs": diagram.finite().tolist(),
            }
            for dim, diagram in diagrams_sample.items()
        }
        (out_dir / "diagram_sample.json").write_text(
            json.dumps(serialisable, indent=2)
        )

    LOGGER.info("training report exported to %s", out_dir)


# --------------------------------------------------------------------------- #
# Pipeline entry point
# --------------------------------------------------------------------------- #


def _load_flows_from_archive(archive_path: Path) -> List[Dict[str, object]]:
    flows: List[Dict[str, object]] = []
    with archive_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                flows.append(json.loads(line))
    return flows


def _load_packet_array(
    npz: np.lib.npyio.NpzFile, flow_id: str, key: str
) -> Optional[np.ndarray]:
    k = f"{flow_id}/{key}"
    if k not in npz.files:
        return None
    return npz[k]


def build_tensor_set(
    campaign_root: Path,
    hyperparameters: Optional[Mapping[str, object]] = None,
) -> Tuple[List[WindowTensor], Dict[str, Dict[int, PersistenceDiagram]]]:
    hp = dict(hyperparameters or TMN_HYPERPARAMETERS)
    flows_path = campaign_root / "flows.jsonl"
    npz_path = campaign_root / "packet_archive.npz"
    if not flows_path.exists() or not npz_path.exists():
        raise FileNotFoundError(
            "campaign archive must contain flows.jsonl and packet_archive.npz"
        )
    window_size = int(hp["window_size_packets"])
    stride = int(hp["window_stride_packets"])
    max_points = int(hp["max_points_per_window"])
    bins = int(hp["persistence_image_bins"])
    sigma = float(hp["persistence_image_sigma"])

    flows = _load_flows_from_archive(flows_path)
    npz = np.load(npz_path, allow_pickle=False)

    # We collected diagrams first so that image ranges could be computed
    # across the entire campaign before any window was rasterised.
    raw_windows: List[Dict[str, object]] = []
    diagrams_by_flow: Dict[str, Dict[int, PersistenceDiagram]] = {}
    filtration = VietorisRipsFiltration(max_points=max_points, max_dim=1)

    for entry in flows:
        flow_id = str(entry["flow_id"])
        label = int(entry["label"])
        protocol = Protocol(str(entry["protocol"]))
        sizes = _load_packet_array(npz, flow_id, "sizes")
        iats = _load_packet_array(npz, flow_id, "iats")
        directions = _load_packet_array(npz, flow_id, "directions")
        if sizes is None or iats is None or directions is None:
            LOGGER.warning("missing packet arrays for flow %s", flow_id)
            continue
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
            raw_windows.append(
                {
                    "window": window,
                    "diagrams": diagrams,
                    "cloud": cloud,
                }
            )
            diagrams_by_flow.setdefault(flow_id, diagrams)

    if not raw_windows:
        return [], {}

    h0_all = [rw["diagrams"][0] for rw in raw_windows]  # type: ignore[index]
    h1_all = [rw["diagrams"][1] for rw in raw_windows]  # type: ignore[index]
    ranges_h0 = _birth_persistence_ranges(h0_all)
    ranges_h1 = _birth_persistence_ranges(h1_all)

    tensors: List[WindowTensor] = []
    for rw in raw_windows:
        window: FeatureWindow = rw["window"]  # type: ignore[assignment]
        diagrams: Dict[int, PersistenceDiagram] = rw["diagrams"]  # type: ignore[assignment]
        img0 = persistence_image(diagrams[0], bins, sigma, *ranges_h0)
        img1 = persistence_image(diagrams[1], bins, sigma, *ranges_h1)
        tensors.append(
            WindowTensor(
                image_h0=img0,
                image_h1=img1,
                sequence=window.features.T.astype(np.float32),
                label=window.label,
                flow_id=window.flow_id,
                start_index=window.start_index,
            )
        )

    # We stashed the ranges on the tensor set through a side-effect on the
    # trainer hyperparameters dict, so the caller can record them in the
    # manifest without recomputing.
    hp["__ranges_h0"] = ranges_h0
    hp["__ranges_h1"] = ranges_h1
    return tensors, diagrams_by_flow


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the MIRAGE-DoH Topological Memory Network on a campaign "
            "archive produced by the extended experimentation suite."
        )
    )
    parser.add_argument(
        "--campaign-root",
        type=Path,
        required=True,
        help="directory containing flows.jsonl and packet_archive.npz",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    hp = dict(TMN_HYPERPARAMETERS)
    tensors, diagrams = build_tensor_set(args.campaign_root, hp)
    if not tensors:
        LOGGER.error("no windows were extracted; aborting")
        return 1

    ranges_h0 = hp.pop("__ranges_h0", ((0.0, 1.0), (0.0, 1.0)))
    ranges_h1 = hp.pop("__ranges_h1", ((0.0, 1.0), (0.0, 1.0)))

    trainer = TMNTrainer(hyperparameters=hp)
    report = trainer.fit(tensors, seed=args.seed)

    model_state = None
    if trainer.model is not None and _TORCH_AVAILABLE:
        model_state = trainer.model.state_dict()

    diagrams_sample = None
    if diagrams:
        first_id = next(iter(diagrams))
        diagrams_sample = diagrams[first_id]

    export_training_report(
        out_dir=args.out,
        report=report,
        n_tensors=len(tensors),
        image_ranges_h0=ranges_h0,
        image_ranges_h1=ranges_h1,
        model_state=model_state,
        diagrams_sample=diagrams_sample,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
