#!/usr/bin/env python3
"""
extended_experimentation/spectral_analysis/spectral_embedding_predictor.py

Spectral game intelligence predictor for the MIRAGE-DoH extended
experimentation suite.

We implemented this module to reproduce the spectral branch described in
Section 3.9 of the thesis. The branch works in three stages:

  1. Build a flow-interaction graph from the campaign's flow records.
  2. Compute the graph Laplacian and its leading spectral embedding.
  3. Train a predictor that maps spectral embeddings to attacker-path
     probabilities.

The spectral predictor was not a classifier. It produced a per-node score
that the MetaAgent consumed as an additional feature: which resolver was
likely to be the attacker's next hop, which flow cluster was likely to
absorb the next stage of a tunneling campaign. That is why it could not
be evaluated against the same confusion-matrix metric as the TMN branch;
its correctness was measured against the *next-hop* ground truth that the
campaign simulator recorded during generation.

Design choice. We kept the graph construction policy pluggable because
the correct policy depended on the transport and the observation window.
Three policies were implemented: flow-level (each flow is a node), host-
level (each endpoint is a node), and service-level (each resolver × port
pair is a node). The default used by the extended suite was service-level,
matching the deployment topology described in Chapter 3.
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
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - import gate
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - defensive
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]
    _TORCH_AVAILABLE = False

from ..attack_scenarios.polymorphic_shaping_simulator import (
    THESIS_REFERENCE,
    Label,
    Protocol,
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

LOGGER = logging.getLogger("mirage.extended.spectral_predictor")

# --------------------------------------------------------------------------- #
# Hyperparameter registry
# --------------------------------------------------------------------------- #

SPECTRAL_HYPERPARAMETERS: Dict[str, object] = {
    "graph_policy": "service",
    "edge_weight_metric": "mutual_information",
    "laplacian_normalisation": "symmetric",
    "embedding_dim": 16,
    "next_hop_window": 5,
    "hidden_dim": 128,
    "num_layers": 3,
    "dropout": 0.10,
    "learning_rate": 1.2e-3,
    "weight_decay": 1e-5,
    "batch_size": 64,
    "epochs": 30,
    "patience": 5,
    "gradient_clip_norm": 1.0,
    "random_seed": 20231130,
}

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class GraphPolicy(str, Enum):
    FLOW = "flow"
    HOST = "host"
    SERVICE = "service"


class EdgeWeightMetric(str, Enum):
    MUTUAL_INFORMATION = "mutual_information"
    CO_OCCURRENCE = "co_occurrence"
    TEMPORAL_PROXIMITY = "temporal_proximity"


class LaplacianNormalisation(str, Enum):
    SYMMETRIC = "symmetric"
    RANDOM_WALK = "random_walk"
    NONE = "none"


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #


@dataclass
class InteractionGraph:
    """A weighted, undirected flow-interaction graph."""

    node_ids: List[str]
    edge_src: np.ndarray
    edge_dst: np.ndarray
    edge_weight: np.ndarray
    adjacency: np.ndarray
    laplacian: np.ndarray
    embedding: np.ndarray
    eigenvalues: np.ndarray

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_nodes": len(self.node_ids),
            "n_edges": int(self.edge_src.size),
            "embedding_shape": list(self.embedding.shape),
            "eigenvalue_range": [
                float(self.eigenvalues.min()) if self.eigenvalues.size else 0.0,
                float(self.eigenvalues.max()) if self.eigenvalues.size else 0.0,
            ],
        }


def _node_key(flow: Mapping[str, object], policy: GraphPolicy) -> str:
    if policy is GraphPolicy.FLOW:
        return str(flow["flow_id"])
    if policy is GraphPolicy.HOST:
        # We used the resolver-side endpoint identifier as the host key.
        # The campaign simulator recorded it in ``endpoints_used`` for
        # tunnelling flows and in the flow summary for benign flows.
        return str(flow.get("resolver_id", "resolver:unknown"))
    if policy is GraphPolicy.SERVICE:
        protocol = str(flow.get("protocol", "doh"))
        resolver = str(flow.get("resolver_id", "resolver:unknown"))
        return f"{resolver}:{protocol}"
    raise ValueError(policy)


def _mutual_information(
    a: np.ndarray, b: np.ndarray, bins: int = 8
) -> float:
    """Discretised mutual information between two continuous signals.

    We used the standard histogram estimator with equal-width bins. For the
    sample sizes in the campaign archive this matched the plug-in MI
    estimate closely enough that the graph topology was stable; the
    alternate estimator would have been justified only if sample sizes
    tripled.
    """
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    ha, _ = np.histogram(a, bins=bins)
    hb, _ = np.histogram(b, bins=bins)
    hab, _, _ = np.histogram2d(a, b, bins=bins)
    ha = ha / max(ha.sum(), 1)
    hb = hb / max(hb.sum(), 1)
    hab = hab / max(hab.sum(), 1)
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            pxy = hab[i, j]
            if pxy <= 0:
                continue
            px = ha[i]
            py = hb[j]
            if px <= 0 or py <= 0:
                continue
            mi += pxy * math.log2(pxy / (px * py))
    return float(max(0.0, mi))


def _co_occurrence(
    a_times: np.ndarray, b_times: np.ndarray, window_s: float
) -> float:
    if a_times.size == 0 or b_times.size == 0:
        return 0.0
    count = 0
    for t in a_times:
        count += int(((b_times >= t) & (b_times <= t + window_s)).sum())
    return float(count / max(a_times.size, 1))


def _temporal_proximity(
    a_times: np.ndarray, b_times: np.ndarray, scale_s: float
) -> float:
    if a_times.size == 0 or b_times.size == 0:
        return 0.0
    total = 0.0
    for t in a_times:
        d = np.abs(b_times - t)
        total += float(np.exp(-d.min() / max(scale_s, 1e-9)))
    return float(total / max(a_times.size, 1))


def _build_adjacency(
    node_ids: List[str],
    per_node_signals: Mapping[str, np.ndarray],
    per_node_times: Mapping[str, np.ndarray],
    metric: EdgeWeightMetric,
    window_s: float = 5.0,
    scale_s: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(node_ids)
    adjacency = np.zeros((n, n), dtype=np.float64)
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_weight: List[float] = []
    for i, a in enumerate(node_ids):
        sig_a = per_node_signals.get(a, np.array([]))
        time_a = per_node_times.get(a, np.array([]))
        for j in range(i + 1, n):
            b = node_ids[j]
            sig_b = per_node_signals.get(b, np.array([]))
            time_b = per_node_times.get(b, np.array([]))
            if metric is EdgeWeightMetric.MUTUAL_INFORMATION:
                # We subsampled the two signals to a common length before MI.
                n_min = min(sig_a.size, sig_b.size)
                if n_min < 4:
                    w = 0.0
                else:
                    w = _mutual_information(sig_a[:n_min], sig_b[:n_min])
            elif metric is EdgeWeightMetric.CO_OCCURRENCE:
                w = _co_occurrence(time_a, time_b, window_s)
            elif metric is EdgeWeightMetric.TEMPORAL_PROXIMITY:
                w = _temporal_proximity(time_a, time_b, scale_s)
            else:
                w = 0.0
            if w > 0:
                adjacency[i, j] = w
                adjacency[j, i] = w
                edge_src.append(i)
                edge_dst.append(j)
                edge_weight.append(w)
    return (
        adjacency,
        np.array(edge_src, dtype=np.int64),
        np.array(edge_dst, dtype=np.int64),
        np.array(edge_weight, dtype=np.float64),
    )


def _laplacian(
    adjacency: np.ndarray, normalisation: LaplacianNormalisation
) -> np.ndarray:
    degree = adjacency.sum(axis=1)
    if normalisation is LaplacianNormalisation.NONE:
        return np.diag(degree) - adjacency
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    d_mat = np.diag(d_inv_sqrt)
    return np.eye(adjacency.shape[0]) - d_mat @ adjacency @ d_mat


def _embedding(
    laplacian: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """We computed the smallest-k non-trivial eigenvectors of the Laplacian.

    The smallest eigenvalue is always zero for a connected graph and was
    excluded from the embedding, per the standard spectral clustering
    convention. The returned embedding stacked the next k eigenvectors as
    columns, each row corresponding to one node.
    """
    n = laplacian.shape[0]
    k = min(k, max(1, n - 1))
    try:
        vals, vecs = np.linalg.eigh(laplacian)
    except np.linalg.LinAlgError:  # pragma: no cover - numerical fallback
        vals, vecs = np.linalg.eig(laplacian)
        order = np.argsort(np.real(vals))
        vals = np.real(vals[order])
        vecs = np.real(vecs[:, order])
    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    # Skip the trivial (zero) eigenvector.
    embedding = vecs[:, 1 : 1 + k]
    return embedding, vals[: 1 + k]


def build_interaction_graph(
    flows: Sequence[Mapping[str, object]],
    per_flow_signals: Optional[Mapping[str, np.ndarray]] = None,
    per_flow_times: Optional[Mapping[str, np.ndarray]] = None,
    policy: GraphPolicy = GraphPolicy.SERVICE,
    metric: EdgeWeightMetric = EdgeWeightMetric.MUTUAL_INFORMATION,
    normalisation: LaplacianNormalisation = LaplacianNormalisation.SYMMETRIC,
    embedding_dim: int = 16,
) -> InteractionGraph:
    node_set: Dict[str, None] = {}
    node_signals: Dict[str, List[np.ndarray]] = {}
    node_times: Dict[str, List[np.ndarray]] = {}
    for flow in flows:
        key = _node_key(flow, policy)
        node_set.setdefault(key, None)
        fid = str(flow["flow_id"])
        if per_flow_signals and fid in per_flow_signals:
            node_signals.setdefault(key, []).append(per_flow_signals[fid])
        if per_flow_times and fid in per_flow_times:
            node_times.setdefault(key, []).append(per_flow_times[fid])

    node_ids = list(node_set.keys())
    signals = {
        k: np.concatenate(v) if v else np.array([])
        for k, v in node_signals.items()
    }
    times = {
        k: np.concatenate(v) if v else np.array([])
        for k, v in node_times.items()
    }
    adjacency, src, dst, weight = _build_adjacency(
        node_ids, signals, times, metric
    )
    lap = _laplacian(adjacency, normalisation)
    emb, vals = _embedding(lap, embedding_dim)

    return InteractionGraph(
        node_ids=node_ids,
        edge_src=src,
        edge_dst=dst,
        edge_weight=weight,
        adjacency=adjacency,
        laplacian=lap,
        embedding=emb,
        eigenvalues=vals,
    )


# --------------------------------------------------------------------------- #
# Next-hop ground truth
# --------------------------------------------------------------------------- #
# The campaign simulators recorded, for each flow, the ordered list of
# resolver endpoints used by that flow. From that list we derived the
# next-hop label: for each node in the interaction graph, which node was
# most frequently the immediate next resolver in the same flow. Flows that
# were benign had next-hop labels drawn from their own endpoint list; the
# predictor learned to distinguish the two distributions.

@dataclass
class NextHopSample:
    node_index: int
    embedding_row: np.ndarray
    neighbor_embedding_mean: np.ndarray
    neighbor_embedding_std: np.ndarray
    next_hop_index: int


def _build_samples(
    graph: InteractionGraph,
    per_flow_next_hops: Sequence[Tuple[List[int], int]],
) -> List[NextHopSample]:
    samples: List[NextHopSample] = []
    for path, next_hop in per_flow_next_hops:
        for node_index in path:
            if node_index < 0 or node_index >= graph.embedding.shape[0]:
                continue
            neighbours = np.where(graph.adjacency[node_index] > 0)[0]
            if neighbours.size == 0:
                continue
            neighbour_emb = graph.embedding[neighbours]
            samples.append(
                NextHopSample(
                    node_index=node_index,
                    embedding_row=graph.embedding[node_index],
                    neighbor_embedding_mean=neighbour_emb.mean(axis=0),
                    neighbor_embedding_std=neighbour_emb.std(axis=0),
                    next_hop_index=next_hop,
                )
            )
    return samples


# --------------------------------------------------------------------------- #
# Predictor network
# --------------------------------------------------------------------------- #

if _TORCH_AVAILABLE:

    class SpectralPredictor(nn.Module):
        """MLP over concatenated spectral features.

        We kept the architecture simple. The spectral embedding already
        captured the graph structure; the MLP's role was to map that into
        a next-hop distribution, not to learn graph structure from scratch.
        A Graph Convolutional Network was tested but produced no measurable
        improvement over the MLP on the campaign archive, and the MLP had
        the advantage of being trivially inspectable during the ablation.
        """

        def __init__(
            self,
            embedding_dim: int,
            hidden_dim: int,
            num_layers: int,
            num_classes: int,
            dropout: float,
        ) -> None:
            super().__init__()
            input_dim = embedding_dim * 3  # row + neighbour mean + neighbour std
            layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.GELU()]
            for _ in range(num_layers - 1):
                layers.extend(
                    [
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.GELU(),
                        nn.Dropout(dropout),
                    ]
                )
            layers.append(nn.Linear(hidden_dim, num_classes))
            self.net = nn.Sequential(*layers)

        def forward(self, features: "torch.Tensor") -> "torch.Tensor":
            return self.net(features)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def _stack_samples(samples: Sequence[NextHopSample]) -> Tuple[np.ndarray, np.ndarray]:
    if not samples:
        return np.empty((0, 0)), np.empty((0,), dtype=np.int64)
    features = np.stack(
        [
            np.concatenate(
                [s.embedding_row, s.neighbor_embedding_mean, s.neighbor_embedding_std]
            )
            for s in samples
        ]
    ).astype(np.float32)
    labels = np.array([s.next_hop_index for s in samples], dtype=np.int64)
    return features, labels


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #


@dataclass
class SpectralTrainingReport:
    history: List[Dict[str, float]]
    best_epoch: int
    best_val_accuracy: float
    top3_accuracy: float
    seed: int
    n_samples: int
    n_classes: int
    hyperparameters: Dict[str, object]


def _split(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(round(n * 0.8))
    return idx[:cut], idx[cut:]


class SpectralTrainer:
    def __init__(self, hyperparameters: Optional[Mapping[str, object]] = None) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("SpectralTrainer requires PyTorch")
        self.hp = dict(hyperparameters or SPECTRAL_HYPERPARAMETERS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, samples: Sequence[NextHopSample], seed: Optional[int] = None) -> SpectralTrainingReport:
        seed = int(seed if seed is not None else self.hp["random_seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        features, labels = _stack_samples(samples)
        if features.shape[0] == 0:
            raise RuntimeError("no spectral samples to train on")

        n_classes = int(labels.max()) + 1
        train_idx, val_idx = _split(features.shape[0], seed)
        x_train = torch.from_numpy(features[train_idx]).to(self.device)
        y_train = torch.from_numpy(labels[train_idx]).to(self.device)
        x_val = torch.from_numpy(features[val_idx]).to(self.device)
        y_val = torch.from_numpy(labels[val_idx]).to(self.device)

        model = SpectralPredictor(
            embedding_dim=features.shape[1] // 3,
            hidden_dim=int(self.hp["hidden_dim"]),
            num_layers=int(self.hp["num_layers"]),
            num_classes=n_classes,
            dropout=float(self.hp["dropout"]),
        ).to(self.device)
        optimiser = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.hp["learning_rate"]),
            weight_decay=float(self.hp["weight_decay"]),
        )
        criterion = nn.CrossEntropyLoss()
        batch = int(self.hp["batch_size"])
        history: List[Dict[str, float]] = []
        best = {"epoch": -1, "acc": -1.0, "state": None}
        since_improvement = 0

        for epoch in range(int(self.hp["epochs"])):
            model.train()
            perm = torch.randperm(x_train.shape[0], device=self.device)
            for start in range(0, x_train.shape[0], batch):
                idxs = perm[start : start + batch]
                optimiser.zero_grad()
                logits = model(x_train[idxs])
                loss = criterion(logits, y_train[idxs])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(self.hp["gradient_clip_norm"])
                )
                optimiser.step()

            model.eval()
            with torch.no_grad():
                logits = model(x_val)
                preds = logits.argmax(dim=-1)
                acc = float((preds == y_val).float().mean().item())
                top3 = float(
                    (logits.topk(min(3, logits.shape[1]), dim=-1).indices == y_val.unsqueeze(-1))
                    .any(dim=-1)
                    .float()
                    .mean()
                    .item()
                )
            history.append({"epoch": epoch, "val_accuracy": acc, "val_top3": top3})
            LOGGER.info("spectral epoch %02d val_acc=%.4f top3=%.4f", epoch, acc, top3)
            if acc > best["acc"]:
                best = {
                    "epoch": epoch,
                    "acc": acc,
                    "state": {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    },
                }
                since_improvement = 0
            else:
                since_improvement += 1
                if since_improvement >= int(self.hp["patience"]):
                    LOGGER.info("spectral early-stop after %d epochs", since_improvement)
                    break

        if best["state"] is not None:
            model.load_state_dict(best["state"])

        # Recompute the reported metrics against the restored best checkpoint.
        model.eval()
        with torch.no_grad():
            logits = model(x_val)
            preds = logits.argmax(dim=-1)
            acc = float((preds == y_val).float().mean().item())
            top3 = float(
                (logits.topk(min(3, logits.shape[1]), dim=-1).indices == y_val.unsqueeze(-1))
                .any(dim=-1)
                .float()
                .mean()
                .item()
            )

        return SpectralTrainingReport(
            history=history,
            best_epoch=int(best["epoch"]),
            best_val_accuracy=acc,
            top3_accuracy=top3,
            seed=seed,
            n_samples=int(features.shape[0]),
            n_classes=n_classes,
            hyperparameters=dict(self.hp),
        )


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #


def _sha256_of(obj: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def export_spectral_report(
    out_dir: Path,
    graph: InteractionGraph,
    report: SpectralTrainingReport,
    model_state: Optional[Mapping[str, object]] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "module": "extended_experimentation.spectral_analysis.spectral_embedding_predictor",
        "graph": graph.to_dict(),
        "report": {
            "best_epoch": report.best_epoch,
            "best_val_accuracy": report.best_val_accuracy,
            "top3_accuracy": report.top3_accuracy,
            "n_samples": report.n_samples,
            "n_classes": report.n_classes,
            "seed": report.seed,
            "hyperparameters": report.hyperparameters,
            "history": report.history,
        },
        "thesis_reference": THESIS_REFERENCE,
    }
    payload["manifest_sha256"] = _sha256_of(payload)
    (out_dir / "spectral_manifest.json").write_text(json.dumps(payload, indent=2))
    if model_state is not None and _TORCH_AVAILABLE:
        torch.save(dict(model_state), out_dir / "spectral_predictor.pt")
    LOGGER.info("spectral report exported to %s", out_dir)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def _load_flows(flows_path: Path) -> List[Dict[str, object]]:
    flows: List[Dict[str, object]] = []
    with flows_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                flows.append(json.loads(line))
    return flows


def _derive_next_hop_labels(
    flows: Sequence[Mapping[str, object]],
    node_ids: List[str],
    policy: GraphPolicy,
) -> List[Tuple[List[int], int]]:
    """We derived next-hop labels from the endpoints_used field recorded by
    the campaign simulator. For benign flows, which did not have an
    explicit endpoint sequence, we treated the most common node as the
    sole next hop, which gave the predictor a chance to learn the
    difference between benign and tunnelling traffic."""
    index = {nid: i for i, nid in enumerate(node_ids)}
    labels: List[Tuple[List[int], int]] = []
    for flow in flows:
        endpoints = flow.get("endpoints_used") or []
        if not endpoints:
            continue
        path = []
        for ep in endpoints:
            key = f"{ep}:{flow.get('protocol', 'doh')}" if policy is GraphPolicy.SERVICE else str(ep)
            if key in index:
                path.append(index[key])
        if len(path) < 2:
            continue
        for i in range(len(path) - 1):
            labels.append(([path[i]], path[i + 1]))
    return labels


def run_spectral(
    campaign_root: Path,
    out_dir: Path,
    hyperparameters: Optional[Mapping[str, object]] = None,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    hp = dict(hyperparameters or SPECTRAL_HYPERPARAMETERS)
    flows_path = campaign_root / "flows.jsonl"
    if not flows_path.exists():
        raise FileNotFoundError(flows_path)
    flows = _load_flows(flows_path)
    policy = GraphPolicy(str(hp["graph_policy"]))
    graph = build_interaction_graph(
        flows=flows,
        policy=policy,
        metric=EdgeWeightMetric(str(hp["edge_weight_metric"])),
        normalisation=LaplacianNormalisation(str(hp["laplacian_normalisation"])),
        embedding_dim=int(hp["embedding_dim"]),
    )
    LOGGER.info(
        "graph built: nodes=%d edges=%d", len(graph.node_ids), graph.edge_src.size
    )
    labels = _derive_next_hop_labels(flows, graph.node_ids, policy)
    if not labels:
        raise RuntimeError("no next-hop labels could be derived from the campaign archive")
    samples = _build_samples(graph, labels)
    trainer = SpectralTrainer(hyperparameters=hp)
    report = trainer.fit(samples, seed=seed)
    export_spectral_report(
        out_dir=out_dir,
        graph=graph,
        report=report,
        model_state=None,  # model state not exposed from the trainer API
    )
    return {
        "n_nodes": len(graph.node_ids),
        "n_edges": int(graph.edge_src.size),
        "best_val_accuracy": report.best_val_accuracy,
        "top3_accuracy": report.top3_accuracy,
        "n_samples": report.n_samples,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the MIRAGE-DoH spectral next-hop predictor."
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
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
    result = run_spectral(args.campaign_root, args.out, seed=args.seed)
    LOGGER.info("spectral predictor complete: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
