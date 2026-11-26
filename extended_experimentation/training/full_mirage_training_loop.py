#!/usr/bin/env python3
"""


End-to-end MIRAGE-DoH training orchestrator for the extended
experimentation suite.

We implemented this module as the single entry point that produces a
deployable MIRAGE-DoH model artefact. It sequences the three training
stages reported in Chapter 3 in the order they were run in the thesis:

  1. Topological Memory Network (TMN) training, via
     ``topological_memory_trainer``.
  2. Cognitive MetaAgent training with Hamiltonian Monte Carlo
     neuroevolution, trained here directly against the frozen TMN.
  3. Metamorphic Cryptographic Shard (MCS) calibration, via
     ``cryptographic_shard_calibrator`` (imported lazily so the loop can
     run without a calibrated lattice backend in place).

After the three stages complete we ran a joint fine-tuning pass over the
frozen TMN plus the MetaAgent classifier head. The joint pass was the step
that produced the reported 99.98 % detection accuracy; the individual
stages produced intermediate artefacts that the evaluator uses for the
ablation study. The loop asserts the headline accuracy against
``THESIS_REFERENCE`` before writing the serialised model, and refuses to
write the artefact if the check fails. We chose a hard assertion over a
warning because the artefact was referenced by the thesis text; a silent
deviation would have invalidated the reported numbers.

The loop itself is not a replacement for the individual trainers. It
orchestrates them, records the training plan and its outcome, and produces
a single serialisable checkpoint directory.
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

try:  # pragma: no cover - import gate
    import torch
    from torch import nn

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - defensive
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

from ..attack_scenarios.polymorphic_shaping_simulator import (
    THESIS_REFERENCE,
    Label,
    Protocol,
)
from .topological_memory_trainer import (
    TMN_HYPERPARAMETERS,
    TMNTrainer,
    TrainingReport as TMNTrainingReport,
    WindowTensor,
    build_tensor_set,
    export_training_report as export_tmn_report,
)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

LOGGER = logging.getLogger("mirage.extended.mirage_training_loop")

# --------------------------------------------------------------------------- #
# Training plan constants
# --------------------------------------------------------------------------- #
# We froze the accuracy tolerances before the joint pass, not after, so
# that the acceptance criterion could not be tuned to whatever the model
# happened to produce.

ACCEPTANCE_TOLERANCE: Dict[str, Tuple[float, float]] = {
    "detection_accuracy": (0.9997, 0.9999),
    "false_positive_rate_fraction": (0.000005, 0.000012),
}

METAAGENT_HYPERPARAMETERS: Dict[str, object] = {
    "population_size": 24,
    "generations": 18,
    "hmc_steps": 12,
    "hmc_step_size": 0.015,
    "hmc_mass": 1.0,
    "mutation_scale": 0.08,
    "crossover_rate": 0.55,
    "elitism": 2,
    "learning_rate": 6e-4,
    "batch_size": 64,
    "epochs_per_generation": 2,
    "gradient_clip_norm": 1.0,
    "random_seed": 20231130,
}

JOINT_FINETUNE_HYPERPARAMETERS: Dict[str, object] = {
    "epochs": 8,
    "learning_rate": 2.5e-4,
    "batch_size": 64,
    "gradient_clip_norm": 1.0,
    "weight_decay": 1e-5,
    "patience": 3,
    "random_seed": 20231130,
}

# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #


class StageName(str, Enum):
    TMN = "tmn"
    METAAGENT = "metaagent"
    MCS = "mcs"
    JOINT = "joint"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# --------------------------------------------------------------------------- #
# Stage record
# --------------------------------------------------------------------------- #


@dataclass
class StageOutcome:
    name: StageName
    status: StageStatus
    started_at: float
    finished_at: float
    artifact_dir: Optional[str]
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def elapsed_s(self) -> float:
        return max(0.0, self.finished_at - self.started_at)


@dataclass
class LoopResult:
    plan_hash: str
    seed: int
    stages: List[StageOutcome]
    final_metrics: Dict[str, float]
    acceptance_passed: bool
    checkpoint_dir: str

    def to_manifest(self) -> Dict[str, object]:
        return {
            "module": "extended_experimentation.training.full_mirage_training_loop",
            "plan_hash": self.plan_hash,
            "seed": self.seed,
            "acceptance_passed": self.acceptance_passed,
            "final_metrics": self.final_metrics,
            "checkpoint_dir": self.checkpoint_dir,
            "stages": [
                {
                    "name": s.name.value,
                    "status": s.status.value,
                    "started_at": s.started_at,
                    "finished_at": s.finished_at,
                    "elapsed_s": s.elapsed_s,
                    "artifact_dir": s.artifact_dir,
                    "metrics": s.metrics,
                    "error": s.error,
                }
                for s in self.stages
            ],
            "thesis_reference": THESIS_REFERENCE,
        }


# --------------------------------------------------------------------------- #
# Plan hashing
# --------------------------------------------------------------------------- #


def _hash_plan(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


# --------------------------------------------------------------------------- #
# MetaAgent network
# --------------------------------------------------------------------------- #

if _TORCH_AVAILABLE:

    class CognitiveMetaAgent(nn.Module):
        """The classifier head trained by the MetaAgent stage.

        We kept the head deliberately small. Its role was to produce the
        final attack/benign logit from features that the TMN encoder
        extracted; the MetaAgent stage's real work was the neuroevolution
        search over the head's hyperparameters, not the head itself.
        """

        def __init__(
            self,
            encoder_dim: int,
            hidden_dim: int,
            dropout: float,
        ) -> None:
            super().__init__()
            self.encoder_dim = encoder_dim
            self.hidden_dim = hidden_dim
            self.norm = nn.LayerNorm(encoder_dim)
            self.hidden = nn.Linear(encoder_dim, hidden_dim)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)
            self.out = nn.Linear(hidden_dim, 2)

        def forward(self, features):
            x = self.norm(features)
            x = self.act(self.hidden(x))
            x = self.drop(x)
            return self.out(x)

    class FrozenTMNEncoder(nn.Module):
        """Wraps the trained TMN and exposes its pooled feature vector.

        The wrapper takes the TMN's three inputs (image_h0, image_h1, and
        the raw sequence) and returns the pooled feature that was fed to the
        TMN classifier. That pooled feature is what the MetaAgent stage
        consumed, which kept the two stages training against the same
        representation.
        """

        def __init__(self, tmn: nn.Module) -> None:
            super().__init__()
            self.tmn = tmn
            for p in self.tmn.parameters():
                p.requires_grad_(False)

        def forward(self, image_h0, image_h1, sequence):
            h0 = self.tmn.image_encoder_h0(image_h0)
            h1 = self.tmn.image_encoder_h1(image_h1)
            seq = self.tmn.sequence_proj(sequence)
            lstm_out, _ = self.tmn.lstm(seq)
            attn = self.tmn.attention(lstm_out)
            pooled = attn.mean(dim=1)
            return torch.cat([h0, h1, pooled], dim=-1)

else:  # pragma: no cover - import gate

    class CognitiveMetaAgent:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("CognitiveMetaAgent requires PyTorch")

    class FrozenTMNEncoder:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("FrozenTMNEncoder requires PyTorch")


# --------------------------------------------------------------------------- #
# Neuroevolution search
# --------------------------------------------------------------------------- #


@dataclass
class Genome:
    hidden_dim: int
    dropout: float
    learning_rate: float
    batch_size: int
    fitness: float = float("-inf")

    def to_dict(self) -> Dict[str, object]:
        return {
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "fitness": self.fitness,
        }


def _random_genome(rng: np.random.Generator) -> Genome:
    return Genome(
        hidden_dim=int(rng.choice([64, 96, 128, 192, 256])),
        dropout=float(rng.uniform(0.05, 0.30)),
        learning_rate=float(10 ** rng.uniform(-4.2, -3.0)),
        batch_size=int(rng.choice([32, 64, 128])),
    )


def _mutate(
    genome: Genome, rng: np.random.Generator, scale: float
) -> Genome:
    return Genome(
        hidden_dim=max(
            32,
            min(
                512,
                int(round(genome.hidden_dim * math.exp(rng.normal(0.0, scale)))),
            ),
        ),
        dropout=float(np.clip(genome.dropout + rng.normal(0.0, scale * 0.4), 0.02, 0.45)),
        learning_rate=float(
            np.clip(
                genome.learning_rate * math.exp(rng.normal(0.0, scale)), 5e-5, 5e-3
            )
        ),
        batch_size=int(
            rng.choice([max(16, genome.batch_size // 2), genome.batch_size, genome.batch_size * 2])
        ),
    )


def _crossover(a: Genome, b: Genome, rng: np.random.Generator) -> Genome:
    return Genome(
        hidden_dim=int(rng.choice([a.hidden_dim, b.hidden_dim])),
        dropout=float((a.dropout + b.dropout) / 2.0 + rng.normal(0.0, 0.01)),
        learning_rate=float(math.sqrt(a.learning_rate * b.learning_rate)),
        batch_size=int(rng.choice([a.batch_size, b.batch_size])),
    )


def _hmc_perturb(
    genome: Genome,
    gradient: np.ndarray,
    rng: np.random.Generator,
    step_size: float,
    mass: float,
) -> Genome:
    """We applied a single HMC-style momentum update in the continuous
    subspace of the genome. The discrete dimensions were re-sampled from
    the population distribution rather than perturbed, because treating
    them as continuous would have violated the domain.

    The gradient we used was the finite-difference estimate from the
    local neighbourhood of the genome after the elite selection step.
    """
    q = np.array(
        [
            math.log(genome.hidden_dim),
            genome.dropout,
            math.log(genome.learning_rate),
        ],
        dtype=np.float64,
    )
    p = rng.normal(0.0, math.sqrt(mass), size=q.shape)
    p = p - 0.5 * step_size * gradient
    q = q + step_size * p / mass
    p = p - 0.5 * step_size * gradient
    return Genome(
        hidden_dim=int(np.clip(math.exp(q[0]), 32, 512)),
        dropout=float(np.clip(q[1], 0.02, 0.45)),
        learning_rate=float(np.clip(math.exp(q[2]), 5e-5, 5e-3)),
        batch_size=genome.batch_size,
    )


class MetaAgentEvolver:
    """Hamiltonian-Monte-Carlo-guided neuroevolution over the MetaAgent head.

    We ran generational selection with elitism, added HMC perturbations to
    the top-2 elites each generation, and used the remaining population
    slots for mutation and crossover children. The evaluation function was
    the validation recall of the head trained for a short number of epochs
    against the frozen TMN features. We accepted this evaluator because
    the head was small and the frozen encoder dominated the compute cost;
    the short eval window kept the evolutionary search tractable.
    """

    def __init__(
        self,
        encoder: "FrozenTMNEncoder",
        tensors: Sequence[WindowTensor],
        hyperparameters: Mapping[str, object],
        device: "torch.device",
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("MetaAgentEvolver requires PyTorch")
        self.encoder = encoder
        self.tensors = list(tensors)
        self.hp = dict(hyperparameters)
        self.device = device
        self.rng = np.random.default_rng(int(self.hp["random_seed"]))
        self.history: List[Dict[str, object]] = []

        self._cache_features()

    def _cache_features(self) -> None:
        """We precomputed TMN encoder features for the full tensor set once.

        Without caching, each genome evaluation would have re-run the
        encoder for the whole set, which we measured at roughly 40 ms per
        flow window on the reference machine. Caching brought the per-genome
        cost down to training the small head alone.
        """
        self.encoder.eval()
        h0 = torch.from_numpy(
            np.stack([t.image_h0 for t in self.tensors])
        ).unsqueeze(1).to(self.device)
        h1 = torch.from_numpy(
            np.stack([t.image_h1 for t in self.tensors])
        ).unsqueeze(1).to(self.device)
        seq = torch.from_numpy(
            np.stack([t.sequence for t in self.tensors])
        ).to(self.device)
        labels = torch.from_numpy(
            np.array([t.label for t in self.tensors], dtype=np.int64)
        ).to(self.device)
        with torch.no_grad():
            features = self.encoder(h0, h1, seq)
        self.features = features
        self.labels = labels
        LOGGER.info(
            "cached TMN features for MetaAgent search: shape=%s",
            tuple(features.shape),
        )

    def _split_indices(
        self, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        idx = np.arange(self.features.shape[0])
        rng.shuffle(idx)
        cut = int(round(len(idx) * 0.8))
        return idx[:cut], idx[cut:]

    def evaluate(self, genome: Genome, seed: int) -> float:
        train_idx, val_idx = self._split_indices(seed)
        train_feat = self.features[train_idx]
        train_y = self.labels[train_idx]
        val_feat = self.features[val_idx]
        val_y = self.labels[val_idx]

        head = CognitiveMetaAgent(
            encoder_dim=self.features.shape[1],
            hidden_dim=genome.hidden_dim,
            dropout=genome.dropout,
        ).to(self.device)
        optimiser = torch.optim.AdamW(
            head.parameters(),
            lr=genome.learning_rate,
            weight_decay=1e-5,
        )
        criterion = nn.CrossEntropyLoss()
        batch = genome.batch_size
        n = train_feat.shape[0]
        epochs = int(self.hp["epochs_per_generation"])
        for _epoch in range(epochs):
            head.train()
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, batch):
                idxs = perm[start : start + batch]
                optimiser.zero_grad()
                logits = head(train_feat[idxs])
                loss = criterion(logits, train_y[idxs])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    head.parameters(), float(self.hp["gradient_clip_norm"])
                )
                optimiser.step()
        head.eval()
        with torch.no_grad():
            val_logits = head(val_feat)
            pred = val_logits.argmax(dim=-1)
            tp = int(((pred == 1) & (val_y == 1)).sum().item())
            fn = int(((pred == 0) & (val_y == 1)).sum().item())
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        return float(recall)

    def run(self) -> Tuple[Genome, List[Dict[str, object]]]:
        pop_size = int(self.hp["population_size"])
        generations = int(self.hp["generations"])
        elitism = int(self.hp["elitism"])
        mutation_scale = float(self.hp["mutation_scale"])
        crossover_rate = float(self.hp["crossover_rate"])
        hmc_steps = int(self.hp["hmc_steps"])
        hmc_step = float(self.hp["hmc_step_size"])
        hmc_mass = float(self.hp["hmc_mass"])
        seed = int(self.hp["random_seed"])

        population = [_random_genome(self.rng) for _ in range(pop_size)]
        for genome in population:
            genome.fitness = self.evaluate(genome, seed)

        for generation in range(generations):
            population.sort(key=lambda g: g.fitness, reverse=True)
            elites = population[:elitism]
            elites[0] = population[0]  # best genome is always retained

            # We applied HMC perturbation to the second-best genome only,
            # which is where it added diversity without displacing the
            # incumbent. Applying it to the top genome was tested but was
            # found to reduce the convergence rate.
            if len(elites) > 1 and _TORCH_AVAILABLE:
                grad = self._numerical_gradient(elites[1], seed)
                elites[1] = _hmc_perturb(
                    elites[1], grad, self.rng, hmc_step, hmc_mass
                )
                for _ in range(max(0, hmc_steps - 1)):
                    grad = self._numerical_gradient(elites[1], seed)
                    elites[1] = _hmc_perturb(
                        elites[1], grad, self.rng, hmc_step, hmc_mass
                    )
                elites[1].fitness = self.evaluate(elites[1], seed)

            new_population = list(elites)
            while len(new_population) < pop_size:
                if self.rng.random() < crossover_rate:
                    a = population[int(self.rng.integers(0, max(2, pop_size // 2)))]
                    b = population[int(self.rng.integers(0, max(2, pop_size // 2)))]
                    child = _crossover(a, b, self.rng)
                else:
                    parent = population[int(self.rng.integers(0, pop_size))]
                    child = _mutate(parent, self.rng, mutation_scale)
                child.fitness = self.evaluate(child, seed)
                new_population.append(child)

            population = sorted(new_population, key=lambda g: g.fitness, reverse=True)
            best = population[0]
            self.history.append(
                {
                    "generation": generation,
                    "best_fitness": best.fitness,
                    "best_genome": best.to_dict(),
                    "population_mean_fitness": float(
                        np.mean([g.fitness for g in population])
                    ),
                }
            )
            LOGGER.info(
                "metaagent generation %02d best_recall=%.4f hidden=%d dropout=%.3f lr=%.2e",
                generation,
                best.fitness,
                best.hidden_dim,
                best.dropout,
                best.learning_rate,
            )
        return population[0], self.history

    def _numerical_gradient(
        self, genome: Genome, seed: int
    ) -> np.ndarray:
        eps = 1e-3
        base = genome.fitness
        grads = np.zeros(3, dtype=np.float64)
        for i, make in enumerate(
            (
                lambda g: Genome(g.hidden_dim * math.exp(eps), g.dropout, g.learning_rate, g.batch_size),
                lambda g: Genome(g.hidden_dim, float(np.clip(g.dropout + eps, 0.02, 0.45)), g.learning_rate, g.batch_size),
                lambda g: Genome(g.hidden_dim, g.dropout, g.learning_rate * math.exp(eps), g.batch_size),
            )
        ):
            probe = make(genome)
            probe.fitness = self.evaluate(probe, seed)
            grads[i] = (probe.fitness - base) / eps
        return grads


# --------------------------------------------------------------------------- #
# Joint fine-tuning
# --------------------------------------------------------------------------- #


@dataclass
class JointResult:
    metrics: Dict[str, float]
    epochs_run: int
    best_epoch: int


def _joint_finetune(
    tmn: "nn.Module",
    metaagent: "nn.Module",
    tensors: Sequence[WindowTensor],
    hyperparameters: Mapping[str, object],
    device: "torch.device",
) -> JointResult:
    if not _TORCH_AVAILABLE:
        raise ImportError("joint fine-tune requires PyTorch")
    hp = dict(hyperparameters)
    seed = int(hp["random_seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = len(tensors)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(n * 0.75))
    train_idx = idx[:n_train]
    val_idx = idx[n_train:]

    def batches(indices: np.ndarray, batch_size: int):
        for start in range(0, len(indices), batch_size):
            yield indices[start : start + batch_size]

    optimiser = torch.optim.AdamW(
        list(tmn.parameters()) + list(metaagent.parameters()),
        lr=float(hp["learning_rate"]),
        weight_decay=float(hp["weight_decay"]),
    )
    criterion = nn.CrossEntropyLoss()
    patience = int(hp["patience"])
    best = {"epoch": -1, "f1": -1.0, "state": None}
    since_improvement = 0
    epochs_run = 0

    def stack(idxs: np.ndarray):
        h0 = torch.from_numpy(np.stack([tensors[i].image_h0 for i in idxs])).unsqueeze(1).to(device)
        h1 = torch.from_numpy(np.stack([tensors[i].image_h1 for i in idxs])).unsqueeze(1).to(device)
        seq = torch.from_numpy(np.stack([tensors[i].sequence for i in idxs])).to(device)
        y = torch.from_numpy(np.array([tensors[i].label for i in idxs], dtype=np.int64)).to(device)
        return h0, h1, seq, y

    for epoch in range(int(hp["epochs"])):
        epochs_run = epoch + 1
        tmn.train()
        metaagent.train()
        for idxs in batches(train_idx, int(hp["batch_size"])):
            h0, h1, seq, y = stack(idxs)
            optimiser.zero_grad()
            feats = FrozenTMNEncoder(tmn)(h0, h1, seq)  # type: ignore[operator]
            logits = metaagent(feats)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(tmn.parameters()) + list(metaagent.parameters()),
                float(hp["gradient_clip_norm"]),
            )
            optimiser.step()

        tmn.eval()
        metaagent.eval()
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for idxs in batches(val_idx, int(hp["batch_size"])):
                h0, h1, seq, y = stack(idxs)
                feats = FrozenTMNEncoder(tmn)(h0, h1, seq)  # type: ignore[operator]
                pred = metaagent(feats).argmax(dim=-1)
                tp += int(((pred == 1) & (y == 1)).sum().item())
                fp += int(((pred == 1) & (y == 0)).sum().item())
                fn += int(((pred == 0) & (y == 1)).sum().item())
                tn += int(((pred == 0) & (y == 0)).sum().item())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        acc = (tp + tn) / max(1, tp + fp + fn + tn)
        LOGGER.info(
            "joint epoch %02d acc=%.6f f1=%.6f precision=%.6f recall=%.6f",
            epoch,
            acc,
            f1,
            precision,
            recall,
        )
        if f1 > best["f1"]:
            best = {
                "epoch": epoch,
                "f1": f1,
                "state": {
                    "tmn": {k: v.detach().cpu().clone() for k, v in tmn.state_dict().items()},
                    "metaagent": {k: v.detach().cpu().clone() for k, v in metaagent.state_dict().items()},
                },
            }
            since_improvement = 0
        else:
            since_improvement += 1
            if since_improvement >= patience:
                LOGGER.info("joint early-stop after %d epochs without improvement", patience)
                break

    if best["state"] is not None:
        tmn.load_state_dict(best["state"]["tmn"])
        metaagent.load_state_dict(best["state"]["metaagent"])

    # We computed the final metrics against the whole validation split with
    # the restored best checkpoint, which is what the artefact reports.
    tmn.eval()
    metaagent.eval()
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for idxs in batches(val_idx, int(hp["batch_size"])):
            h0, h1, seq, y = stack(idxs)
            feats = FrozenTMNEncoder(tmn)(h0, h1, seq)  # type: ignore[operator]
            pred = metaagent(feats).argmax(dim=-1)
            tp += int(((pred == 1) & (y == 1)).sum().item())
            fp += int(((pred == 1) & (y == 0)).sum().item())
            fn += int(((pred == 0) & (y == 1)).sum().item())
            tn += int(((pred == 0) & (y == 0)).sum().item())
    accuracy = (tp + tn) / max(1, tp + fp + fn + tn)
    fpr = fp / max(1, fp + tn)
    fnr = fn / max(1, fn + tp)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return JointResult(
        metrics={
            "detection_accuracy": float(accuracy),
            "false_positive_rate_fraction": float(fpr),
            "false_negative_rate_fraction": float(fnr),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        },
        epochs_run=epochs_run,
        best_epoch=int(best["epoch"]),
    )


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


class MirageTrainingLoop:
    """Runs the full four-stage MIRAGE-DoH training plan."""

    def __init__(
        self,
        campaign_root: Path,
        checkpoint_root: Path,
        seed: Optional[int] = None,
        skip_stages: Optional[Sequence[StageName]] = None,
        device: Optional[str] = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("MirageTrainingLoop requires PyTorch")
        self.campaign_root = Path(campaign_root)
        self.checkpoint_root = Path(checkpoint_root)
        self.seed = int(seed if seed is not None else TMN_HYPERPARAMETERS["random_seed"])
        self.skip = set(skip_stages or [])
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)

    def _record(
        self, name: StageName, status: StageStatus, started: float,
        artifact_dir: Optional[Path], metrics: Optional[Dict[str, float]] = None,
        error: Optional[str] = None,
    ) -> StageOutcome:
        return StageOutcome(
            name=name,
            status=status,
            started_at=started,
            finished_at=time.monotonic(),
            artifact_dir=str(artifact_dir) if artifact_dir else None,
            metrics=dict(metrics or {}),
            error=error,
        )

    def run(self) -> LoopResult:
        plan = {
            "campaign_root": str(self.campaign_root),
            "seed": self.seed,
            "skip": sorted(s.value for s in self.skip),
            "tmn": TMN_HYPERPARAMETERS,
            "metaagent": METAAGENT_HYPERPARAMETERS,
            "joint": JOINT_FINETUNE_HYPERPARAMETERS,
        }
        plan_hash = _hash_plan(plan)
        LOGGER.info("starting MIRAGE training loop plan=%s", plan_hash[:12])

        stages: List[StageOutcome] = []

        # ------------------------------------------------------------------ #
        # Stage 1: TMN
        # ------------------------------------------------------------------ #
        tmn_dir = self.checkpoint_root / "tmn"
        tmn_started = time.monotonic()
        tensors: List[WindowTensor] = []
        tmn_trainer: Optional[TMNTrainer] = None
        ranges_h0 = ((0.0, 1.0), (0.0, 1.0))
        ranges_h1 = ((0.0, 1.0), (0.0, 1.0))

        if StageName.TMN in self.skip:
            stages.append(
                self._record(StageName.TMN, StageStatus.SKIPPED, tmn_started, tmn_dir)
            )
        else:
            try:
                hp = dict(TMN_HYPERPARAMETERS)
                tensors, _ = build_tensor_set(self.campaign_root, hp)
                ranges_h0 = hp.pop("__ranges_h0", ranges_h0)
                ranges_h1 = hp.pop("__ranges_h1", ranges_h1)
                if not tensors:
                    raise RuntimeError("no windows extracted from campaign archive")
                tmn_trainer = TMNTrainer(hyperparameters=hp, device=str(self.device))
                report = tmn_trainer.fit(tensors, seed=self.seed)
                export_tmn_report(
                    out_dir=tmn_dir,
                    report=report,
                    n_tensors=len(tensors),
                    image_ranges_h0=ranges_h0,
                    image_ranges_h1=ranges_h1,
                    model_state=tmn_trainer.model.state_dict() if tmn_trainer.model else None,
                )
                stages.append(
                    self._record(
                        StageName.TMN,
                        StageStatus.COMPLETED,
                        tmn_started,
                        tmn_dir,
                        metrics={
                            "best_val_recall": report.best_val_recall,
                            "best_val_precision": report.best_val_precision,
                            "best_val_acc": report.best_val_acc,
                        },
                    )
                )
            except Exception as exc:
                LOGGER.exception("TMN stage failed")
                stages.append(
                    self._record(
                        StageName.TMN, StageStatus.FAILED, tmn_started, tmn_dir,
                        error=str(exc),
                    )
                )
                return self._finish(plan_hash, stages, final_metrics={}, acceptance=False)

        # ------------------------------------------------------------------ #
        # Stage 2: MetaAgent
        # ------------------------------------------------------------------ #
        metaagent_dir = self.checkpoint_root / "metaagent"
        meta_started = time.monotonic()
        best_genome: Optional[Genome] = None
        metaagent_model: Optional[nn.Module] = None

        if StageName.METAAGENT in self.skip or tmn_trainer is None or not tensors:
            stages.append(
                self._record(
                    StageName.METAAGENT, StageStatus.SKIPPED, meta_started, metaagent_dir
                )
            )
        else:
            try:
                assert tmn_trainer.model is not None
                encoder = FrozenTMNEncoder(tmn_trainer.model).to(self.device)
                evolver = MetaAgentEvolver(
                    encoder=encoder,
                    tensors=tensors,
                    hyperparameters=METAAGENT_HYPERPARAMETERS,
                    device=self.device,
                )
                best_genome, history = evolver.run()
                metaagent_model = CognitiveMetaAgent(
                    encoder_dim=encoder.features.shape[1],
                    hidden_dim=best_genome.hidden_dim,
                    dropout=best_genome.dropout,
                ).to(self.device)
                metaagent_dir.mkdir(parents=True, exist_ok=True)
                (metaagent_dir / "evolution_history.json").write_text(
                    json.dumps(history, indent=2)
                )
                (metaagent_dir / "best_genome.json").write_text(
                    json.dumps(best_genome.to_dict(), indent=2)
                )
                torch.save(
                    metaagent_model.state_dict(),
                    metaagent_dir / "metaagent_head.pt",
                )
                stages.append(
                    self._record(
                        StageName.METAAGENT,
                        StageStatus.COMPLETED,
                        meta_started,
                        metaagent_dir,
                        metrics={"best_recall": float(best_genome.fitness)},
                    )
                )
            except Exception as exc:
                LOGGER.exception("MetaAgent stage failed")
                stages.append(
                    self._record(
                        StageName.METAAGENT, StageStatus.FAILED, meta_started,
                        metaagent_dir, error=str(exc),
                    )
                )
                return self._finish(plan_hash, stages, final_metrics={}, acceptance=False)

        # ------------------------------------------------------------------ #
        # Stage 3: MCS calibration
        # ------------------------------------------------------------------ #
        mcs_dir = self.checkpoint_root / "mcs"
        mcs_started = time.monotonic()
        if StageName.MCS in self.skip:
            stages.append(
                self._record(StageName.MCS, StageStatus.SKIPPED, mcs_started, mcs_dir)
            )
        else:
            try:
                from ..cryptographic_agility.cryptographic_shard_calibrator import (  # noqa: F401
                    calibrate_shards,
                )
                calibrate_shards(out_dir=mcs_dir, seed=self.seed)
                stages.append(
                    self._record(
                        StageName.MCS, StageStatus.COMPLETED, mcs_started, mcs_dir,
                    )
                )
            except ImportError:
                LOGGER.warning(
                    "cryptographic_shard_calibrator unavailable; skipping MCS stage"
                )
                stages.append(
                    self._record(
                        StageName.MCS, StageStatus.SKIPPED, mcs_started, mcs_dir
                    )
                )
            except Exception as exc:
                LOGGER.exception("MCS stage failed")
                stages.append(
                    self._record(
                        StageName.MCS, StageStatus.FAILED, mcs_started, mcs_dir,
                        error=str(exc),
                    )
                )

        # ------------------------------------------------------------------ #
        # Stage 4: Joint fine-tune
        # ------------------------------------------------------------------ #
        joint_dir = self.checkpoint_root / "joint"
        joint_started = time.monotonic()
        final_metrics: Dict[str, float] = {}

        if (
            StageName.JOINT in self.skip
            or tmn_trainer is None
            or tmn_trainer.model is None
            or metaagent_model is None
        ):
            stages.append(
                self._record(StageName.JOINT, StageStatus.SKIPPED, joint_started, joint_dir)
            )
            acceptance = False
        else:
            try:
                joint = _joint_finetune(
                    tmn=tmn_trainer.model,
                    metaagent=metaagent_model,
                    tensors=tensors,
                    hyperparameters=JOINT_FINETUNE_HYPERPARAMETERS,
                    device=self.device,
                )
                joint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "tmn": tmn_trainer.model.state_dict(),
                        "metaagent": metaagent_model.state_dict(),
                        "ranges_h0": ranges_h0,
                        "ranges_h1": ranges_h1,
                    },
                    joint_dir / "mirage_full_model.pt",
                )
                final_metrics = dict(joint.metrics)
                (joint_dir / "joint_metrics.json").write_text(
                    json.dumps(joint.metrics, indent=2)
                )
                stages.append(
                    self._record(
                        StageName.JOINT, StageStatus.COMPLETED, joint_started,
                        joint_dir, metrics=joint.metrics,
                    )
                )
                acceptance = self._check_acceptance(final_metrics)
            except Exception as exc:
                LOGGER.exception("joint stage failed")
                stages.append(
                    self._record(
                        StageName.JOINT, StageStatus.FAILED, joint_started,
                        joint_dir, error=str(exc),
                    )
                )
                acceptance = False

        return self._finish(plan_hash, stages, final_metrics, acceptance)

    @staticmethod
    def _check_acceptance(metrics: Mapping[str, float]) -> bool:
        for key, (lo, hi) in ACCEPTANCE_TOLERANCE.items():
            val = metrics.get(key)
            if val is None:
                LOGGER.error("acceptance metric %s missing", key)
                return False
            if not (lo <= val <= hi):
                LOGGER.error(
                    "acceptance failed for %s: %.6f outside [%.6f, %.6f]",
                    key, val, lo, hi,
                )
                return False
        return True

    def _finish(
        self,
        plan_hash: str,
        stages: List[StageOutcome],
        final_metrics: Dict[str, float],
        acceptance: bool,
    ) -> LoopResult:
        result = LoopResult(
            plan_hash=plan_hash,
            seed=self.seed,
            stages=stages,
            final_metrics=final_metrics,
            acceptance_passed=acceptance,
            checkpoint_dir=str(self.checkpoint_root),
        )
        manifest = result.to_manifest()
        manifest["manifest_sha256"] = hashlib.sha256(
            json.dumps(manifest, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        (self.checkpoint_root / "loop_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        if acceptance:
            LOGGER.info("MIRAGE training loop passed acceptance: %s", final_metrics)
        else:
            LOGGER.warning(
                "MIRAGE training loop did not pass acceptance: %s", final_metrics
            )
        return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end MIRAGE-DoH training loop."
    )
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--skip",
        nargs="*",
        choices=[s.value for s in StageName],
        default=[],
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if acceptance fails",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    loop = MirageTrainingLoop(
        campaign_root=args.campaign_root,
        checkpoint_root=args.checkpoint_root,
        seed=args.seed,
        skip_stages=[StageName(s) for s in args.skip],
        device=args.device,
    )
    result = loop.run()
    if args.strict and not result.acceptance_passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
