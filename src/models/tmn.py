"""
tmn.py — Topological Memory Networks (TMN).

Thesis reference: Chapter 3, Section 3.6.1 and 3.6.2, Equations 6-10,
and Section 3.15.

Implements:
- Vietoris-Rips filtration and persistent homology (Eq. 6-8).
- LSTM-Attention hybrid for temporal topological memory (Eq. 9).
- Multi-hop persistence tracking (Eq. 10).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import gudhi
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mirage.utils.math_utils import (
    attention_weight,
    multi_hop_persistence,
    persistence_betti_numbers,
    persistence_diagram,
    simplicial_filtration,
    TMN_BETTI_MIN,
    TMN_HOP_HORIZON,
)

logger = logging.getLogger(__name__)


class TMNAttentionLSTM(nn.Module):
    """Equation 9: LSTM-Attention hybrid for the TMN memory state.

    Config per configs/tmn_hyperparameters.yaml §2:
    input_size=46, hidden_size=128, num_layers=2, dropout=0.20.
    """

    def __init__(self, input_size: int = 46, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Attention mechanism: linear layer to compute attention scores.
        self.attention = nn.Linear(hidden_size, 1, bias=False)
        # LSTM.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        # Output head: classify as benign (0) or malicious (1).
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention-weighted pooling.

        Args:
            x: (batch_size, sequence_length, input_size)

        Returns:
            (logits, attention_weights)
        """
        # LSTM over the sequence.
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        # Attention scores.
        scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(scores, dim=1)  # (batch, seq_len, 1)
        # Weighted sum.
        context = (attn_weights * lstm_out).sum(dim=1)  # (batch, hidden_size)
        # Classify.
        logits = self.classifier(context)
        return logits, attn_weights.squeeze(-1)


class TopologicalMemoryNetwork:
    """Full TMN orchestrator.

    Manages the persistent homology computation over incoming traffic
    windows, the LSTM-Attention memory, and the multi-hop persistence
    score used for tunneling detection.
    """

    def __init__(
        self,
        input_size: int = 46,
        hidden_size: int = 128,
        hop_horizon: int = TMN_HOP_HORIZON,
        betti_min: int = TMN_BETTI_MIN,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TMNAttentionLSTM(input_size=input_size, hidden_size=hidden_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=5e-4)
        self.hop_horizon = hop_horizon
        self.betti_min = betti_min
        # Rolling window of past features for the LSTM.
        self.feature_history: List[np.ndarray] = []

    def compute_persistence(self, points: np.ndarray) -> np.ndarray:
        """Compute the persistence diagram for a point cloud (Eq. 8).

        Uses gudhi to construct the Vietoris-Rips complex and ripser to
        extract the persistence diagram in dimensions 0 and 1.
        """
        # gudhi expects a distance matrix. Compute it from the points.
        from scipy.spatial.distance import cdist
        distances = cdist(points, points)

        rips_complex = gudhi.RipsComplex(distance_matrix=distances, max_edge_length=100.0)
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
        simplex_tree.compute_persistence()

        # Extract the persistence diagram (birth, death) pairs for H_1.
        persistence_pairs = simplex_tree.persistence()
        h1_pairs = [(b, d) for (dim, (b, d)) in persistence_pairs if dim == 1 and np.isfinite(d)]
        return np.array(h1_pairs) if h1_pairs else np.zeros((0, 2))

    def analyze(self, flow_features: np.ndarray) -> Dict[str, float]:
        """Analyze a flow's topological properties.

        Args:
            flow_features: (sequence_length, 46) features for one flow.

        Returns:
            Dict with keys: betti_1, persistence_score, is_tunnel.
        """
        # Point cloud from the feature sequence.
        points = flow_features.astype(np.float64)
        distances = np.sqrt(((points[:, None, :] - points[None, :, :]) ** 2).sum(-1))
        # Persistence diagram.
        scales = np.linspace(0.0, distances.max(), 256)
        dgm = persistence_diagram(distances, scales)

        # Betti-1 number.
        _, b_1 = persistence_betti_numbers(distances, scale=scales.max() * 0.5)
        # Multi-hop persistence score (Eq. 10).
        multi_hop = multi_hop_persistence([dgm], decay_lambda=0.5)

        return {
            "betti_1": b_1,
            "persistence_score": multi_hop,
            "is_tunnel": b_1 >= self.betti_min and multi_hop > 0.8,
        }

    def predict(self, sequence: np.ndarray) -> Tuple[int, float]:
        """Predict whether a sequence of features is a tunnel using the LSTM.

        Args:
            sequence: (sequence_length, 46) features.

        Returns:
            (predicted_class, confidence)
        """
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(sequence, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, _ = self.model(x_t)
            probs = torch.softmax(logits, dim=-1)
            pred = logits.argmax(dim=-1).item()
            conf = probs[0, pred].item()
        return pred, conf

    def train(self, X_train: np.ndarray, y_train: np.ndarray, num_epochs: int = 100, batch_size: int = 32) -> Dict[str, List[float]]:
        """Train the TMN on sequences of features."""
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loss_fn = nn.CrossEntropyLoss()
        history = {"loss": [], "accuracy": []}

        for epoch in range(num_epochs):
            epoch_loss, correct, total = 0.0, 0, 0
            self.model.train()
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                logits, _ = self.model(x_batch)
                loss = loss_fn(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(x_batch)
                correct += (logits.argmax(-1) == y_batch).sum().item()
                total += len(y_batch)
            avg_loss = epoch_loss / total
            acc = correct / total
            history["loss"].append(avg_loss)
            history["accuracy"].append(acc)
            logger.info("TMN epoch %d/%d loss=%.4f acc=%.4f", epoch + 1, num_epochs, avg_loss, acc)
        return history
