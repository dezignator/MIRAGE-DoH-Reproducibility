"""
cnn_lstm.py — CNN-LSTM baseline for encrypted DNS traffic classification.

Thesis reference: Chapter 3, Table 8 (accuracy 95.5%, FPR 0.15%,
latency 210 µs, throughput 8.7 M QPS).

Architecture follows the standard encrypted-traffic CNN-LSTM from the
literature (Wang et al., ISI 2017):
    - 2x Conv1d (64, 128 filters, kernel 3, ReLU, BatchNorm)
    - MaxPool1d (size 2)
    - 2x LSTM (128 hidden units, dropout 0.2)
    - Dense (64) + Dropout + Dense (1, sigmoid)

The input is treated as a sequence of 46 feature vectors per flow.
Because the raw CSV is flat (1 row = 46 features), the input is reshaped
to (batch_size, 46, 1) for the 1D convolution.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class CNNLSTMClassifier(nn.Module):
    """CNN-LSTM binary classifier."""

    def __init__(self, input_dim: int = 46, dropout: float = 0.2):
        super().__init__()
        # Input shape: (batch, input_dim, 1)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2)
        # LSTM input: (batch, seq_len, features)
        # After conv+pool, seq_len is input_dim // 2, features is 128
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim) -> reshape to (batch, 1, input_dim)
        x = x.unsqueeze(1)
        # Conv blocks
        x = torch.relu(self.bn1(self.conv1(x)))  # (batch, 64, 46)
        x = torch.relu(self.bn2(self.conv2(x)))  # (batch, 128, 46)
        x = self.pool(x)                          # (batch, 128, 23)
        # LSTM needs (batch, seq_len, features)
        x = x.transpose(1, 2)                     # (batch, 23, 128)
        # LSTM
        lstm_out, _ = self.lstm(x)                # (batch, 23, 128)
        # Take the last timestep
        x = lstm_out[:, -1, :]                    # (batch, 128)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))               # (batch, 64)
        x = self.dropout(x)
        x = self.fc2(x)                           # (batch, 1)
        return self.sigmoid(x).squeeze(-1)        # (batch,)


class CNNLSTMBaseline:
    """Wrapper exposing the consistent train / predict / evaluate API."""

    def __init__(
        self,
        input_dim: int = 46,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        num_epochs: int = 50,
        patience: int = 10,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CNNLSTMClassifier(input_dim=input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience
        self.criterion = nn.BCELoss()

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, List[float]]:
        """Train with early stopping on validation loss."""
        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.num_epochs):
            # --- Train epoch ---
            self.model.train()
            epoch_loss, correct, total = 0.0, 0, 0
            for x_batch, y_batch in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(x_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item() * len(x_batch)
                correct += ((preds > 0.5).float() == y_batch).sum().item()
                total += len(y_batch)

            train_loss = epoch_loss / total
            train_acc = correct / total
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # --- Validation epoch ---
            if X_val is not None:
                val_loss, val_acc = self._evaluate_split(X_val, y_val)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                logger.info(
                    "CNN-LSTM epoch %d/%d  train_loss=%.4f  train_acc=%.4f  val_loss=%.4f  val_acc=%.4f",
                    epoch + 1, self.num_epochs, train_loss, train_acc, val_loss, val_acc,
                )

                if epochs_without_improvement >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break
            else:
                logger.info("CNN-LSTM epoch %d/%d  train_loss=%.4f  train_acc=%.4f",
                            epoch + 1, self.num_epochs, train_loss, train_acc)

        return history

    def _evaluate_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
            preds = self.model(x_t)
            loss = self.criterion(preds, y_t).item()
            acc = ((preds > 0.5).float() == y_t).float().mean().item()
        return loss, acc

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities for class 1 (malicious)."""
        self.model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            return self.model(x_t).cpu().numpy()

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Return metrics in Table 8 format (accuracy, FPR, latency, throughput)."""
        from mirage.utils.metrics import compute_all_table_8_metrics

        # --- Latency measurement ---
        # Warmup 10 iterations, then measure 100 iterations.
        X_sample = X_test[:min(len(X_test), 1000)]
        for _ in range(10):
            self.predict(X_sample[:min(len(X_sample), 64)])
        start = time.perf_counter()
        n_reps = 100
        for _ in range(n_reps):
            self.predict(X_sample)
        elapsed = time.perf_counter() - start
        latencies_ms = [(elapsed / n_reps) * 1000.0] * len(X_test)

        # --- Classification metrics ---
        probs = self.predict(X_test)
        preds = (probs > 0.5).astype(int)
        metrics = compute_all_table_8_metrics(
            y_true=y_test,
            y_pred=preds,
            y_prob=probs,
            latencies_ms=latencies_ms,
            num_queries=len(X_test),
            elapsed_seconds=elapsed / n_reps,
        )
        return metrics
