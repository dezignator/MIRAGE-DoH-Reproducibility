"""
random_forest.py — Random Forest baseline for encrypted DNS traffic classification.

Thesis reference: Chapter 3, Table 8 (accuracy 92.3%, FPR 0.42%,
latency 155 µs, throughput 4.2 M QPS).

Uses scikit-learn's RandomForestClassifier with the standard hyperparameter
search space from configs/baselines_config.yaml §3.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)


class RandomForestBaseline:
    """Wrapper exposing the consistent train / predict / evaluate API."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        criterion: str = "gini",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight=None,  # set to "balanced" only if class imbalance requires it
        )
        self.is_fitted = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the Random Forest. No validation loop (RF does not use epochs)."""
        logger.info("Training Random Forest on %d samples...", len(X_train))
        start = time.perf_counter()
        self.model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        logger.info("Random Forest trained in %.2f s", elapsed)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted probabilities for class 1 (malicious)."""
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained. Call .train() first.")
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Return metrics in Table 8 format."""
        from mirage.utils.metrics import compute_all_table_8_metrics

        # --- Latency measurement ---
        # Random Forest has extremely low latency per prediction; we measure
        # by predicting the full test set once and dividing by n_queries.
        # This is a lower bound on latency (batch prediction is faster than
        # one-at-a-time prediction). For the thesis Table 8 comparison, the
        # 155 us figure is the mean per-query latency at batch_size=1; we
        # replicate that here.
        latencies_ms = []
        n_measure = min(len(X_test), 1000)
        for i in range(n_measure):
            start = time.perf_counter()
            _ = self.model.predict_proba(X_test[i:i+1])
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

        # --- Classification metrics ---
        probs = self.predict(X_test)
        preds = (probs > 0.5).astype(int)
        total_elapsed = sum(latencies_ms) / 1000.0
        metrics = compute_all_table_8_metrics(
            y_true=y_test,
            y_pred=preds,
            y_prob=probs,
            latencies_ms=latencies_ms,
            num_queries=len(X_test),
            elapsed_seconds=total_elapsed / max(len(X_test) - n_measure, 1),
        )
        return metrics
