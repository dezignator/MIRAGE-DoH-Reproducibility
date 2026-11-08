"""
Baselines package.

Exports:
    CNNLSTMBaseline      (Table 8: accuracy 95.5%, FPR 0.15%)
    RandomForestBaseline (Table 8: accuracy 92.3%, FPR 0.42%)
"""

from mirage.baselines.cnn_lstm import CNNLSTMBaseline
from mirage.baselines.random_forest import RandomForestBaseline

__all__ = [
    "CNNLSTMBaseline",
    "RandomForestBaseline",
]
