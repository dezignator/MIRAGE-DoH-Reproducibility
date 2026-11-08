"""
sgi.py — Spectral Game Intelligence (SGI).

Thesis reference: Chapter 3, Section 3.9.1-3.9.3, Equations 17-21,
Tables 6, 7, 9, and 12.

Implements:
- The graph Laplacian and spectral embedding (Eq. 17, 18).
- The Deep Q-Network (Eq. 19) with hidden layers [256, 128, 64].
- The temporal edge weight (Eq. 20) with tau=100 ms.
- The reward function (Eq. 21).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from mirage.utils.math_utils import (
    dqn_policy,
    edge_weight,
    graph_laplacian,
    reward_function,
    spectral_embedding,
    SGI_K_DIM,
    SGI_TAU_MS,
)

logger = logging.getLogger(__name__)


class DQNetwork(nn.Module):
    """Eq. 19: Deep Q-Network with [256, 128, 64] hidden layers (Section 3.9.2)."""

    def __init__(self, input_dim: int = SGI_K_DIM, num_actions: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectralGameIntelligence:
    """Full SGI orchestrator.

    Manages the DQN, the spectral state construction, and the mitigation
    actions defined in Section 3.9.3.
    """

    def __init__(
        self,
        k: int = SGI_K_DIM,
        tau_ms: float = SGI_TAU_MS,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 100000,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.k = k
        self.tau_ms = tau_ms
        self.gamma = gamma
        self.q_network = DQNetwork(input_dim=k).to(self.device)
        self.target_network = DQNetwork(input_dim=k).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Replay buffer.
        self.replay_buffer: Deque = deque(maxlen=100000)
        self.target_update_interval = 1000
        self.batch_size = 32
        self.steps = 0

        # Epsilon-greedy schedule.
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps

    def build_graph(self, flows: np.ndarray) -> np.ndarray:
        """Build the traffic graph (Section 3.9.3).

        Args:
            flows: (n_flows, feature_dim) matrix. Row i is flow i.

        Returns:
            (n_flows, n_flows) adjacency matrix.
        """
        n = len(flows)
        # Temporal proximity: each flow's timestamp is feature 0 (inter_arrival).
        timestamps = flows[:, 0]
        adjacency = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                delta_t = abs(timestamps[i] - timestamps[j])
                w = edge_weight(delta_t, tau=self.tau_ms)
                adjacency[i, j] = w
                adjacency[j, i] = w
        return adjacency

    def compute_spectral_state(self, flows: np.ndarray) -> np.ndarray:
        """Compute the spectral embedding X_spectral (Eq. 18).

        Args:
            flows: (n_flows, feature_dim) matrix.

        Returns:
            (k,) spectral embedding vector (mean over the k eigenvectors).
        """
        adjacency = self.build_graph(flows)
        L = graph_laplacian(adjacency)
        try:
            X_spec = spectral_embedding(L, k=self.k)  # (k, n)
            return X_spec.mean(axis=1)  # (k,)
        except np.linalg.LinAlgError:
            return np.zeros(self.k)

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, 4)  # explore
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(state_t)
        return int(q_values.argmax().item())

    def update(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> float:
        """Perform one DQN learning step."""
        self.replay_buffer.append((state, action, reward, next_state, done))
        self.steps += 1

        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch = np.random.choice(len(self.replay_buffer), size=self.batch_size, replace=False)
        states = torch.tensor(np.array([self.replay_buffer[i][0] for i in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([self.replay_buffer[i][1] for i in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([self.replay_buffer[i][2] for i in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([self.replay_buffer[i][3] for i in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([self.replay_buffer[i][4] for i in batch], dtype=torch.float32, device=self.device)

        # Current Q-values.
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # Target Q-values.
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            logger.info("SGI target network updated at step %d", self.steps)

        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)
        return float(loss.item())
