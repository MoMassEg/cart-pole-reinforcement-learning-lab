"""
Deep Q-Network (DQN) implementation from scratch using NumPy.
Featuring:
- 2-hidden-layer MLP (128x128) with He/Kaiming initialization
- Double DQN (DDQN) decoupling action selection from value estimation
- Huber loss gradient clipping
- Exact Adam optimizer with analytical backpropagation
- Experience Replay Memory
"""

import os
import random
from typing import Tuple, List, Dict
import numpy as np


class DQNAgent:
    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        hidden_dim: int = 128,
        lr: float = 7e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.99,
        target_update_freq: int = 200,
        buffer_capacity: int = 50000,
        batch_size: int = 64
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size

        # Weight initialization (He / Kaiming normal)
        self.W1 = (np.random.randn(state_dim, hidden_dim) * np.sqrt(2.0 / state_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = (np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(hidden_dim, dtype=np.float32)
        self.W3 = (np.random.randn(hidden_dim, action_dim) * np.sqrt(2.0 / hidden_dim)).astype(np.float32)
        self.b3 = np.zeros(action_dim, dtype=np.float32)

        # Target network weights
        self.t_W1 = np.copy(self.W1)
        self.t_b1 = np.copy(self.b1)
        self.t_W2 = np.copy(self.W2)
        self.t_b2 = np.copy(self.b2)
        self.t_W3 = np.copy(self.W3)
        self.t_b3 = np.copy(self.b3)

        # Adam optimizer state
        self.params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.step_cnt = 0
        self.global_step = 0

        # Experience replay buffer
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []

    def forward(self, x: np.ndarray, target: bool = False):
        """Forward pass through policy or target network."""
        if not isinstance(x, np.ndarray):
            x = np.array(x, dtype=np.float32)

        if target:
            W1, b1, W2, b2, W3, b3 = self.t_W1, self.t_b1, self.t_W2, self.t_b2, self.t_W3, self.t_b3
        else:
            W1, b1, W2, b2, W3, b3 = self.W1, self.b1, self.W2, self.b2, self.W3, self.b3

        if x.ndim == 1:
            x = x.reshape(1, -1)

        z1 = np.dot(x, W1) + b1
        h1 = np.maximum(0.0, z1)

        z2 = np.dot(h1, W2) + b2
        h2 = np.maximum(0.0, z2)

        out = np.dot(h2, W3) + b3
        return out, (x, z1, h1, z2, h2)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Select action using epsilon-greedy policy or deterministic greedy policy."""
        if not evaluate and np.random.rand() < self.epsilon:
            return random.randrange(self.action_dim)
        q_vals, _ = self.forward(state)
        return int(np.argmax(q_vals[0]))

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get estimated Q-values for all actions given state (for UI telemetry)."""
        q_vals, _ = self.forward(state)
        return q_vals[0]

    def sync_target(self):
        """Copy policy network parameters to target network."""
        self.t_W1 = np.copy(self.W1)
        self.t_b1 = np.copy(self.b1)
        self.t_W2 = np.copy(self.W2)
        self.t_b2 = np.copy(self.b2)
        self.t_W3 = np.copy(self.W3)
        self.t_b3 = np.copy(self.b3)

    def step(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        """Record transition and perform one optimization step."""
        self.global_step += 1
        self.buffer.append((
            np.array(state, dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done)
        ))
        if len(self.buffer) > self.buffer_capacity:
            self.buffer.pop(0)

        loss = self.train_step()

        if self.global_step % self.target_update_freq == 0:
            self.sync_target()

        return loss

    def train_step(self) -> float:
        """Sample mini-batch, calculate Double DQN targets, and update with Adam."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        idx = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        states = np.array([b[0] for b in batch], dtype=np.float32)
        actions = np.array([b[1] for b in batch], dtype=np.int64)
        rewards = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones = np.array([b[4] for b in batch], dtype=np.float32)

        # Current Q from online network
        q_pred, cache = self.forward(states)
        x, z1, h1, z2, h2 = cache

        # Double DQN:
        # 1. Best next action chosen by online network
        q_next_online, _ = self.forward(next_states)
        best_a = np.argmax(q_next_online, axis=1)

        # 2. Target Q evaluated by target network at best_a
        q_next_target, _ = self.forward(next_states, target=True)
        q_target_val = q_next_target[np.arange(self.batch_size), best_a]

        # 3. Bellman target: y = r + gamma * Q_target(s', a*) * (1 - done)
        y = rewards + (self.gamma * q_target_val * (1.0 - dones))

        # TD Error and Huber gradient clipping
        td_error = q_pred[np.arange(self.batch_size), actions] - y
        clipped_error = np.clip(td_error, -1.0, 1.0)
        loss = float(np.mean(0.5 * (clipped_error ** 2)))

        # Analytical gradients through backpropagation
        d_out = np.zeros_like(q_pred)
        d_out[np.arange(self.batch_size), actions] = clipped_error / self.batch_size

        dW3 = np.dot(h2.T, d_out)
        db3 = np.sum(d_out, axis=0)
        dh2 = np.dot(d_out, self.W3.T)

        dz2 = dh2 * (z2 > 0.0)
        dW2 = np.dot(h1.T, dz2)
        db2 = np.sum(dz2, axis=0)
        dh1 = np.dot(dz2, self.W2.T)

        dz1 = dh1 * (z1 > 0.0)
        dW1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0)

        grads = [dW1, db1, dW2, db2, dW3, db3]

        # Adam optimizer update
        self.step_cnt += 1
        lr_t = self.lr * np.sqrt(1.0 - (0.999 ** self.step_cnt)) / (1.0 - (0.9 ** self.step_cnt))
        for i in range(len(self.params)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * grads[i]
            self.v[i] = 0.999 * self.v[i] + 0.001 * (grads[i] ** 2)
            self.params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-8)

        return loss

    def decay_epsilon(self):
        """Exponential decay of epsilon."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """Save model parameters to .npz file."""
        np.savez(
            filepath,
            W1=self.W1, b1=self.b1,
            W2=self.W2, b2=self.b2,
            W3=self.W3, b3=self.b3,
            epsilon=np.array(self.epsilon, dtype=np.float32),
            global_step=np.array(self.global_step, dtype=np.int64)
        )

    def load(self, filepath: str):
        """Load model parameters from .npz file."""
        if not filepath.endswith(".npz"):
            if os.path.exists(filepath + ".npz"):
                filepath = filepath + ".npz"

        data = np.load(filepath)
        self.W1[:] = data["W1"]
        self.b1[:] = data["b1"]
        self.W2[:] = data["W2"]
        self.b2[:] = data["b2"]
        self.W3[:] = data["W3"]
        self.b3[:] = data["b3"]
        self.sync_target()
        if "epsilon" in data:
            self.epsilon = float(data["epsilon"])
        if "global_step" in data:
            self.global_step = int(data["global_step"])
