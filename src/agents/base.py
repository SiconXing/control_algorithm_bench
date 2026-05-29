"""Base class and shared utilities for RL agents."""

from abc import ABC, abstractmethod
from collections import deque
import random
from typing import Any, Deque, Optional, Tuple

import numpy as np
import torch


class BaseAgent(ABC):
    """Abstract RL agent."""

    def __init__(self, name: str = "BaseAgent"):
        self.name = name
        self.train_step = 0

    @abstractmethod
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> Any:
        ...

    def store_transition(
        self, state: np.ndarray, action: Any, reward: float,
        next_state: np.ndarray, done: bool,
    ) -> None:
        pass

    @abstractmethod
    def update(self) -> dict:
        ...

    def save(self, path: str) -> None:
        torch.save(self.__dict__, path)

    def load(self, path: str) -> None:
        self.__dict__.update(torch.load(path, weights_only=False))


class ReplayBuffer:
    """Fixed-size experience replay buffer."""

    def __init__(self, capacity: int = 10000):
        self.buffer: Deque = deque(maxlen=capacity)

    def push(
        self, state: np.ndarray, action: int, reward: float,
        next_state: np.ndarray, done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class PPOBuffer:
    """On-policy buffer for PPO."""

    def __init__(self):
        self.states: list = []
        self.actions: list = []
        self.rewards: list = []
        self.log_probs: list = []
        self.values: list = []
        self.dones: list = []

    def push(self, state, action, reward, log_prob, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def size(self) -> int:
        return len(self.states)

    def get_all(self):
        return (
            torch.FloatTensor(np.array(self.states)),
            torch.FloatTensor(np.array(self.actions)),
            torch.FloatTensor(self.rewards),
            torch.FloatTensor(self.log_probs),
            torch.FloatTensor(self.values),
            torch.FloatTensor(self.dones),
        )


def mlp(sizes: list, activation=torch.nn.ReLU, output_activation=None):
    """Build a simple MLP."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(activation())
        elif output_activation is not None:
            layers.append(output_activation())
    return torch.nn.Sequential(*layers)
