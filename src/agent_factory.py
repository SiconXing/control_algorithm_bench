import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_layers=None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 64]
        layers = []
        prev = state_size
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, action_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        device = get_device()
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones, dtype=float)).to(device),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layers=None,
        lr: float = 0.001,
        gamma: float = 0.99,
        batch_size: int = 64,
        target_update: int = 20,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.998,
        memory_capacity: int = 20000,
    ):
        if hidden_layers is None:
            hidden_layers = [64, 64]
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        self.device = get_device()
        self.q_network = DQN(state_size, action_size, hidden_layers).to(self.device)
        self.target_network = DQN(state_size, action_size, hidden_layers).to(self.device)
        self._sync_target()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer(memory_capacity)

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def _sync_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def act(self, state) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            q = self.q_network(torch.FloatTensor(state).unsqueeze(0).to(self.device))
        return q.argmax().item()

    def train_step(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def q_values(self, states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.q_network(states.to(self.device))
