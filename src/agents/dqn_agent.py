"""DQN agent with experience replay and target network."""

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .base import BaseAgent, ReplayBuffer, mlp


class DQNAgent(BaseAgent):
    """DQN + Double DQN for discrete action spaces."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_layers: list = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        memory_size: int = 10000,
        target_update_freq: int = 100,
        double_dqn: bool = True,
    ):
        super().__init__(name="DQN")
        if hidden_layers is None:
            hidden_layers = [128, 64]

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.double_dqn = double_dqn

        self.q_net = mlp([state_dim, *hidden_layers, action_dim])
        self.target_net = mlp([state_dim, *hidden_layers, action_dim])
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer(memory_size)

    def select_action(self, state: np.ndarray, eval_mode: bool = False):
        if not eval_mode and random.random() < self.epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def update(self) -> dict:
        if len(self.memory) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size
        )
        batch_idx = torch.arange(self.batch_size)

        # Current Q values
        q_values = self.q_net(states)
        q_current = q_values[batch_idx, actions]

        # Target Q values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use online net to select action, target net to evaluate
                next_actions = self.q_net(next_states).argmax(dim=1)
                q_next = self.target_net(next_states)[batch_idx, next_actions]
            else:
                q_next = self.target_net(next_states).max(dim=1).values
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = self.loss_fn(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {"loss": loss.item(), "epsilon": self.epsilon}
