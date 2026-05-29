"""PPO agent with clipped objective and GAE."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Categorical

from .base import BaseAgent, PPOBuffer, mlp


class ActorCritic(nn.Module):
    """Shared backbone, separate actor/critic heads."""

    def __init__(
        self, state_dim: int, action_dim: int,
        hidden: list, continuous: bool = True,
    ):
        super().__init__()
        self.continuous = continuous
        self.backbone = mlp([state_dim, *hidden])

        if continuous:
            self.mean_head = nn.Linear(hidden[-1], action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.logits_head = nn.Linear(hidden[-1], action_dim)

        self.value_head = nn.Linear(hidden[-1], 1)

    def forward(self, state):
        features = self.backbone(state)
        value = self.value_head(features).squeeze(-1)

        if self.continuous:
            mean = self.mean_head(features)
            std = self.log_std.exp().expand_as(mean)
            dist = Normal(mean, std)
        else:
            logits = self.logits_head(features)
            dist = Categorical(logits=logits)
        return dist, value


class PPOAgent(BaseAgent):
    """PPO for continuous or discrete action spaces."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        continuous: bool = True,
        hidden_layers: list = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        max_grad_norm: float = 0.5,
    ):
        super().__init__(name="PPO")
        if hidden_layers is None:
            hidden_layers = [128, 64]

        self.continuous = continuous
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm

        self.ac = ActorCritic(state_dim, action_dim, hidden_layers, continuous)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        self.buffer = PPOBuffer()

    def select_action(self, state: np.ndarray, eval_mode: bool = False):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            dist, value = self.ac(state_t)
            if eval_mode:
                action = dist.mean if self.continuous else dist.probs.argmax(-1)
            else:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        a = action.squeeze(0).numpy()
        return a, log_prob.item(), value.item()

    def store_transition(self, state, action, reward, next_state, done):
        pass  # PPO stores in buffer externally — see store_experience

    def store_experience(self, state, action, reward, log_prob, value, done):
        self.buffer.push(state, action, reward, log_prob, value, done)

    def _compute_gae(self, rewards, values, dones, last_val):
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0.0
        values = torch.cat([values, torch.tensor([last_val])])
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages)
        returns = advantages + values[:-1]
        return advantages, returns

    def update(self) -> dict:
        if self.buffer.size() < self.batch_size:
            return {}

        states, actions, rewards, log_probs_old, values, dones = self.buffer.get_all()

        # Get last value for bootstrap
        with torch.no_grad():
            _, last_val = self.ac(states[-1:])
            last_val = last_val.item()

        advantages, returns = self._compute_gae(
            rewards.tolist(), values.tolist(), dones.tolist(), last_val
        )
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        n_samples = states.size(0)

        for _ in range(self.ppo_epochs):
            indices = torch.randperm(n_samples)
            for start in range(0, n_samples, self.batch_size):
                idx = indices[start:start + self.batch_size]
                s_batch = states[idx]
                a_batch = actions[idx]
                adv_batch = advantages[idx]
                ret_batch = returns[idx]
                lp_old_batch = log_probs_old[idx]

                dist, values_pred = self.ac(s_batch)
                log_prob = dist.log_prob(a_batch).sum(dim=-1)
                entropy = dist.entropy().mean()

                # PPO clipped objective
                ratio = (log_prob - lp_old_batch).exp()
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv_batch
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(values_pred, ret_batch)

                loss = (
                    actor_loss
                    + self.value_coef * critic_loss
                    - self.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()

        self.buffer.clear()
        self.train_step += 1
        return {"loss": total_loss / max(1, n_samples // self.batch_size * self.ppo_epochs)}
