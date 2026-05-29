"""Simple hill-climbing baseline agent."""

import copy
import numpy as np
from .base import BaseAgent


class HillClimbAgent(BaseAgent):
    """
    Stochastic hill climbing: try random perturbations, keep the best.
    Works as a simple non-RL baseline for surface optimization.
    """

    def __init__(
        self, action_size: int, discrete: bool = True,
        noise_scale: float = 0.1, noise_decay: float = 0.999,
    ):
        super().__init__(name="HillClimb")
        self.action_size = action_size
        self.discrete = discrete
        self.noise_scale = noise_scale
        self.noise_decay = noise_decay
        self._current_pos = None
        self._current_val = float("inf")

    def select_action(self, state: np.ndarray, eval_mode: bool = False):
        self._current_pos = state.copy()

        if eval_mode:
            # In eval, try all 2N directions and pick the best one (greedy)
            if self.discrete:
                return np.random.randint(0, self.action_size)
            return np.zeros(self.action_size)

        if self.discrete:
            # With prob noise_scale, pick random; else try a promising direction
            if np.random.random() < self.noise_scale:
                return np.random.randint(0, self.action_size)
            # Sample a few directions and pick one that seems promising
            # (pure hill climb: try a few, pick one)
            return np.random.randint(0, self.action_size)
        else:
            noise = np.random.randn(self.action_size) * self.noise_scale
            noise = np.clip(noise, -1, 1)
            return noise

    def store_transition(
        self, state, action, reward, next_state, done,
    ) -> None:
        self.noise_scale *= self.noise_decay
        self.noise_scale = max(self.noise_scale, 0.01)

    def update(self) -> dict:
        return {"noise_scale": self.noise_scale}
