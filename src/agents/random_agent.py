import numpy as np
from .base import BaseAgent


class RandomAgent(BaseAgent):
    """Uniform random action baseline."""

    def __init__(self, action_size: int, discrete: bool = True):
        super().__init__(name="Random")
        self.action_size = action_size
        self.discrete = discrete

    def select_action(self, state: np.ndarray, eval_mode: bool = False):
        if self.discrete:
            return np.random.randint(0, self.action_size)
        return np.random.uniform(-1, 1, size=self.action_size)

    def update(self) -> dict:
        return {}
