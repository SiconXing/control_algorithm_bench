"""
High-dimensional surface optimization RL environment.

Provides a Gymnasium-compatible interface wrapping a C++/pybind11 backend.
The agent navigates an N-dimensional surface to find its minimum or maximum.

Functions available (9 total):
    sphere          - Simple convex quadratic
    rastrigin       - Highly multimodal with cosine modulation
    ackley          - Flat outer region, steep central hole
    rosenbrock      - Banana-shaped valley
    griewank        - Product-of-cos + quadratic
    levy            - Complex multimodal
    michalewicz     - Steep narrow valleys
    custom_sin      - User-defined sinusoidal surface
    quadratic_well  - Shifted quadratic well

Usage:
    from envs import make_env

    env = make_env("rastrigin", dim=5, mode="minimize")
    obs, info = env.reset()
    for _ in range(200):
        action = env.action_space.sample()  # or your policy
        obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any, List

import numpy as np

# Ensure the compiled .pyd/.so is discoverable
_pkg_dir = Path(__file__).parent
if str(_pkg_dir) not in sys.path:
    sys.path.insert(0, str(_pkg_dir))

if TYPE_CHECKING:
    # Placeholder so static analyzers see the module name.
    # The real .pyd/.so is loaded at runtime via the "else" branch.
    class _CppModule:
        SurfaceType: Any = None
        OptMode: Any = None
        SurfaceOptimizationEnv: Any = None

    _cpp: _CppModule = _CppModule()
else:
    from . import _surface_env_cpp as _cpp


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
SURFACE_TYPES = {
    "sphere": _cpp.SurfaceType.SPHERE,
    "rastrigin": _cpp.SurfaceType.RASTRIGIN,
    "ackley": _cpp.SurfaceType.ACKLEY,
    "rosenbrock": _cpp.SurfaceType.ROSENBROCK,
    "griewank": _cpp.SurfaceType.GRIEWANK,
    "levy": _cpp.SurfaceType.LEVY,
    "michalewicz": _cpp.SurfaceType.MICHALEWICZ,
    "custom_sin": _cpp.SurfaceType.CUSTOM_SIN,
    "quadratic_well": _cpp.SurfaceType.QUADRATIC_WELL,
}

OPT_MODES = {
    "minimize": _cpp.OptMode.MINIMIZE,
    "maximize": _cpp.OptMode.MAXIMIZE,
}

# Known function characteristics for reference
FUNCTION_INFO = {
    "sphere": {
        "optimum": 0.0,
        "opt_point": "origin",
        "typical_bounds": (-5.12, 5.12),
        "difficulty": "easy",
    },
    "rastrigin": {
        "optimum": 0.0,
        "opt_point": "origin",
        "typical_bounds": (-5.12, 5.12),
        "difficulty": "hard",
    },
    "ackley": {
        "optimum": 0.0,
        "opt_point": "origin",
        "typical_bounds": (-32.768, 32.768),
        "difficulty": "medium",
    },
    "rosenbrock": {
        "optimum": 0.0,
        "opt_point": "all-ones",
        "typical_bounds": (-2.048, 2.048),
        "difficulty": "hard",
    },
    "griewank": {
        "optimum": 0.0,
        "opt_point": "origin",
        "typical_bounds": (-600, 600),
        "difficulty": "medium",
    },
    "levy": {
        "optimum": 0.0,
        "opt_point": "all-ones",
        "typical_bounds": (-10, 10),
        "difficulty": "hard",
    },
    "michalewicz": {
        "optimum": "varies with dim",
        "opt_point": "varies with dim",
        "typical_bounds": (0, 3.1416),
        "difficulty": "hard",
    },
    "custom_sin": {
        "optimum": "user-defined",
        "opt_point": "user-defined",
        "typical_bounds": (-5, 5),
        "difficulty": "custom",
    },
    "quadratic_well": {
        "optimum": 0.0,
        "opt_point": "bounds center",
        "typical_bounds": (-5, 5),
        "difficulty": "easy",
    },
}


def list_functions() -> dict:
    """Return metadata for all available benchmark functions."""
    return dict(FUNCTION_INFO)


# ---------------------------------------------------------------------------
# Gymnasium-style environment
# ---------------------------------------------------------------------------
class SurfaceEnv:
    """
    Gymnasium-compatible RL environment for high-dimensional surface
    optimization.

    State space:  N-dimensional continuous position
    Action space: 2N discrete directional movements, or N-dimensional
                  continuous vector

    Parameters
    ----------
    func : str
        Name of the benchmark function (see SURFACE_TYPES).
    dim : int
        Number of dimensions (N >= 1).
    mode : str
        "minimize" or "maximize".
    step_size : float
        Magnitude of each step.
    bounds : tuple[float, float]
        (low, high) domain bounds.
    max_steps : int
        Max steps per episode.
    convergence_tol : float
        Early-stop when |f(x) - optimal| < tol.
    action_type : str
        "discrete" (2N actions) or "continuous" (N-dim vector).
    seed : int
        Random seed (0 = random).
    **kwargs
        Passed to the C++ constructor.
    """

    def __init__(
        self,
        func: str = "sphere",
        dim: int = 2,
        mode: str = "minimize",
        step_size: float = 0.1,
        bounds: Tuple[float, float] = (-5.0, 5.0),
        max_steps: int = 200,
        convergence_tol: float = 1e-4,
        action_type: str = "discrete",
        seed: int = 0,
        **kwargs,
    ):
        if func not in SURFACE_TYPES:
            raise ValueError(
                f"Unknown function '{func}'. Available: {list(SURFACE_TYPES)}"
            )
        if mode not in OPT_MODES:
            raise ValueError(f"Unknown mode '{mode}'. Use 'minimize' or 'maximize'.")

        self._func_name = func
        self._dim = dim
        self._mode_str = mode
        self._action_type = action_type
        self._max_steps = max_steps

        self._env = _cpp.SurfaceOptimizationEnv(
            dim,
            SURFACE_TYPES[func],
            OPT_MODES[mode],
            step_size,
            bounds[0],
            bounds[1],
            convergence_tol,
            max_steps,
            seed,
        )

        # Build observation & action spaces
        low = np.full(dim, bounds[0], dtype=np.float32)
        high = np.full(dim, bounds[1], dtype=np.float32)
        self.observation_space = _Box(low, high)

        if action_type == "discrete":
            self.action_space = _Discrete(2 * dim)
        else:
            self.action_space = _Box(
                -np.ones(dim, dtype=np.float32),
                np.ones(dim, dtype=np.float32),
            )

    # -- properties ----------------------------------------------------------
    @property
    def optimal_value(self) -> float:
        """The global optimum (min or max) value of the surface."""
        return self._env.optimal_value

    @property
    def optimal_position(self) -> np.ndarray:
        """The position of the global optimum."""
        return np.array(self._env.optimal_position, dtype=np.float64)

    @property
    def position(self) -> np.ndarray:
        """Current agent position."""
        return np.array(self._env.position, dtype=np.float64)

    @property
    def function_name(self) -> str:
        return self._env.function_name

    # -- RL interface --------------------------------------------------------
    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        obs = self._env.reset()
        return np.array(obs, dtype=np.float32), {}

    def step(
        self, action
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        if self._action_type == "continuous":
            action = np.asarray(action, dtype=np.float64)
            obs, reward, done, fval = self._env.step_continuous(action)
        else:
            obs, reward, done, fval = self._env.step(int(action))

        info = {
            "function_value": fval,
            "distance_to_optimal": abs(fval - self._env.optimal_value),
        }
        return np.array(obs, dtype=np.float32), float(reward), done, False, info

    # -- helpers -------------------------------------------------------------
    def set_custom_coefficients(self, coeffs: List[float]) -> None:
        """Set coefficients for 'custom_sin' surface."""
        self._env.set_custom_coefficients(list(coeffs))

    def set_custom_optimal(
        self, position: List[float], value: float
    ) -> None:
        """Manually set the optimal point/value for 'custom_sin'."""
        self._env.set_custom_optimal(list(position), float(value))

    def close(self) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"SurfaceEnv(func={self._func_name!r}, dim={self._dim}, "
            f"mode={self._mode_str!r}, action={self._action_type})"
        )


# ---------------------------------------------------------------------------
# Minimal space stubs (avoid gymnasium dependency when not needed)
# ---------------------------------------------------------------------------
class _Box:
    def __init__(self, low, high, dtype=np.float32):
        self.low = np.asarray(low, dtype=dtype)
        self.high = np.asarray(high, dtype=dtype)
        self.shape = self.low.shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(self.dtype)

    def contains(self, x) -> bool:
        x = np.asarray(x)
        return bool(np.all(x >= self.low) and np.all(x <= self.high))

    def __repr__(self):
        return f"Box({self.low}, {self.high})"


class _Discrete:
    def __init__(self, n: int):
        self.n = n
        self.shape = ()

    def sample(self) -> int:
        return np.random.randint(0, self.n)

    def contains(self, x) -> bool:
        return 0 <= int(x) < self.n

    def __repr__(self):
        return f"Discrete({self.n})"


# ---------------------------------------------------------------------------
# Gymnasium wrapper (requires: pip install gymnasium)
# ---------------------------------------------------------------------------
def make_gym_env(func: str = "sphere", dim: int = 2, **kwargs):
    """Create a full gymnasium.Env (requires gymnasium installed)."""
    try:
        import gymnasium as gym
    except ImportError:
        raise ImportError("gymnasium is required. Install with: pip install gymnasium")

    class _GymSurfaceEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self._env = SurfaceEnv(func=func, dim=dim, **kwargs)
            self.observation_space = gym.spaces.Box(
                low=self._env.observation_space.low,
                high=self._env.observation_space.high,
                dtype=np.float32,
            )
            if self._env._action_type == "discrete":
                self.action_space = gym.spaces.Discrete(2 * dim)
            else:
                self.action_space = gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(dim,), dtype=np.float32,
                )

        def reset(self, *, seed=None, options=None):
            obs, info = self._env.reset(seed=seed, options=options)
            return obs, info

        def step(self, action):
            obs, reward, terminated, truncated, info = self._env.step(action)
            return obs, reward, terminated, truncated, info

        def render(self):
            pass

        def close(self):
            self._env.close()

    return _GymSurfaceEnv()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def make_env(
    func: str = "sphere",
    dim: int = 2,
    mode: str = "minimize",
    step_size: float = 0.1,
    bounds: Tuple[float, float] = (-5.0, 5.0),
    max_steps: int = 200,
    action_type: str = "discrete",
    seed: int = 0,
    gymnasium: bool = False,
    **kwargs,
) -> SurfaceEnv:
    """
    Create a surface optimization environment.

    Parameters
    ----------
    func : str
        Benchmark function name.
    dim : int
        Dimensionality (>= 1).
    mode : str
        "minimize" or "maximize".
    step_size : float
        Step magnitude per action.
    bounds : tuple
        (low, high) for each dimension.
    max_steps : int
        Episode length limit.
    action_type : str
        "discrete" (2N directions) or "continuous" (N-dim vector).
    seed : int
        Random seed.
    gymnasium : bool
        If True, return a gymnasium.Env (requires gymnasium installed).

    Returns
    -------
    SurfaceEnv or gymnasium.Env
    """
    # if gymnasium:
    #     return make_gym_env(
    #         func=func, dim=dim, mode=mode, step_size=step_size,
    #         bounds=bounds, max_steps=max_steps, action_type=action_type,
    #         seed=seed, **kwargs,
    #     )
    return SurfaceEnv(
        func=func, dim=dim, mode=mode, step_size=step_size,
        bounds=bounds, max_steps=max_steps, action_type=action_type,
        seed=seed, **kwargs,
    )
