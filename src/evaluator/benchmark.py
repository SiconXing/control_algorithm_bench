"""Orchestrate batch benchmarks across functions, dimensions, and agents."""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.envs import make_env, list_functions, SurfaceEnv
from src.agents import (
    BaseAgent,
    RandomAgent,
    HillClimbAgent,
    DQNAgent,
    PPOAgent,
)
from .runner import train_agent, evaluate_agent


def _create_agent(
    name: str, env: SurfaceEnv, **kwargs
) -> BaseAgent:
    """Agent factory."""
    state_dim = env._dim
    action_dim = env.action_space.n if env._action_type == "discrete" else env._dim
    continuous = env._action_type == "continuous"

    if name == "random":
        return RandomAgent(action_dim, discrete=not continuous)
    elif name == "hill_climb":
        return HillClimbAgent(action_dim, discrete=not continuous, **kwargs)
    elif name == "dqn":
        return DQNAgent(state_dim, action_dim, **kwargs)
    elif name == "ppo":
        return PPOAgent(state_dim, action_dim, continuous=continuous, **kwargs)
    else:
        raise ValueError(f"Unknown agent: {name}")


class BenchmarkSuite:
    """Run a set of benchmarks and tabulate results."""

    def __init__(self, episodes: int = 500, verbose: bool = True):
        self.episodes = episodes
        self.verbose = verbose
        self.results: List[dict] = []

    def run(
        self,
        functions: Optional[List[str]] = None,
        dims: Optional[List[int]] = None,
        agents: Optional[List[str]] = None,
        modes: Optional[List[str]] = None,
        **env_kwargs,
    ) -> List[dict]:
        """Run benchmark across all combinations."""
        functions = functions or ["sphere", "rastrigin", "ackley"]
        dims = dims or [3, 5]
        agents_list = agents or ["random", "dqn", "ppo"]
        modes = modes or ["minimize"]
        self.results = []

        total_n = len(functions) * len(dims) * len(agents_list) * len(modes)
        count = 0

        for func in functions:
            for dim in dims:
                for mode in modes:
                    for agent_name in agents_list:
                        count += 1

                        action_type = "discrete" if agent_name != "ppo" else "continuous"
                        if agent_name == "dqn":
                            action_type = "discrete"

                        info = list_functions()[func]
                        bounds = info.get("typical_bounds", (-5.0, 5.0))

                        env = make_env(
                            func, dim=dim, mode=mode,
                            bounds=bounds, max_steps=200,
                            action_type=action_type,
                        )

                        agent = _create_agent(agent_name, env)

                        title = f"[{count}/{total_n}] {func} dim={dim} {mode} {agent_name}"
                        if self.verbose:
                            print(f"\n{'='*60}")
                            print(title)
                            print(f"Optimal value: {env.optimal_value:.6f}")
                            print(f"Action space: {env.action_space}")
                            print(f"{'='*60}")

                        t0 = time.time()
                        history = train_agent(
                            env, agent, episodes=self.episodes,
                            eval_every=max(10, self.episodes // 10),
                            verbose=self.verbose,
                        )
                        elapsed = time.time() - t0

                        final = history["final"]
                        self.results.append({
                            "function": func,
                            "dim": dim,
                            "mode": mode,
                            "agent": agent_name,
                            "optimal_value": env.optimal_value,
                            "best_value": final["mean_best_value"],
                            "distance": final["mean_distance"],
                            "eval_reward": final["mean_reward"],
                            "elapsed_sec": elapsed,
                            "history": history,
                        })

                        env.close()

        return self.results

    def summary(self) -> str:
        """Print a summary table of results."""
        if not self.results:
            return "No results yet."

        lines = []
        header = (
            f"{'Function':<20s} {'Dim':>4s} {'Mode':>9s} {'Agent':>10s} "
            f"{'Optimal':>10s} {'BestFound':>12s} {'Distance':>10s} {'Time':>8s}"
        )
        sep = "-" * len(header)
        lines.extend([sep, header, sep])

        for r in self.results:
            lines.append(
                f"{r['function']:<20s} {r['dim']:>4d} {r['mode']:>9s} "
                f"{r['agent']:>10s} {r['optimal_value']:>10.4f} "
                f"{r['best_value']:>12.4f} {r['distance']:>10.4f} "
                f"{r['elapsed_sec']:>7.1f}s"
            )
        lines.append(sep)
        return "\n".join(lines)

    def plot(self, output_dir: str = "results") -> dict:
        """Generate all visualization plots. Returns dict of plot_name -> Path."""
        from .visualize import plot_all
        return plot_all(self.results, output_dir)

    def to_dataframe(self):
        """Return results as a pandas DataFrame."""
        try:
            import pandas as pd
            rows = []
            for r in self.results:
                row = dict(r)
                del row["history"]
                rows.append(row)
            return pd.DataFrame(rows)
        except ImportError:
            return None
