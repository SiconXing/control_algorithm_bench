"""Episode runner: train/eval a single agent on a single environment."""

import time
from typing import Optional

import numpy as np

from src.envs import make_env, SurfaceEnv
from src.agents import BaseAgent


def run_episode(
    env: SurfaceEnv,
    agent: BaseAgent,
    max_steps: Optional[int] = None,
    train: bool = True,
    render: bool = False,
) -> dict:
    """Run one episode. Returns metrics dict."""
    max_steps = max_steps or env._max_steps
    obs, _ = env.reset()
    total_reward = 0.0
    steps = 0
    best_value = float("inf") if "minimize" in str(env._env.mode) else float("-inf")

    for step in range(max_steps):
        raw_action = agent.select_action(obs, eval_mode=not train)

        # PPO returns (action_array, log_prob, value); others return plain action
        if isinstance(raw_action, tuple) and len(raw_action) == 3:
            act_arr, log_prob, value = raw_action
        else:
            act_arr = raw_action

        # Step the environment with the plain action array/ scalar
        if env._action_type == "continuous":
            next_obs, reward, done, _, info = env.step(act_arr)
        else:
            next_obs, reward, done, _, info = env.step(int(act_arr))

        # Store experience for training
        if hasattr(agent, "store_experience") and isinstance(raw_action, tuple) and len(raw_action) == 3:
            agent.store_experience(obs, act_arr, reward, log_prob, value, done)
        else:
            agent.store_transition(obs, act_arr, reward, next_obs, done)
        total_reward += reward
        steps += 1

        # Track best function value found
        fval = info["function_value"]
        if "minimize" in str(env._env.mode):
            if fval < best_value:
                best_value = fval
        else:
            if fval > best_value:
                best_value = fval

        obs = next_obs
        if done:
            break

    distance = abs(best_value - env.optimal_value)
    return {
        "total_reward": total_reward,
        "steps": steps,
        "best_value": best_value,
        "optimal_value": env.optimal_value,
        "distance_to_optimal": distance,
    }


def train_agent(
    env: SurfaceEnv,
    agent: BaseAgent,
    episodes: int = 500,
    eval_every: int = 50,
    eval_episodes: int = 5,
    verbose: bool = True,
) -> dict:
    """Train an agent and evaluate periodically."""
    history = {"train_reward": [], "eval_reward": [], "eval_distance": [], "time": []}
    t_start = time.time()

    for ep in range(1, episodes + 1):
        metrics = run_episode(env, agent, train=True)
        history["train_reward"].append(metrics["total_reward"])

        if ep % eval_every == 0:
            eval_metrics = evaluate_agent(env, agent, n_episodes=eval_episodes)
            history["eval_reward"].append(eval_metrics["mean_reward"])
            history["eval_distance"].append(eval_metrics["mean_distance"])
            history["time"].append(time.time() - t_start)

            if verbose:
                print(
                    f"  Ep {ep:5d}/{episodes}  "
                    f"train_r={metrics['total_reward']:8.2f}  "
                    f"eval_r={eval_metrics['mean_reward']:8.2f}  "
                    f"best_dist={eval_metrics['mean_distance']:.6f}  "
                    f"time={history['time'][-1]:6.1f}s"
                )

        # Agent update (DQN does online updates, PPO updates after buffer fills)

    # Final evaluation
    final = evaluate_agent(env, agent, n_episodes=10)
    history["final"] = final
    history["total_time"] = time.time() - t_start
    return history


def evaluate_agent(env: SurfaceEnv, agent: BaseAgent, n_episodes: int = 10) -> dict:
    """Evaluate agent over multiple episodes (no training)."""
    rewards, distances, best_values = [], [], []

    for _ in range(n_episodes):
        metrics = run_episode(env, agent, train=False)
        rewards.append(metrics["total_reward"])
        distances.append(metrics["distance_to_optimal"])
        best_values.append(metrics["best_value"])

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_distance": float(np.mean(distances)),
        "std_distance": float(np.std(distances)),
        "mean_best_value": float(np.mean(best_values)),
    }
