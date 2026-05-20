"""
Function-Fitting RL Benchmark — Entry Point

Usage:
    python main.py                        Single function demo (step_single)
    python main.py --func step_multi      Train on a specific function
    python main.py --batch                Run batch experiments (all functions)
    python main.py --batch --quick        Quick batch experiment
"""
import argparse
import os
import sys

import function_env

from src.piecewise_functions import FUNCTION_LIST, FUNCTION_BY_NAME
from src.agent_factory import DQNAgent
from train import train, evaluate_function_fit, plot_progress, plot_function_fit


def run_demo(func_name="step_single", episodes=800):
    """Single function training demo."""
    func_meta = FUNCTION_BY_NAME.get(func_name)
    if func_meta is None:
        print(f"Unknown function: {func_name}")
        print(f"Available: {[f['name'] for f in FUNCTION_LIST]}")
        sys.exit(1)

    env = function_env.FunctionFittingEnv(func_meta["id"], n_actions=100, episode_length=200)
    print(f"\nFunction-fitting RL demo")
    print(f"  Function: {env.get_function_name()}  ({func_meta['difficulty']})")
    print(f"  State size: {env.get_state_size()}")
    print(f"  Action size: {env.get_action_size()}  (y ∈ [{env.get_y_min():.3f}, {env.get_y_max():.3f}])")

    agent = DQNAgent(
        state_size=env.get_state_size(),
        action_size=env.get_action_size(),
        hidden_layers=[64, 64],
    )

    agent, rewards, losses = train(env, agent, episodes=episodes)

    # Evaluate function fit
    xs, y_true, y_pred, mae = evaluate_function_fit(env, agent, n_samples=1000)
    print(f"\n  Function fit MAE: {mae:.4f}")

    # Save plots
    os.makedirs("results", exist_ok=True)
    plot_progress(rewards, losses, save_path="results/training_progress.png")
    plot_function_fit(xs, y_true, y_pred, mae, env.get_function_name(),
                      save_path=f"results/{env.get_function_name()}.png")
    print(f"  Plots saved: results/training_progress.png, results/{env.get_function_name()}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Function-fitting RL benchmark")
    parser.add_argument("--func", type=str, default="step_single",
                        help="Function name to train on (default: step_single)")
    parser.add_argument("--batch", action="store_true", help="Run batch experiments")
    parser.add_argument("--quick", action="store_true", help="Quick mode for batch experiments")
    parser.add_argument("--episodes", type=int, default=800, help="Number of training episodes")
    args = parser.parse_args()

    if args.batch:
        from experiments import run_batch
        run_batch(episodes=args.episodes, quick=args.quick)
    else:
        run_demo(func_name=args.func, episodes=args.episodes)
