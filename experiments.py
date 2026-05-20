"""
Batch experiment runner for function-fitting RL benchmarks.

Usage:
    python experiments.py              # Full experiments (all functions)
    python experiments.py --quick      # Quick mode (300 episodes per function)
    python experiments.py --func step_single  # Single function test
"""
import argparse
import os
import sys
import time

import numpy as np
import function_env

from src.piecewise_functions import FUNCTION_LIST, PY_FUNCTIONS
from src.agent_factory import DQNAgent
from train import train, evaluate_function_fit, plot_function_fit


def run_single_experiment(func_id, episodes=800, verbose=True, model_dir="models"):
    """Train and evaluate on a single function. Returns results dict."""
    env = function_env.FunctionFittingEnv(func_id, n_actions=100, episode_length=200)
    func_name = env.get_function_name()
    func_meta = next(f for f in FUNCTION_LIST if f["id"] == func_id)

    agent = DQNAgent(
        state_size=env.get_state_size(),
        action_size=env.get_action_size(),
        hidden_layers=[64, 64],
        lr=0.001,
        gamma=0.99,
        batch_size=64,
        target_update=20,
    )

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Function: {func_name}  ({func_meta['difficulty']})")
        print(f"  Episodes: {episodes}")
        print(f"{'='*60}")

    start = time.time()
    agent, rewards, losses = train(env, agent, episodes=episodes, verbose=verbose, model_dir=model_dir)
    elapsed = time.time() - start

    xs, y_true, y_pred, mae = evaluate_function_fit(env, agent, n_samples=1000)

    # Convergence: episode where 100-ep moving avg first reaches % of final
    if len(rewards) >= 100:
        smooth = np.convolve(rewards, np.ones(100) / 100, mode="valid")
        final_avg = smooth[-1]
        threshold = final_avg * 0.9
        converged_at = next((i for i, v in enumerate(smooth) if v >= threshold), len(smooth))
    else:
        converged_at = len(rewards)

    result = {
        "func_name": func_name,
        "difficulty": func_meta["difficulty"],
        "mae": mae,
        "best_reward": max(rewards) if rewards else 0,
        "final_avg_reward": np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards),
        "converged_at": converged_at + 100,  # offset for smoothing window
        "total_episodes": episodes,
        "elapsed_sec": elapsed,
        "rewards": rewards,
        "losses": losses,
        "xs": xs,
        "y_true": y_true,
        "y_pred": y_pred,
    }

    return result


def run_batch(episodes=800, quick=False, func_filter=None):
    """Run experiments for all (or filtered) functions."""
    if quick:
        episodes = 300

    funcs = FUNCTION_LIST
    if func_filter:
        funcs = [f for f in FUNCTION_LIST if f["name"] == func_filter]
        if not funcs:
            print(f"Unknown function: {func_filter}")
            print(f"Available: {[f['name'] for f in FUNCTION_LIST]}")
            sys.exit(1)

    results = []
    for func_info in funcs:
        result = run_single_experiment(func_info["id"], episodes=episodes)
        results.append(result)

        # Save individual function plot
        plot_function_fit(
            result["xs"], result["y_true"], result["y_pred"],
            result["mae"], result["func_name"],
            save_path=f"results/{result['func_name']}.png"
        )

    # ── Print summary table ──────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"{'Function':<25} {'Difficulty':<10} {'MAE':>8} {'BestReward':>12} {'Conv@Ep':>8} {'Time(s)':>8}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["mae"]):
        print(f"{r['func_name']:<25} {r['difficulty']:<10} {r['mae']:>8.4f} {r['best_reward']:>12.2f} {r['converged_at']:>8} {r['elapsed_sec']:>8.1f}")

    # ── Summary plot ─────────────────────────────────────────────────
    plot_summary(results)

    return results


def plot_summary(results):
    """Grid plot of all functions: true curve vs agent prediction."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        n = len(results)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5))
        axes = axes.flatten() if n > 1 else [axes]

        for i, r in enumerate(results):
            ax = axes[i]
            ax.plot(r["xs"], r["y_true"], color="black", linewidth=1.5, label="true")
            ax.scatter(r["xs"][::30], r["y_pred"][::30], color="tab:red", s=5, alpha=0.5, label="pred")
            ax.set_title(f"{r['func_name']}  MAE={r['mae']:.3f}", fontsize=9)
            ax.set_xlim(0, 1)
            if i >= (rows - 1) * cols:
                ax.set_xlabel("x")
            if i % cols == 0:
                ax.set_ylabel("y")
            ax.grid(True, alpha=0.2)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("Function Fitting Results — True vs Agent Prediction", fontsize=13, y=1.01)
        os.makedirs("results", exist_ok=True)
        plt.tight_layout()
        plt.savefig("results/summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Summary plot saved: results/summary.png")
    except ImportError:
        print("  (install matplotlib for summary plot)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Function-fitting RL benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick mode (300 episodes)")
    parser.add_argument("--func", type=str, default=None, help="Run a single function by name")
    parser.add_argument("--episodes", type=int, default=800, help="Number of training episodes")
    args = parser.parse_args()

    run_batch(episodes=args.episodes, quick=args.quick, func_filter=args.func)
