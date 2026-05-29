"""Visualization: training curves, convergence plots, agent comparison."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend

# Consistent color palette
AGENT_COLORS: Dict[str, str] = {
    "random": "#95a5a6",
    "hill_climb": "#3498db",
    "dqn": "#e74c3c",
    "ppo": "#2ecc71",
}

AGENT_MARKERS: Dict[str, str] = {
    "random": "s",
    "hill_climb": "D",
    "dqn": "o",
    "ppo": "^",
}


def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)


def plot_training_reward(
    results: List[dict],
    save_path: str = "results/training_reward.png",
    smooth: int = 10,
) -> Path:
    """Training reward vs episode, one subplot per function."""
    by_func: Dict[str, list] = {}
    for r in results:
        by_func.setdefault(r["function"], []).append(r)

    n_funcs = len(by_func)
    cols = min(2, n_funcs)
    rows = math.ceil(n_funcs / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), squeeze=False)

    for idx, (func, entries) in enumerate(sorted(by_func.items())):
        ax = axes[idx // cols][idx % cols]
        for entry in entries:
            agent_name = entry["agent"]
            color = AGENT_COLORS.get(agent_name, "#333333")
            rewards = np.array(entry["history"]["train_reward"])
            # Smooth with moving average
            if smooth > 1 and len(rewards) >= smooth:
                kernel = np.ones(smooth) / smooth
                rewards = np.convolve(rewards, kernel, mode="valid")
            xs = np.arange(len(rewards))
            ax.plot(xs, rewards, color=color, linewidth=1.2,
                    label=agent_name, alpha=0.85)
        _style_ax(ax, title=f"{func} (dim={entries[0]['dim']})",
                  xlabel="Episode", ylabel="Reward")
        ax.legend(fontsize=8)

    # Hide unused subplots
    for idx in range(n_funcs, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Path(save_path)


def plot_convergence(
    results: List[dict],
    save_path: str = "results/convergence.png",
) -> Path:
    """Distance-to-optimal vs training time, grouped by function."""
    by_func: Dict[str, list] = {}
    for r in results:
        by_func.setdefault(r["function"], []).append(r)

    n_funcs = len(by_func)
    cols = min(2, n_funcs)
    rows = math.ceil(n_funcs / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), squeeze=False)

    for idx, (func, entries) in enumerate(sorted(by_func.items())):
        ax = axes[idx // cols][idx % cols]
        best_dist = float("inf")
        for entry in entries:
            agent_name = entry["agent"]
            color = AGENT_COLORS.get(agent_name, "#333333")
            marker = AGENT_MARKERS.get(agent_name, "x")
            hist = entry["history"]
            # Use eval distance; fall back to final if no eval history
            dists = hist.get("eval_distance", [])
            times = hist.get("time", [])
            if dists and times:
                ax.plot(times, dists, color=color, linewidth=1.5,
                        marker=marker, markersize=4, markevery=max(1, len(dists) // 8),
                        label=agent_name, alpha=0.85)
                best_dist = min(best_dist, min(dists))
        # Set y-limit to show meaningful convergence range
        if best_dist < float("inf"):
            ax.set_ylim(bottom=-0.05 * best_dist,
                        top=min(best_dist * 5, ax.get_ylim()[1]))
        _style_ax(ax, title=f"{func} (dim={entries[0]['dim']})",
                  xlabel="Time (s)", ylabel="Distance to optimal")
        ax.legend(fontsize=8)

    for idx in range(n_funcs, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Path(save_path)


def plot_comparison_bars(
    results: List[dict],
    save_path: str = "results/comparison.png",
) -> Path:
    """Grouped bar chart: final distance-to-optimal per agent per function."""
    by_func: Dict[str, list] = {}
    for r in results:
        by_func.setdefault(r["function"], []).append(r)

    func_names = sorted(by_func.keys())
    agent_names = sorted(
        {r["agent"] for r in results},
        key=lambda a: ["random", "hill_climb", "dqn", "ppo"].index(a)
        if a in ["random", "hill_climb", "dqn", "ppo"] else 999,
    )

    x = np.arange(len(func_names))
    n_agents = len(agent_names)
    width = 0.8 / n_agents

    fig, ax = plt.subplots(figsize=(max(8, 3 * len(func_names)), 5))

    for i, agent_name in enumerate(agent_names):
        distances = []
        for func in func_names:
            entries = [r for r in results
                       if r["function"] == func and r["agent"] == agent_name]
            dist = entries[0]["distance"] if entries else 0
            distances.append(dist)
        color = AGENT_COLORS.get(agent_name, "#333333")
        bars = ax.bar(x + i * width, distances, width, label=agent_name,
                      color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        # Annotate values on bars
        for bar, val in zip(bars, distances):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(distances),
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x + width * (n_agents - 1) / 2)
    ax.set_xticklabels([f"{f}\n(dim={results[0]['dim']})" for f in func_names], fontsize=9)
    _style_ax(ax, title="Agent Comparison — Distance to Optimal (lower = better)",
              xlabel="", ylabel="Distance to Optimal")
    ax.legend(fontsize=9)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Path(save_path)


def plot_summary_panel(
    results: List[dict],
    save_path: str = "results/summary.png",
) -> Path:
    """All-in-one summary: rewards + convergence + bars."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # ---- Panel A: Training reward curves ----
    ax_a = fig.add_subplot(gs[0, :2])
    by_func_agent: Dict[str, list] = {}
    for r in results:
        key = r["function"]
        by_func_agent.setdefault(key, []).append(r)
    for func, entries in sorted(by_func_agent.items()):
        for entry in entries:
            agent_name = entry["agent"]
            color = AGENT_COLORS.get(agent_name, "#333333")
            rewards = np.array(entry["history"]["train_reward"])
            if len(rewards) >= 10:
                kernel = np.ones(10) / 10
                rewards = np.convolve(rewards, kernel, mode="valid")
            ax_a.plot(rewards, color=color, linewidth=1.0, alpha=0.7,
                      label=f"{func}/{agent_name}")
    _style_ax(ax_a, title="Training Reward (smoothed)", xlabel="Episode", ylabel="Reward")
    ax_a.legend(fontsize=6, ncol=2, loc="lower right")

    # ---- Panel B: Convergence ----
    ax_b = fig.add_subplot(gs[0, 2])
    for r in results:
        agent_name = r["agent"]
        color = AGENT_COLORS.get(agent_name, "#333333")
        dists = r["history"].get("eval_distance", [])
        times = r["history"].get("time", [])
        if dists and times:
            ax_b.plot(times, dists, color=color, linewidth=1.0, alpha=0.6,
                      label=f"{r['function']}/{agent_name}")
    _style_ax(ax_b, title="Distance to Optimal vs Time", xlabel="Time (s)", ylabel="Distance")
    ax_b.legend(fontsize=6, ncol=2, loc="upper right")

    # ---- Panel C: Bar chart comparison ----
    ax_c = fig.add_subplot(gs[1, :])
    # Group by function, then agent
    func_names = sorted({r["function"] for r in results})
    agent_names = sorted(
        {r["agent"] for r in results},
        key=lambda a: ["random", "hill_climb", "dqn", "ppo"].index(a)
        if a in ["random", "hill_climb", "dqn", "ppo"] else 999,
    )
    x = np.arange(len(func_names))
    width = 0.8 / len(agent_names)
    for i, agent_name in enumerate(agent_names):
        distances = []
        for func in func_names:
            entries = [r for r in results
                       if r["function"] == func and r["agent"] == agent_name]
            distances.append(entries[0]["distance"] if entries else 0)
        ax_c.bar(x + i * width, distances, width, label=agent_name,
                 color=AGENT_COLORS.get(agent_name, "#333333"),
                 alpha=0.85, edgecolor="white", linewidth=0.5)
    ax_c.set_xticks(x + width * (len(agent_names) - 1) / 2)
    ax_c.set_xticklabels(func_names, fontsize=9)
    _style_ax(ax_c, title=f"Final Distance to Optimal (dim={results[0]['dim']})",
              xlabel="", ylabel="Distance")
    ax_c.legend(fontsize=9)

    fig.suptitle("Surface Optimization RL Benchmark — Summary",
                 fontsize=14, fontweight="bold", y=0.98)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return Path(save_path)


def plot_all(results: List[dict], output_dir: str = "results") -> Dict[str, Path]:
    """Generate all plots. Returns dict of plot_name -> Path."""
    out = Path(output_dir)
    paths = {}
    paths["training_reward"] = plot_training_reward(results, str(out / "training_reward.png"))
    paths["convergence"] = plot_convergence(results, str(out / "convergence.png"))
    paths["comparison"] = plot_comparison_bars(results, str(out / "comparison.png"))
    paths["summary"] = plot_summary_panel(results, str(out / "summary.png"))
    return paths
