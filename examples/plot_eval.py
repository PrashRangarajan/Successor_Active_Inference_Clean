"""Plotting functions for evaluation benchmarks.

Generates paper-quality figures from .npy data files produced by run_eval.py.

Directory structure:
    data/eval/<env>/       — .npy data files
    figures/eval/<env>/    — output figures

Can be run standalone:
    python examples/plot_eval.py

Or imported by run_eval.py / run_eval_acrobot.py / run_eval_mountaincar.py.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import json
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-poster")


# ==================== Helpers ====================


def get_distance(grid_size, walls, start, goal):
    """BFS shortest-path distance on gridworld respecting walls.

    Args:
        grid_size: Size of the grid
        walls: List of (x, y) wall positions
        start: (x, y) start position
        goal: (x, y) goal position

    Returns:
        Integer distance, or -1 if unreachable
    """
    wall_set = set(walls)
    if start in wall_set or goal in wall_set:
        return -1

    # BFS
    queue = deque([(start, 0)])
    visited = {start}

    # Actions: left, right, up, down
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        (x, y), dist = queue.popleft()
        if (x, y) == goal:
            return dist

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < grid_size and 0 <= ny < grid_size
                    and (nx, ny) not in wall_set and (nx, ny) not in visited):
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))

    return -1  # Unreachable


# ==================== Plot Functions ====================


def plot_SR_rewards(args, data_dir="data/eval/gridworld", save_dir="figures/eval/gridworld"):
    """Plot reward curves with confidence bands (Hierarchy vs Flat vs Q-Learning)."""
    os.makedirs(save_dir, exist_ok=True)
    eps_range = args.episodes

    SR_rewards = np.load(os.path.join(data_dir, "SR_rewards_hierarchy.npy"))[:, :len(eps_range)]
    mean_SR_rewards = np.mean(SR_rewards, axis=0)
    std_SR_rewards = np.std(SR_rewards, axis=0) / np.sqrt(len(SR_rewards))

    SR_rewards2 = np.load(os.path.join(data_dir, "SR_rewards_flat.npy"))[:, :len(eps_range)]
    mean_SR_rewards2 = np.mean(SR_rewards2, axis=0)
    std_SR_rewards2 = np.std(SR_rewards2, axis=0) / np.sqrt(len(SR_rewards2))

    # Load Q-learning rewards if available
    Q_rewards_path = os.path.join(data_dir, "Q_rewards.npy")
    has_q_learning = os.path.exists(Q_rewards_path)
    if has_q_learning:
        Q_rewards = np.load(Q_rewards_path)[:, :len(eps_range)]
        mean_Q_rewards = np.mean(Q_rewards, axis=0)
        std_Q_rewards = np.std(Q_rewards, axis=0) / np.sqrt(len(Q_rewards))

    fig = plt.figure(figsize=(14, 10))
    plt.plot(eps_range, mean_SR_rewards, label="Hierarchy")
    plt.fill_between(
        eps_range,
        mean_SR_rewards - std_SR_rewards,
        mean_SR_rewards + std_SR_rewards,
        alpha=0.5,
    )
    plt.plot(eps_range, mean_SR_rewards2, label="Flat")
    plt.fill_between(
        eps_range,
        mean_SR_rewards2 - std_SR_rewards2,
        mean_SR_rewards2 + std_SR_rewards2,
        alpha=0.5,
    )
    if has_q_learning:
        plt.plot(eps_range, mean_Q_rewards, label="Q-Learning")
        plt.fill_between(
            eps_range,
            mean_Q_rewards - std_Q_rewards,
            mean_Q_rewards + std_Q_rewards,
            alpha=0.5,
        )

    plt.xlabel("Number of Training Episodes", fontsize=28)
    plt.ylabel("Total Reward", fontsize=28)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "reward_obtained.png"), format="png")
    plt.close()
    print(f"  Saved {save_dir}/reward_obtained.png")


def plot_SR_values(args, data_dir="data/eval/gridworld", save_dir="figures/eval/gridworld"):
    """Plot SR convergence: goal value distance + successor matrix distance."""
    os.makedirs(save_dir, exist_ok=True)
    eps_range = args.episodes

    SR_values = np.load(os.path.join(data_dir, "SR_values_hierarchy.npy"))[:, :len(eps_range)]
    SR_values2 = np.load(os.path.join(data_dir, "SR_values_flat.npy"))[:, :len(eps_range)]
    SR_succ = np.load(os.path.join(data_dir, "SR_succ_hierarchy.npy"))[:, :len(eps_range)]
    SR_succ2 = np.load(os.path.join(data_dir, "SR_succ_flat.npy"))[:, :len(eps_range)]
    SR_succ_macro = np.load(os.path.join(data_dir, "SR_succ_macro.npy"))[:, :len(eps_range)]

    # Goal value distance plot
    mean_SR_values = np.mean(SR_values, axis=0)
    mean_SR_values2 = np.mean(SR_values2, axis=0)
    std_SR_values = np.std(SR_values, axis=0) / np.sqrt(len(SR_values))
    std_SR_values2 = np.std(SR_values2, axis=0) / np.sqrt(len(SR_values2))

    fig = plt.figure(figsize=(14, 10))
    plt.plot(eps_range, mean_SR_values, label="Macro")
    plt.fill_between(
        eps_range,
        mean_SR_values - std_SR_values,
        mean_SR_values + std_SR_values,
        alpha=0.5,
    )
    plt.plot(eps_range, mean_SR_values2, label="Micro")
    plt.fill_between(
        eps_range,
        mean_SR_values2 - std_SR_values2,
        mean_SR_values2 + std_SR_values2,
        alpha=0.5,
    )
    plt.xlabel("Number of Training Episodes", fontsize=28)
    plt.ylabel("Value distance", fontsize=28)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "goal_values.png"), format="png")
    plt.close()
    print(f"  Saved {save_dir}/goal_values.png")

    # Successor matrix distance plot
    mean_SR_succ2 = np.mean(SR_succ2, axis=0)
    mean_SR_succ_macro = np.mean(SR_succ_macro, axis=0)
    std_SR_succ2 = np.std(SR_succ2, axis=0) / np.sqrt(len(SR_succ2))
    std_SR_succ_macro = np.std(SR_succ_macro, axis=0) / np.sqrt(len(SR_succ_macro))

    fig = plt.figure(figsize=(14, 10))
    plt.plot(eps_range, mean_SR_succ2, label="Micro")
    plt.fill_between(
        eps_range,
        mean_SR_succ2 - std_SR_succ2,
        mean_SR_succ2 + std_SR_succ2,
        alpha=0.5,
    )
    plt.plot(eps_range, mean_SR_succ_macro, label="Macro")
    plt.fill_between(
        eps_range,
        mean_SR_succ_macro - std_SR_succ_macro,
        mean_SR_succ_macro + std_SR_succ_macro,
        alpha=0.5,
    )
    plt.xlabel("Number of Training Episodes", fontsize=28)
    plt.ylabel("Successor distance", fontsize=28)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "successor_values.png"), format="png")
    plt.close()
    print(f"  Saved {save_dir}/successor_values.png")


def plot_SR_steps(args, data_dir="data/eval/gridworld", save_dir="figures/eval/gridworld"):
    """Plot number of steps to success."""
    os.makedirs(save_dir, exist_ok=True)
    eps_range = args.episodes

    SR_steps = np.load(os.path.join(data_dir, "SR_steps_hierarchy.npy"))
    SR_steps2 = np.load(os.path.join(data_dir, "SR_steps_flat.npy"))
    mean_SR_steps = np.mean(SR_steps, axis=0)
    mean_SR_steps2 = np.mean(SR_steps2, axis=0)
    std_SR_steps = np.std(SR_steps, axis=0) / np.sqrt(len(SR_steps))
    std_SR_steps2 = np.std(SR_steps2, axis=0) / np.sqrt(len(SR_steps2))

    fig = plt.figure(figsize=(14, 10))
    plt.plot(eps_range, mean_SR_steps, label="Hierarchy")
    plt.fill_between(
        eps_range,
        mean_SR_steps - std_SR_steps,
        mean_SR_steps + std_SR_steps,
        alpha=0.5,
    )
    plt.plot(eps_range, mean_SR_steps2, label="Flat")
    plt.fill_between(
        eps_range,
        mean_SR_steps2 - std_SR_steps2,
        mean_SR_steps2 + std_SR_steps2,
        alpha=0.5,
    )
    plt.title("Number of Steps to Success", fontsize=30)
    plt.xlabel("Training Episodes", fontsize=28)
    plt.ylabel("Mean Steps", fontsize=28)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "num_steps.png"), format="png")
    plt.close()
    print(f"  Saved {save_dir}/num_steps.png")


def plot_SR_times(args, data_dir="data/eval/gridworld", save_dir="figures/eval/gridworld"):
    """Plot processing time comparison."""
    os.makedirs(save_dir, exist_ok=True)
    eps_range = args.episodes

    SR_times = np.load(os.path.join(data_dir, "SR_times_hierarchy.npy"))
    SR_times2 = np.load(os.path.join(data_dir, "SR_times_flat.npy"))
    mean_SR_times = np.mean(SR_times, axis=0)
    mean_SR_times2 = np.mean(SR_times2, axis=0)
    std_SR_times = np.std(SR_times, axis=0) / np.sqrt(len(SR_times))
    std_SR_times2 = np.std(SR_times2, axis=0) / np.sqrt(len(SR_times2))

    fig = plt.figure(figsize=(14, 10))
    plt.plot(eps_range, mean_SR_times, label="Hierarchy")
    plt.fill_between(
        eps_range,
        mean_SR_times - std_SR_times,
        mean_SR_times + std_SR_times,
        alpha=0.5,
    )
    plt.plot(eps_range, mean_SR_times2, label="Flat")
    plt.fill_between(
        eps_range,
        mean_SR_times2 - std_SR_times2,
        mean_SR_times2 + std_SR_times2,
        alpha=0.5,
    )
    plt.xlabel("Training Episodes", fontsize=28)
    plt.ylabel("Processing Time (s)", fontsize=28)
    plt.legend(fontsize=40, loc="upper left")
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "processing_times.png"), format="png")
    plt.close()
    print(f"  Saved {save_dir}/processing_times.png")


def plot_SR_distances(args, GOALS, data_dir="data/eval/gridworld", save_dir="figures/eval/gridworld"):
    """Plot planning steps vs BFS distance from goal."""
    os.makedirs(save_dir, exist_ok=True)

    SR_dists = np.load(os.path.join(data_dir, "SR_dists_hierarchy.npy"))
    SR_dists2 = np.load(os.path.join(data_dir, "SR_dists_flat.npy"))
    mean_SR_dists = np.mean(SR_dists, axis=0)
    mean_SR_dists2 = np.mean(SR_dists2, axis=0)
    std_SR_dists = np.std(SR_dists, axis=0) / np.sqrt(len(SR_dists))
    std_SR_dists2 = np.std(SR_dists2, axis=0) / np.sqrt(len(SR_dists2))

    grid_size = args.grid_size
    walls = args.walls
    xs = [get_distance(grid_size, walls, (0, 0), g) for g in GOALS]

    fig = plt.figure(figsize=(14, 10))
    plt.plot(xs, mean_SR_dists, label="Hierarchy")
    plt.fill_between(
        xs,
        mean_SR_dists - std_SR_dists,
        mean_SR_dists + std_SR_dists,
        alpha=0.5,
    )
    plt.plot(xs, mean_SR_dists2, label="Flat")
    plt.fill_between(
        xs,
        mean_SR_dists2 - std_SR_dists2,
        mean_SR_dists2 + std_SR_dists2,
        alpha=0.5,
    )
    plt.xlabel("Distance From Goal", fontsize=28)
    plt.ylabel("Number of Planning Steps", fontsize=28)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "num_dist.png"), format="png")
    plt.close()
    print(f"  Saved {save_dir}/num_dist.png")


def plot_relative_stability(args, data_dir="data/eval/gridworld", save_dir="figures/eval/gridworld"):
    """Plot relative stability bar chart (lower is better)."""
    os.makedirs(save_dir, exist_ok=True)

    hierarchy_path = os.path.join(data_dir, "SR_relative_stability_hierarchy.npy")
    flat_path = os.path.join(data_dir, "SR_relative_stability_flat.npy")
    q_learning_path = os.path.join(data_dir, "Q_relative_stability.npy")
    single_path = os.path.join(data_dir, "SR_relative_stability.npy")

    labels, means, sems = [], [], []

    if os.path.exists(hierarchy_path):
        st = np.load(hierarchy_path)
        labels.append("Hierarchy")
        means.append(float(np.mean(st)))
        sems.append(float(np.std(st) / np.sqrt(len(st))))

    if os.path.exists(flat_path):
        st = np.load(flat_path)
        labels.append("Flat")
        means.append(float(np.mean(st)))
        sems.append(float(np.std(st) / np.sqrt(len(st))))

    if os.path.exists(q_learning_path):
        st = np.load(q_learning_path)
        labels.append("Q-Learning")
        means.append(float(np.mean(st)))
        sems.append(float(np.std(st) / np.sqrt(len(st))))

    if len(labels) == 0 and os.path.exists(single_path):
        st = np.load(single_path)
        labels = ["Relative Stability"]
        means = [float(np.mean(st))]
        sems = [float(np.std(st) / np.sqrt(len(st)))]

    if len(labels) == 0:
        print("  No stability data found — skipping relative_stability plot")
        return

    # Match colors from the reward plot (matplotlib default color cycle)
    color_map = {
        "Hierarchy": "C0",   # Blue
        "Flat": "C1",        # Orange
        "Q-Learning": "C2",  # Green
    }
    bar_colors = [color_map.get(label, "C0") for label in labels]

    fig = plt.figure(figsize=(10, 8))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=sems, capsize=8, color=bar_colors)
    plt.xticks(x, labels, fontsize=26)
    plt.yticks(fontsize=26)
    plt.ylabel("Relative Stability", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "relative_stability.png"), format="png")
    plt.close()
    print(f"  Saved {save_dir}/relative_stability.png")


# ==================== Main ====================


if __name__ == "__main__":
    data_dir = "data/eval/gridworld"
    save_dir = "figures/eval/gridworld"

    # Load saved args
    args_path = os.path.join(data_dir, "args.json")
    if os.path.exists(args_path):
        with open(args_path, "r") as f:
            saved = json.load(f)
            saved["walls"] = [tuple(w) for w in saved["walls"]]
            saved["init_loc"] = tuple(saved["init_loc"])
            saved["goal_loc"] = tuple(saved["goal_loc"])
            args = argparse.Namespace(**saved)
        print(f"Loaded args from {args_path}")
    else:
        # Use defaults
        grid_size = 9
        eps = [50, 100, 200, 300, 400, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
        WALLS = (
            [(1, x) for x in range(grid_size // 2 + 2)]
            + [(3, x) for x in range(grid_size // 2 - 2, grid_size)]
            + [(5, x) for x in range(grid_size // 2 + 2)]
            + [(7, x) for x in range(grid_size // 2 - 2, grid_size)]
        )
        args = argparse.Namespace(
            grid_size=grid_size,
            n_macro=4,
            init_loc=(0, 0),
            goal_loc=(grid_size - 1, grid_size - 1),
            goal_val=100,
            n_runs=20,
            walls=WALLS,
            episodes=eps,
        )

    GOALS = [(0, 3), (0, 6), (2, 4), (3, 0), (4, 4), (5, 8), (6, 3), (7, 0), (8, 4), (8, 8)]

    os.makedirs(save_dir, exist_ok=True)

    print("Generating evaluation plots...")

    # Always try these (main experiment outputs)
    if os.path.exists(os.path.join(data_dir, "SR_rewards_hierarchy.npy")):
        plot_SR_rewards(args, data_dir=data_dir, save_dir=save_dir)

    if os.path.exists(os.path.join(data_dir, "SR_values_hierarchy.npy")):
        plot_SR_values(args, data_dir=data_dir, save_dir=save_dir)

    plot_relative_stability(args, data_dir=data_dir, save_dir=save_dir)

    # Optional plots (require specific data files)
    if os.path.exists(os.path.join(data_dir, "SR_steps_hierarchy.npy")):
        plot_SR_steps(args, data_dir=data_dir, save_dir=save_dir)

    if os.path.exists(os.path.join(data_dir, "SR_times_hierarchy.npy")):
        plot_SR_times(args, data_dir=data_dir, save_dir=save_dir)

    if os.path.exists(os.path.join(data_dir, "SR_dists_hierarchy.npy")):
        plot_SR_distances(args, GOALS, data_dir=data_dir, save_dir=save_dir)

    print("\nDone!")
