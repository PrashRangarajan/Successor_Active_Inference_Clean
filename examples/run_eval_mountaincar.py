"""Evaluation: Mountain Car benchmark comparing Hierarchy vs Flat.

Runs repeated experiments across training checkpoints to compare
hierarchical vs flat active inference on the continuous Mountain Car environment.

Saves .npy data files to data/mountaincar/ and figures to figures/eval_mountaincar/.

Usage:
    # Run experiments:
    python examples/run_eval_mountaincar.py --train

    # Plot from saved data:
    python examples/run_eval_mountaincar.py

    # Quick test (2 seeds × 3 checkpoints):
    python examples/run_eval_mountaincar.py --train --quick
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import json
import time

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-poster")

from core import HierarchicalSRAgent
from environments.mountain_car import MountainCarAdapter


# ==================== Utilities ====================


def relative_stability_paper_style(returns, Ke=100, smooth_window=1, eps=1e-8):
    """Paper-style 'Relative Stability' (lower is better)."""
    r = np.asarray(returns, dtype=float).reshape(-1)
    if r.size == 0:
        return np.nan
    Ke = int(min(max(1, Ke), r.size))
    w = r[-Ke:]

    if smooth_window is not None and int(smooth_window) > 1 and w.size >= int(smooth_window):
        k = int(smooth_window)
        kernel = np.ones(k, dtype=float) / k
        w_smooth = np.convolve(w, kernel, mode="same")
    else:
        w_smooth = w

    best = np.max(w)
    denom = np.abs(best) + eps
    return float(np.mean(np.abs((w_smooth - best) / denom)))


# ==================== Agent Factory ====================


def create_mountaincar_agent(n_pos_bins, n_vel_bins, n_clusters,
                             num_episodes, gamma=0.95, learning_rate=0.05,
                             use_replay=True, n_replay_epochs=10,
                             test_max_steps=500):
    """Create a fresh Mountain Car SR agent trained for exactly num_episodes.

    Returns:
        (agent, test_adapter) tuple
    """
    # Training environment
    env_train = gym.make('MountainCar-v0')
    adapter_train = MountainCarAdapter(
        env_train,
        n_pos_bins=n_pos_bins,
        n_vel_bins=n_vel_bins,
    )

    agent = HierarchicalSRAgent(
        adapter=adapter_train,
        n_clusters=n_clusters,
        gamma=gamma,
        learning_rate=learning_rate,
        learn_from_experience=True,
        use_replay=use_replay,
        n_replay_epochs=n_replay_epochs,
    )
    agent.set_goal(None, reward=100.0, default_cost=-1.0)
    agent.learn_environment(num_episodes)

    # Switch to test environment with longer episode limit
    env_test = gym.make('MountainCar-v0', max_episode_steps=test_max_steps)
    adapter_test = MountainCarAdapter(
        env_test,
        n_pos_bins=n_pos_bins,
        n_vel_bins=n_vel_bins,
    )
    agent.adapter = adapter_test

    return agent, adapter_test


# ==================== Experiment ====================


def mountaincar_rewards_experiment(args):
    """Main experiment: rewards across training checkpoints for Hierarchy vs Flat.

    Returns:
        Tuple of (SR_rewards_hier, SR_rewards_flat, SR_steps_hier, SR_steps_flat)
    """
    n_trials = len(args.episodes)

    SR_rewards_hier = np.zeros((args.n_runs, n_trials))
    SR_rewards_flat = np.zeros((args.n_runs, n_trials))
    SR_steps_hier = np.zeros((args.n_runs, n_trials))
    SR_steps_flat = np.zeros((args.n_runs, n_trials))

    init_state = [-0.5, 0.0]

    for n in range(args.n_runs):
        print("x" * 40)
        print(f"Run: {n + 1}/{args.n_runs}")
        print("x" * 40)

        for trial in range(n_trials):
            num_episodes = args.episodes[trial]

            print()
            print("+" * 25)
            print(f"{num_episodes} training episodes")
            print("+" * 25)

            # Retry loop for rare LinAlgError during spectral clustering
            while True:
                try:
                    agent, adapter = create_mountaincar_agent(
                        args.n_pos_bins, args.n_vel_bins,
                        args.n_clusters, num_episodes,
                        gamma=args.gamma,
                        use_replay=args.use_replay,
                        n_replay_epochs=args.n_replay_epochs,
                        test_max_steps=args.test_max_steps,
                    )
                except (np.linalg.LinAlgError, ValueError) as e:
                    print(f"  Error: {e} — retrying...")
                    continue
                else:
                    break

            # Evaluate hierarchy
            print("\nHierarchy")
            agent.reset_episode(init_state=init_state)
            result_hier = agent.run_episode_hierarchical(max_steps=args.test_max_steps)

            # Evaluate flat (same agent, same M)
            print("\nFlat")
            agent.reset_episode(init_state=init_state)
            result_flat = agent.run_episode_flat(max_steps=args.test_max_steps)

            SR_rewards_hier[n, trial] = result_hier["reward"]
            SR_rewards_flat[n, trial] = result_flat["reward"]
            SR_steps_hier[n, trial] = result_hier["steps"]
            SR_steps_flat[n, trial] = result_flat["steps"]

            print(f"  Hier: reward={result_hier['reward']:.1f}, "
                  f"steps={result_hier['steps']}, goal={result_hier['reached_goal']}")
            print(f"  Flat: reward={result_flat['reward']:.1f}, "
                  f"steps={result_flat['steps']}, goal={result_flat['reached_goal']}")

    return SR_rewards_hier, SR_rewards_flat, SR_steps_hier, SR_steps_flat


# ==================== Plotting ====================


def plot_mountaincar_rewards(args, save_dir="figures/eval_mountaincar"):
    """Plot reward curves with confidence bands (Hierarchy vs Flat)."""
    os.makedirs(save_dir, exist_ok=True)
    eps_range = args.episodes

    hier = np.load("data/mountaincar/SR_rewards_hierarchy.npy")[:, :len(eps_range)]
    flat = np.load("data/mountaincar/SR_rewards_flat.npy")[:, :len(eps_range)]

    mean_hier = np.mean(hier, axis=0)
    std_hier = np.std(hier, axis=0) / np.sqrt(len(hier))
    mean_flat = np.mean(flat, axis=0)
    std_flat = np.std(flat, axis=0) / np.sqrt(len(flat))

    fig = plt.figure(figsize=(14, 10))
    plt.plot(eps_range, mean_hier, label="Hierarchy")
    plt.fill_between(eps_range, mean_hier - std_hier, mean_hier + std_hier, alpha=0.5)
    plt.plot(eps_range, mean_flat, label="Flat")
    plt.fill_between(eps_range, mean_flat - std_flat, mean_flat + std_flat, alpha=0.5)

    plt.xlabel("Number of Training Episodes", fontsize=28)
    plt.ylabel("Total Reward", fontsize=28)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mountaincar_reward.png"), format="png")
    plt.close()
    print(f"  Saved {save_dir}/mountaincar_reward.png")


def plot_mountaincar_steps(args, save_dir="figures/eval_mountaincar"):
    """Plot steps-to-goal curves (Hierarchy vs Flat)."""
    os.makedirs(save_dir, exist_ok=True)
    eps_range = args.episodes

    hier = np.load("data/mountaincar/SR_steps_hierarchy.npy")[:, :len(eps_range)]
    flat = np.load("data/mountaincar/SR_steps_flat.npy")[:, :len(eps_range)]

    mean_hier = np.mean(hier, axis=0)
    std_hier = np.std(hier, axis=0) / np.sqrt(len(hier))
    mean_flat = np.mean(flat, axis=0)
    std_flat = np.std(flat, axis=0) / np.sqrt(len(flat))

    fig = plt.figure(figsize=(14, 10))
    plt.plot(eps_range, mean_hier, label="Hierarchy")
    plt.fill_between(eps_range, mean_hier - std_hier, mean_hier + std_hier, alpha=0.5)
    plt.plot(eps_range, mean_flat, label="Flat")
    plt.fill_between(eps_range, mean_flat - std_flat, mean_flat + std_flat, alpha=0.5)

    plt.xlabel("Number of Training Episodes", fontsize=28)
    plt.ylabel("Steps to Goal", fontsize=28)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mountaincar_steps.png"), format="png")
    plt.close()
    print(f"  Saved {save_dir}/mountaincar_steps.png")


def plot_mountaincar_stability(args, save_dir="figures/eval_mountaincar"):
    """Plot relative stability bar chart (Hierarchy vs Flat)."""
    os.makedirs(save_dir, exist_ok=True)

    hier_path = "data/mountaincar/SR_relative_stability_hierarchy.npy"
    flat_path = "data/mountaincar/SR_relative_stability_flat.npy"

    labels, means, sems = [], [], []

    if os.path.exists(hier_path):
        st = np.load(hier_path)
        labels.append("Hierarchy")
        means.append(float(np.mean(st)))
        sems.append(float(np.std(st) / np.sqrt(len(st))))

    if os.path.exists(flat_path):
        st = np.load(flat_path)
        labels.append("Flat")
        means.append(float(np.mean(st)))
        sems.append(float(np.std(st) / np.sqrt(len(st))))

    if len(labels) == 0:
        print("  No stability data found — skipping")
        return

    color_map = {"Hierarchy": "C0", "Flat": "C1"}
    bar_colors = [color_map.get(label, "C0") for label in labels]

    fig = plt.figure(figsize=(10, 8))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=sems, capsize=8, color=bar_colors)
    plt.xticks(x, labels, fontsize=26)
    plt.yticks(fontsize=26)
    plt.ylabel("Relative Stability", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mountaincar_relative_stability.png"), format="png")
    plt.close()
    print(f"  Saved {save_dir}/mountaincar_relative_stability.png")


# ==================== Main ====================


if __name__ == "__main__":
    # Mountain Car configuration
    n_pos_bins = 10
    n_vel_bins = 10
    n_clusters = 6
    gamma = 0.95
    nruns = 5
    eps = [500, 1000, 2000, 4000, 6000, 8000, 10000]
    test_max_steps = 500

    parser = argparse.ArgumentParser(description="Mountain Car Eval: Hierarchy vs Flat")
    parser.add_argument("--train", action="store_true", help="Run experiments")
    parser.add_argument("--quick", action="store_true", help="Quick test")
    parser.add_argument("--n_runs", type=int, default=nruns)
    args_cli = parser.parse_args()

    if args_cli.quick:
        eps = [1000, 4000, 8000]
        nruns = 2

    args = argparse.Namespace(
        n_pos_bins=n_pos_bins,
        n_vel_bins=n_vel_bins,
        n_clusters=n_clusters,
        gamma=gamma,
        n_runs=args_cli.n_runs if not args_cli.quick else nruns,
        episodes=eps,
        test_max_steps=test_max_steps,
        use_replay=True,
        n_replay_epochs=10,
    )

    if args_cli.train:
        os.makedirs("data/mountaincar/", exist_ok=True)

        # Save args
        with open("data/mountaincar/args.json", "w") as f:
            json.dump(vars(args), f, indent=2)

        print("=" * 60)
        print("MOUNTAIN CAR EVAL: Hierarchy vs Flat")
        print("=" * 60)
        print(f"State space: {n_pos_bins * n_vel_bins} states")
        print(f"Runs: {args.n_runs}, Checkpoints: {args.episodes}")

        t0 = time.time()
        SR_rewards_hier, SR_rewards_flat, SR_steps_hier, SR_steps_flat = \
            mountaincar_rewards_experiment(args)
        elapsed = time.time() - t0
        print(f"\nExperiment completed in {elapsed:.0f}s")

        # Compute relative stability
        SR_rel_stability_hier = np.array([
            relative_stability_paper_style(SR_rewards_hier[i, :])
            for i in range(SR_rewards_hier.shape[0])
        ])
        SR_rel_stability_flat = np.array([
            relative_stability_paper_style(SR_rewards_flat[i, :])
            for i in range(SR_rewards_flat.shape[0])
        ])

        # Save data
        np.save("data/mountaincar/SR_rewards_hierarchy.npy", SR_rewards_hier)
        np.save("data/mountaincar/SR_rewards_flat.npy", SR_rewards_flat)
        np.save("data/mountaincar/SR_steps_hierarchy.npy", SR_steps_hier)
        np.save("data/mountaincar/SR_steps_flat.npy", SR_steps_flat)
        np.save("data/mountaincar/SR_relative_stability_hierarchy.npy", SR_rel_stability_hier)
        np.save("data/mountaincar/SR_relative_stability_flat.npy", SR_rel_stability_flat)
        print("\nSaved all data to data/mountaincar/")

    else:
        # Load saved args
        if os.path.exists("data/mountaincar/args.json"):
            with open("data/mountaincar/args.json", "r") as f:
                saved = json.load(f)
                args = argparse.Namespace(**saved)
            print(f"Loaded args: {args}")
        else:
            print("No saved args found. Run with --train first.")

    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    os.makedirs("figures/eval_mountaincar", exist_ok=True)

    if os.path.exists("data/mountaincar/SR_rewards_hierarchy.npy"):
        plot_mountaincar_rewards(args)

    if os.path.exists("data/mountaincar/SR_steps_hierarchy.npy"):
        plot_mountaincar_steps(args)

    plot_mountaincar_stability(args)

    print("\nDone! Figures saved to figures/eval_mountaincar/")
