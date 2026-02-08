"""Evaluation: Key Gridworld benchmark comparing Hierarchy vs Flat vs Q-Learning.

Runs repeated experiments across training checkpoints to compare
hierarchical vs flat active inference and Q-learning on the Key Gridworld
environment. The agent must pick up a key before reaching the goal
(augmented state space).

Saves .npy data files to data/eval/key_gridworld/ and figures to figures/eval/key_gridworld/.

Usage:
    # Run experiments:
    python examples/run_eval_key_gridworld.py --train

    # Plot from saved data:
    python examples/run_eval_key_gridworld.py

    # Quick test (2 seeds × 3 checkpoints):
    python examples/run_eval_key_gridworld.py --train --quick
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
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-poster")

from core import HierarchicalSRAgent
from core.q_learning import QLearningAgent
from environments.key_gridworld import KeyGridworldAdapter
from unified_env import KeyGridworld as SR_Gridworld


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


def create_key_gridworld_agent(grid_size, walls, key_loc, n_clusters,
                                goal_spec, num_episodes, gamma=0.99,
                                learning_rate=0.05, has_pickup_action=True,
                                use_replay=True, n_replay_epochs=10):
    """Create a fresh Key Gridworld SR agent trained for exactly num_episodes.

    Returns:
        (agent, adapter) tuple
    """
    env = SR_Gridworld(grid_size, key_loc=key_loc, pickup=has_pickup_action)
    env.set_walls(walls)

    adapter = KeyGridworldAdapter(
        env, grid_size, has_pickup_action=has_pickup_action,
    )

    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=n_clusters,
        gamma=gamma,
        learning_rate=learning_rate,
        learn_from_experience=True,
        use_replay=use_replay,
        n_replay_epochs=n_replay_epochs,
    )
    agent.set_goal(goal_spec, reward=100.0)
    agent.learn_environment(num_episodes)

    return agent, adapter


def create_key_gridworld_q_agent(grid_size, walls, key_loc, goal_spec,
                                  has_pickup_action=True, gamma=0.99):
    """Create a fresh Q-learning agent for Key Gridworld (not yet trained).

    The Q-learning agent's learn() IS incremental, so we create one per run
    and call learn(delta) at each checkpoint.

    Returns:
        (q_agent, adapter) tuple
    """
    env = SR_Gridworld(grid_size, key_loc=key_loc, pickup=has_pickup_action)
    env.set_walls(walls)

    adapter = KeyGridworldAdapter(
        env, grid_size, has_pickup_action=has_pickup_action,
    )

    goal_states = adapter.get_goal_states(goal_spec)
    C = adapter.create_goal_prior(goal_states, reward=100.0, default_cost=-0.1)
    # Flatten C to 1D for Q-learning agent (augmented state space is (N, 2))
    C_flat = C.flatten('F')

    q_agent = QLearningAgent(
        adapter=adapter,
        goal_states=goal_states,
        C=C_flat,
        gamma=gamma,
    )
    return q_agent, adapter


# ==================== Experiment ====================


def key_gridworld_rewards_experiment(args):
    """Main experiment: rewards across training checkpoints for Hierarchy vs Flat vs Q-Learning.

    Returns:
        Tuple of (SR_rewards_hier, SR_rewards_flat, Q_rewards,
                  SR_steps_hier, SR_steps_flat, Q_steps)
    """
    n_trials = len(args.episodes)

    SR_rewards_hier = np.zeros((args.n_runs, n_trials))
    SR_rewards_flat = np.zeros((args.n_runs, n_trials))
    Q_rewards = np.zeros((args.n_runs, n_trials))
    SR_steps_hier = np.zeros((args.n_runs, n_trials))
    SR_steps_flat = np.zeros((args.n_runs, n_trials))
    Q_steps = np.zeros((args.n_runs, n_trials))

    init_state = args.init_loc + (0,)  # Start without key

    for n in range(args.n_runs):
        print("x" * 40)
        print(f"Run: {n + 1}/{args.n_runs}")
        print("x" * 40)

        # Create Q-learning agent (incremental — one per run)
        q_agent, q_adapter = create_key_gridworld_q_agent(
            args.grid_size, args.walls, args.key_loc,
            args.goal_spec,
            has_pickup_action=args.has_pickup_action,
            gamma=args.gamma,
        )

        for trial in range(n_trials):
            num_episodes = args.episodes[trial]

            # Q-learning: incremental training
            if trial == 0:
                q_delta = num_episodes
            else:
                q_delta = num_episodes - args.episodes[trial - 1]

            print()
            print("+" * 25)
            print(f"{num_episodes} training episodes")
            print("+" * 25)

            # Retry loop for rare LinAlgError during spectral clustering
            while True:
                try:
                    agent, adapter = create_key_gridworld_agent(
                        args.grid_size, args.walls, args.key_loc,
                        args.n_clusters, args.goal_spec, num_episodes,
                        gamma=args.gamma,
                        has_pickup_action=args.has_pickup_action,
                        use_replay=args.use_replay,
                        n_replay_epochs=args.n_replay_epochs,
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

            # Q-learning: train incrementally, then evaluate
            print("\nQ-Learning")
            q_agent.learn(q_delta)
            result_q = q_agent.run_episode(init_state=init_state, max_steps=args.test_max_steps)

            SR_rewards_hier[n, trial] = result_hier["reward"]
            SR_rewards_flat[n, trial] = result_flat["reward"]
            Q_rewards[n, trial] = result_q["reward"]
            SR_steps_hier[n, trial] = result_hier["steps"]
            SR_steps_flat[n, trial] = result_flat["steps"]
            Q_steps[n, trial] = result_q["steps"]

            print(f"  Hier: reward={result_hier['reward']:.1f}, "
                  f"steps={result_hier['steps']}, goal={result_hier['reached_goal']}")
            print(f"  Flat: reward={result_flat['reward']:.1f}, "
                  f"steps={result_flat['steps']}, goal={result_flat['reached_goal']}")
            print(f"  Q:    reward={result_q['reward']:.1f}, "
                  f"steps={result_q['steps']}, goal={result_q['reached_goal']}")

    return SR_rewards_hier, SR_rewards_flat, Q_rewards, SR_steps_hier, SR_steps_flat, Q_steps


# ==================== Plotting ====================


def plot_key_gridworld_rewards(args, data_dir="data/eval/key_gridworld",
                                save_dir="figures/eval/key_gridworld"):
    """Plot reward curves with confidence bands (Hierarchy vs Flat vs Q-Learning)."""
    os.makedirs(save_dir, exist_ok=True)
    eps_range = args.episodes

    hier = np.load(os.path.join(data_dir, "SR_rewards_hierarchy.npy"))[:, :len(eps_range)]
    flat = np.load(os.path.join(data_dir, "SR_rewards_flat.npy"))[:, :len(eps_range)]

    mean_hier = np.mean(hier, axis=0)
    std_hier = np.std(hier, axis=0) / np.sqrt(len(hier))
    mean_flat = np.mean(flat, axis=0)
    std_flat = np.std(flat, axis=0) / np.sqrt(len(flat))

    # Load Q-learning rewards if available
    Q_rewards_path = os.path.join(data_dir, "Q_rewards.npy")
    has_q_learning = os.path.exists(Q_rewards_path)
    if has_q_learning:
        q_data = np.load(Q_rewards_path)[:, :len(eps_range)]
        mean_q = np.mean(q_data, axis=0)
        std_q = np.std(q_data, axis=0) / np.sqrt(len(q_data))

    fig = plt.figure(figsize=(14, 10))
    plt.plot(eps_range, mean_hier, label="Hierarchy")
    plt.fill_between(eps_range, mean_hier - std_hier, mean_hier + std_hier, alpha=0.5)
    plt.plot(eps_range, mean_flat, label="Flat")
    plt.fill_between(eps_range, mean_flat - std_flat, mean_flat + std_flat, alpha=0.5)
    if has_q_learning:
        plt.plot(eps_range, mean_q, label="Q-Learning")
        plt.fill_between(eps_range, mean_q - std_q, mean_q + std_q, alpha=0.5)

    plt.xlabel("Number of Training Episodes", fontsize=28)
    plt.ylabel("Total Reward", fontsize=28)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "key_gridworld_reward.png"), format="png")
    plt.close()
    print(f"  Saved {save_dir}/key_gridworld_reward.png")


def plot_key_gridworld_steps(args, data_dir="data/eval/key_gridworld",
                              save_dir="figures/eval/key_gridworld"):
    """Plot steps-to-goal curves (Hierarchy vs Flat vs Q-Learning)."""
    os.makedirs(save_dir, exist_ok=True)
    eps_range = args.episodes

    hier = np.load(os.path.join(data_dir, "SR_steps_hierarchy.npy"))[:, :len(eps_range)]
    flat = np.load(os.path.join(data_dir, "SR_steps_flat.npy"))[:, :len(eps_range)]

    mean_hier = np.mean(hier, axis=0)
    std_hier = np.std(hier, axis=0) / np.sqrt(len(hier))
    mean_flat = np.mean(flat, axis=0)
    std_flat = np.std(flat, axis=0) / np.sqrt(len(flat))

    # Load Q-learning steps if available
    Q_steps_path = os.path.join(data_dir, "Q_steps.npy")
    has_q_learning = os.path.exists(Q_steps_path)
    if has_q_learning:
        q_data = np.load(Q_steps_path)[:, :len(eps_range)]
        mean_q = np.mean(q_data, axis=0)
        std_q = np.std(q_data, axis=0) / np.sqrt(len(q_data))

    fig = plt.figure(figsize=(14, 10))
    plt.plot(eps_range, mean_hier, label="Hierarchy")
    plt.fill_between(eps_range, mean_hier - std_hier, mean_hier + std_hier, alpha=0.5)
    plt.plot(eps_range, mean_flat, label="Flat")
    plt.fill_between(eps_range, mean_flat - std_flat, mean_flat + std_flat, alpha=0.5)
    if has_q_learning:
        plt.plot(eps_range, mean_q, label="Q-Learning")
        plt.fill_between(eps_range, mean_q - std_q, mean_q + std_q, alpha=0.5)

    plt.xlabel("Number of Training Episodes", fontsize=28)
    plt.ylabel("Steps to Goal", fontsize=28)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "key_gridworld_steps.png"), format="png")
    plt.close()
    print(f"  Saved {save_dir}/key_gridworld_steps.png")


def plot_key_gridworld_stability(args, data_dir="data/eval/key_gridworld",
                                  save_dir="figures/eval/key_gridworld"):
    """Plot relative stability bar chart (Hierarchy vs Flat vs Q-Learning)."""
    os.makedirs(save_dir, exist_ok=True)

    hier_path = os.path.join(data_dir, "SR_relative_stability_hierarchy.npy")
    flat_path = os.path.join(data_dir, "SR_relative_stability_flat.npy")
    q_learning_path = os.path.join(data_dir, "Q_relative_stability.npy")

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

    if os.path.exists(q_learning_path):
        st = np.load(q_learning_path)
        labels.append("Q-Learning")
        means.append(float(np.mean(st)))
        sems.append(float(np.std(st) / np.sqrt(len(st))))

    if len(labels) == 0:
        print("  No stability data found — skipping")
        return

    color_map = {"Hierarchy": "C0", "Flat": "C1", "Q-Learning": "C2"}
    bar_colors = [color_map.get(label, "C0") for label in labels]

    fig = plt.figure(figsize=(10, 8))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=sems, capsize=8, color=bar_colors)
    plt.xticks(x, labels, fontsize=26)
    plt.yticks(fontsize=26)
    plt.ylabel("Relative Stability", fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "key_gridworld_relative_stability.png"), format="png")
    plt.close()
    print(f"  Saved {save_dir}/key_gridworld_relative_stability.png")


# ==================== Main ====================


if __name__ == "__main__":
    # Key Gridworld configuration
    grid_size = 5
    n_clusters = 5
    gamma = 0.99
    nruns = 10
    eps = [100, 200, 500, 750, 1000, 1500, 2000, 3000, 4000]
    test_max_steps = 100
    has_pickup_action = True

    init_loc = (0, 0)
    key_loc = (3, 0)
    goal_loc = (grid_size - 1, grid_size - 1)
    goal_spec = (goal_loc[0], goal_loc[1], 1)  # Must have key

    # Walls (same as run_key_gridworld.py)
    walls = (
        [(1, x) for x in range(grid_size // 2 + 1)] +
        [(4, 2)] +
        [(3, x) for x in range(grid_size // 2 - 1, grid_size) if x != grid_size // 2 + 1]
    )

    parser = argparse.ArgumentParser(description="Key Gridworld Eval: Hierarchy vs Flat vs Q-Learning")
    parser.add_argument("--train", action="store_true", help="Run experiments")
    parser.add_argument("--quick", action="store_true", help="Quick test")
    parser.add_argument("--n_runs", type=int, default=nruns)
    args_cli = parser.parse_args()

    if args_cli.quick:
        eps = [500, 1500, 3000]
        nruns = 2

    args = argparse.Namespace(
        grid_size=grid_size,
        n_clusters=n_clusters,
        gamma=gamma,
        n_runs=args_cli.n_runs if not args_cli.quick else nruns,
        episodes=eps,
        test_max_steps=test_max_steps,
        has_pickup_action=has_pickup_action,
        init_loc=init_loc,
        key_loc=key_loc,
        goal_loc=goal_loc,
        goal_spec=goal_spec,
        walls=walls,
        use_replay=True,
        n_replay_epochs=10,
    )

    data_dir = "data/eval/key_gridworld"
    save_dir = "figures/eval/key_gridworld"

    if args_cli.train:
        os.makedirs(data_dir, exist_ok=True)

        # Save args
        args_save = vars(args).copy()
        args_save["walls"] = [list(w) for w in args_save["walls"]]
        args_save["init_loc"] = list(args_save["init_loc"])
        args_save["key_loc"] = list(args_save["key_loc"])
        args_save["goal_loc"] = list(args_save["goal_loc"])
        args_save["goal_spec"] = list(args_save["goal_spec"])
        with open(os.path.join(data_dir, "args.json"), "w") as f:
            json.dump(args_save, f, indent=2)

        n_base_states = grid_size ** 2
        n_aug_states = n_base_states * 2  # has_key in {0, 1}

        print("=" * 60)
        print("KEY GRIDWORLD EVAL: Hierarchy vs Flat vs Q-Learning")
        print("=" * 60)
        print(f"Grid: {grid_size}x{grid_size}, Augmented states: {n_aug_states}")
        print(f"Key at: {key_loc}, Goal at: {goal_loc} (requires key)")
        print(f"Runs: {args.n_runs}, Checkpoints: {args.episodes}")

        t0 = time.time()
        SR_rewards_hier, SR_rewards_flat, Q_rewards, SR_steps_hier, SR_steps_flat, Q_steps = \
            key_gridworld_rewards_experiment(args)
        elapsed = time.time() - t0
        print(f"\nExperiment completed in {elapsed:.0f}s")

        # Compute relative stability for all three agent types
        SR_rel_stability_hier = np.array([
            relative_stability_paper_style(SR_rewards_hier[i, :])
            for i in range(SR_rewards_hier.shape[0])
        ])
        SR_rel_stability_flat = np.array([
            relative_stability_paper_style(SR_rewards_flat[i, :])
            for i in range(SR_rewards_flat.shape[0])
        ])
        Q_rel_stability = np.array([
            relative_stability_paper_style(Q_rewards[i, :])
            for i in range(Q_rewards.shape[0])
        ])

        # Save data
        np.save(os.path.join(data_dir, "SR_rewards_hierarchy.npy"), SR_rewards_hier)
        np.save(os.path.join(data_dir, "SR_rewards_flat.npy"), SR_rewards_flat)
        np.save(os.path.join(data_dir, "Q_rewards.npy"), Q_rewards)
        np.save(os.path.join(data_dir, "SR_steps_hierarchy.npy"), SR_steps_hier)
        np.save(os.path.join(data_dir, "SR_steps_flat.npy"), SR_steps_flat)
        np.save(os.path.join(data_dir, "Q_steps.npy"), Q_steps)
        np.save(os.path.join(data_dir, "SR_relative_stability_hierarchy.npy"), SR_rel_stability_hier)
        np.save(os.path.join(data_dir, "SR_relative_stability_flat.npy"), SR_rel_stability_flat)
        np.save(os.path.join(data_dir, "Q_relative_stability.npy"), Q_rel_stability)
        print(f"\nSaved all data to {data_dir}/")

    else:
        # Load saved args
        args_path = os.path.join(data_dir, "args.json")
        if os.path.exists(args_path):
            with open(args_path, "r") as f:
                saved = json.load(f)
                saved["walls"] = [tuple(w) for w in saved["walls"]]
                saved["init_loc"] = tuple(saved["init_loc"])
                saved["key_loc"] = tuple(saved["key_loc"])
                saved["goal_loc"] = tuple(saved["goal_loc"])
                saved["goal_spec"] = tuple(saved["goal_spec"])
                args = argparse.Namespace(**saved)

            # Reconcile episodes list with actual data shape.
            # args.json may be stale if a --quick run overwrote .npy files
            # but not args.json (or vice versa).
            ref_path = os.path.join(data_dir, "SR_rewards_hierarchy.npy")
            if os.path.exists(ref_path):
                n_data_cols = np.load(ref_path).shape[1]
                if n_data_cols != len(args.episodes):
                    print(f"  Warning: args.json lists {len(args.episodes)} episodes "
                          f"but data has {n_data_cols} checkpoints.")
                    print(f"  Re-run with --train to regenerate consistent data.")
                    # Use evenly spaced placeholder x-values so plotting still works
                    args.episodes = list(np.linspace(
                        args.episodes[0], args.episodes[-1], n_data_cols, dtype=int
                    ))
                    print(f"  Using interpolated episode labels: {args.episodes}")

            print(f"Loaded args: {args}")
        else:
            print("No saved args found. Run with --train first.")

    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(os.path.join(data_dir, "SR_rewards_hierarchy.npy")):
        plot_key_gridworld_rewards(args, data_dir=data_dir, save_dir=save_dir)

    if os.path.exists(os.path.join(data_dir, "SR_steps_hierarchy.npy")):
        plot_key_gridworld_steps(args, data_dir=data_dir, save_dir=save_dir)

    plot_key_gridworld_stability(args, data_dir=data_dir, save_dir=save_dir)

    print(f"\nDone! Figures saved to {save_dir}/")
