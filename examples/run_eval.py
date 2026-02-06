"""Evaluation: Multi-trial benchmark comparing Hierarchy vs Flat vs Q-Learning.

Runs repeated experiments across training checkpoints to produce paper-quality
figures comparing:
  - Total reward vs training episodes
  - SR convergence (M distance from true M)
  - Relative stability
  - Planning steps vs goal distance

Saves .npy data files to data/ and figures to figures/eval/.

Usage:
    # Run experiments (slow — 20 seeds × 15 checkpoints):
    python examples/run_eval.py --train

    # Plot from saved data (fast):
    python examples/run_eval.py

    # Quick test (2 seeds × 2 checkpoints):
    python examples/run_eval.py --train --quick
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import json
import time
from copy import deepcopy

import numpy as np

from core import HierarchicalSRAgent
from core.q_learning import QLearningAgent
from environments.gridworld import GridworldAdapter
from unified_env import StandardGridworld as SR_Gridworld


# ==================== Utilities ====================


def relative_stability_paper_style(returns, Ke=100, smooth_window=1, eps=1e-8):
    """Paper-style 'Relative Stability' (lower is better).

    Computes mean absolute relative deviation from the best return within the
    final window.

    Args:
        returns: 1D sequence of evaluation returns over training
        Ke: Window size from the end (clipped to len(returns))
        smooth_window: If >1, apply moving average before computing deviations
        eps: Small constant for numerical stability
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    if r.size == 0:
        return np.nan
    Ke = int(min(max(1, Ke), r.size))
    w = r[-Ke:]

    if smooth_window is not None and int(smooth_window) > 1 and w.size >= int(
        smooth_window
    ):
        k = int(smooth_window)
        kernel = np.ones(k, dtype=float) / k
        w_smooth = np.convolve(w, kernel, mode="same")
    else:
        w_smooth = w

    best = np.max(w)
    denom = np.abs(best) + eps
    return float(np.mean(np.abs((w_smooth - best) / denom)))


# ==================== Agent Factories ====================


def create_sr_agent(grid_size, walls, n_clusters, goal_loc, goal_val,
                    num_episodes, gamma=0.99, learning_rate=0.05):
    """Create a fresh SR agent trained for exactly num_episodes.

    Because learn_environment() is NOT incremental (each call creates fresh
    B/M matrices), we must instantiate a new agent for each training checkpoint.

    Returns:
        (agent, adapter) tuple
    """
    env = SR_Gridworld(grid_size)
    env.set_walls(walls)
    adapter = GridworldAdapter(env, grid_size)

    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=n_clusters,
        gamma=gamma,
        learning_rate=learning_rate,
        learn_from_experience=True,  # Must learn from experience for convergence experiments
    )
    agent.set_goal(goal_loc, reward=goal_val)
    agent.learn_environment(num_episodes)
    return agent, adapter


def create_q_agent(grid_size, walls, goal_loc, goal_val, gamma=0.99):
    """Create a fresh Q-learning agent (not yet trained).

    The Q-learning agent's learn() IS incremental, so we create one per run
    and call learn(delta) at each checkpoint.

    Returns:
        (q_agent, adapter) tuple
    """
    env = SR_Gridworld(grid_size)
    env.set_walls(walls)
    adapter = GridworldAdapter(env, grid_size)

    goal_states = adapter.get_goal_states(goal_loc)
    C = adapter.create_goal_prior(goal_states, reward=goal_val, default_cost=-0.1)

    q_agent = QLearningAgent(
        adapter=adapter,
        goal_states=goal_states,
        C=C,
        gamma=gamma,
    )
    return q_agent, adapter


def compute_true_references(grid_size, walls, n_clusters, goal_loc, goal_val,
                            gamma=0.99):
    """Compute analytical ground-truth matrices for convergence metrics.

    Uses learn_from_experience=False to get exact M via (I - gamma*B_avg)^{-1}.

    Returns:
        (true_M, true_V, true_M_macro) tuple
    """
    env = SR_Gridworld(grid_size)
    env.set_walls(walls)
    adapter = GridworldAdapter(env, grid_size)

    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=n_clusters,
        gamma=gamma,
        learn_from_experience=False,  # Analytical computation
    )
    agent.set_goal(goal_loc, reward=goal_val)
    agent.learn_environment(num_episodes=1000)  # Episodes only used for adjacency learning

    true_M = deepcopy(agent.M)
    true_V = true_M @ agent.C
    true_M_macro = deepcopy(agent.M_macro)

    return true_M, true_V, true_M_macro


# ==================== Experiments ====================


def SR_rewards_values(args):
    """Main experiment: rewards + SR convergence across training checkpoints.

    For each run:
      - Computes analytical true M, V, M_macro once
      - Creates a single Q-learning agent (incremental learning)
      - At each checkpoint, creates fresh SR agents (non-incremental learning)
      - Evaluates all three agent types

    Returns:
        Tuple of (SR_vals, SR_vals2, SR_succ, SR_succ2, SR_succ_macro,
                  SR_rewards, SR_rewards2, Q_rewards)
    """
    n_trials = len(args.episodes)
    grid_size = args.grid_size

    # Allocate result arrays
    SR_vals = np.zeros((args.n_runs, n_trials))       # V distance (hierarchy)
    SR_vals2 = np.zeros((args.n_runs, n_trials))      # V distance (flat)
    SR_succ = np.zeros((args.n_runs, n_trials))       # M distance (hierarchy)
    SR_succ2 = np.zeros((args.n_runs, n_trials))      # M distance (flat)
    SR_succ_macro = np.zeros((args.n_runs, n_trials))  # M_macro distance
    SR_rewards = np.zeros((args.n_runs, n_trials))     # reward (hierarchy)
    SR_rewards2 = np.zeros((args.n_runs, n_trials))    # reward (flat)
    Q_rewards = np.zeros((args.n_runs, n_trials))      # reward (Q-learning)

    for n in range(args.n_runs):
        print("x" * 40)
        print(f"Run: {n + 1}/{args.n_runs}")
        print("x" * 40)

        # Compute analytical ground truth once per run
        true_M, true_V, true_M_macro = compute_true_references(
            grid_size, args.walls, args.n_macro, args.goal_loc,
            args.goal_val, gamma=0.99,
        )

        # Create Q-learning agent (incremental — one per run)
        q_agent, q_adapter = create_q_agent(
            grid_size, args.walls, args.goal_loc, args.goal_val
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
                    # Fresh SR agent for hierarchy evaluation
                    agent1, _ = create_sr_agent(
                        grid_size, args.walls, args.n_macro,
                        args.goal_loc, args.goal_val, num_episodes,
                    )

                    # Fresh SR agent for flat evaluation (independent learning)
                    agent2, _ = create_sr_agent(
                        grid_size, args.walls, args.n_macro,
                        args.goal_loc, args.goal_val, num_episodes,
                    )
                except np.linalg.LinAlgError:
                    print("  LinAlgError — retrying...")
                    continue
                else:
                    break

            # Evaluate hierarchy
            print("\nHierarchy")
            agent1.reset_episode(init_state=0)
            result_hier = agent1.run_episode_hierarchical(max_steps=200)

            # Evaluate flat
            print("\nFlat")
            agent2.reset_episode(init_state=0)
            result_flat = agent2.run_episode_flat(max_steps=200)

            # Q-learning: train incrementally, then evaluate
            print("\nQ-Learning")
            q_agent.learn(q_delta)
            result_q = q_agent.run_episode(init_state=0, max_steps=200)

            # Store rewards
            SR_rewards[n, trial] = result_hier["reward"]
            SR_rewards2[n, trial] = result_flat["reward"]
            Q_rewards[n, trial] = result_q["reward"]

            # Store convergence metrics
            V1 = agent1.M @ agent1.C
            V2 = agent2.M @ agent2.C
            SR_vals[n, trial] = np.linalg.norm(V1 - true_V)
            SR_vals2[n, trial] = np.linalg.norm(V2 - true_V)

            SR_succ[n, trial] = np.linalg.norm(agent1.M - true_M) / np.linalg.norm(true_M)
            SR_succ2[n, trial] = np.linalg.norm(agent2.M - true_M) / np.linalg.norm(true_M)
            SR_succ_macro[n, trial] = np.linalg.norm(
                agent1.M_macro - true_M_macro
            ) / np.linalg.norm(true_M_macro)

            print(f"  Hier: reward={result_hier['reward']:.1f}, "
                  f"steps={result_hier['steps']}, goal={result_hier['reached_goal']}")
            print(f"  Flat: reward={result_flat['reward']:.1f}, "
                  f"steps={result_flat['steps']}, goal={result_flat['reached_goal']}")
            print(f"  Q:    reward={result_q['reward']:.1f}, "
                  f"steps={result_q['steps']}, goal={result_q['reached_goal']}")

    return (SR_vals, SR_vals2, SR_succ, SR_succ2, SR_succ_macro,
            SR_rewards, SR_rewards2, Q_rewards)


def SR_distances(args, GOALS):
    """Goal-distance experiment: planning steps vs distance to goal.

    For each goal location, trains fresh agents and measures steps to reach.

    Returns:
        (SR_dists, SR_dists2) — hierarchy and flat steps, shape (n_runs, n_goals)
    """
    SR_dists = []
    SR_dists2 = []
    num_episodes = 1500

    print("\nComparing Hierarchy and Flat for different goal distances")

    for n in range(args.n_runs):
        print(f"\nRun: {n + 1}/{args.n_runs}")
        SR_dist = []
        SR_dist2 = []

        for goal_loc in GOALS:
            print(f"  Goal = {goal_loc}")

            # Fresh hierarchy agent
            agent1, _ = create_sr_agent(
                args.grid_size, args.walls, args.n_macro,
                goal_loc, args.goal_val, num_episodes,
            )
            agent1.reset_episode(init_state=0)
            result1 = agent1.run_episode_hierarchical(max_steps=200)

            # Fresh flat agent
            agent2, _ = create_sr_agent(
                args.grid_size, args.walls, args.n_macro,
                goal_loc, args.goal_val, num_episodes,
            )
            agent2.reset_episode(init_state=0)
            result2 = agent2.run_episode_flat(max_steps=200)

            SR_dist.append(result1["steps"])
            SR_dist2.append(result2["steps"])

        SR_dists.append(np.array(SR_dist))
        SR_dists2.append(np.array(SR_dist2))

    return np.array(SR_dists), np.array(SR_dists2)


# ==================== Main ====================


if __name__ == "__main__":
    # Default configuration (matches legacy experiment)
    grid_size = 9
    n_macro = 4
    init_loc = (0, 0)
    goal_loc = (grid_size - 1, grid_size - 1)
    nruns = 20
    eps = [50, 100, 200, 300, 400, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]
    GOALS = [(0, 3), (0, 6), (2, 4), (3, 0), (4, 4), (5, 8), (6, 3), (7, 0), (8, 4), (8, 8)]
    goal_val = 100

    # Serpentine wall pattern
    WALLS = (
        [(1, x) for x in range(grid_size // 2 + 2)]
        + [(3, x) for x in range(grid_size // 2 - 2, grid_size)]
        + [(5, x) for x in range(grid_size // 2 + 2)]
        + [(7, x) for x in range(grid_size // 2 - 2, grid_size)]
    )

    parser = argparse.ArgumentParser(description="Evaluation: Hierarchy vs Flat vs Q-Learning")
    parser.add_argument("--train", action="store_true", help="Run experiments (default: plot from saved data)")
    parser.add_argument("--quick", action="store_true", help="Quick test with small parameters")
    parser.add_argument("--n_runs", type=int, default=nruns, help="Number of seeds")
    parser.add_argument("--no_distances", action="store_true", help="Skip distance experiment")
    args_cli = parser.parse_args()

    # Build args namespace (matches legacy format for plotting compatibility)
    if args_cli.quick:
        eps = [500, 1000]
        nruns = 2

    args = argparse.Namespace(
        grid_size=grid_size,
        n_macro=n_macro,
        init_loc=init_loc,
        goal_loc=goal_loc,
        goal_val=goal_val,
        n_runs=args_cli.n_runs if not args_cli.quick else nruns,
        walls=WALLS,
        episodes=eps,
    )

    if args_cli.train:
        os.makedirs("data/", exist_ok=True)

        # Save args for later plotting
        args_save = vars(args).copy()
        args_save["walls"] = [list(w) for w in args_save["walls"]]
        args_save["init_loc"] = list(args_save["init_loc"])
        args_save["goal_loc"] = list(args_save["goal_loc"])
        with open("data/args.json", "w") as f:
            json.dump(args_save, f, indent=2)

        # Run main experiment
        print("=" * 60)
        print("MAIN EXPERIMENT: Rewards + SR Convergence")
        print("=" * 60)
        (SR_vals, SR_vals2, SR_succ, SR_succ2, SR_succ_macro,
         SR_rewards, SR_rewards2, Q_rewards) = SR_rewards_values(args)

        # Compute relative stability for all three agent types
        SR_rel_stability_hierarchy = np.array([
            relative_stability_paper_style(SR_rewards[i, :])
            for i in range(SR_rewards.shape[0])
        ])
        SR_rel_stability_flat = np.array([
            relative_stability_paper_style(SR_rewards2[i, :])
            for i in range(SR_rewards2.shape[0])
        ])
        Q_rel_stability = np.array([
            relative_stability_paper_style(Q_rewards[i, :])
            for i in range(Q_rewards.shape[0])
        ])

        # Save all data (filenames match legacy for backward compatibility)
        np.save("data/SR_values_hierarchy.npy", SR_vals)
        np.save("data/SR_values_flat.npy", SR_vals2)
        np.save("data/SR_succ_hierarchy.npy", SR_succ)
        np.save("data/SR_succ_flat.npy", SR_succ2)
        np.save("data/SR_succ_macro.npy", SR_succ_macro)
        np.save("data/SR_rewards_hierarchy.npy", SR_rewards)
        np.save("data/SR_rewards_flat.npy", SR_rewards2)
        np.save("data/SR_relative_stability_hierarchy.npy", SR_rel_stability_hierarchy)
        np.save("data/SR_relative_stability_flat.npy", SR_rel_stability_flat)
        np.save("data/Q_rewards.npy", Q_rewards)
        np.save("data/Q_relative_stability.npy", Q_rel_stability)

        print("\nSaved all data to data/")

        # Distance experiment (optional)
        if not args_cli.no_distances:
            print("\n" + "=" * 60)
            print("DISTANCE EXPERIMENT")
            print("=" * 60)
            SR_dists, SR_dists2 = SR_distances(args, GOALS)
            np.save("data/SR_dists_hierarchy.npy", SR_dists)
            np.save("data/SR_dists_flat.npy", SR_dists2)
            print("\nSaved distance data to data/")

    else:
        # Load saved args
        if os.path.exists("data/args.json"):
            with open("data/args.json", "r") as f:
                saved = json.load(f)
                # Convert lists back to tuples where needed
                saved["walls"] = [tuple(w) for w in saved["walls"]]
                saved["init_loc"] = tuple(saved["init_loc"])
                saved["goal_loc"] = tuple(saved["goal_loc"])
                args = argparse.Namespace(**saved)
            print(f"Loaded args: {args}")
        else:
            print("No saved args found. Run with --train first.")

    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    from examples.plot_eval import (
        plot_SR_rewards,
        plot_SR_values,
        plot_relative_stability,
        plot_SR_distances,
    )

    os.makedirs("figures/eval", exist_ok=True)

    plot_SR_rewards(args)
    plot_SR_values(args)
    plot_relative_stability(args)

    if os.path.exists("data/SR_dists_hierarchy.npy"):
        plot_SR_distances(args, GOALS)

    print("\nDone! Figures saved to figures/eval/")
