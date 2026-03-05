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

import os

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import json
import time
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

try:
    plt.style.use("seaborn-v0_8-poster")
except OSError:
    plt.style.use("seaborn-poster")

from core import HierarchicalSRAgent
from core.q_learning import QLearningAgent
from core.eval_utils import (
    relative_stability,
    compute_stability_array,
    plot_reward_curves,
    plot_step_curves,
    plot_stability_bars,
    save_eval_data,
    load_eval_args,
)
from environments.key_gridworld import KeyGridworldAdapter
from examples.configs import KEY_GRIDWORLD, SHARED
from unified_env import KeyGridworld as SR_Gridworld

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

# ==================== Main ====================

if __name__ == "__main__":
    # Key Gridworld configuration (from centralized config)
    grid_size = KEY_GRIDWORLD["grid_size"]
    n_clusters = KEY_GRIDWORLD["n_clusters"]
    gamma = KEY_GRIDWORLD["gamma"]
    nruns = KEY_GRIDWORLD["eval_n_runs"]
    eps = list(KEY_GRIDWORLD["eval_episodes"])
    test_max_steps = KEY_GRIDWORLD["test_max_steps"]
    has_pickup_action = KEY_GRIDWORLD["has_pickup_action"]

    init_loc = KEY_GRIDWORLD["init_loc"]
    key_loc = KEY_GRIDWORLD["key_loc"]
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
        eps = list(KEY_GRIDWORLD["eval_quick_episodes"])
        nruns = KEY_GRIDWORLD["eval_quick_n_runs"]

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
        use_replay=SHARED["use_replay"],
        n_replay_epochs=SHARED["n_replay_epochs"],
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
        SR_rel_stability_hier = compute_stability_array(SR_rewards_hier)
        SR_rel_stability_flat = compute_stability_array(SR_rewards_flat)
        Q_rel_stability = compute_stability_array(Q_rewards)

        # Save data
        save_eval_data(data_dir, {
            "SR_rewards_hierarchy": SR_rewards_hier,
            "SR_rewards_flat": SR_rewards_flat,
            "Q_rewards": Q_rewards,
            "SR_steps_hierarchy": SR_steps_hier,
            "SR_steps_flat": SR_steps_flat,
            "Q_steps": Q_steps,
            "SR_relative_stability_hierarchy": SR_rel_stability_hier,
            "SR_relative_stability_flat": SR_rel_stability_flat,
            "Q_relative_stability": Q_rel_stability,
        })

    else:
        # Load saved args
        args = load_eval_args(data_dir, tuple_keys=["walls", "init_loc", "key_loc", "goal_loc", "goal_spec"])
        if args is None:
            print("No saved args found. Run with --train first.")

    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(os.path.join(data_dir, "SR_rewards_hierarchy.npy")):
        data = OrderedDict()
        data["Hierarchy"] = np.load(os.path.join(data_dir, "SR_rewards_hierarchy.npy"))
        data["Flat"] = np.load(os.path.join(data_dir, "SR_rewards_flat.npy"))
        q_path = os.path.join(data_dir, "Q_rewards.npy")
        if os.path.exists(q_path):
            data["Q-Learning"] = np.load(q_path)
        plot_reward_curves(args.episodes, data, os.path.join(save_dir, "key_gridworld_reward.png"))

    if os.path.exists(os.path.join(data_dir, "SR_steps_hierarchy.npy")):
        data = OrderedDict()
        data["Hierarchy"] = np.load(os.path.join(data_dir, "SR_steps_hierarchy.npy"))
        data["Flat"] = np.load(os.path.join(data_dir, "SR_steps_flat.npy"))
        q_path = os.path.join(data_dir, "Q_steps.npy")
        if os.path.exists(q_path):
            data["Q-Learning"] = np.load(q_path)
        plot_step_curves(args.episodes, data, os.path.join(save_dir, "key_gridworld_steps.png"))

    stability_data = OrderedDict()
    hier_stab_path = os.path.join(data_dir, "SR_relative_stability_hierarchy.npy")
    flat_stab_path = os.path.join(data_dir, "SR_relative_stability_flat.npy")
    q_stab_path = os.path.join(data_dir, "Q_relative_stability.npy")
    if os.path.exists(hier_stab_path):
        stability_data["Hierarchy"] = np.load(hier_stab_path)
    if os.path.exists(flat_stab_path):
        stability_data["Flat"] = np.load(flat_stab_path)
    if os.path.exists(q_stab_path):
        stability_data["Q-Learning"] = np.load(q_stab_path)
    plot_stability_bars(stability_data, os.path.join(save_dir, "key_gridworld_relative_stability.png"))

    print(f"\nDone! Figures saved to {save_dir}/")
