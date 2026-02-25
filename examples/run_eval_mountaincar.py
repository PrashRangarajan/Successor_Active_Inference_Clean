"""Evaluation: Mountain Car benchmark comparing Hierarchy vs Flat vs Q-Learning.

Runs repeated experiments across training checkpoints to compare
hierarchical vs flat active inference and Q-learning on the continuous
Mountain Car environment.

Saves .npy data files to data/eval/mountaincar/ and figures to figures/eval/mountaincar/.

Usage:
    # Run experiments:
    python examples/run_eval_mountaincar.py --train

    # Plot from saved data:
    python examples/run_eval_mountaincar.py

    # Quick test (2 seeds × 3 checkpoints):
    python examples/run_eval_mountaincar.py --train --quick
"""

import os

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import json
import time
from collections import OrderedDict

import numpy as np
import gymnasium as gym
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
from environments.mountain_car import MountainCarAdapter
from examples.configs import MOUNTAINCAR, SHARED

# ==================== Agent Factory ====================

def create_mountaincar_agent(n_pos_bins, n_vel_bins, n_clusters,
                             num_episodes, gamma=0.95, learning_rate=0.05,
                             use_replay=True, n_replay_epochs=10,
                             test_max_steps=500,
                             train_smooth_steps=10, test_smooth_steps=1):
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
        train_smooth_steps=train_smooth_steps,
        test_smooth_steps=test_smooth_steps,
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

def create_mountaincar_q_agent(n_pos_bins, n_vel_bins, gamma=0.95,
                                test_max_steps=500):
    """Create a fresh Q-learning agent for Mountain Car (not yet trained).

    The Q-learning agent's learn() IS incremental, so we create one per run
    and call learn(delta) at each checkpoint.

    Returns:
        (q_agent, adapter) tuple
    """
    env = gym.make('MountainCar-v0', max_episode_steps=test_max_steps)
    adapter = MountainCarAdapter(env, n_pos_bins=n_pos_bins, n_vel_bins=n_vel_bins)

    goal_states = adapter.get_goal_states(None)
    C = adapter.create_goal_prior(goal_states, reward=100.0, default_cost=-1.0)

    q_agent = QLearningAgent(
        adapter=adapter,
        goal_states=goal_states,
        C=C,
        gamma=gamma,
        epsilon_decay=0.999,   # Slower decay for continuous env exploration
        epsilon_end=0.1,       # Maintain exploration to avoid Q-value instability
    )
    return q_agent, adapter

# ==================== Experiment ====================

def mountaincar_rewards_experiment(args):
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

    init_state = [-0.5, 0.0]

    for n in range(args.n_runs):
        print("x" * 40)
        print(f"Run: {n + 1}/{args.n_runs}")
        print("x" * 40)

        # Create Q-learning agent (incremental — one per run)
        q_agent, q_adapter = create_mountaincar_q_agent(
            args.n_pos_bins, args.n_vel_bins,
            gamma=args.gamma,
            test_max_steps=args.test_max_steps,
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
                    agent, adapter = create_mountaincar_agent(
                        args.n_pos_bins, args.n_vel_bins,
                        args.n_clusters, num_episodes,
                        gamma=args.gamma,
                        use_replay=args.use_replay,
                        n_replay_epochs=args.n_replay_epochs,
                        test_max_steps=args.test_max_steps,
                        train_smooth_steps=args.train_smooth_steps,
                        test_smooth_steps=args.test_smooth_steps,
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
    # Mountain Car configuration (from centralized config)
    n_pos_bins = MOUNTAINCAR["n_pos_bins"]
    n_vel_bins = MOUNTAINCAR["n_vel_bins"]
    n_clusters = MOUNTAINCAR["n_clusters"]
    gamma = MOUNTAINCAR["gamma"]
    nruns = MOUNTAINCAR["eval_n_runs"]
    eps = list(MOUNTAINCAR["eval_episodes"])
    test_max_steps = MOUNTAINCAR["test_max_steps"]

    parser = argparse.ArgumentParser(description="Mountain Car Eval: Hierarchy vs Flat vs Q-Learning")
    parser.add_argument("--train", action="store_true", help="Run experiments")
    parser.add_argument("--quick", action="store_true", help="Quick test")
    parser.add_argument("--n_runs", type=int, default=nruns)
    args_cli = parser.parse_args()

    if args_cli.quick:
        eps = list(MOUNTAINCAR["eval_quick_episodes"])
        nruns = MOUNTAINCAR["eval_quick_n_runs"]

    args = argparse.Namespace(
        n_pos_bins=n_pos_bins,
        n_vel_bins=n_vel_bins,
        n_clusters=n_clusters,
        gamma=gamma,
        n_runs=args_cli.n_runs if not args_cli.quick else nruns,
        episodes=eps,
        test_max_steps=test_max_steps,
        train_smooth_steps=MOUNTAINCAR["train_smooth_steps"],
        test_smooth_steps=MOUNTAINCAR["test_smooth_steps"],
        use_replay=SHARED["use_replay"],
        n_replay_epochs=SHARED["n_replay_epochs"],
    )

    data_dir = "data/eval/mountaincar"
    save_dir = "figures/eval/mountaincar"

    if args_cli.train:
        os.makedirs(data_dir, exist_ok=True)

        # Save args
        with open(os.path.join(data_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

        print("=" * 60)
        print("MOUNTAIN CAR EVAL: Hierarchy vs Flat vs Q-Learning")
        print("=" * 60)
        print(f"State space: {n_pos_bins * n_vel_bins} states")
        print(f"Runs: {args.n_runs}, Checkpoints: {args.episodes}")

        t0 = time.time()
        SR_rewards_hier, SR_rewards_flat, Q_rewards, SR_steps_hier, SR_steps_flat, Q_steps = \
            mountaincar_rewards_experiment(args)
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
        args = load_eval_args(data_dir)
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
        plot_reward_curves(args.episodes, data, os.path.join(save_dir, "mountaincar_reward.png"))

    if os.path.exists(os.path.join(data_dir, "SR_steps_hierarchy.npy")):
        data = OrderedDict()
        data["Hierarchy"] = np.load(os.path.join(data_dir, "SR_steps_hierarchy.npy"))
        data["Flat"] = np.load(os.path.join(data_dir, "SR_steps_flat.npy"))
        q_path = os.path.join(data_dir, "Q_steps.npy")
        if os.path.exists(q_path):
            data["Q-Learning"] = np.load(q_path)
        plot_step_curves(args.episodes, data, os.path.join(save_dir, "mountaincar_steps.png"))

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
    plot_stability_bars(stability_data, os.path.join(save_dir, "mountaincar_relative_stability.png"))

    print(f"\nDone! Figures saved to {save_dir}/")
