"""Evaluation: Pendulum benchmark comparing Hierarchy vs Flat.

Runs repeated experiments across training checkpoints to compare
hierarchical vs flat active inference on the Pendulum swing-up task.

Saves .npy data files to data/eval/pendulum/ and figures to figures/eval/pendulum/.

Usage:
    # Run experiments:
    python examples/run_eval_pendulum.py --train

    # Plot from saved data:
    python examples/run_eval_pendulum.py

    # Quick test (2 seeds × 3 checkpoints):
    python examples/run_eval_pendulum.py --train --quick
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
from core.eval_utils import (
    relative_stability,
    compute_stability_array,
    plot_reward_curves,
    plot_step_curves,
    plot_stability_bars,
    save_eval_data,
    load_eval_args,
)
from environments.pendulum import PendulumAdapter
from examples.configs import PENDULUM, SHARED

# ==================== Agent Factory ====================

def create_pendulum_agent(n_theta_bins, n_omega_bins, n_torque_bins, n_clusters,
                          num_episodes, gamma=0.95, learning_rate=0.05,
                          use_replay=True, n_replay_epochs=10,
                          test_max_steps=200,
                          train_smooth_steps=5, test_smooth_steps=5):
    """Create a fresh Pendulum SR agent trained for exactly num_episodes.

    Returns:
        (agent, test_adapter) tuple
    """
    # Training environment
    env_train = gym.make('Pendulum-v1')
    adapter_train = PendulumAdapter(
        env_train,
        n_theta_bins=n_theta_bins,
        n_omega_bins=n_omega_bins,
        n_torque_bins=n_torque_bins,
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
    # Shaped goal: mirrors Pendulum-v1 reward -(θ² + 0.1·ω²)
    C_shaped = adapter_train.create_shaped_prior(scale=10.0)
    agent.set_shaped_goal(C_shaped, goal_threshold=-1.0)
    agent.learn_environment(num_episodes)

    # Switch to test environment (Pendulum has default 200-step limit)
    env_test = gym.make('Pendulum-v1')
    adapter_test = PendulumAdapter(
        env_test,
        n_theta_bins=n_theta_bins,
        n_omega_bins=n_omega_bins,
        n_torque_bins=n_torque_bins,
    )
    agent.adapter = adapter_test

    return agent, adapter_test

# ==================== Experiment ====================

def pendulum_rewards_experiment(args):
    """Main experiment: rewards across training checkpoints for Hierarchy vs Flat.

    Returns:
        Tuple of (SR_rewards_hier, SR_rewards_flat, SR_steps_hier, SR_steps_flat)
    """
    n_trials = len(args.episodes)

    SR_rewards_hier = np.zeros((args.n_runs, n_trials))
    SR_rewards_flat = np.zeros((args.n_runs, n_trials))
    SR_steps_hier = np.zeros((args.n_runs, n_trials))
    SR_steps_flat = np.zeros((args.n_runs, n_trials))

    init_state = [np.pi, 0.0]  # Hanging down, zero velocity

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
                    agent, adapter = create_pendulum_agent(
                        args.n_theta_bins, args.n_omega_bins,
                        args.n_torque_bins, args.n_clusters,
                        num_episodes,
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

            SR_rewards_hier[n, trial] = result_hier["reward"]
            SR_rewards_flat[n, trial] = result_flat["reward"]
            SR_steps_hier[n, trial] = result_hier["steps"]
            SR_steps_flat[n, trial] = result_flat["steps"]

            print(f"  Hier: reward={result_hier['reward']:.1f}, "
                  f"steps={result_hier['steps']}, goal={result_hier['reached_goal']}")
            print(f"  Flat: reward={result_flat['reward']:.1f}, "
                  f"steps={result_flat['steps']}, goal={result_flat['reached_goal']}")

    return SR_rewards_hier, SR_rewards_flat, SR_steps_hier, SR_steps_flat

# ==================== Main ====================

if __name__ == "__main__":
    # Pendulum configuration (from centralized config)
    n_theta_bins = PENDULUM["n_theta_bins"]
    n_omega_bins = PENDULUM["n_omega_bins"]
    n_torque_bins = PENDULUM["n_torque_bins"]
    n_clusters = PENDULUM["n_clusters"]
    gamma = PENDULUM["gamma"]
    nruns = PENDULUM["eval_n_runs"]
    eps = list(PENDULUM["eval_episodes"])
    test_max_steps = PENDULUM["test_max_steps"]

    parser = argparse.ArgumentParser(description="Pendulum Eval: Hierarchy vs Flat")
    parser.add_argument("--train", action="store_true", help="Run experiments")
    parser.add_argument("--quick", action="store_true", help="Quick test")
    parser.add_argument("--n_runs", type=int, default=nruns)
    args_cli = parser.parse_args()

    if args_cli.quick:
        eps = list(PENDULUM["eval_quick_episodes"])
        nruns = PENDULUM["eval_quick_n_runs"]

    args = argparse.Namespace(
        n_theta_bins=n_theta_bins,
        n_omega_bins=n_omega_bins,
        n_torque_bins=n_torque_bins,
        n_clusters=n_clusters,
        gamma=gamma,
        n_runs=args_cli.n_runs if not args_cli.quick else nruns,
        episodes=eps,
        test_max_steps=test_max_steps,
        train_smooth_steps=PENDULUM.get("train_smooth_steps", 5),
        test_smooth_steps=PENDULUM.get("test_smooth_steps", 5),
        use_replay=SHARED["use_replay"],
        n_replay_epochs=SHARED["n_replay_epochs"],
    )

    data_dir = "data/eval/pendulum"
    save_dir = "figures/eval/pendulum"

    if args_cli.train:
        os.makedirs(data_dir, exist_ok=True)

        # Save args
        with open(os.path.join(data_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

        print("=" * 60)
        print("PENDULUM EVAL: Hierarchy vs Flat")
        print("=" * 60)
        print(f"State space: {n_theta_bins * n_omega_bins} states")
        print(f"Actions: {n_torque_bins} discrete torques")
        print(f"Runs: {args.n_runs}, Checkpoints: {args.episodes}")

        t0 = time.time()
        SR_rewards_hier, SR_rewards_flat, SR_steps_hier, SR_steps_flat = \
            pendulum_rewards_experiment(args)
        elapsed = time.time() - t0
        print(f"\nExperiment completed in {elapsed:.0f}s")

        # Compute relative stability
        SR_rel_stability_hier = compute_stability_array(SR_rewards_hier)
        SR_rel_stability_flat = compute_stability_array(SR_rewards_flat)

        # Save data
        save_eval_data(data_dir, {
            "SR_rewards_hierarchy": SR_rewards_hier,
            "SR_rewards_flat": SR_rewards_flat,
            "SR_steps_hierarchy": SR_steps_hier,
            "SR_steps_flat": SR_steps_flat,
            "SR_relative_stability_hierarchy": SR_rel_stability_hier,
            "SR_relative_stability_flat": SR_rel_stability_flat,
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
        plot_reward_curves(args.episodes, data, os.path.join(save_dir, "pendulum_reward.png"))

    if os.path.exists(os.path.join(data_dir, "SR_steps_hierarchy.npy")):
        data = OrderedDict()
        data["Hierarchy"] = np.load(os.path.join(data_dir, "SR_steps_hierarchy.npy"))
        data["Flat"] = np.load(os.path.join(data_dir, "SR_steps_flat.npy"))
        plot_step_curves(args.episodes, data, os.path.join(save_dir, "pendulum_steps.png"))

    stability_data = OrderedDict()
    hier_stab_path = os.path.join(data_dir, "SR_relative_stability_hierarchy.npy")
    flat_stab_path = os.path.join(data_dir, "SR_relative_stability_flat.npy")
    if os.path.exists(hier_stab_path):
        stability_data["Hierarchy"] = np.load(hier_stab_path)
    if os.path.exists(flat_stab_path):
        stability_data["Flat"] = np.load(flat_stab_path)
    plot_stability_bars(stability_data, os.path.join(save_dir, "pendulum_relative_stability.png"))

    print(f"\nDone! Figures saved to {save_dir}/")
