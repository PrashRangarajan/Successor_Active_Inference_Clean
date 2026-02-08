"""Evaluation: CartPole benchmark (experimental).

CartPole is a survival task — success = steps survived (not goal-reaching).
This script evaluates the SR agent's policy using gym terminated/truncated
signals, NOT the agent's internal goal check.

Compares SR agent performance against a random baseline.

Saves .npy data files to data/eval/cartpole/ and figures to figures/eval/cartpole/.

Usage:
    # Run experiments:
    python examples/run_eval_cartpole.py --train

    # Plot from saved data:
    python examples/run_eval_cartpole.py

    # Quick test (2 seeds × 3 checkpoints):
    python examples/run_eval_cartpole.py --train --quick
"""

import os

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*step.*terminated.*")

import argparse
import json
import time
from collections import OrderedDict

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-poster")

from core import HierarchicalSRAgent
from core.eval_utils import (
    relative_stability,
    compute_stability_array,
    plot_reward_curves,
    plot_stability_bars,
    save_eval_data,
    load_eval_args,
)
from environments.cartpole import CartPoleAdapter
from examples.configs import CARTPOLE, SHARED

# ==================== Agent Factory ====================

def create_cartpole_agent(n_pos_bins, n_vel_bins, n_angle_bins, n_ang_vel_bins,
                          n_clusters, num_episodes, gamma=0.99, learning_rate=0.05,
                          use_replay=True, n_replay_epochs=10,
                          test_max_steps=500):
    """Create a fresh CartPole SR agent trained for exactly num_episodes.

    Returns:
        (agent, test_adapter) tuple
    """
    # Training environment
    env_train = gym.make('CartPole-v1')
    adapter_train = CartPoleAdapter(
        env_train,
        n_pos_bins=n_pos_bins,
        n_vel_bins=n_vel_bins,
        n_angle_bins=n_angle_bins,
        n_ang_vel_bins=n_ang_vel_bins,
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
    env_test = gym.make('CartPole-v1', max_episode_steps=test_max_steps)
    adapter_test = CartPoleAdapter(
        env_test,
        n_pos_bins=n_pos_bins,
        n_vel_bins=n_vel_bins,
        n_angle_bins=n_angle_bins,
        n_ang_vel_bins=n_ang_vel_bins,
    )
    agent.adapter = adapter_test

    return agent, adapter_test

def run_evaluation_episode(agent, adapter, init_state, max_steps):
    """Run an episode using the agent's policy.

    Uses gym terminated/truncated signals to detect episode end (NOT _is_at_goal),
    since CartPole is a survival task.

    Returns:
        Number of steps survived.
    """
    adapter.reset(init_state)
    V = adapter.multiply_M_C(agent.M, agent.C)

    done = False
    steps = 0

    while not done and steps < max_steps:
        state_onehot = adapter._current_state

        # Compute expected value for each action
        V_adj = []
        for act in range(adapter.n_actions):
            s_next_dist = adapter.multiply_B_s(agent.B, state_onehot, act)
            V_adj.append(float(s_next_dist @ V))

        best_action = np.argmax(V_adj)

        _, _, terminated, truncated, _ = adapter.step_with_info(best_action)
        done = terminated or truncated
        steps += 1

    return steps

def run_random_episode(adapter, init_state, max_steps):
    """Run an episode with random actions (baseline).

    Returns:
        Number of steps survived.
    """
    adapter.reset(init_state)

    done = False
    steps = 0

    while not done and steps < max_steps:
        action = np.random.randint(adapter.n_actions)
        _, _, terminated, truncated, _ = adapter.step_with_info(action)
        done = terminated or truncated
        steps += 1

    return steps

# ==================== Experiment ====================

def cartpole_steps_experiment(args):
    """Main experiment: steps survived across training checkpoints.

    For CartPole, we measure steps survived (not reward/goal-reaching).
    Also runs random baseline episodes for comparison.

    Returns:
        Tuple of (SR_steps, random_steps)
    """
    n_trials = len(args.episodes)

    SR_steps = np.zeros((args.n_runs, n_trials))
    random_steps = np.zeros((args.n_runs,))

    init_state = [0.0, 0.0, 0.0, 0.0]

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
                    agent, adapter = create_cartpole_agent(
                        args.n_pos_bins, args.n_vel_bins,
                        args.n_angle_bins, args.n_ang_vel_bins,
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

            # Evaluate SR agent (average of 5 test episodes for stability)
            test_steps = []
            for _ in range(5):
                s = run_evaluation_episode(agent, adapter, init_state, args.test_max_steps)
                test_steps.append(s)
            avg_steps = np.mean(test_steps)

            SR_steps[n, trial] = avg_steps

            print(f"  SR agent: {avg_steps:.0f} avg steps "
                  f"(range {min(test_steps)}-{max(test_steps)})")

        # Random baseline (5 episodes)
        env_rand = gym.make('CartPole-v1', max_episode_steps=args.test_max_steps)
        adapter_rand = CartPoleAdapter(
            env_rand,
            n_pos_bins=args.n_pos_bins,
            n_vel_bins=args.n_vel_bins,
            n_angle_bins=args.n_angle_bins,
            n_ang_vel_bins=args.n_ang_vel_bins,
        )
        rand_steps = [run_random_episode(adapter_rand, init_state, args.test_max_steps)
                      for _ in range(5)]
        random_steps[n] = np.mean(rand_steps)
        print(f"\n  Random baseline: {np.mean(rand_steps):.0f} avg steps")
        env_rand.close()

    return SR_steps, random_steps

# ==================== Main ====================

if __name__ == "__main__":
    # CartPole configuration (from centralized config)
    n_pos_bins = CARTPOLE["n_pos_bins"]
    n_vel_bins = CARTPOLE["n_vel_bins"]
    n_angle_bins = CARTPOLE["n_angle_bins"]
    n_ang_vel_bins = CARTPOLE["n_ang_vel_bins"]
    n_clusters = CARTPOLE["n_clusters"]
    gamma = CARTPOLE["gamma"]
    nruns = CARTPOLE["eval_n_runs"]
    eps = list(CARTPOLE["eval_episodes"])
    test_max_steps = CARTPOLE["test_max_steps"]

    parser = argparse.ArgumentParser(description="CartPole Eval (Experimental)")
    parser.add_argument("--train", action="store_true", help="Run experiments")
    parser.add_argument("--quick", action="store_true", help="Quick test")
    parser.add_argument("--n_runs", type=int, default=nruns)
    args_cli = parser.parse_args()

    if args_cli.quick:
        eps = list(CARTPOLE["eval_quick_episodes"])
        nruns = CARTPOLE["eval_quick_n_runs"]

    args = argparse.Namespace(
        n_pos_bins=n_pos_bins,
        n_vel_bins=n_vel_bins,
        n_angle_bins=n_angle_bins,
        n_ang_vel_bins=n_ang_vel_bins,
        n_clusters=n_clusters,
        gamma=gamma,
        n_runs=args_cli.n_runs if not args_cli.quick else nruns,
        episodes=eps,
        test_max_steps=test_max_steps,
        use_replay=SHARED["use_replay"],
        n_replay_epochs=SHARED["n_replay_epochs"],
    )

    data_dir = "data/eval/cartpole"
    save_dir = "figures/eval/cartpole"

    if args_cli.train:
        os.makedirs(data_dir, exist_ok=True)

        # Save args
        with open(os.path.join(data_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

        n_states = n_pos_bins * n_vel_bins * n_angle_bins * n_ang_vel_bins
        print("=" * 60)
        print("CARTPOLE EVAL (Experimental): SR Agent vs Random")
        print("=" * 60)
        print(f"State space: {n_states} states")
        print(f"Actions: 2 (push left, push right)")
        print(f"Runs: {args.n_runs}, Checkpoints: {args.episodes}")
        print(f"Note: CartPole is a survival task — measuring steps survived")

        t0 = time.time()
        SR_steps, random_steps = cartpole_steps_experiment(args)
        elapsed = time.time() - t0
        print(f"\nExperiment completed in {elapsed:.0f}s")

        # Compute relative stability
        SR_rel_stability = compute_stability_array(SR_steps)

        # Save data
        save_eval_data(data_dir, {
            "SR_steps": SR_steps,
            "random_steps": random_steps,
            "SR_relative_stability": SR_rel_stability,
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

    if os.path.exists(os.path.join(data_dir, "SR_steps.npy")):
        data = OrderedDict()
        data["SR Agent"] = np.load(os.path.join(data_dir, "SR_steps.npy"))
        plot_reward_curves(args.episodes, data, os.path.join(save_dir, "cartpole_steps.png"),
                           ylabel="Steps Survived")

    stability_data = OrderedDict()
    sr_stab_path = os.path.join(data_dir, "SR_relative_stability.npy")
    if os.path.exists(sr_stab_path):
        stability_data["SR Agent"] = np.load(sr_stab_path)
    plot_stability_bars(stability_data, os.path.join(save_dir, "cartpole_relative_stability.png"))

    print(f"\nDone! Figures saved to {save_dir}/")
