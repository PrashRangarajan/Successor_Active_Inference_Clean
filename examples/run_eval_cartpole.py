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

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*step.*terminated.*")

import argparse
import json
import time

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-poster")

from core import HierarchicalSRAgent
from environments.cartpole import CartPoleAdapter


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


# ==================== Plotting ====================


def plot_cartpole_steps(args, save_dir="figures/eval/cartpole"):
    """Plot steps survived across training checkpoints."""
    os.makedirs(save_dir, exist_ok=True)
    eps_range = args.episodes

    sr = np.load("data/eval/cartpole/SR_steps.npy")[:, :len(eps_range)]
    rand = np.load("data/eval/cartpole/random_steps.npy")

    mean_sr = np.mean(sr, axis=0)
    std_sr = np.std(sr, axis=0) / np.sqrt(len(sr))
    mean_rand = float(np.mean(rand))
    std_rand = float(np.std(rand) / np.sqrt(len(rand)))

    fig = plt.figure(figsize=(14, 10))
    plt.plot(eps_range, mean_sr, label="SR Agent", marker='o')
    plt.fill_between(eps_range, mean_sr - std_sr, mean_sr + std_sr, alpha=0.3)

    # Random baseline as horizontal line
    plt.axhline(y=mean_rand, color='gray', linestyle='--', alpha=0.7,
                label=f'Random ({mean_rand:.0f}±{std_rand:.0f})')
    plt.axhspan(mean_rand - std_rand, mean_rand + std_rand,
                color='gray', alpha=0.1)

    plt.xlabel("Number of Training Episodes", fontsize=28)
    plt.ylabel("Steps Survived", fontsize=28)
    plt.title("CartPole: Steps Survived (Experimental)", fontsize=28)
    plt.legend(fontsize=22)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cartpole_steps.png"), format="png")
    plt.close()
    print(f"  Saved {save_dir}/cartpole_steps.png")


def plot_cartpole_stability(args, save_dir="figures/eval/cartpole"):
    """Plot relative stability bar chart (SR vs Random)."""
    os.makedirs(save_dir, exist_ok=True)

    sr_path = "data/eval/cartpole/SR_relative_stability.npy"

    if not os.path.exists(sr_path):
        print("  No stability data found — skipping")
        return

    sr_st = np.load(sr_path)

    labels = ["SR Agent"]
    means = [float(np.mean(sr_st))]
    sems = [float(np.std(sr_st) / np.sqrt(len(sr_st)))]

    fig = plt.figure(figsize=(8, 8))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=sems, capsize=8, color='C0')
    plt.xticks(x, labels, fontsize=26)
    plt.yticks(fontsize=26)
    plt.ylabel("Relative Stability", fontsize=20)
    plt.title("CartPole: Relative Stability", fontsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "cartpole_relative_stability.png"), format="png")
    plt.close()
    print(f"  Saved {save_dir}/cartpole_relative_stability.png")


# ==================== Main ====================


if __name__ == "__main__":
    # CartPole configuration
    n_pos_bins = 6
    n_vel_bins = 6
    n_angle_bins = 8
    n_ang_vel_bins = 6
    n_clusters = 6
    gamma = 0.99
    nruns = 5
    eps = [1000, 2000, 4000, 6000, 8000, 10000]
    test_max_steps = 500

    parser = argparse.ArgumentParser(description="CartPole Eval (Experimental)")
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
        n_angle_bins=n_angle_bins,
        n_ang_vel_bins=n_ang_vel_bins,
        n_clusters=n_clusters,
        gamma=gamma,
        n_runs=args_cli.n_runs if not args_cli.quick else nruns,
        episodes=eps,
        test_max_steps=test_max_steps,
        use_replay=True,
        n_replay_epochs=10,
    )

    if args_cli.train:
        os.makedirs("data/eval/cartpole/", exist_ok=True)

        # Save args
        with open("data/eval/cartpole/args.json", "w") as f:
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
        SR_rel_stability = np.array([
            relative_stability_paper_style(SR_steps[i, :])
            for i in range(SR_steps.shape[0])
        ])

        # Save data
        np.save("data/eval/cartpole/SR_steps.npy", SR_steps)
        np.save("data/eval/cartpole/random_steps.npy", random_steps)
        np.save("data/eval/cartpole/SR_relative_stability.npy", SR_rel_stability)
        print("\nSaved all data to data/eval/cartpole/")

    else:
        # Load saved args
        if os.path.exists("data/eval/cartpole/args.json"):
            with open("data/eval/cartpole/args.json", "r") as f:
                saved = json.load(f)
                args = argparse.Namespace(**saved)
            print(f"Loaded args: {args}")
        else:
            print("No saved args found. Run with --train first.")

    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    os.makedirs("figures/eval/cartpole", exist_ok=True)

    if os.path.exists("data/eval/cartpole/SR_steps.npy"):
        plot_cartpole_steps(args)

    plot_cartpole_stability(args)

    print("\nDone! Figures saved to figures/eval/cartpole/")
