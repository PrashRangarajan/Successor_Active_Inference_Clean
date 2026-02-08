"""Evaluation: Smooth stepping comparison for Mountain Car.

Compares 4 configurations of (train_smooth_steps, test_smooth_steps):
  1. train=1,  test=1   (no smooth stepping at all)
  2. train=10, test=1   (smooth train, single-step test — previous default)
  3. train=1,  test=10  (single-step train, smooth test)
  4. train=10, test=10  (smooth everywhere)

For each configuration, trains multiple seeds and evaluates both hierarchical
and flat policies.  Reports success rate and average steps.

Usage:
    # Run experiments:
    python examples/run_eval_smooth_stepping.py --train

    # Plot from saved data:
    python examples/run_eval_smooth_stepping.py

    # Quick test (2 seeds):
    python examples/run_eval_smooth_stepping.py --train --quick
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import time

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-poster")

from core import HierarchicalSRAgent
from environments.mountain_car import MountainCarAdapter

# ==================== Configuration ====================

CONFIGS = [
    {"name": "train=1, test=1",   "train_smooth": 1,  "test_smooth": 1},
    {"name": "train=10, test=1",  "train_smooth": 10, "test_smooth": 1},
    {"name": "train=1, test=10",  "train_smooth": 1,  "test_smooth": 10},
    {"name": "train=10, test=10", "train_smooth": 10, "test_smooth": 10},
]


# ==================== Agent Factory ====================


def create_agent(train_smooth_steps, test_smooth_steps,
                 num_episodes=4000, test_max_steps=500,
                 n_pos_bins=10, n_vel_bins=10, n_clusters=6):
    """Create a trained Mountain Car agent with specified smooth stepping."""
    env_train = gym.make('MountainCar-v0', max_episode_steps=500)
    adapter_train = MountainCarAdapter(
        env_train, n_pos_bins=n_pos_bins, n_vel_bins=n_vel_bins,
    )

    agent = HierarchicalSRAgent(
        adapter=adapter_train,
        n_clusters=n_clusters,
        gamma=0.95,
        learning_rate=0.05,
        learn_from_experience=True,
        use_replay=True,
        n_replay_epochs=10,
        train_smooth_steps=train_smooth_steps,
        test_smooth_steps=test_smooth_steps,
    )
    agent.set_goal(None, reward=100.0, default_cost=-1.0)
    agent.learn_environment(num_episodes)

    # Switch to test environment
    env_test = gym.make('MountainCar-v0', max_episode_steps=test_max_steps)
    adapter_test = MountainCarAdapter(
        env_test, n_pos_bins=n_pos_bins, n_vel_bins=n_vel_bins,
    )
    agent.adapter = adapter_test

    return agent, adapter_test


# ==================== Experiment ====================


def run_experiment(args):
    """Run the 4-config comparison experiment."""
    n_configs = len(CONFIGS)
    n_test = args.n_test_episodes
    init_state = [-0.5, 0.0]

    # Results: [n_runs, n_configs] for each metric
    results = {
        'hier_success': np.zeros((args.n_runs, n_configs)),
        'hier_steps':   np.zeros((args.n_runs, n_configs)),
        'flat_success': np.zeros((args.n_runs, n_configs)),
        'flat_steps':   np.zeros((args.n_runs, n_configs)),
    }

    for ci, config in enumerate(CONFIGS):
        print(f"\n{'='*60}")
        print(f"Config {ci+1}/{n_configs}: {config['name']}")
        print(f"{'='*60}")

        for seed in range(args.n_runs):
            np.random.seed(seed * 1000 + ci)
            t0 = time.time()

            print(f"  Seed {seed+1}/{args.n_runs} ... ", end="", flush=True)

            agent, adapter = create_agent(
                train_smooth_steps=config['train_smooth'],
                test_smooth_steps=config['test_smooth'],
                num_episodes=args.num_train_episodes,
                test_max_steps=args.test_max_steps,
            )

            # Evaluate hierarchical policy
            hier_successes = []
            hier_steps_list = []
            for _ in range(n_test):
                agent.reset_episode(init_state=init_state)
                result = agent.run_episode_hierarchical(max_steps=args.test_max_steps)
                hier_successes.append(float(result['reached_goal']))
                hier_steps_list.append(result['steps'])

            # Evaluate flat policy
            flat_successes = []
            flat_steps_list = []
            for _ in range(n_test):
                agent.reset_episode(init_state=init_state)
                result = agent.run_episode_flat(max_steps=args.test_max_steps)
                flat_successes.append(float(result['reached_goal']))
                flat_steps_list.append(result['steps'])

            results['hier_success'][seed, ci] = np.mean(hier_successes)
            results['hier_steps'][seed, ci] = np.mean(hier_steps_list)
            results['flat_success'][seed, ci] = np.mean(flat_successes)
            results['flat_steps'][seed, ci] = np.mean(flat_steps_list)

            elapsed = time.time() - t0
            print(f"hier={np.mean(hier_successes)*100:.0f}% "
                  f"flat={np.mean(flat_successes)*100:.0f}%  ({elapsed:.1f}s)")

            adapter.env.close()

    return results


# ==================== Plotting ====================


def plot_comparison(results, save_dir):
    """Generate grouped bar charts comparing the 4 configurations."""
    os.makedirs(save_dir, exist_ok=True)

    config_names = [c['name'] for c in CONFIGS]
    n_configs = len(config_names)
    x = np.arange(n_configs)
    width = 0.35

    # --- Success Rate ---
    fig, ax = plt.subplots(figsize=(12, 6))
    hier_mean = results['hier_success'].mean(axis=0) * 100
    hier_std = results['hier_success'].std(axis=0) * 100
    flat_mean = results['flat_success'].mean(axis=0) * 100
    flat_std = results['flat_success'].std(axis=0) * 100

    bars1 = ax.bar(x - width/2, hier_mean, width, yerr=hier_std,
                   label='Hierarchical', color='steelblue', capsize=4)
    bars2 = ax.bar(x + width/2, flat_mean, width, yerr=flat_std,
                   label='Flat', color='coral', capsize=4)

    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Mountain Car: Smooth Stepping Comparison — Success Rate')
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f'{h:.0f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    path = os.path.join(save_dir, "smooth_stepping_success.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # --- Average Steps ---
    fig, ax = plt.subplots(figsize=(12, 6))
    hier_mean_s = results['hier_steps'].mean(axis=0)
    hier_std_s = results['hier_steps'].std(axis=0)
    flat_mean_s = results['flat_steps'].mean(axis=0)
    flat_std_s = results['flat_steps'].std(axis=0)

    bars1 = ax.bar(x - width/2, hier_mean_s, width, yerr=hier_std_s,
                   label='Hierarchical', color='steelblue', capsize=4)
    bars2 = ax.bar(x + width/2, flat_mean_s, width, yerr=flat_std_s,
                   label='Flat', color='coral', capsize=4)

    ax.set_ylabel('Average Steps (physics)')
    ax.set_title('Mountain Car: Smooth Stepping Comparison — Steps to Goal')
    ax.set_xticks(x)
    ax.set_xticklabels(config_names, fontsize=11)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(f'{h:.0f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    path = os.path.join(save_dir, "smooth_stepping_steps.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def print_summary(results):
    """Print a summary table of results."""
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<22} {'Hier Succ%':>10} {'Hier Steps':>11} "
          f"{'Flat Succ%':>10} {'Flat Steps':>11}")
    print("-" * 70)
    for ci, config in enumerate(CONFIGS):
        hs = results['hier_success'][:, ci].mean() * 100
        hst = results['hier_steps'][:, ci].mean()
        fs = results['flat_success'][:, ci].mean() * 100
        fst = results['flat_steps'][:, ci].mean()
        print(f"{config['name']:<22} {hs:>9.1f}% {hst:>11.1f} "
              f"{fs:>9.1f}% {fst:>11.1f}")
    print(f"{'='*70}")


# ==================== Main ====================


def main():
    parser = argparse.ArgumentParser(
        description="Smooth stepping comparison for Mountain Car")
    parser.add_argument('--train', action='store_true',
                        help="Run experiments (otherwise just plot)")
    parser.add_argument('--quick', action='store_true',
                        help="Quick test mode (2 seeds, 3 test episodes)")
    parser.add_argument('--n-runs', type=int, default=5,
                        help="Number of random seeds per config")
    parser.add_argument('--num-train-episodes', type=int, default=4000,
                        help="Training episodes per seed")
    parser.add_argument('--n-test-episodes', type=int, default=5,
                        help="Test episodes per seed per policy")
    parser.add_argument('--test-max-steps', type=int, default=500,
                        help="Max steps per test episode")
    args = parser.parse_args()

    if args.quick:
        args.n_runs = 2
        args.n_test_episodes = 3
        args.num_train_episodes = 3000

    data_dir = "data/eval/mountaincar_smooth"
    fig_dir = "figures/eval/mountaincar_smooth"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    if args.train:
        print("Running smooth stepping comparison experiment...")
        results = run_experiment(args)

        # Save raw data
        for key, arr in results.items():
            np.save(os.path.join(data_dir, f"{key}.npy"), arr)
        print(f"\nSaved raw data to {data_dir}/")

        print_summary(results)
        plot_comparison(results, fig_dir)
    else:
        # Load and plot
        results = {}
        for key in ['hier_success', 'hier_steps', 'flat_success', 'flat_steps']:
            path = os.path.join(data_dir, f"{key}.npy")
            if not os.path.exists(path):
                print(f"No data found at {path}. Run with --train first.")
                return
            results[key] = np.load(path)

        print_summary(results)
        plot_comparison(results, fig_dir)


if __name__ == '__main__':
    main()
