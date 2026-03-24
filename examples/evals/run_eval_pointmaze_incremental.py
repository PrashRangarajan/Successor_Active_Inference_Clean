"""Incremental eval: PointMaze Medium & Large.

Single continuous training run per seed with periodic evaluation.
Much faster than the standard eval which retrains from scratch at
each checkpoint.

Usage:
    python examples/evals/run_eval_pointmaze_incremental.py --maze medium
    python examples/evals/run_eval_pointmaze_incremental.py --maze large
    python examples/evals/run_eval_pointmaze_incremental.py --maze medium --quick
"""

import os
import argparse
import json
import time
from collections import OrderedDict

import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-poster")

import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

from core import HierarchicalSRAgent
from core.eval_utils import (
    AGENT_COLORS,
    compute_stability_array,
    plot_reward_curves,
    plot_step_curves,
    save_eval_data,
)
from environments.point_maze import PointMazeAdapter
from examples.configs import POINTMAZE, POINTMAZE_MEDIUM, POINTMAZE_LARGE, SHARED

MAZE_CONFIGS = {
    "umaze": POINTMAZE,
    "medium": POINTMAZE_MEDIUM,
    "large": POINTMAZE_LARGE,
}


def _find_start_position(adapter, corner="top-left"):
    """Find the first navigable bin center in the specified corner."""
    x_centers, y_centers = adapter.get_bin_centers()
    wall_set = set(adapter.get_wall_indices())
    n_x, n_y = adapter.n_x_bins, adapter.n_y_bins

    if corner == "top-left":
        y_range = range(n_y - 1, -1, -1)
        x_range = range(n_x)
    else:
        y_range = range(n_y)
        x_range = range(n_x - 1, -1, -1)

    for yi in y_range:
        for xi in x_range:
            idx = adapter.state_space.state_to_index((xi, yi))
            if idx not in wall_set:
                return [float(x_centers[xi]), float(y_centers[yi])]
    raise RuntimeError("No navigable bin found")


def run_incremental_eval(config, episodes, n_runs):
    """Run incremental training with periodic evaluation.

    Instead of retraining from scratch at each checkpoint, we train
    one agent incrementally and evaluate at each checkpoint.
    """
    n_trials = len(episodes)

    SR_steps_hier = np.zeros((n_runs, n_trials))
    SR_steps_flat = np.zeros((n_runs, n_trials))
    SR_rewards_hier = np.zeros((n_runs, n_trials))
    SR_rewards_flat = np.zeros((n_runs, n_trials))
    SR_success_hier = np.zeros((n_runs, n_trials))
    SR_success_flat = np.zeros((n_runs, n_trials))

    # Pick a goal far from the top-left start.
    # For Medium, replan_goals[0] is Bot-right (fine).
    # For Large, replan_goals[0] is Top-left (same as start!), so use Bot-right.
    goal_idx = config.get("eval_single_goal_idx", 0)
    goal_cell = np.array(config["replan_goals"][goal_idx]["cell"])
    goal_options = {"goal_cell": goal_cell}
    print(f"Goal: {config['replan_goals'][goal_idx]['label']} (cell={goal_cell.tolist()})")

    for n in range(n_runs):
        print("x" * 40)
        print(f"Run: {n + 1}/{n_runs}")
        print("x" * 40)

        # Create fresh agent + adapter for this seed
        env = gym.make(
            config["maze_id"],
            render_mode=None,
            max_episode_steps=config["test_max_steps"],
            continuing_task=False,
        )
        adapter = PointMazeAdapter(
            env, n_x_bins=config["n_x_bins"], n_y_bins=config["n_y_bins"],
        )
        agent = HierarchicalSRAgent(
            adapter=adapter,
            n_clusters=config["n_clusters"],
            gamma=config["gamma"],
            learning_rate=config["learning_rate"],
            learn_from_experience=True,
            use_replay=SHARED["use_replay"],
            n_replay_epochs=SHARED["n_replay_epochs"],
            train_smooth_steps=config["train_smooth_steps"],
            test_smooth_steps=config["test_smooth_steps"],
        )

        # Set goal
        adapter.reset(reset_options={"goal_cell": goal_cell})
        agent.set_goal(None, reward=config["reward"],
                       default_cost=config["default_cost"])

        test_start = _find_start_position(adapter, corner="top-left")

        # Initial training up to first checkpoint
        prev_eps = 0
        for trial, num_eps in enumerate(episodes):
            delta = num_eps - prev_eps
            print(f"\n  Checkpoint {num_eps} episodes (delta={delta})")

            # Incremental training
            if trial == 0:
                agent.learn_environment(num_eps)
            else:
                agent.learn_environment_incremental(delta)

            prev_eps = num_eps

            # Evaluate hierarchy
            adapter.reset(init_state=test_start, reset_options=goal_options)
            agent.set_goal(None, reward=config["reward"],
                           default_cost=config["default_cost"])
            agent._compute_macro_preference()
            agent.current_state = agent._get_planning_state()
            result_hier = agent.run_episode_hierarchical(
                max_steps=config["test_max_steps"],
            )

            # Evaluate flat
            adapter.reset(init_state=test_start, reset_options=goal_options)
            agent.set_goal(None, reward=config["reward"],
                           default_cost=config["default_cost"])
            agent._compute_macro_preference()
            agent.current_state = agent._get_planning_state()
            result_flat = agent.run_episode_flat(
                max_steps=config["test_max_steps"],
            )

            SR_steps_hier[n, trial] = result_hier["steps"]
            SR_steps_flat[n, trial] = result_flat["steps"]
            SR_rewards_hier[n, trial] = result_hier["reward"]
            SR_rewards_flat[n, trial] = result_flat["reward"]
            SR_success_hier[n, trial] = 1.0 if result_hier["reached_goal"] else 0.0
            SR_success_flat[n, trial] = 1.0 if result_flat["reached_goal"] else 0.0

            print(f"    Hier: steps={result_hier['steps']}, "
                  f"goal={result_hier['reached_goal']}, "
                  f"macro={result_hier.get('macro_decisions', '?')}")
            print(f"    Flat: steps={result_flat['steps']}, "
                  f"goal={result_flat['reached_goal']}")

        env.close()

    return (SR_steps_hier, SR_steps_flat,
            SR_rewards_hier, SR_rewards_flat,
            SR_success_hier, SR_success_flat)


def plot_success_curves(eps_range, data_dict, save_path):
    """Plot success rate curves per checkpoint."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig = plt.figure(figsize=(14, 10))
    for label, data in data_dict.items():
        arr = data[:, :len(eps_range)]
        mean = np.mean(arr, axis=0)
        sem = np.std(arr, axis=0) / np.sqrt(len(arr))
        color = AGENT_COLORS.get(label, None)
        plt.plot(eps_range, mean, 'o-', label=label, color=color, markersize=8)
        plt.fill_between(eps_range, mean - sem, mean + sem, alpha=0.3, color=color)
    plt.xlabel("Number of Training Episodes", fontsize=28)
    plt.ylabel("Success Rate", fontsize=28)
    plt.ylim(-0.05, 1.05)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig(save_path, format="png")
    plt.close()
    print(f"  Saved {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PointMaze Incremental Eval: Hierarchy vs Flat"
    )
    parser.add_argument("--maze", type=str, required=True,
                        choices=["umaze", "medium", "large"])
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (2 runs, fewer checkpoints)")
    parser.add_argument("--n_runs", type=int, default=None)
    args = parser.parse_args()

    config = MAZE_CONFIGS[args.maze]
    maze_label = args.maze.capitalize()

    if args.quick:
        episodes = list(config["eval_quick_episodes"])
        n_runs = config.get("eval_quick_n_runs", 2)
    else:
        episodes = list(config["eval_episodes"])
        n_runs = config.get("eval_n_runs", 5)

    if args.n_runs is not None:
        n_runs = args.n_runs

    data_dir = f"data/eval/pointmaze/{args.maze}_incremental"
    save_dir = f"figures/eval/pointmaze/{args.maze}_incremental"
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # Save config
    with open(os.path.join(data_dir, "args.json"), "w") as f:
        json.dump({"episodes": episodes, "n_runs": n_runs,
                    "maze": args.maze, "maze_id": config["maze_id"]}, f, indent=2)

    print("=" * 60)
    print(f"POINTMAZE INCREMENTAL EVAL ({maze_label}): Hierarchy vs Flat")
    print("=" * 60)
    print(f"Maze: {config['maze_id']}")
    print(f"Bins: {config['n_x_bins']}x{config['n_y_bins']}, "
          f"Clusters: {config['n_clusters']}")
    print(f"Runs: {n_runs}, Checkpoints: {episodes}")

    t0 = time.time()
    (SR_steps_hier, SR_steps_flat,
     SR_rewards_hier, SR_rewards_flat,
     SR_success_hier, SR_success_flat) = run_incremental_eval(
        config, episodes, n_runs
    )
    elapsed = time.time() - t0
    print(f"\nExperiment completed in {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Compute stability
    SR_rel_stability_hier = compute_stability_array(SR_steps_hier)
    SR_rel_stability_flat = compute_stability_array(SR_steps_flat)

    # Save data
    save_eval_data(data_dir, {
        "SR_steps_hierarchy": SR_steps_hier,
        "SR_steps_flat": SR_steps_flat,
        "SR_rewards_hierarchy": SR_rewards_hier,
        "SR_rewards_flat": SR_rewards_flat,
        "SR_success_hierarchy": SR_success_hier,
        "SR_success_flat": SR_success_flat,
        "SR_relative_stability_hierarchy": SR_rel_stability_hier,
        "SR_relative_stability_flat": SR_rel_stability_flat,
    })

    # ==================== Plots ====================
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    # Reward curves
    reward_data = OrderedDict([
        ("Hierarchy", SR_rewards_hier),
        ("Flat", SR_rewards_flat),
    ])
    plot_reward_curves(
        episodes, reward_data,
        os.path.join(save_dir, f"pointmaze_{args.maze}_reward.png"),
        ylabel="Average Reward",
    )

    # Step curves
    step_data = OrderedDict([
        ("Hierarchy", SR_steps_hier),
        ("Flat", SR_steps_flat),
    ])
    plot_step_curves(
        episodes, step_data,
        os.path.join(save_dir, f"pointmaze_{args.maze}_steps.png"),
    )

    # Success curves
    success_data = OrderedDict([
        ("Hierarchy", SR_success_hier),
        ("Flat", SR_success_flat),
    ])
    plot_success_curves(
        episodes, success_data,
        os.path.join(save_dir, f"pointmaze_{args.maze}_success.png"),
    )

    print(f"\nDone! Figures saved to {save_dir}/")
