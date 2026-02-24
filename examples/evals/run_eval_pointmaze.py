"""Evaluation: PointMaze benchmark comparing Hierarchy vs Flat.

Runs repeated experiments across training checkpoints to compare
hierarchical vs flat active inference on PointMaze environments.
Supports UMaze (default), Medium, and Large variants.

Saves .npy data files to data/eval/pointmaze/<maze>/ and
figures to figures/eval/pointmaze/<maze>/.

Usage:
    # Run experiments (UMaze, default):
    python examples/evals/run_eval_pointmaze.py --train

    # Quick test (2 seeds × 3 checkpoints):
    python examples/evals/run_eval_pointmaze.py --train --quick

    # Medium or Large maze:
    python examples/evals/run_eval_pointmaze.py --train --maze medium
    python examples/evals/run_eval_pointmaze.py --train --maze large

    # Plot from saved data:
    python examples/evals/run_eval_pointmaze.py
    python examples/evals/run_eval_pointmaze.py --maze medium
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
    plot_stability_bars,
    save_eval_data,
    load_eval_args,
)
from environments.point_maze import PointMazeAdapter
from examples.configs import POINTMAZE, POINTMAZE_MEDIUM, POINTMAZE_LARGE, SHARED

MAZE_CONFIGS = {
    "umaze": POINTMAZE,
    "medium": POINTMAZE_MEDIUM,
    "large": POINTMAZE_LARGE,
}

# ==================== Agent Factory ====================


def create_pointmaze_agent(config, num_episodes):
    """Create a fresh PointMaze SR agent trained for exactly num_episodes.

    Returns:
        (agent, adapter) tuple
    """
    env = gym.make(
        config["maze_id"],
        render_mode=None,
        max_episode_steps=500,
        continuing_task=False,
    )
    adapter = PointMazeAdapter(
        env,
        n_x_bins=config["n_x_bins"],
        n_y_bins=config["n_y_bins"],
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

    # Set goal to first replan_goals entry for the learning phase
    goal_cell = np.array(config["replan_goals"][0]["cell"])
    adapter.reset(reset_options={"goal_cell": goal_cell})
    agent.set_goal(None, reward=config["reward"],
                   default_cost=config["default_cost"])

    agent.learn_environment(num_episodes)
    return agent, adapter


def _reset_for_eval(agent, adapter, config, test_start, goal_options):
    """Reset agent + adapter for a test episode with fixed start and goal."""
    adapter.reset(init_state=test_start, reset_options=goal_options)
    agent.set_goal(None, reward=config["reward"],
                   default_cost=config["default_cost"])
    agent._compute_macro_preference()
    agent.current_state = agent._get_planning_state()


def _find_start_position(adapter, corner="top-left"):
    """Find the first navigable bin center in the specified corner."""
    x_centers, y_centers = adapter.get_bin_centers()
    wall_set = set(adapter.get_wall_indices())
    n_x = adapter.n_x_bins
    n_y = adapter.n_y_bins

    if corner == "top-left":
        y_range = range(n_y - 1, -1, -1)
        x_range = range(n_x)
    elif corner == "bottom-right":
        y_range = range(n_y)
        x_range = range(n_x - 1, -1, -1)
    else:
        y_range = range(n_y - 1, -1, -1)
        x_range = range(n_x)

    for yi in y_range:
        for xi in x_range:
            idx = adapter.state_space.state_to_index((xi, yi))
            if idx not in wall_set:
                return [float(x_centers[xi]), float(y_centers[yi])]

    raise RuntimeError("No navigable bin found")


# ==================== Experiment ====================


def pointmaze_experiment(args, config):
    """Main experiment: steps and success across training checkpoints.

    Returns:
        Tuple of (SR_steps_hier, SR_steps_flat,
                  SR_rewards_hier, SR_rewards_flat,
                  SR_success_hier, SR_success_flat)
    """
    n_trials = len(args.episodes)

    SR_steps_hier = np.zeros((args.n_runs, n_trials))
    SR_steps_flat = np.zeros((args.n_runs, n_trials))
    SR_rewards_hier = np.zeros((args.n_runs, n_trials))
    SR_rewards_flat = np.zeros((args.n_runs, n_trials))
    SR_success_hier = np.zeros((args.n_runs, n_trials))
    SR_success_flat = np.zeros((args.n_runs, n_trials))

    goal_cell = np.array(config["replan_goals"][0]["cell"])
    goal_options = {"goal_cell": goal_cell}

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
                    agent, adapter = create_pointmaze_agent(config, num_episodes)
                except (np.linalg.LinAlgError, ValueError) as e:
                    print(f"  Error: {e} — retrying...")
                    continue
                else:
                    break

            # Determine start position (opposite corner from goal)
            test_start = _find_start_position(adapter, corner="top-left")

            # Evaluate hierarchy
            print("\nHierarchy")
            _reset_for_eval(agent, adapter, config, test_start, goal_options)
            result_hier = agent.run_episode_hierarchical(
                max_steps=config["test_max_steps"],
            )

            # Evaluate flat (same agent, same M)
            print("Flat")
            _reset_for_eval(agent, adapter, config, test_start, goal_options)
            result_flat = agent.run_episode_flat(
                max_steps=config["test_max_steps"],
            )

            SR_steps_hier[n, trial] = result_hier["steps"]
            SR_steps_flat[n, trial] = result_flat["steps"]
            SR_rewards_hier[n, trial] = result_hier["reward"]
            SR_rewards_flat[n, trial] = result_flat["reward"]
            SR_success_hier[n, trial] = 1.0 if result_hier["reached_goal"] else 0.0
            SR_success_flat[n, trial] = 1.0 if result_flat["reached_goal"] else 0.0

            print(f"  Hier: steps={result_hier['steps']}, "
                  f"goal={result_hier['reached_goal']}, "
                  f"macro={result_hier['macro_decisions']}")
            print(f"  Flat: steps={result_flat['steps']}, "
                  f"goal={result_flat['reached_goal']}")

    return (SR_steps_hier, SR_steps_flat,
            SR_rewards_hier, SR_rewards_flat,
            SR_success_hier, SR_success_flat)


# ==================== Success Rate Plot ====================


def plot_success_curves(eps_range, data_dict, save_path):
    """Plot success rate curves per checkpoint.

    Args:
        eps_range: List of training episode counts (x-axis).
        data_dict: OrderedDict of {label: (n_runs, n_trials) binary array}.
    """
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


# ==================== Main ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PointMaze Eval: Hierarchy vs Flat"
    )
    parser.add_argument("--train", action="store_true", help="Run experiments")
    parser.add_argument("--quick", action="store_true", help="Quick test")
    parser.add_argument("--maze", type=str, default="umaze",
                        choices=["umaze", "medium", "large"],
                        help="Maze variant (default: umaze)")
    parser.add_argument("--n_runs", type=int, default=None)
    args_cli = parser.parse_args()

    config = MAZE_CONFIGS[args_cli.maze]
    maze_label = args_cli.maze.capitalize()

    nruns = config["eval_n_runs"]
    eps = list(config["eval_episodes"])

    if args_cli.quick:
        eps = list(config["eval_quick_episodes"])
        nruns = config["eval_quick_n_runs"]

    if args_cli.n_runs is not None:
        nruns = args_cli.n_runs

    args = argparse.Namespace(
        n_runs=nruns,
        episodes=eps,
        maze=args_cli.maze,
    )

    data_dir = f"data/eval/pointmaze/{args_cli.maze}"
    save_dir = f"figures/eval/pointmaze/{args_cli.maze}"

    if args_cli.train:
        os.makedirs(data_dir, exist_ok=True)

        # Save args + config info
        save_info = {**vars(args), "maze_id": config["maze_id"],
                     "n_x_bins": config["n_x_bins"],
                     "n_y_bins": config["n_y_bins"],
                     "n_clusters": config["n_clusters"]}
        with open(os.path.join(data_dir, "args.json"), "w") as f:
            json.dump(save_info, f, indent=2)

        n_navigable = config["n_x_bins"] * config["n_y_bins"]  # approx
        print("=" * 60)
        print(f"POINTMAZE EVAL ({maze_label}): Hierarchy vs Flat")
        print("=" * 60)
        print(f"Maze: {config['maze_id']}")
        print(f"Bins: {config['n_x_bins']}×{config['n_y_bins']}, "
              f"Clusters: {config['n_clusters']}")
        print(f"Runs: {args.n_runs}, Checkpoints: {args.episodes}")

        t0 = time.time()
        (SR_steps_hier, SR_steps_flat,
         SR_rewards_hier, SR_rewards_flat,
         SR_success_hier, SR_success_flat) = pointmaze_experiment(args, config)
        elapsed = time.time() - t0
        print(f"\nExperiment completed in {elapsed:.0f}s")

        # Compute relative stability
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

    else:
        # Load saved args
        args = load_eval_args(data_dir)
        if args is None:
            print("No saved args found. Run with --train first.")
            exit(1)

    # ==================== Generate Plots ====================
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    # Steps curves
    if os.path.exists(os.path.join(data_dir, "SR_steps_hierarchy.npy")):
        data = OrderedDict()
        data["Hierarchy"] = np.load(os.path.join(data_dir, "SR_steps_hierarchy.npy"))
        data["Flat"] = np.load(os.path.join(data_dir, "SR_steps_flat.npy"))
        plot_step_curves(
            args.episodes, data,
            os.path.join(save_dir, f"pointmaze_{args_cli.maze}_steps.png"),
        )

    # Reward curves
    if os.path.exists(os.path.join(data_dir, "SR_rewards_hierarchy.npy")):
        data = OrderedDict()
        data["Hierarchy"] = np.load(os.path.join(data_dir, "SR_rewards_hierarchy.npy"))
        data["Flat"] = np.load(os.path.join(data_dir, "SR_rewards_flat.npy"))
        plot_reward_curves(
            args.episodes, data,
            os.path.join(save_dir, f"pointmaze_{args_cli.maze}_reward.png"),
        )

    # Success rate curves
    if os.path.exists(os.path.join(data_dir, "SR_success_hierarchy.npy")):
        data = OrderedDict()
        data["Hierarchy"] = np.load(os.path.join(data_dir, "SR_success_hierarchy.npy"))
        data["Flat"] = np.load(os.path.join(data_dir, "SR_success_flat.npy"))
        plot_success_curves(
            args.episodes, data,
            os.path.join(save_dir, f"pointmaze_{args_cli.maze}_success.png"),
        )

    # Stability bars
    stability_data = OrderedDict()
    hier_stab = os.path.join(data_dir, "SR_relative_stability_hierarchy.npy")
    flat_stab = os.path.join(data_dir, "SR_relative_stability_flat.npy")
    if os.path.exists(hier_stab):
        stability_data["Hierarchy"] = np.load(hier_stab)
    if os.path.exists(flat_stab):
        stability_data["Flat"] = np.load(flat_stab)
    if stability_data:
        plot_stability_bars(
            stability_data,
            os.path.join(save_dir, f"pointmaze_{args_cli.maze}_stability.png"),
        )

    print(f"\nDone! Figures saved to {save_dir}/")
