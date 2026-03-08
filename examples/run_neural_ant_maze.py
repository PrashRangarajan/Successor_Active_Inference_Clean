"""Neural Successor Feature agent on AntMaze (gymnasium-robotics).

Quadruped locomotion + maze navigation: an 8-DOF Ant robot walks through
corridors and walls to reach a goal position. Uses ActionConditionedSFNetwork
to handle 6561 discrete actions (3^8 joint torque grid).

This script:
  - Trains the hierarchical neural SF agent with two-phase training
  - Evaluates from default and diverse starts
  - Plots training curves and evaluation metrics

Requires: pip install gymnasium-robotics

Usage:
    # Full training + evaluation:
    python examples/run_neural_ant_maze.py --train

    # Quick smoke test:
    python examples/run_neural_ant_maze.py --train --quick

    # Plot from saved data (no training):
    python examples/run_neural_ant_maze.py
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np

import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

from environments.mujoco.ant_maze import AntMazeAdapter
from core.neural.hierarchical_agent import HierarchicalNeuralSRAgent
from examples.configs import NEURAL_ANTMAZE
from examples.neural_experiment import (
    setup_device, train_two_phase, train_to_checkpoint,
    plot_training_curves, save_training_log, load_training_log,
    plot_checkpoint_curves,
)


# ==================== Reward Shaping ====================

def ant_maze_distance_reward(obs):
    """Dense shaped reward based on distance to goal.

    Observation layout (109D):
        obs[0:105]   = body state (qpos, qvel, contact forces)
        obs[105:107] = achieved_goal (ant x, y)
        obs[107:109] = desired_goal (goal x, y)

    Returns:
        Scalar reward clipped to [-5, 5].
    """
    ant_pos = obs[105:107]
    goal_pos = obs[107:109]
    dist = np.linalg.norm(ant_pos - goal_pos)

    # Base: negative distance
    reward = -dist

    # Proximity bonus
    if dist < 2.0:
        reward += 1.0
    if dist < 1.0:
        reward += 2.0
    if dist < 0.5:
        reward += 5.0

    return float(np.clip(reward, -5.0, 5.0))


# ==================== Agent Factory ====================

def create_agent(cfg, render_mode=None, device='cpu'):
    """Create a fresh hierarchical neural SF agent for AntMaze.

    Uses ActionConditionedSFNetwork to handle 6561 discrete actions.

    Args:
        cfg: Config dict (NEURAL_ANTMAZE).
        render_mode: 'rgb_array' for video, None for speed.
        device: Torch device ('cpu' or 'cuda').

    Returns:
        (agent, adapter) tuple.
    """
    adapter = AntMazeAdapter(
        maze_id=cfg["maze_id"],
        n_bins_per_joint=cfg["n_bins_per_joint"],
        render_mode=render_mode,
        max_episode_steps=cfg["steps_per_episode"],
    )

    agent = HierarchicalNeuralSRAgent(
        adapter=adapter,
        sf_dim=cfg["sf_dim"],
        hidden_sizes=cfg["hidden_sizes"],
        gamma=cfg["gamma"],
        lr=cfg["lr"],
        lr_w=cfg["lr_w"],
        batch_size=cfg["batch_size"],
        buffer_size=cfg["buffer_size"],
        target_update_freq=cfg["target_update_freq"],
        tau=cfg["tau"],
        epsilon_start=cfg["epsilon_start"],
        epsilon_end=cfg["epsilon_end"],
        epsilon_decay_steps=cfg["epsilon_decay_steps"],
        sf_network_cls=cfg["sf_network_cls"],
        n_clusters=4,
        cluster_method='kmeans',
        cluster_on='observations',
        n_cluster_samples=5000,
        adjacency_episodes=500,
        adjacency_episode_length=100,
        device=device,
        use_per=cfg.get("use_per", False),
        per_alpha=cfg.get("per_alpha", 0.6),
        per_beta_start=cfg.get("per_beta_start", 0.4),
        per_beta_end=cfg.get("per_beta_end", 1.0),
        use_episodic_replay=cfg.get("use_episodic_replay", False),
        episodic_replay_episodes=cfg.get("episodic_replay_episodes", 2),
    )

    # Use distance-based shaping for richer gradient signal
    agent.set_goal(
        goal_spec=None,
        reward=cfg["reward"],
        default_cost=cfg["default_cost"],
        use_env_reward=False,
        terminal_bonus=cfg.get("terminal_bonus", 0.0),
        reward_shaping_fn=ant_maze_distance_reward,
    )

    return agent, adapter


def train_agent(agent, cfg, ep_diverse, ep_fixed):
    """Two-phase training: diverse exploration then mixed training."""
    train_two_phase(agent, cfg, ep_diverse, ep_fixed)


# ==================== Evaluation ====================

def evaluate_agent(agent, cfg, n_eval=10, diverse=False):
    """Evaluate trained agent on AntMaze.

    For maze navigation, success is measured by reaching the goal
    within the step limit.

    Returns:
        List of result dicts.
    """
    test_adapter = AntMazeAdapter(
        maze_id=cfg["maze_id"],
        n_bins_per_joint=cfg["n_bins_per_joint"],
        max_episode_steps=cfg["test_max_steps"],
    )
    agent.adapter = test_adapter

    results = []
    for i in range(n_eval):
        if diverse:
            test_adapter.sample_random_state()
            init_state = test_adapter.get_state_for_reset()
        else:
            init_state = None

        result = agent.run_episode(
            init_state=init_state,
            max_steps=cfg["test_max_steps"],
        )
        results.append(result)

    return results


def print_eval_results(results, label=""):
    """Pretty-print evaluation results for maze navigation."""
    print(f"\n  {label} Results ({len(results)} episodes):")
    for i, r in enumerate(results):
        status = "GOAL" if r.get('reached_goal', False) else "timeout"
        print(f"    Episode {i+1}: {r['steps']:4d} steps, "
              f"reward={r['reward']:.1f}, {status}")

    avg_steps = np.mean([r['steps'] for r in results])
    avg_reward = np.mean([r['reward'] for r in results])
    successes = sum(1 for r in results if r.get('reached_goal', False))
    print(f"\n  Success rate: {successes}/{len(results)} "
          f"({100 * successes / len(results):.0f}%)")
    print(f"  Avg steps: {avg_steps:.1f}")
    print(f"  Avg reward: {avg_reward:.1f}")

    return avg_reward, avg_steps


# ==================== Plotting ====================

def _plot_checkpoint_curves(episodes_list, avg_rewards, avg_steps, save_dir):
    """Plot evaluation metrics across training checkpoints."""
    plot_checkpoint_curves(
        episodes_list,
        {'Avg Episode Reward': avg_rewards,
         'Avg Episode Steps': avg_steps},
        save_dir, env_name='AntMaze')


# ==================== Multi-Checkpoint Experiment ====================

def checkpoint_experiment(cfg, episodes_list, n_runs, n_eval, device='cpu'):
    """Train incrementally, evaluating at each checkpoint.

    Instead of retraining from scratch at each episode budget, trains
    one agent through the full schedule and evaluates at intermediate
    points.  This reduces total training from O(sum(episodes_list))
    to O(max(episodes_list)) per run.
    """
    sorted_eps = sorted(episodes_list)
    max_eps = sorted_eps[-1]
    ep_diverse = int(max_eps * 0.4)
    ep_fixed = max_eps - ep_diverse

    avg_rewards_arr = np.zeros((n_runs, len(episodes_list)))
    avg_steps_arr = np.zeros((n_runs, len(episodes_list)))

    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Run {run+1}/{n_runs}")
        print(f"{'='*60}")

        agent, adapter = create_agent(cfg, device=device)
        trained, boundary_done = 0, False

        for target in sorted_eps:
            print(f"\n--- Checkpoint: {target} episodes ---")
            trained, boundary_done = train_to_checkpoint(
                agent, cfg, target, trained,
                ep_diverse, ep_fixed, boundary_done)

            results = evaluate_agent(agent, cfg, n_eval)
            idx = episodes_list.index(target)
            avg_rewards_arr[run, idx] = np.mean([r['reward'] for r in results])
            avg_steps_arr[run, idx] = np.mean([r['steps'] for r in results])

            print(f"  => Avg reward: {avg_rewards_arr[run, idx]:.1f}, "
                  f"Avg steps: {avg_steps_arr[run, idx]:.1f}")

    return avg_rewards_arr, avg_steps_arr


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Neural SF AntMaze — Train, Evaluate")
    parser.add_argument("--train", action="store_true",
                        help="Run training experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with fewer episodes")
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--n-eval", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device ('cpu' or 'cuda')")
    args_cli = parser.parse_args()

    device = setup_device(args_cli.device)

    cfg = NEURAL_ANTMAZE

    data_dir = "data/eval/neural_ant_maze"
    fig_dir = args_cli.save_dir or "figures/eval/neural_ant_maze"
    checkpoint_dir = "data/neural_ant_maze"

    if args_cli.quick:
        episodes = list(cfg["eval_quick_episodes"])
        n_runs = args_cli.n_runs or cfg["eval_quick_n_runs"]
        n_eval = 5
    else:
        episodes = list(cfg["eval_episodes"])
        n_runs = args_cli.n_runs or cfg["eval_n_runs"]
        n_eval = args_cli.n_eval

    # ==================== Full Training Run ====================
    if args_cli.train:
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)

        print("=" * 60)
        print("NEURAL SF ANTMAZE")
        print("=" * 60)
        n_actions = cfg["n_bins_per_joint"] ** 8
        print(f"Config: sf_dim={cfg['sf_dim']}, hidden={cfg['hidden_sizes']}, "
              f"n_bins={cfg['n_bins_per_joint']} ({n_actions} actions)")
        print(f"Network: {cfg['sf_network_cls']}")
        print(f"Maze: {cfg['maze_id']}")
        print(f"Device: {device}")

        with open(os.path.join(data_dir, "args.json"), "w") as f:
            json.dump({
                "episodes": episodes,
                "n_runs": n_runs,
                "n_eval": n_eval,
                "config": {k: v if not isinstance(v, tuple) else list(v)
                           for k, v in cfg.items()},
            }, f, indent=2)

        # --- Single full training run ---
        print("\n" + "=" * 60)
        print("PHASE A: Full training run")
        print("=" * 60)

        ep_diverse = cfg["train_episodes_diverse"]
        ep_fixed = cfg["train_episodes_fixed"]
        if args_cli.quick:
            ep_diverse = 50
            ep_fixed = 100

        agent, adapter = create_agent(cfg, device=device)
        print(f"Obs dim: {adapter.obs_dim}, Actions: {adapter.n_actions}")

        t0 = time.time()
        train_agent(agent, cfg, ep_diverse, ep_fixed)
        train_time = time.time() - t0
        print(f"\nTraining time: {train_time:.1f}s")

        os.makedirs(checkpoint_dir, exist_ok=True)
        agent.save(os.path.join(checkpoint_dir, "checkpoint.pt"))
        print(f"Saved checkpoint")

        save_training_log(agent.training_log, data_dir)

        print("\n--- Training Curves ---")
        plot_training_curves(agent.training_log, fig_dir, env_name='AntMaze')

        print("\n--- Default Start Evaluation ---")
        results_default = evaluate_agent(agent, cfg, n_eval, diverse=False)
        print_eval_results(results_default, "Default Start")

        print("\n--- Diverse Start Evaluation ---")
        results_diverse = evaluate_agent(agent, cfg, n_eval, diverse=True)
        print_eval_results(results_diverse, "Diverse Start")

        # --- Multi-checkpoint experiment (skip in quick mode) ---
        if not args_cli.quick:
            print("\n" + "=" * 60)
            print("PHASE B: Checkpoint evaluation (learning curves)")
            print("=" * 60)

            t0 = time.time()
            avg_rewards, avg_steps = checkpoint_experiment(
                cfg, episodes, n_runs, n_eval, device=device)
            exp_time = time.time() - t0
            print(f"\nCheckpoint experiment: {exp_time:.0f}s")

            np.save(os.path.join(data_dir, "checkpoint_rewards.npy"),
                    avg_rewards)
            np.save(os.path.join(data_dir, "checkpoint_steps.npy"),
                    avg_steps)
            np.save(os.path.join(data_dir, "checkpoint_episodes.npy"),
                    np.array(episodes))

            print("\n--- Checkpoint Curves ---")
            _plot_checkpoint_curves(episodes, avg_rewards, avg_steps, fig_dir)

    else:
        print("=" * 60)
        print("PLOTTING FROM SAVED DATA")
        print("=" * 60)
        os.makedirs(fig_dir, exist_ok=True)

    # --- Always try to plot from saved data ---
    training_log = load_training_log(data_dir)
    if training_log:
        print("\n--- Training Curves (from saved data) ---")
        plot_training_curves(training_log, fig_dir, env_name='AntMaze')

    ckpt_rewards_path = os.path.join(data_dir, "checkpoint_rewards.npy")
    if os.path.exists(ckpt_rewards_path):
        print("\n--- Checkpoint Curves (from saved data) ---")
        avg_rewards = np.load(ckpt_rewards_path)
        avg_steps = np.load(os.path.join(data_dir, "checkpoint_steps.npy"))
        eps_list = np.load(
            os.path.join(data_dir, "checkpoint_episodes.npy")).tolist()
        _plot_checkpoint_curves(eps_list, avg_rewards, avg_steps, fig_dir)

    print(f"\nDone! Figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
