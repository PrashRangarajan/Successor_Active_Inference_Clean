"""Neural Successor Feature agent on PointMaze (gymnasium-robotics).

Trains a neural SF agent on PointMaze UMaze — a 2D maze navigation task
where a point mass must reach a goal position through corridors and walls.

The agent sees 6D continuous observations [x, y, vx, vy, goal_x, goal_y]
and uses the same 8 discrete directional actions as the tabular version.
Goal checking uses the tabular adapter's spatial binning for consistency.

Training uses two phases with phase-boundary management:
1. Diverse exploration (100%): random starts to build the SF representation
2. Goal-focused (30% diverse): mostly fixed-start to learn navigation

Requires: pip install gymnasium-robotics

Usage:
    python examples/run_neural_point_maze.py
    python examples/run_neural_point_maze.py --quick
    python examples/run_neural_point_maze.py --device cuda
"""

import argparse
import os
import sys
import time

# Ensure imports resolve from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

from environments.point_maze import PointMazeAdapter
from core.neural.continuous_adapter import ContinuousAdapter
from core.neural.agent import NeuralSRAgent
from examples.configs import NEURAL_POINTMAZE


def make_maze_distance_reward(adapter):
    """Create a reward function using maze-aware (BFS) shortest-path distance.

    In a U-maze, Euclidean distance is deceptive — it points through walls.
    BFS distance follows the actual corridors, giving a correct gradient
    that guides the agent around the U to the goal.

    The adapter holds the discretized bin grid and wall set, so the BFS
    runs on the same structure and is cached for efficiency.

    Args:
        adapter: PointMazeAdapter (has maze_distance() method).

    Returns:
        Reward function: obs → float.
    """
    # Cache the max possible maze distance for normalization
    _max_dist = [None]

    def maze_distance_reward(obs):
        """Dense reward based on BFS maze-path distance to goal.

        Observation layout (6D): [x, y, vx, vy, goal_x, goal_y]
        """
        maze_dist = adapter.maze_distance(obs)

        # Lazy-init max distance (first call after reset)
        if _max_dist[0] is None:
            goal_bin = adapter.discretize_obs(obs[4:6])
            dist_map = adapter._bfs_from_goal(goal_bin)
            if dist_map:
                x_bw = (adapter._x_range[1] - adapter._x_range[0]) / adapter.n_x_bins
                y_bw = (adapter._y_range[1] - adapter._y_range[0]) / adapter.n_y_bins
                bin_size = (x_bw + y_bw) / 2.0
                _max_dist[0] = max(dist_map.values()) * bin_size
            else:
                _max_dist[0] = 5.0  # fallback

        # Normalize to [-1, 0] range, then add bonuses
        max_d = max(_max_dist[0], 0.01)
        reward = -maze_dist / max_d  # in [-1, 0]

        # Euclidean proximity bonus (fine for close range)
        eucl_dist = np.linalg.norm(obs[:2] - obs[4:6])
        if eucl_dist < 1.0:
            reward += 1.0
        if eucl_dist < 0.45:
            reward += 5.0

        return float(np.clip(reward, -5.0, 5.0))

    return maze_distance_reward


def main():
    parser = argparse.ArgumentParser(
        description="Neural SF agent on PointMaze")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with fewer episodes")
    parser.add_argument("--n-eval", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--save-dir", type=str,
                        default="data/neural_point_maze",
                        help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device ('cpu' or 'cuda')")
    args = parser.parse_args()

    cfg = NEURAL_POINTMAZE

    # ==================== Setup ====================
    print("=" * 60)
    print("Neural Successor Feature Agent -- PointMaze")
    print("=" * 60)

    env_train = gym.make(
        cfg["maze_id"],
        max_episode_steps=cfg["steps_per_episode"],
    )

    base_adapter = PointMazeAdapter(
        env_train,
        n_x_bins=cfg["n_x_bins"],
        n_y_bins=cfg["n_y_bins"],
    )
    adapter = ContinuousAdapter(base_adapter)

    # Reset once to populate desired_goal
    adapter.reset()

    print(f"Observation dim: {adapter.obs_dim} (6D: x, y, vx, vy, gx, gy)")
    print(f"Actions: {adapter.n_actions} (8 directional forces)")
    print(f"SF dim: {cfg['sf_dim']}")
    print(f"Maze: {cfg['maze_id']}")
    print(f"Bins: {cfg['n_x_bins']}x{cfg['n_y_bins']} "
          f"({base_adapter.n_states - len(base_adapter.get_wall_indices())} "
          f"navigable)")
    goal = base_adapter._desired_goal
    if goal is not None:
        print(f"Goal: ({goal[0]:.2f}, {goal[1]:.2f})")
    print()

    # ==================== Create Agent ====================
    import torch
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'
    print(f"Device: {device}")

    agent = NeuralSRAgent(
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
        device=device,
    )

    # Build maze-aware (BFS) reward shaping — the key fix for U-maze.
    # Euclidean distance points through walls; BFS follows corridors.
    reward_fn = make_maze_distance_reward(base_adapter)

    agent.set_goal(
        goal_spec=None,
        reward=cfg["reward"],
        default_cost=cfg["default_cost"],
        use_env_reward=True,
        terminal_bonus=cfg.get("terminal_bonus", 0.0),
        reward_shaping_fn=reward_fn,
    )

    # ==================== Two-Phase Training ====================
    ep1 = cfg["train_episodes_phase1"]
    ep2 = cfg["train_episodes_phase2"]
    frac2 = cfg["diverse_fraction_phase2"]
    if args.quick:
        ep1 = 200
        ep2 = 300
        args.n_eval = 5

    t0 = time.time()

    # Phase 1: Diverse exploration — build SF representation of maze
    print(f"Phase 1: Diverse exploration ({ep1} episodes, 100% diverse)")
    agent.learn_environment(
        num_episodes=ep1,
        steps_per_episode=cfg["steps_per_episode"],
        diverse_start=True,
        log_interval=max(1, ep1 // 5),
    )

    os.makedirs(args.save_dir, exist_ok=True)
    agent.save(os.path.join(args.save_dir, "checkpoint_phase1.pt"))

    # Phase boundary management
    agent.truncate_buffer(keep_fraction=cfg["buffer_keep_phase2"])
    agent.reset_epsilon(
        new_start=cfg["epsilon_phase2_start"],
        new_decay_steps=cfg["epsilon_phase2_decay_steps"],
    )
    agent.reset_lr(
        sf_lr=cfg["lr"] * cfg["lr_phase2_fraction"],
        rw_lr=cfg["lr_w"] * cfg["lr_phase2_fraction"],
        decay_steps=ep2 * cfg["steps_per_episode"],
    )

    # Phase 2: Goal-focused — mostly fixed start
    print(f"\nPhase 2: Goal-focused ({ep2} episodes, {frac2:.0%} diverse)")
    agent.learn_environment(
        num_episodes=ep2,
        steps_per_episode=cfg["steps_per_episode"],
        diverse_start=True,
        diverse_fraction=frac2,
        log_interval=max(1, ep2 // 5),
    )

    train_time = time.time() - t0
    print(f"\nTotal training time: {train_time:.1f}s")

    # ==================== Save ====================
    save_path = os.path.join(args.save_dir, "checkpoint.pt")
    agent.save(save_path)
    print(f"Saved checkpoint to {save_path}")

    # ==================== Evaluate ====================
    print(f"\nEvaluating ({args.n_eval} episodes)...")

    # Create a separate test env (no step limit truncation for evaluation)
    env_test = gym.make(
        cfg["maze_id"],
        max_episode_steps=cfg["test_max_steps"],
    )
    test_base = PointMazeAdapter(
        env_test,
        n_x_bins=cfg["n_x_bins"],
        n_y_bins=cfg["n_y_bins"],
    )
    agent.adapter = ContinuousAdapter(test_base)
    agent.goal_states = test_base.get_goal_states()

    results = []
    for i in range(args.n_eval):
        result = agent.run_episode(
            init_state=None,
            max_steps=cfg["test_max_steps"],
        )
        results.append(result)
        status = "GOAL" if result['reached_goal'] else "timeout"
        print(f"  Episode {i + 1}: {result['steps']:4d} steps, "
              f"reward={result['reward']:.1f}, {status}")

    # ==================== Summary ====================
    steps = [r['steps'] for r in results]
    successes = [r['reached_goal'] for r in results]
    print(f"\n{'=' * 40}")
    print(f"Success rate: {sum(successes)}/{len(successes)} "
          f"({100 * sum(successes) / len(successes):.0f}%)")
    print(f"Avg steps (all): {np.mean(steps):.1f} +/- {np.std(steps):.1f}")
    if any(successes):
        success_steps = [s for s, ok in zip(steps, successes) if ok]
        print(f"Avg steps (success): {np.mean(success_steps):.1f} "
              f"+/- {np.std(success_steps):.1f}")
    print(f"{'=' * 40}")

    env_train.close()
    env_test.close()


if __name__ == "__main__":
    main()
