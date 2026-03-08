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

# Use EGL for headless MuJoCo rendering (no X11 display needed)
os.environ.setdefault("MUJOCO_GL", "egl")

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
from examples.neural_experiment import (
    setup_device, plot_training_curves, save_training_log,
)


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

        # Normalize to [-1, 0] range with smooth proximity bonus
        max_d = max(_max_dist[0], 0.01)
        reward = -maze_dist / max_d  # in [-1, 0]

        # Smooth proximity bonus (exponential, no discontinuous jumps)
        eucl_dist = np.linalg.norm(obs[:2] - obs[4:6])
        reward += np.exp(-2.0 * eucl_dist)  # smooth ~1.0 at goal, ~0 far away

        return float(np.clip(reward, -1.0, 1.0))

    return maze_distance_reward


def main():
    parser = argparse.ArgumentParser(
        description="Neural SF agent on PointMaze")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with fewer episodes")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only evaluate from saved checkpoint (no training)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file (default: <save-dir>/checkpoint.pt)")
    parser.add_argument("--n-eval", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--save-dir", type=str,
                        default="data/neural_point_maze",
                        help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device ('cpu' or 'cuda')")
    parser.add_argument("--resume-phase", type=int, default=0,
                        choices=[0, 1, 2, 3],
                        help="Resume from end of given phase (0=start fresh, "
                             "1=skip Phase 1, 2=skip Phase 1+2, 3=skip to eval)")
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
    device = setup_device(args.device)
    print(f"Device: {device}")

    # HER config
    her_goal_indices = tuple(cfg.get("her_goal_indices", [4, 6]))

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
        use_per=cfg.get("use_per", False),
        per_alpha=cfg.get("per_alpha", 0.6),
        per_beta_start=cfg.get("per_beta_start", 0.4),
        per_beta_end=cfg.get("per_beta_end", 1.0),
        use_episodic_replay=cfg.get("use_episodic_replay", False),
        episodic_replay_episodes=cfg.get("episodic_replay_episodes", 2),
        use_her=cfg.get("use_her", False),
        her_k=cfg.get("her_k", 4),
        her_goal_indices=her_goal_indices,
        train_every=cfg.get("train_every", 1),
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

    # ==================== Train or Load ====================
    if args.eval_only:
        ckpt = args.checkpoint or os.path.join(args.save_dir, "checkpoint.pt")
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found at {ckpt}")
            return
        agent.load(ckpt)
        print(f"Loaded checkpoint from {ckpt}")
    else:
        ep1 = cfg["train_episodes_phase1"]
        ep2 = cfg["train_episodes_phase2"]
        frac2 = cfg["diverse_fraction_phase2"]
        use_staged = cfg.get("staged_learning", False)
        consolidation_eps = cfg.get("consolidation_episodes", 1000)
        consolidation_replay = cfg.get("consolidation_episodic_replay", 10)
        if args.quick:
            ep1 = 200
            ep2 = 300
            consolidation_eps = 100
            args.n_eval = 5

        os.makedirs(args.save_dir, exist_ok=True)
        resume = args.resume_phase

        # ---- Resume from checkpoint if requested ----
        if resume >= 1:
            ckpt_map = {
                1: "checkpoint_phase1.pt",
                2: "checkpoint_phase2.pt",
                3: "checkpoint_phase3.pt",
            }
            ckpt_path = os.path.join(args.save_dir, ckpt_map[resume])
            if not os.path.exists(ckpt_path):
                print(f"Checkpoint not found: {ckpt_path}")
                return
            agent.load(ckpt_path)
            print(f"Resumed from {ckpt_path} (skipping phases 1-{resume})")

        t0 = time.time()

        # ---- Phase 1: Diverse exploration — build SF representation ----
        if resume < 1:
            print(f"Phase 1: Diverse exploration ({ep1} episodes, 100% diverse)")
            agent.learn_environment(
                num_episodes=ep1,
                steps_per_episode=cfg["steps_per_episode"],
                diverse_start=True,
                log_interval=max(1, ep1 // 5),
            )
            agent.save(os.path.join(args.save_dir, "checkpoint_phase1.pt"))

        # ---- Phase 2: SF consolidation (staged learning) ----
        # Freeze w so φ/ψ can stabilize without chasing a moving reward signal.
        # Mirrors tabular agent's 10-epoch replay pass over M.
        if resume < 2 and use_staged and consolidation_eps > 0:
            print(f"\nPhase 2: SF consolidation ({consolidation_eps} episodes, "
                  f"w frozen, {consolidation_replay} episodic replays)")
            agent.freeze_reward_weights()

            # Boost episodic replay for consolidation
            old_episodic = agent._episodic_replay_episodes
            agent._episodic_replay_episodes = consolidation_replay

            agent.learn_environment(
                num_episodes=consolidation_eps,
                steps_per_episode=cfg["steps_per_episode"],
                diverse_start=True,
                log_interval=max(1, consolidation_eps // 5),
            )

            # Restore settings
            agent._episodic_replay_episodes = old_episodic
            agent.unfreeze_reward_weights()
            agent.save(os.path.join(args.save_dir, "checkpoint_phase2.pt"))

        # ---- Phase 3: Goal-focused — mostly fixed start ----
        if resume < 3:
            # Phase boundary: truncate buffer, reset schedule
            if resume < 2:
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

            # If staged, freeze φ so only w adapts to the goal.
            if use_staged:
                agent.freeze_sf()
                print(f"\nPhase 3: Goal-focused ({ep2} episodes, {frac2:.0%} diverse, "
                      f"φ frozen, w learning)")
            else:
                print(f"\nPhase 2: Goal-focused ({ep2} episodes, {frac2:.0%} diverse)")

            agent.learn_environment(
                num_episodes=ep2,
                steps_per_episode=cfg["steps_per_episode"],
                diverse_start=True,
                diverse_fraction=frac2,
                log_interval=max(1, ep2 // 5),
            )

            if use_staged:
                agent.unfreeze_sf()

            agent.save(os.path.join(args.save_dir, "checkpoint_phase3.pt"))

        train_time = time.time() - t0
        print(f"\nTotal training time: {train_time:.1f}s")

        # ==================== Save ====================
        save_path = os.path.join(args.save_dir, "checkpoint.pt")
        agent.save(save_path)
        print(f"Saved checkpoint to {save_path}")

        # Save training log data
        save_training_log(agent.training_log, args.save_dir)

        # Plot training curves
        fig_dir = args.save_dir.replace("data/", "figures/")
        print("\n--- Training Curves ---")
        plot_training_curves(agent.training_log, fig_dir, env_name='PointMaze')

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
    test_base.reset()
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
