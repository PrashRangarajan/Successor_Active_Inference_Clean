"""Example: Running hierarchical SR agent on PointMaze.

Demonstrates the hierarchical agent on a maze environment where spatial
bottleneck structure (corridors, doorways) gives hierarchy a natural
advantage.  Generates cluster heatmap, trajectory plots, and video.

Requires: pip install gymnasium-robotics
"""

import os

import numpy as np
import imageio

from core import HierarchicalSRAgent
from environments.point_maze import PointMazeAdapter
from examples.configs import POINTMAZE
from core.eval_utils import plot_planning_steps_bars, plot_planning_cost_bars

import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


def run_point_maze_example():
    """Run the hierarchical SR agent on PointMaze UMaze."""

    # Configuration
    n_x_bins = POINTMAZE["n_x_bins"]
    n_y_bins = POINTMAZE["n_y_bins"]
    n_clusters = POINTMAZE["n_clusters"]

    # Create gymnasium-robotics environment
    env = gym.make(
        POINTMAZE["maze_id"],
        render_mode="rgb_array",
        max_episode_steps=500,
    )

    # Wrap with adapter
    adapter = PointMazeAdapter(env, n_x_bins=n_x_bins, n_y_bins=n_y_bins)
    print(f"State space: {adapter.n_states} total bins, "
          f"{adapter.n_states - len(adapter.get_wall_indices())} navigable")
    adapter.print_bin_wall_map()

    # Create agent
    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=n_clusters,
        gamma=POINTMAZE["gamma"],
        learning_rate=POINTMAZE["learning_rate"],
        learn_from_experience=True,
        train_smooth_steps=POINTMAZE["train_smooth_steps"],
        test_smooth_steps=POINTMAZE["test_smooth_steps"],
    )

    # Set goal — PointMaze picks a random goal on reset.
    # Reset first to populate desired_goal.
    adapter.reset()
    goal = adapter._desired_goal
    print(f"\nGoal location: ({goal[0]:.2f}, {goal[1]:.2f})")
    agent.set_goal(None, reward=POINTMAZE["reward"],
                   default_cost=POINTMAZE["default_cost"])

    # Learn environment
    print("\n" + "=" * 50)
    print("LEARNING PHASE")
    print("=" * 50)
    agent.learn_environment(num_episodes=POINTMAZE["train_episodes"])

    # Run test episodes from a fixed start position (top-left room)
    # so results are comparable across runs.
    test_start = [-1.5, 1.5]  # UMaze cell (row=1, col=1)
    print(f"\nTest start position: ({test_start[0]}, {test_start[1]})")

    print("\n" + "=" * 50)
    print("TEST PHASE - Hierarchical Policy")
    print("=" * 50)
    adapter.reset(init_state=test_start)
    agent.current_state = agent._get_planning_state()
    result = agent.run_episode_hierarchical(max_steps=POINTMAZE["test_max_steps"])
    print(f"  Steps: {result['steps']}, Reached goal: {result['reached_goal']}, "
          f"Macro decisions: {result['macro_decisions']}")

    print("\n" + "=" * 50)
    print("TEST PHASE - Flat Policy")
    print("=" * 50)
    adapter.reset(init_state=test_start)
    agent.current_state = agent._get_planning_state()
    result_flat = agent.run_episode_flat(max_steps=POINTMAZE["test_max_steps"])
    print(f"  Steps: {result_flat['steps']}, Reached goal: {result_flat['reached_goal']}")

    print("\n" + "=" * 50)
    print("COMPARISON")
    print("=" * 50)
    print(f"Hierarchical: {result['steps']} steps, goal: {result['reached_goal']}")
    print(f"Flat:          {result_flat['steps']} steps, goal: {result_flat['reached_goal']}")

    # Planning steps comparison
    from collections import OrderedDict
    adapter.reset(init_state=test_start)
    agent.current_state = agent._get_planning_state()
    result_reentrant = agent.run_episode_hierarchical_reentrant(
        max_steps=POINTMAZE["test_max_steps"],
    )
    print(f"Reentrant: {result_reentrant['steps']} steps, "
          f"macro_decisions={result_reentrant['macro_decisions']}, "
          f"planning_steps={result_reentrant['planning_steps']}")

    os.makedirs("figures/pointmaze", exist_ok=True)

    steps_data = OrderedDict([
        ("Hierarchy", result_reentrant['planning_steps']),
        ("Flat", result_flat['planning_steps']),
    ])
    plot_planning_steps_bars(
        steps_data,
        save_path="figures/pointmaze/planning_steps.png",
    )

    # Effective navigable states for cost comparison
    n_navigable = adapter.n_states - len(adapter.get_wall_indices())
    plot_planning_cost_bars(
        n_states=n_navigable,
        n_clusters=n_clusters,
        save_path="figures/pointmaze/planning_cost.png",
    )

    # ==================== Visualization ====================
    print("\n" + "=" * 50)
    print("VISUALIZATION")
    print("=" * 50)

    agent.view_matrices(save_dir="figures/pointmaze/matrices", learned=True)
    print("  Saved matrix visualizations")

    agent.visualize_clusters(save_dir="figures/pointmaze/clustering")

    # Maze-aware cluster plot (shows walls + cluster coloring)
    adapter.plot_clusters_on_maze(
        agent, save_path="figures/pointmaze/maze_clusters.png",
    )

    agent.plot_macro_action_heatmap(
        save_path="figures/pointmaze/macro_actions.png",
    )

    agent.visualize_policy(save_dir="figures/pointmaze/macro_action_network")

    agent.plot_value_function(save_path="figures/pointmaze/value_function.png")
    agent.plot_policy(save_path="figures/pointmaze/policy.png")
    agent.plot_value_with_policy(
        save_path="figures/pointmaze/value_with_policy.png",
    )

    # ==================== Record Episode + Trajectory ====================
    print("\n" + "=" * 50)
    print("RECORDING VIDEO & TRAJECTORY")
    print("=" * 50)

    # Start from top-left room (far from goal) for a meaningful trajectory.
    # UMaze cell (row=1, col=1) → x = 1 - 2.5 = -1.5, y = 2.5 - 1 = 1.5
    start_xy = [-1.5, 1.5]
    print(f"  Tracking start: ({start_xy[0]}, {start_xy[1]})")

    frames, x_positions, y_positions, actions = run_episode_with_tracking(
        agent, adapter, max_steps=POINTMAZE["test_max_steps"],
        start_position=start_xy,
    )
    print(f"  Trajectory: {len(actions)} actions, {len(frames)} frames, "
          f"reached goal: {agent._is_at_goal()}")

    if frames and len(frames) > 1:
        video_path = "figures/pointmaze/pointmaze_episode.mp4"
        imageio.mimsave(video_path, frames, fps=30, macro_block_size=1)
        print(f"  Video saved: {video_path} ({len(frames)} frames)")

    if len(x_positions) > 2:
        agent.plot_trajectory_with_macro_states(
            x_positions, y_positions,
            save_path="figures/pointmaze/trajectory_macro_state.png",
            color_by='macro_state',
        )
        agent.plot_trajectory_with_macro_states(
            x_positions, y_positions,
            save_path="figures/pointmaze/trajectory_macro_action.png",
            color_by='macro_action',
        )
        # plot_trajectory_with_actions uses (dim0, dim1) as axes — works for
        # (x, y) spatial trajectories, not just phase-space.
        agent.plot_trajectory_with_actions(
            x_positions, y_positions, actions,
            save_path="figures/pointmaze/trajectory_actions.png",
        )

        if frames and len(frames) > 1:
            agent.plot_stage_state_diagram(
                frames, x_positions, y_positions,
                save_path="figures/pointmaze/pointmaze_stages.png",
            )
            agent.generate_combined_video(
                frames, x_positions, y_positions,
                save_path="figures/pointmaze/pointmaze_combined.mp4",
                color_by='macro_action',
            )
    else:
        print("  Skipped trajectory plots (too few points — agent started near goal)")

    env.close()
    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)


def run_episode_with_tracking(agent, adapter, max_steps=300,
                               start_position=None):
    """Run an episode capturing frames and (x, y) positions.

    Uses the agent's flat policy for consistent trajectory recording.

    Args:
        agent: Trained HierarchicalSRAgent.
        adapter: PointMazeAdapter.
        max_steps: Maximum physics steps.
        start_position: Optional [x, y] start position. If None, uses
            Gym's random initialization.

    Returns:
        (frames, x_positions, y_positions, actions_taken) tuple.
    """
    frames = []
    x_positions = []
    y_positions = []
    actions_taken = []

    adapter.reset(init_state=start_position)
    agent.current_state = agent._get_planning_state()
    V = adapter.multiply_M_C(agent.M, agent.C)
    steps = 0

    while steps < max_steps and not agent._is_at_goal():
        obs = adapter.get_current_obs()
        x_positions.append(float(obs[0]))
        y_positions.append(float(obs[1]))

        frame = adapter.env.render()
        if frame is not None:
            frames.append(frame)

        action = agent._select_micro_action(V)
        actions_taken.append(action)

        n_phys = agent._step_with_smooth(action, agent.test_smooth_steps)
        agent.current_state = agent._get_planning_state()
        steps += n_phys

    # Final state
    obs = adapter.get_current_obs()
    x_positions.append(float(obs[0]))
    y_positions.append(float(obs[1]))
    frame = adapter.env.render()
    if frame is not None:
        frames.append(frame)

    return frames, x_positions, y_positions, actions_taken


if __name__ == '__main__':
    run_point_maze_example()
