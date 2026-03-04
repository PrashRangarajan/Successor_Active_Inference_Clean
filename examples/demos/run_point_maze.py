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
    # continuing_task=False so that Gym returns terminated=True when the
    # ball reaches the goal.  This lets _step_with_smooth stop immediately
    # rather than overshooting.
    render_kwargs = {}
    if "render_width" in POINTMAZE:
        render_kwargs["width"] = POINTMAZE["render_width"]
    if "render_height" in POINTMAZE:
        render_kwargs["height"] = POINTMAZE["render_height"]

    env = gym.make(
        POINTMAZE["maze_id"],
        render_mode="rgb_array",
        max_episode_steps=POINTMAZE["test_max_steps"],
        continuing_task=False,
        **render_kwargs,
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

    # Set goal in the bottom-left room (far from start in top-left)
    # to guarantee the hierarchy has to navigate through the corridor.
    # goal_cell=[3, 1] places the goal in maze row 3, col 1.
    # We pass this to ALL future resets so the Gym rendering shows the
    # correct red goal marker.
    goal_cell = np.array([3, 1])
    goal_options = {"goal_cell": goal_cell}
    adapter.reset(reset_options=goal_options)
    goal = adapter._desired_goal
    print(f"\nGoal location: ({goal[0]:.2f}, {goal[1]:.2f})")
    agent.set_goal(None, reward=POINTMAZE["reward"],
                   default_cost=POINTMAZE["default_cost"])

    # Learn environment
    print("\n" + "=" * 50)
    print("LEARNING PHASE")
    print("=" * 50)
    agent.learn_environment(num_episodes=POINTMAZE["train_episodes"])

    # Run test episodes from a fixed start position (top-left room).
    # Use a bin CENTER to avoid landing on wall boundaries.
    # Bin (4, 15) center = (-1.375, 1.375) — top-left navigable bin.
    test_start = [-1.375, 1.375]
    print(f"\nTest start position: ({test_start[0]}, {test_start[1]})")

    print("\n" + "=" * 50)
    print("TEST PHASE - Hierarchical Policy")
    print("=" * 50)
    adapter.reset(init_state=test_start, reset_options=goal_options)
    agent.current_state = agent._get_planning_state()
    result = agent.run_episode_hierarchical(max_steps=POINTMAZE["test_max_steps"])
    print(f"  Steps: {result['steps']}, Reached goal: {result['reached_goal']}, "
          f"Macro decisions: {result['macro_decisions']}")

    print("\n" + "=" * 50)
    print("TEST PHASE - Flat Policy")
    print("=" * 50)
    adapter.reset(init_state=test_start, reset_options=goal_options)
    agent.current_state = agent._get_planning_state()
    result_flat = agent.run_episode_flat(max_steps=POINTMAZE["test_max_steps"])
    print(f"  Steps: {result_flat['steps']}, Reached goal: {result_flat['reached_goal']}")

    print("\n" + "=" * 50)
    print("COMPARISON")
    print("=" * 50)
    print(f"Hierarchical: {result['steps']} steps, goal: {result['reached_goal']}")
    print(f"Flat:          {result_flat['steps']} steps, goal: {result_flat['reached_goal']}")

    # Planning steps comparison
    # Use the actual hierarchical result (which navigates via macro actions)
    # rather than the reentrant method (which follows the flat V = M·C policy
    # and gets stuck due to vanishing value gradients in the U-corridor).
    from collections import OrderedDict

    os.makedirs("figures/demos/pointmaze", exist_ok=True)

    steps_data = OrderedDict([
        ("Hierarchy", result['planning_steps']),
        ("Flat", result_flat['planning_steps']),
    ])
    plot_planning_steps_bars(
        steps_data,
        save_path="figures/demos/pointmaze/planning_steps.png",
    )

    # Effective navigable states for cost comparison
    n_navigable = adapter.n_states - len(adapter.get_wall_indices())
    plot_planning_cost_bars(
        n_states=n_navigable,
        n_clusters=n_clusters,
        save_path="figures/demos/pointmaze/planning_cost.png",
    )

    # ==================== Visualization ====================
    print("\n" + "=" * 50)
    print("VISUALIZATION")
    print("=" * 50)

    agent.view_matrices(save_dir="figures/demos/pointmaze/matrices", learned=True)
    print("  Saved matrix visualizations")

    agent.visualize_clusters(save_dir="figures/demos/pointmaze/clustering")

    # Maze-aware cluster plot (shows walls + cluster coloring)
    # Pass the original goal so the star appears at the correct location,
    # even though adapter._desired_goal may have changed during later resets.
    adapter.plot_clusters_on_maze(
        agent, save_path="figures/demos/pointmaze/maze_clusters.png",
        goal_xy=goal,
    )

    agent.plot_macro_action_heatmap(
        save_path="figures/demos/pointmaze/macro_actions.png",
    )

    agent.visualize_policy(save_dir="figures/demos/pointmaze/macro_action_network")

    agent.plot_value_function(save_path="figures/demos/pointmaze/value_function.png")
    agent.plot_policy(save_path="figures/demos/pointmaze/policy.png")
    agent.plot_value_with_policy(
        save_path="figures/demos/pointmaze/value_with_policy.png",
    )

    # ==================== Record Episode + Trajectory ====================
    print("\n" + "=" * 50)
    print("RECORDING VIDEO & TRAJECTORY")
    print("=" * 50)

    # Start from top-left room (far from goal) for a meaningful trajectory.
    # Use a bin center to avoid wall boundaries.
    start_xy = [-1.375, 1.375]
    print(f"  Tracking start: ({start_xy[0]}, {start_xy[1]})")

    frames, x_positions, y_positions, actions = run_episode_with_tracking(
        agent, adapter, max_steps=POINTMAZE["test_max_steps"],
        start_position=start_xy, reset_options=goal_options,
    )
    print(f"  Trajectory: {len(actions)} actions, {len(frames)} frames, "
          f"reached goal: {agent._is_at_goal()}")

    if frames and len(frames) > 1:
        video_path = "figures/demos/pointmaze/pointmaze_episode.mp4"
        imageio.mimsave(video_path, frames, fps=30, macro_block_size=1)
        print(f"  Video saved: {video_path} ({len(frames)} frames)")

    if len(x_positions) > 2:
        agent.plot_trajectory_with_macro_states(
            x_positions, y_positions,
            save_path="figures/demos/pointmaze/trajectory_macro_state.png",
            color_by='macro_state',
        )
        agent.plot_trajectory_with_macro_states(
            x_positions, y_positions,
            save_path="figures/demos/pointmaze/trajectory_macro_action.png",
            color_by='macro_action',
        )
        # plot_trajectory_with_actions uses (dim0, dim1) as axes — works for
        # (x, y) spatial trajectories, not just phase-space.
        agent.plot_trajectory_with_actions(
            x_positions, y_positions, actions,
            save_path="figures/demos/pointmaze/trajectory_actions.png",
        )

        if frames and len(frames) > 1:
            agent.plot_stage_state_diagram(
                frames, x_positions, y_positions,
                save_path="figures/demos/pointmaze/pointmaze_stages.png",
            )
            agent.generate_combined_video(
                frames, x_positions, y_positions,
                save_path="figures/demos/pointmaze/pointmaze_combined.mp4",
                color_by='macro_action',
            )
    else:
        print("  Skipped trajectory plots (too few points — agent started near goal)")

    env.close()
    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)


def run_episode_with_tracking(agent, adapter, max_steps=300,
                               start_position=None, reset_options=None,
                               goal_xy=None):
    """Run an episode using the HIERARCHICAL policy while capturing frames.

    Mirrors agent.run_episode_hierarchical() but records (x, y) positions
    and rendered frames at each micro step for video generation.

    Args:
        agent: Trained HierarchicalSRAgent.
        adapter: PointMazeAdapter.
        max_steps: Maximum physics steps.
        start_position: Optional [x, y] start position.
        reset_options: Optional dict forwarded to env.reset() (e.g. goal_cell).
        goal_xy: Optional [x, y] to pin the goal position after reset.
                 PointMaze randomizes the exact continuous position within
                 a cell on each reset; passing goal_xy overrides that so
                 every run uses the identical goal for fair comparison.

    Returns:
        (frames, x_positions, y_positions, actions_taken) tuple.
    """
    frames = []
    x_positions = []
    y_positions = []
    actions_taken = []

    adapter.reset(init_state=start_position, reset_options=reset_options)
    if goal_xy is not None:
        adapter._desired_goal = np.array(goal_xy, dtype=np.float64)
        adapter._agent_goal_xy = np.array(goal_xy, dtype=np.float64)
    agent.current_state = agent._get_planning_state()
    steps = 0

    def _capture():
        """Capture current frame and position."""
        obs = adapter.get_current_obs()
        x_positions.append(float(obs[0]))
        y_positions.append(float(obs[1]))
        frame = adapter.env.render()
        if frame is not None:
            frames.append(frame)

    def _step_with_capture(action):
        """Execute one micro action with frame capture."""
        nonlocal steps
        _capture()
        actions_taken.append(action)
        n_phys, _ = agent._step_with_smooth(action, agent.test_smooth_steps)
        agent.current_state = agent._get_planning_state()
        steps += n_phys

    # ---- Macro phase: navigate room-to-room ----
    s_idx = adapter.get_current_state_index()

    goal_macro_states = set()
    for gs in agent.goal_states:
        if gs in agent.micro_to_macro:
            goal_macro_states.add(agent.micro_to_macro[gs])

    while steps < max_steps:
        if s_idx not in agent.micro_to_macro:
            break
        s_macro = agent.micro_to_macro[s_idx]
        if s_macro in goal_macro_states:
            break

        V_macro = agent.M_macro @ agent.C_macro
        best_macro_action = agent._select_macro_action(s_macro, V_macro)
        if best_macro_action is None:
            break

        target_macro = agent.adj_list[s_macro][best_macro_action]

        # Execute macro action (navigate toward bottleneck / target cluster)
        bottleneck = agent.bottleneck_states.get((s_macro, target_macro), [])
        if not bottleneck:
            bottleneck = agent.macro_state_list[target_macro]
        if not bottleneck:
            break

        C_temp = adapter.create_goal_prior(bottleneck, reward=10.0, default_cost=0.0)
        V_temp = adapter.multiply_M_C(agent.M, C_temp)

        while steps < max_steps:
            s_idx = adapter.get_current_state_index()
            if s_idx in bottleneck or agent._is_at_goal():
                break
            if s_idx in agent.micro_to_macro:
                current_macro = agent.micro_to_macro[s_idx]
                if current_macro == target_macro:
                    break
                if current_macro != s_macro:
                    break

            action = agent._select_micro_action(V_temp)
            _step_with_capture(action)

        s_idx = adapter.get_current_state_index()
        if agent._is_at_goal():
            break

    # ---- Micro phase: fine-grained navigation to exact goal ----
    if steps < max_steps and not agent._is_at_goal():
        V = adapter.multiply_M_C(agent.M, agent.C)

        while steps < max_steps:
            if agent._is_at_goal():
                break
            action = agent._select_micro_action(V)
            _step_with_capture(action)

    # Final state capture
    _capture()

    return frames, x_positions, y_positions, actions_taken


if __name__ == '__main__':
    run_point_maze_example()
