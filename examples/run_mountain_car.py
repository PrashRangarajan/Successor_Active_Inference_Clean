"""Example: Running hierarchical SR agent on Mountain Car.

This demonstrates how to use the unified agent with the Mountain Car environment.
Generates cluster heatmap, macro-state trajectory, stage diagram, and video.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import imageio

# Import the unified framework
from core import HierarchicalSRAgent
from environments.mountain_car import MountainCarAdapter

# Import gymnasium
import gymnasium as gym


def run_mountain_car_example():
    """Run the hierarchical SR agent on Mountain Car."""

    # Configuration
    # Note: Mountain Car requires finer discretization and more training than gridworld
    n_pos_bins = 10
    n_vel_bins = 10
    n_clusters = 6
    init_state = [-0.5, 0.0]  # Start in middle with zero velocity

    # Create gymnasium environment with rgb_array rendering for video
    # Override default max_episode_steps (200) — the agent needs more steps
    # to explore the full state space during learning and to reach the goal
    # during testing.  Without this, Gym's TimeLimit wrapper truncates
    # episodes before the car can swing up to position ≥ 0.5.
    env = gym.make("MountainCar-v0", render_mode="rgb_array", max_episode_steps=500)

    # Wrap with adapter
    adapter = MountainCarAdapter(env, n_pos_bins=n_pos_bins, n_vel_bins=n_vel_bins)

    # Create unified agent
    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=n_clusters,
        gamma=0.95,  # Lower gamma works better for Mountain Car
        learning_rate=0.05,
        learn_from_experience=True,  # Must learn for Mountain Car
    )

    # Set goal (rightmost position)
    agent.set_goal(None, reward=100.0, default_cost=-1.0)  # None = default goal

    # Learn environment
    # Mountain Car needs more episodes to explore the state space properly
    print("\n" + "="*50)
    print("LEARNING PHASE")
    print("="*50)
    agent.learn_environment(num_episodes=6000)

    # Run test episodes
    print("\n" + "="*50)
    print("TEST PHASE - Hierarchical Policy")
    print("="*50)

    agent.reset_episode(init_state=init_state)
    result = agent.run_episode_hierarchical(max_steps=500)

    print(f"\nResults:")
    print(f"  Steps taken: {result['steps']}")
    print(f"  Total reward: {result['reward']:.2f}")
    print(f"  Reached goal: {result['reached_goal']}")
    print(f"  Final state: {result['final_state']}")

    # Compare with flat policy
    print("\n" + "="*50)
    print("TEST PHASE - Flat Policy")
    print("="*50)

    agent.reset_episode(init_state=init_state)
    result_flat = agent.run_episode_flat(max_steps=500)

    print(f"\nResults:")
    print(f"  Steps taken: {result_flat['steps']}")
    print(f"  Total reward: {result_flat['reward']:.2f}")
    print(f"  Reached goal: {result_flat['reached_goal']}")
    print(f"  Final state: {result_flat['final_state']}")

    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    print(f"Hierarchical: {result['steps']} steps, reached goal: {result['reached_goal']}")
    print(f"Flat: {result_flat['steps']} steps, reached goal: {result_flat['reached_goal']}")

    # ==================== Visualization ====================
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)

    os.makedirs("figures/mountaincar", exist_ok=True)

    # Matrix visualizations
    agent.view_matrices(save_dir="figures/mountaincar/matrices", learned=True)
    print("  Saved matrix visualizations to figures/mountaincar/matrices/")

    # Cluster heatmap
    agent.visualize_clusters(save_dir="figures/mountaincar/clustering")

    # ==================== Record Episode + Trajectory ====================
    print("\n" + "="*50)
    print("RECORDING VIDEO & TRAJECTORY")
    print("="*50)

    frames, positions, velocities, actions = run_episode_with_tracking(
        agent, adapter, init_state, max_steps=500,
    )

    if frames:
        video_path = "figures/mountaincar/mountain_car_episode.mp4"
        imageio.mimsave(video_path, frames, fps=30, macro_block_size=1)
        print(f"  Video saved to: {video_path} ({len(frames)} frames)")
    else:
        print("  No frames captured")

    if positions:
        # Trajectory colored by macro state
        agent.plot_trajectory_with_macro_states(
            positions, velocities,
            save_path="figures/mountaincar/trajectory_macro.png",
        )

        # Stage diagram (snapshots + phase plot)
        if frames:
            agent.plot_stage_state_diagram(
                frames, positions, velocities,
                save_path="figures/mountaincar/mountaincar_stages.png",
            )

            # Combined vertical video (environment + animated trajectory)
            agent.generate_combined_video(
                frames, positions, velocities,
                save_path="figures/mountaincar/mountain_car_combined.mp4",
            )

    env.close()

    print("\n" + "="*50)
    print("DONE")
    print("="*50)


def run_episode_with_tracking(agent, adapter, init_state, max_steps=500):
    """Run an episode capturing frames, positions, and velocities.

    Uses the agent's own flat policy (``_select_micro_action``) so the
    recorded trajectory matches what ``run_episode_flat`` would produce.

    Returns:
        (frames, positions, velocities, actions) tuple
    """
    frames = []
    positions = []
    velocities = []
    actions_taken = []

    # Reset via the agent so agent.current_state is properly initialised
    agent.reset_episode(init_state=init_state)

    # Compute value function (same as run_episode_flat)
    V = adapter.multiply_M_C(agent.M, agent.C)

    steps = 0

    while steps < max_steps and not agent._is_at_goal():
        # Record continuous state
        obs = adapter.get_current_obs()
        positions.append(float(obs[0]))
        velocities.append(float(obs[1]))

        # Capture frame
        frame = adapter.env.render()
        if frame is not None:
            frames.append(frame)

        # Select action using the agent's micro-level policy
        action = agent._select_micro_action(V)
        actions_taken.append(action)

        # Single step (matches run_episode_flat)
        adapter.step(action)
        agent.current_state = agent._get_planning_state()

        steps += 1

    # Record final state
    obs = adapter.get_current_obs()
    positions.append(float(obs[0]))
    velocities.append(float(obs[1]))
    frame = adapter.env.render()
    if frame is not None:
        frames.append(frame)

    return frames, positions, velocities, actions_taken


if __name__ == '__main__':
    run_mountain_car_example()
