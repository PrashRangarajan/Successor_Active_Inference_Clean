"""Example: Running hierarchical SR agent on Mountain Car.

This demonstrates how to use the unified agent with the Mountain Car environment.
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
    env = gym.make("MountainCar-v0", render_mode="rgb_array")

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

    # Visualize matrices (works for all environments)
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)

    agent.view_matrices(save_dir="figures/mountaincar/matrices", learned=True)
    print("  Saved matrix visualizations to figures/mountaincar/matrices/")

    # Record a video of the flat policy
    print("\n" + "="*50)
    print("RECORDING VIDEO")
    print("="*50)

    os.makedirs("figures/mountaincar", exist_ok=True)
    video_path = "figures/mountaincar/mountain_car_episode.mp4"

    frames = run_episode_with_video(agent, adapter, init_state, max_steps=500)

    if frames:
        imageio.mimsave(video_path, frames, fps=30)
        print(f"Video saved to: {video_path}")
    else:
        print("No frames captured")

    env.close()


def run_episode_with_video(agent, adapter, init_state, max_steps=500):
    """Run an episode and capture frames for video."""
    frames = []

    # Reset
    adapter.reset(init_state)

    # Compute value function
    V = adapter.multiply_M_C(agent.M, agent.C)

    done = False
    steps = 0

    while not done and steps < max_steps:
        # Capture frame
        frame = adapter.env.render()
        if frame is not None:
            frames.append(frame)

        # Get current state
        state_onehot = adapter._current_state
        state_idx = adapter.get_current_state_index()

        # Compute values for each action
        V_adj = []
        for act in range(adapter.n_actions):
            s_next = adapter.multiply_B_s(agent.B, state_onehot, act)
            next_idx = adapter.onehot_to_index(s_next)
            V_adj.append(V[next_idx])

        best_action = np.argmax(V_adj)

        # Take action with smooth stepping
        for _ in range(4):
            _, _, terminated, truncated, _ = adapter.step_with_info(best_action)
            done = terminated or truncated

            # Capture frame after each step
            frame = adapter.env.render()
            if frame is not None:
                frames.append(frame)

            if done or adapter.get_current_state_index() in agent.goal_states:
                done = True
                break

        steps += 1

    return frames


if __name__ == '__main__':
    run_mountain_car_example()
