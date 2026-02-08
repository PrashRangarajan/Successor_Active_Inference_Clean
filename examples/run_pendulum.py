"""Example: Running hierarchical SR agent on Pendulum.

Pendulum-v1 has a continuous action space (torque) which is discretized.
The agent must learn to swing the pendulum up from hanging (θ=π) to
upright (θ=0).  This is a goal-reaching task well suited to the SR
framework: the successor matrix M captures the dynamics, and the goal
prior C encodes the upright target.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import imageio

from core import HierarchicalSRAgent
from environments.pendulum import PendulumAdapter

import gymnasium as gym


def plot_value_function_2d(agent, adapter, save_path="figures/pendulum/value_function_2d.png"):
    """Plot value function as a 2D heatmap over (θ, ω)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    V = adapter.multiply_M_C(agent.M, agent.C)
    theta_centers, omega_centers = adapter.get_bin_centers()

    # Reshape V to 2D grid (theta × omega)
    V_grid = np.zeros((adapter.n_theta_bins, adapter.n_omega_bins))
    for t in range(adapter.n_theta_bins):
        for w in range(adapter.n_omega_bins):
            idx = adapter.state_space.state_to_index((t, w))
            V_grid[t, w] = V[idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(V_grid.T, origin='lower', aspect='auto',
                   extent=[theta_centers[0], theta_centers[-1],
                           omega_centers[0], omega_centers[-1]],
                   cmap='viridis')
    ax.set_xlabel("θ (angle)", fontsize=14)
    ax.set_ylabel("ω (angular velocity)", fontsize=14)
    ax.set_title("Value Function V = M @ C", fontsize=16)
    plt.colorbar(im, ax=ax, label="Value")

    # Mark goal region
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Upright (θ=0)')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Value function heatmap saved to {save_path}")


def run_episode_with_video(agent, adapter, init_state, max_steps=200):
    """Run an episode using the agent's policy and capture frames for video.

    Uses smooth stepping (4 physics steps per decision) for smoother video.
    """
    frames = []

    adapter.reset(init_state)
    V = adapter.multiply_M_C(agent.M, agent.C)

    goal_states_set = set(agent.goal_states)
    done = False
    steps = 0

    while not done and steps < max_steps:
        # Capture frame
        frame = adapter.render()
        if frame is not None:
            frames.append(frame)

        # Select action using agent's policy
        state_onehot = adapter._current_state
        V_adj = []
        for act in range(adapter.n_actions):
            s_next_dist = adapter.multiply_B_s(agent.B, state_onehot, act)
            V_adj.append(float(s_next_dist @ V))

        best_action = np.argmax(V_adj)

        # Take action with smooth stepping
        for _ in range(4):
            _, _, terminated, truncated, _ = adapter.step_with_info(best_action)
            done = terminated or truncated

            frame = adapter.render()
            if frame is not None:
                frames.append(frame)

            if adapter.get_current_state_index() in goal_states_set:
                done = True
                break

            if done:
                break

        steps += 1

    return frames


def run_pendulum_example():
    """Run the hierarchical SR agent on Pendulum."""

    # Configuration
    n_theta_bins = 20
    n_omega_bins = 20
    n_torque_bins = 5
    n_clusters = 4
    init_state = [np.pi, 0.0]  # Hanging down, zero velocity

    # Create gymnasium environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")

    # Wrap with adapter
    adapter = PendulumAdapter(env,
                              n_theta_bins=n_theta_bins,
                              n_omega_bins=n_omega_bins,
                              n_torque_bins=n_torque_bins)

    print("=" * 50)
    print("PENDULUM - Hierarchical Active Inference")
    print("=" * 50)
    print(f"State space: {n_theta_bins} × {n_omega_bins} = {adapter.n_states} states")
    print(f"Actions: {n_torque_bins} discrete torques from "
          f"{adapter._discrete_torques}")
    print(f"Clusters: {n_clusters}")

    # Create agent
    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=n_clusters,
        gamma=0.95,
        learning_rate=0.05,
        learn_from_experience=True,
    )

    # Set goal (upright position)
    agent.set_goal(None, reward=100.0, default_cost=-1.0)

    print(f"\nGoal states ({len(agent.goal_states)} states): "
          f"{[adapter.render_state(g) for g in agent.goal_states]}")

    # Learn environment
    print("\n" + "=" * 50)
    print("LEARNING PHASE")
    print("=" * 50)
    agent.learn_environment(num_episodes=5000)

    # Visualizations
    print("\n" + "=" * 50)
    print("VISUALIZATION")
    print("=" * 50)

    os.makedirs("figures/pendulum", exist_ok=True)
    plot_value_function_2d(agent, adapter)
    agent.visualize_clusters(save_dir="figures/pendulum/clustering")
    agent.view_matrices(save_dir="figures/pendulum/matrices", learned=True)
    print("  Saved matrix visualizations to figures/pendulum/matrices/")

    # Test hierarchical policy
    print("\n" + "=" * 50)
    print("TEST PHASE - Hierarchical Policy")
    print("=" * 50)

    agent.reset_episode(init_state=init_state)
    result = agent.run_episode_hierarchical(max_steps=200)

    print(f"\nHierarchical Results:")
    print(f"  Steps taken: {result['steps']}")
    print(f"  Total reward: {result['reward']:.2f}")
    print(f"  Reached goal: {result['reached_goal']}")
    print(f"  Final state: {result['final_state']}")

    # Test flat policy
    print("\n" + "=" * 50)
    print("TEST PHASE - Flat Policy")
    print("=" * 50)

    agent.reset_episode(init_state=init_state)
    result_flat = agent.run_episode_flat(max_steps=200)

    print(f"\nFlat Results:")
    print(f"  Steps taken: {result_flat['steps']}")
    print(f"  Total reward: {result_flat['reward']:.2f}")
    print(f"  Reached goal: {result_flat['reached_goal']}")
    print(f"  Final state: {result_flat['final_state']}")

    print("\n" + "=" * 50)
    print("COMPARISON")
    print("=" * 50)
    print(f"Hierarchical: {result['steps']} steps, reached goal: {result['reached_goal']}")
    print(f"Flat: {result_flat['steps']} steps, reached goal: {result_flat['reached_goal']}")

    # Record video
    print("\n" + "=" * 50)
    print("RECORDING VIDEO")
    print("=" * 50)

    video_path = "figures/pendulum/pendulum_episode.mp4"
    frames = run_episode_with_video(agent, adapter, init_state, max_steps=200)

    if frames:
        imageio.mimsave(video_path, frames, fps=30, macro_block_size=1)
        print(f"Video saved to: {video_path}")
    else:
        print("No frames captured")

    env.close()


if __name__ == '__main__':
    run_pendulum_example()
