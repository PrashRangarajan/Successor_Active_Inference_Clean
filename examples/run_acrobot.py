"""Example: Running hierarchical SR agent on Acrobot.

This demonstrates how to use the unified agent with the Gymnasium Acrobot-v1 environment.
Acrobot is a 2-link planar robot that must swing the end-effector above a threshold.

The continuous 4D state (theta1, theta2, dtheta1, dtheta2) is discretized into bins.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import imageio

# Import the unified framework
from core import HierarchicalSRAgent
from environments.acrobot import AcrobotAdapter


def run_acrobot_example():
    """Run the hierarchical SR agent on Acrobot."""

    # Configuration
    n_theta_bins = 12    # Number of bins for angles (theta1, theta2)
    n_dtheta_bins = 10   # Number of bins for angular velocities (dtheta1, dtheta2)
    n_clusters = 6       # Number of macro states
    gamma = 0.99
    learning_rate = 0.05
    train_episodes = 5000   # Episodes for learning dynamics
    test_max_steps = 500    # Max steps per test episode

    # Initial state: [theta1, theta2, dtheta1, dtheta2]
    # Start hanging down with zero velocity
    init_state = [0.0, 0.0, 0.0, 0.0]

    print("="*60)
    print("ACROBOT HIERARCHICAL SR AGENT")
    print("="*60)
    print(f"State space: {n_theta_bins}x{n_theta_bins}x{n_dtheta_bins}x{n_dtheta_bins} = "
          f"{n_theta_bins**2 * n_dtheta_bins**2} states")
    print(f"Macro states: {n_clusters}")

    # Create Gymnasium environment
    env = gym.make('Acrobot-v1', render_mode='rgb_array')

    # Wrap with adapter
    adapter = AcrobotAdapter(
        env,
        n_theta_bins=n_theta_bins,
        n_dtheta_bins=n_dtheta_bins
    )

    # Create unified agent
    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=n_clusters,
        gamma=gamma,
        learning_rate=learning_rate,
        learn_from_experience=True,
    )

    # Set goal: default Acrobot terminal condition
    # -cos(theta1) - cos(theta1 + theta2) > 1.0
    agent.set_goal(None, reward=100.0)

    print(f"Goal states: {len(agent.goal_states)} states")

    # ==================== Learning Phase ====================
    print("\n" + "="*50)
    print("LEARNING PHASE")
    print("="*50)

    agent.learn_environment(num_episodes=train_episodes)

    # ==================== Test Phase - Hierarchical ====================
    print("\n" + "="*50)
    print("TEST PHASE - Hierarchical Policy")
    print("="*50)

    # Recreate env with longer episode limit for testing
    env_test = gym.make('Acrobot-v1', render_mode='rgb_array', max_episode_steps=test_max_steps)
    adapter_test = AcrobotAdapter(
        env_test,
        n_theta_bins=n_theta_bins,
        n_dtheta_bins=n_dtheta_bins
    )
    agent.adapter = adapter_test  # Switch to test adapter

    agent.reset_episode(init_state=init_state)
    result_hier = agent.run_episode_hierarchical(max_steps=test_max_steps)

    print(f"\nHierarchical Results:")
    print(f"  Steps taken: {result_hier['steps']}")
    print(f"  Total reward: {result_hier['reward']:.2f}")
    print(f"  Reached goal: {result_hier['reached_goal']}")

    # ==================== Test Phase - Flat ====================
    print("\n" + "="*50)
    print("TEST PHASE - Flat Policy")
    print("="*50)

    agent.reset_episode(init_state=init_state)
    result_flat = agent.run_episode_flat(max_steps=test_max_steps)

    print(f"\nFlat Results:")
    print(f"  Steps taken: {result_flat['steps']}")
    print(f"  Total reward: {result_flat['reward']:.2f}")
    print(f"  Reached goal: {result_flat['reached_goal']}")

    # ==================== Comparison ====================
    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    print(f"Hierarchical: {result_hier['steps']} steps, reached_goal: {result_hier['reached_goal']}")
    print(f"Flat: {result_flat['steps']} steps, reached_goal: {result_flat['reached_goal']}")

    # ==================== Visualization ====================
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)

    os.makedirs("figures/matrices/acrobot", exist_ok=True)
    os.makedirs("figures/clustering/acrobot", exist_ok=True)
    os.makedirs("figures/acrobot", exist_ok=True)

    # Note: Matrix visualization for 4D state space is complex
    # We can save the raw matrices for custom analysis
    if agent.M is not None:
        np.save("figures/acrobot/M_acrobot.npy", agent.M)
        print("  Saved M matrix to figures/acrobot/M_acrobot.npy")

    if agent.B is not None:
        np.save("figures/acrobot/B_acrobot.npy", agent.B)
        print("  Saved B matrix to figures/acrobot/B_acrobot.npy")

    # Plot a 2D projection of the value function (averaging over velocities)
    plot_value_function_2d(agent, "figures/acrobot/value_function_2d.png")

    # Plot macro state distribution
    plot_macro_distribution(agent, "figures/acrobot/macro_distribution.png")

    # Record a video of the agent
    print("\n  Recording video...")
    video_path = "figures/acrobot/acrobot_episode.mp4"
    run_video_episode(agent, adapter_test, video_path, max_steps=test_max_steps)

    print("\n" + "="*50)
    print("DONE")
    print("="*50)


def plot_value_function_2d(agent, save_path: str):
    """Plot 2D projection of value function over (theta1, theta2), averaged over velocities."""
    if agent.M is None or agent.C is None:
        print("  No M or C to visualize")
        return

    V = agent.adapter.multiply_M_C(agent.M, agent.C)

    n_theta = agent.adapter.n_theta_bins
    n_dtheta = agent.adapter.n_dtheta_bins

    # Reshape to 4D and average over velocity dimensions
    V_4d = V.reshape(n_theta, n_theta, n_dtheta, n_dtheta)
    V_2d = np.mean(V_4d, axis=(2, 3))

    plt.figure(figsize=(8, 6))
    plt.imshow(V_2d.T, origin='lower', aspect='equal', cmap='viridis',
               extent=[-np.pi, np.pi, -np.pi, np.pi])
    plt.colorbar(label='Value (avg over velocities)')
    plt.xlabel('theta1 (rad)')
    plt.ylabel('theta2 (rad)')
    plt.title('Value Function V = M @ C\n(averaged over angular velocities)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved value function plot to {save_path}")


def plot_macro_distribution(agent, save_path: str):
    """Plot distribution of states across macro clusters."""
    if not agent.macro_state_list:
        print("  No macro states to visualize")
        return

    n_clusters = agent.n_clusters
    cluster_sizes = [len(states) for states in agent.macro_state_list]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(n_clusters), cluster_sizes, color='steelblue', edgecolor='black')
    plt.xlabel('Macro State')
    plt.ylabel('Number of Micro States')
    plt.title('Macro State Cluster Sizes')
    plt.xticks(range(n_clusters))

    # Add count labels on bars
    for bar, size in zip(bars, cluster_sizes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(size), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved macro distribution plot to {save_path}")


def run_video_episode(agent, adapter, save_path: str, max_steps: int = 300):
    """Run an episode and save video."""
    frames = []

    adapter.reset(init_state=[0.0, 0.0, 0.0, 0.0])

    V = adapter.multiply_M_C(agent.M, agent.C)
    done = False
    steps = 0

    while not done and steps < max_steps:
        # Get current state
        state_onehot = adapter._current_state
        state_idx = adapter.onehot_to_index(state_onehot)

        # Compute values for each action
        V_adj = []
        for act in range(adapter.n_actions):
            s_next = adapter.multiply_B_s(agent.B, state_onehot, act)
            next_idx = adapter.onehot_to_index(s_next)
            V_adj.append(V[next_idx])

        best_action = np.argmax(V_adj)

        # Take action
        adapter.step(best_action)
        frame = adapter.render()
        if frame is not None:
            frames.append(frame)

        # Check terminal
        if adapter.is_terminal():
            done = True

        steps += 1

    if frames:
        imageio.mimsave(save_path, frames, fps=30)
        print(f"  Saved video to {save_path}")
    else:
        print("  No frames to save")


if __name__ == '__main__':
    run_acrobot_example()
