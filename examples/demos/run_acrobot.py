"""Example: Running hierarchical SR agent on Acrobot.

This demonstrates how to use the unified agent with the Gymnasium Acrobot-v1 environment.
Acrobot is a 2-link planar robot that must swing the end-effector above a threshold.

The continuous 4D state (theta1, theta2, dtheta1, dtheta2) is discretized into bins.
"""

import os

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import imageio

# Import the unified framework
from core import HierarchicalSRAgent
from environments.acrobot import AcrobotAdapter
from examples.configs import ACROBOT

def run_acrobot_example():
    """Run the hierarchical SR agent on Acrobot."""

    # Configuration (from centralized config)
    n_theta_bins = ACROBOT["n_theta_bins"]
    n_dtheta_bins = ACROBOT["n_dtheta_bins"]
    n_clusters = ACROBOT["n_clusters"]
    gamma = ACROBOT["gamma"]
    learning_rate = ACROBOT["learning_rate"]
    train_episodes = ACROBOT["train_episodes"]
    test_max_steps = ACROBOT["test_max_steps"]

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
        n_dtheta_bins=n_dtheta_bins,
        goal_velocity_filter=True,  # Reduce goal dilution for better planning
    )

    # Create unified agent
    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=n_clusters,
        gamma=gamma,
        learning_rate=learning_rate,
        learn_from_experience=True,
        train_smooth_steps=10,   # Multi-step to change discrete bins
        test_smooth_steps=10,    # Match training dynamics at test time
    )

    # Set goal: default Acrobot terminal condition
    # -cos(theta1) - cos(theta1 + theta2) > 1.0
    agent.set_goal(None, reward=100.0, default_cost=-1.0)

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
        n_dtheta_bins=n_dtheta_bins,
        goal_velocity_filter=True,
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

    os.makedirs("figures/acrobot", exist_ok=True)

    # Visualize transition and successor matrices (works for all environments)
    agent.view_matrices(save_dir="figures/acrobot/matrices", learned=True)
    print("  Saved matrix visualizations to figures/acrobot/matrices/")

    # Also save raw matrices for custom analysis
    if agent.M is not None:
        np.save("figures/acrobot/M_acrobot.npy", agent.M)
        print("  Saved M matrix to figures/acrobot/M_acrobot.npy")

    if agent.B is not None:
        np.save("figures/acrobot/B_acrobot.npy", agent.B)
        print("  Saved B matrix to figures/acrobot/B_acrobot.npy")

    # Value function, policy, and overlay plots
    agent.plot_value_function(save_path="figures/acrobot/value_function.png")
    agent.plot_policy(save_path="figures/acrobot/policy.png")
    agent.plot_value_with_policy(save_path="figures/acrobot/value_with_policy.png")

    # Plot macro state distribution
    plot_macro_distribution(agent, "figures/acrobot/macro_distribution.png")

    # Record a video of the agent
    print("\n  Recording video...")
    video_path = "figures/acrobot/acrobot_episode.mp4"
    run_video_episode(agent, adapter_test, video_path, max_steps=test_max_steps)

    print("\n" + "="*50)
    print("DONE")
    print("="*50)

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

def run_video_episode(agent, adapter, save_path: str, max_steps: int = 300,
                      smooth_steps: int = 10):
    """Run an episode and save video.

    Mirrors the agent's run_episode_flat() logic so the video matches
    the reported test results:
    - Uses smooth stepping (repeats each action up to smooth_steps times
      until the discrete bin changes), matching training/test dynamics.
    - Handles is_terminal() returning None (goal_velocity_filter mode)
      by falling back to discrete bin check only.
    """
    frames = []
    goal_states_set = set(agent.goal_states)

    adapter.reset(init_state=[0.0, 0.0, 0.0, 0.0])

    V = adapter.multiply_M_C(agent.M, agent.C)
    done = False
    steps = 0

    while not done and steps < max_steps:
        # Get current state
        state_onehot = adapter._current_state
        state_idx = adapter.onehot_to_index(state_onehot)

        # Compute expected value for each action (handles stochastic transitions)
        V_adj = []
        for act in range(adapter.n_actions):
            s_next_dist = adapter.multiply_B_s(agent.B, state_onehot, act)
            V_adj.append(float(s_next_dist @ V))

        best_action = np.argmax(V_adj)

        # Take action with smooth stepping (match agent's test_smooth_steps)
        s_before = adapter.get_current_state_index()
        for _ in range(smooth_steps):
            adapter.step(best_action)
            frame = adapter.render()
            if frame is not None:
                frames.append(frame)
            s_after = adapter.get_current_state_index()
            if s_after != s_before:
                break

        # Check terminal — handle is_terminal() returning None
        s_idx = adapter.get_current_state_index()
        in_goal_bin = s_idx in goal_states_set
        continuous_terminal = adapter.is_terminal()

        if continuous_terminal is None:
            # goal_velocity_filter mode: rely on discrete bin check only
            if in_goal_bin:
                done = True
                print(f"  Step {steps}: Reached goal (discrete bin)")
        elif in_goal_bin and continuous_terminal:
            done = True
            print(f"  Step {steps}: Reached goal (bin + continuous agree)")
        elif continuous_terminal:
            done = True
            print(f"  Step {steps}: Continuous terminal (not in goal bin — edge case)")

        steps += 1

    if not done:
        print(f"  Episode ended at max_steps={max_steps} without reaching goal")

    if frames:
        imageio.mimsave(save_path, frames, fps=30, macro_block_size=1)
        print(f"  Saved video to {save_path} ({steps} steps, {len(frames)} frames)")
    else:
        print("  No frames to save")

if __name__ == '__main__':
    run_acrobot_example()
