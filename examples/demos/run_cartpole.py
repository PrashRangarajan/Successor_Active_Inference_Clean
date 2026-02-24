"""Example: Running hierarchical SR agent on CartPole (experimental).

CartPole-v1 is a survival task — the agent must keep the pole balanced.
This is an *experimental* fit with the SR framework, which assumes
goal-reaching with absorbing terminal states.

We reframe the task:
- Goal = balanced central region (center of position × angle bins)
- Episode ends when the pole falls (Gym terminated/truncated), NOT when the
  agent reaches a "goal" in the SR sense.
- Success metric = number of steps survived.

The continuous 4D state (pos, vel, angle, ang_vel) is discretized into bins.
"""

import os

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import imageio

from core import HierarchicalSRAgent
from environments.cartpole import CartPoleAdapter
from examples.configs import CARTPOLE

def run_cartpole_example():
    """Run the hierarchical SR agent on CartPole."""

    # Configuration (from centralized config)
    n_pos_bins = CARTPOLE["n_pos_bins"]
    n_vel_bins = CARTPOLE["n_vel_bins"]
    n_angle_bins = CARTPOLE["n_angle_bins"]
    n_ang_vel_bins = CARTPOLE["n_ang_vel_bins"]
    n_clusters = CARTPOLE["n_clusters"]
    gamma = CARTPOLE["gamma"]
    learning_rate = CARTPOLE["learning_rate"]
    train_episodes = CARTPOLE["train_episodes"]
    test_max_steps = CARTPOLE["test_max_steps"]

    # Initial state: centered, balanced, zero velocities
    init_state = [0.0, 0.0, 0.0, 0.0]

    n_states = n_pos_bins * n_vel_bins * n_angle_bins * n_ang_vel_bins

    print("=" * 60)
    print("CARTPOLE HIERARCHICAL SR AGENT (Experimental)")
    print("=" * 60)
    print(f"State space: {n_pos_bins}×{n_vel_bins}×{n_angle_bins}×{n_ang_vel_bins} "
          f"= {n_states} states")
    print(f"Actions: 2 (push left, push right)")
    print(f"Macro states: {n_clusters}")
    print(f"Note: CartPole is a survival task — experimental SR fit")

    # Create Gymnasium environment
    env = gym.make('CartPole-v1', render_mode='rgb_array')

    # Wrap with adapter
    adapter = CartPoleAdapter(
        env,
        n_pos_bins=n_pos_bins,
        n_vel_bins=n_vel_bins,
        n_angle_bins=n_angle_bins,
        n_ang_vel_bins=n_ang_vel_bins,
    )

    # Create agent
    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=n_clusters,
        gamma=gamma,
        learning_rate=learning_rate,
        learn_from_experience=True,
    )

    # Sparse reward: balanced central region (same pattern as pendulum sparse).
    # Uses set_shaped_goal so B is NOT made absorbing — the agent retains
    # correct action-dependent dynamics at the balanced states.
    C_sparse = adapter.create_sparse_prior(
        radius=CARTPOLE["sparse_radius"],
        reward=CARTPOLE["sparse_reward"],
        default_cost=CARTPOLE["sparse_default_cost"],
    )
    agent.set_shaped_goal(C_sparse,
                          goal_threshold=CARTPOLE["sparse_goal_threshold"])

    print(f"Goal states: {len(agent.goal_states)} states "
          f"({100 * len(agent.goal_states) / n_states:.1f}% of state space)")

    # ==================== Learning Phase ====================
    print("\n" + "=" * 50)
    print("LEARNING PHASE")
    print("=" * 50)

    agent.learn_environment(num_episodes=train_episodes)

    # ==================== Test Phase - Hierarchical ====================
    print("\n" + "=" * 50)
    print("TEST PHASE - Hierarchical Policy")
    print("=" * 50)

    # Recreate env with longer episode limit for testing
    env_test = gym.make('CartPole-v1', render_mode='rgb_array',
                        max_episode_steps=test_max_steps)
    adapter_test = CartPoleAdapter(
        env_test,
        n_pos_bins=n_pos_bins,
        n_vel_bins=n_vel_bins,
        n_angle_bins=n_angle_bins,
        n_ang_vel_bins=n_ang_vel_bins,
    )
    agent.adapter = adapter_test

    # Run using custom evaluation (gym terminated/truncated, not _is_at_goal)
    steps_hier = run_evaluation_episode(agent, adapter_test, init_state, test_max_steps)

    print(f"\nHierarchical Results:")
    print(f"  Steps survived: {steps_hier}")
    print(f"  Max possible: {test_max_steps}")

    # ==================== Test Phase - Flat ====================
    print("\n" + "=" * 50)
    print("TEST PHASE - Flat Policy")
    print("=" * 50)

    steps_flat = run_evaluation_episode(agent, adapter_test, init_state, test_max_steps)

    print(f"\nFlat Results:")
    print(f"  Steps survived: {steps_flat}")
    print(f"  Max possible: {test_max_steps}")

    # ==================== Comparison ====================
    print("\n" + "=" * 50)
    print("COMPARISON")
    print("=" * 50)
    print(f"Hierarchical: {steps_hier} steps survived")
    print(f"Flat:          {steps_flat} steps survived")

    # ==================== Visualization ====================
    print("\n" + "=" * 50)
    print("VISUALIZATION")
    print("=" * 50)

    os.makedirs("figures/cartpole", exist_ok=True)

    # Matrix visualizations
    agent.view_matrices(save_dir="figures/cartpole/matrices", learned=True)
    print("  Saved matrix visualizations to figures/cartpole/matrices/")

    # Value function, policy, and overlay plots
    agent.plot_value_function(save_path="figures/cartpole/value_function.png")
    agent.plot_policy(save_path="figures/cartpole/policy.png")
    agent.plot_value_with_policy(save_path="figures/cartpole/value_with_policy.png")

    # Macro state distribution
    plot_macro_distribution(agent, "figures/cartpole/macro_distribution.png")

    # Macro action heatmap (target cluster at each state)
    agent.plot_macro_action_heatmap(save_path="figures/cartpole/macro_actions.png")

    # Macro action network (per-transition micro-level policies)
    agent.visualize_policy(save_dir="figures/cartpole/macro_action_network")
    print("  Saved policy visualizations to figures/cartpole/macro_action_network/")

    # ==================== Record Video ====================
    print("\n" + "=" * 50)
    print("RECORDING VIDEO")
    print("=" * 50)

    video_path = "figures/cartpole/cartpole_episode.mp4"
    frames = run_episode_with_video(agent, adapter_test, init_state, test_max_steps)

    if frames:
        imageio.mimsave(video_path, frames, fps=30, macro_block_size=1)
        print(f"Video saved to: {video_path} ({len(frames)} frames)")
    else:
        print("No frames captured")

    env.close()
    env_test.close()

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)

def run_evaluation_episode(agent, adapter, init_state, max_steps):
    """Run an episode using the agent's policy.

    Uses gym terminated/truncated signals to detect episode end (NOT _is_at_goal),
    since CartPole is a survival task where the agent starts in the "goal" state.

    Returns:
        Number of steps survived.
    """
    adapter.reset(init_state)
    V = adapter.multiply_M_C(agent.M, agent.C)

    done = False
    steps = 0

    while not done and steps < max_steps:
        state_onehot = adapter._current_state

        # Compute expected value for each action
        V_adj = []
        for act in range(adapter.n_actions):
            s_next_dist = adapter.multiply_B_s(agent.B, state_onehot, act)
            V_adj.append(float(s_next_dist @ V))

        best_action = np.argmax(V_adj)

        _, _, terminated, truncated, _ = adapter.step_with_info(best_action)
        done = terminated or truncated
        steps += 1

    return steps

def run_episode_with_video(agent, adapter, init_state, max_steps):
    """Run an episode and capture frames for video.

    Uses gym terminated/truncated for episode termination.
    """
    frames = []

    adapter.reset(init_state)
    V = adapter.multiply_M_C(agent.M, agent.C)

    done = False
    steps = 0

    while not done and steps < max_steps:
        # Capture frame
        frame = adapter.render()
        if frame is not None:
            frames.append(frame)

        state_onehot = adapter._current_state

        # Compute expected value for each action
        V_adj = []
        for act in range(adapter.n_actions):
            s_next_dist = adapter.multiply_B_s(agent.B, state_onehot, act)
            V_adj.append(float(s_next_dist @ V))

        best_action = np.argmax(V_adj)

        _, _, terminated, truncated, _ = adapter.step_with_info(best_action)
        done = terminated or truncated
        steps += 1

    # Capture final frame
    frame = adapter.render()
    if frame is not None:
        frames.append(frame)

    return frames

def plot_macro_distribution(agent, save_path):
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
    plt.title('CartPole: Macro State Cluster Sizes')
    plt.xticks(range(n_clusters))

    for bar, size in zip(bars, cluster_sizes):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(size), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved macro distribution plot to {save_path}")

if __name__ == '__main__':
    run_cartpole_example()
