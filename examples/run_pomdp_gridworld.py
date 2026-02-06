"""Example: Hierarchical Active Inference on POMDP Gridworld.

This example demonstrates:
1. Learning in a partially observable environment (noisy observations)
2. Maintaining and updating beliefs over hidden states
3. Using observation entropy in value computation (information gain)
4. Hierarchical planning with belief-based macro states

Key differences from MDP:
- Agent receives noisy observations, not true state
- Agent maintains belief distribution over states
- Value function includes epistemic term (observation uncertainty)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unified_env import StandardGridworld as SR_Gridworld
from environments.pomdp_gridworld import POMDPGridworldAdapter
from core.hierarchical_agent import HierarchicalSRAgent


def run_pomdp_gridworld_example():
    """Run hierarchical active inference on a POMDP gridworld."""

    print("=" * 50)
    print("POMDP GRIDWORLD - Hierarchical Active Inference")
    print("=" * 50)

    # ==================== Environment Setup ====================
    grid_size = 9
    n_clusters = 5

    # Use simpler setup: no walls, just some noisy states
    walls = []  # No walls for simplicity

    # Noisy states in the middle of the grid
    hallway_states = [
        (4, 4),  # center
        (3, 4),
        (5, 4),
        (4, 3),
        (4, 5),
    ]

    # Create environment
    env = SR_Gridworld(grid_size, noise=None)
    env.set_walls(walls)

    # Hallway indices (these states have extra observation noise)
    hallway_indices = [grid_size * pos[0] + pos[1] for pos in hallway_states]

    print(f"Walls set: {len(walls)} wall cells")
    print(f"Noisy states: {hallway_states}")

    # POMDP Mode options:
    # - use_true_state_for_learning=False (default): Full POMDP - learn entirely from beliefs
    #   More challenging but theoretically correct. Agent never sees true state.
    # - use_true_state_for_learning=True: Hybrid - learn from true states, execute with beliefs
    #   Easier learning but still demonstrates belief-based execution.
    adapter = POMDPGridworldAdapter(
        env,
        grid_size=grid_size,
        noise_level=0.2,  # 20% noise: P(correct obs) = 0.8, demonstrates POMDP behavior
        noisy_states=hallway_indices,  # Extra noise in hallways
        noise_spread=3.0,  # How spread out the hallway noise is (gamma scale)
        use_true_state_for_learning=False  # Full POMDP mode
    )

    print(f"\nGrid size: {grid_size}x{grid_size}")
    print(f"Number of states: {adapter.n_states}")
    print(f"Noisy hallway states: {hallway_states}")
    mode = "FULL POMDP (learn from beliefs)" if not adapter._use_true_state_for_learning else "HYBRID (learn from true states)"
    print(f"Learning mode: {mode}")

    # ==================== Agent Setup ====================
    init_loc = (0, 0)
    goal_loc = (grid_size - 1, grid_size - 1)
    goal_states = adapter.get_goal_states(goal_loc)

    print(f"\nInitial location: {init_loc}")
    print(f"Goal location: {goal_loc}")
    print(f"Goal state index: {goal_states}")

    # Create goal prior with information gain term
    beta = 0.1  # Weight for observation entropy
    C = adapter.create_goal_prior_with_info_gain(
        goal_states,
        reward=100.0,
        default_cost=-0.1,
        beta=beta
    )

    # Create hierarchical agent
    # Note: For discrete gridworld, use learn_from_experience=False to use
    # analytical M computation. TD learning of M requires extensive exploration
    # which doesn't happen well with random walks on large grids.
    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=n_clusters,
        gamma=0.99,
        learning_rate=0.05,
        learn_from_experience=False  # Use analytical M for gridworld
    )

    # Set goal
    agent.set_goal(goal_loc, reward=100.0)

    # Override C to include information gain term
    agent.C = C

    # ==================== Learning Phase ====================
    print("\n" + "=" * 50)
    print("LEARNING PHASE")
    print("=" * 50)

    n_episodes = 2000
    print(f"Learning environment dynamics with {n_episodes} episodes...")

    # Use agent's built-in learning (handles SR, clustering, adjacency)
    agent.learn_environment(num_episodes=n_episodes)

    # ==================== Visualize Learning ====================
    visualize_pomdp_learning(adapter, agent, grid_size)

    # ==================== Test Phase ====================
    print("\n" + "=" * 50)
    print("TEST PHASE - Hierarchical Policy")
    print("=" * 50)

    # Run hierarchical episode
    agent.reset_episode(init_state=0)
    result = agent.run_episode_hierarchical(max_steps=200)

    # Check true goal achievement (for evaluation)
    true_final_state = adapter.get_true_state_index()
    true_reached_goal = true_final_state in agent.goal_states

    print(f"\nHierarchical Results:")
    print(f"  Steps taken: {result['steps']}")
    print(f"  Total reward: {result['reward']:.2f}")
    print(f"  Believes reached goal: {result['reached_goal']}")
    print(f"  Actually reached goal: {true_reached_goal}")
    print(f"  Believed final state: {result['final_state']}")
    print(f"  True final state: {adapter.get_true_state()}")

    # Compare with flat policy
    print("\n" + "=" * 50)
    print("TEST PHASE - Flat Policy")
    print("=" * 50)

    agent.reset_episode(init_state=0)
    result_flat = agent.run_episode_flat(max_steps=200)

    # Check true goal achievement (for evaluation)
    true_final_state_flat = adapter.get_true_state_index()
    true_reached_goal_flat = true_final_state_flat in agent.goal_states

    print(f"\nFlat Results:")
    print(f"  Steps taken: {result_flat['steps']}")
    print(f"  Total reward: {result_flat['reward']:.2f}")
    print(f"  Believes reached goal: {result_flat['reached_goal']}")
    print(f"  Actually reached goal: {true_reached_goal_flat}")
    print(f"  Believed final state: {result_flat['final_state']}")
    print(f"  True final state: {adapter.get_true_state()}")

    print("\n" + "=" * 50)
    print("COMPARISON")
    print("=" * 50)
    print(f"Hierarchical: {result['steps']} steps, believes goal: {result['reached_goal']}, actually reached: {true_reached_goal}")
    print(f"Flat: {result_flat['steps']} steps, believes goal: {result_flat['reached_goal']}, actually reached: {true_reached_goal_flat}")

    # ==================== Belief vs True State Analysis ====================
    print("\n" + "=" * 50)
    print("BELIEF vs TRUE STATE (POMDP Analysis)")
    print("=" * 50)

    # Show history of true states vs beliefs
    print("\nSample of trajectory (true state vs believed state):")
    true_states = adapter.state_history
    belief_states = adapter.belief_history
    observations = adapter.observation_history

    # Show first 10 steps
    n_show = min(10, len(true_states))
    print(f"{'Step':>4} | {'True State':>12} | {'Observation':>12} | {'Belief State':>12} | {'Match':>6}")
    print("-" * 60)
    for i in range(n_show):
        true_loc = adapter.render_state(true_states[i])
        obs_loc = adapter.render_state(observations[i])
        belief_loc = adapter.render_state(belief_states[i])
        match = "✓" if true_states[i] == belief_states[i] else "✗"
        print(f"{i:>4} | {str(true_loc):>12} | {str(obs_loc):>12} | {str(belief_loc):>12} | {match:>6}")

    # Count belief errors
    n_errors = sum(1 for t, b in zip(true_states, belief_states) if t != b)
    print(f"\nBelief accuracy: {len(true_states) - n_errors}/{len(true_states)} " +
          f"({100 * (len(true_states) - n_errors) / len(true_states):.1f}%)")
    print(f"Note: Errors occur due to noisy observations - this is expected in a POMDP!")

    return agent, adapter


def visualize_pomdp_learning(adapter, agent, grid_size):
    """Visualize POMDP-specific learning results."""
    os.makedirs("figures/pomdp", exist_ok=True)

    # 1. Observation model entropy
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Observation entropy map
    entropy_map = adapter.get_observation_entropy().reshape(grid_size, grid_size).T
    im1 = axes[0].imshow(entropy_map, cmap='hot')
    axes[0].set_title("Observation Entropy (Higher = Noisier)")
    plt.colorbar(im1, ax=axes[0])

    # Value function
    V = agent.M @ agent.C
    V_map = V.reshape(grid_size, grid_size).T
    im2 = axes[1].imshow(V_map, cmap='coolwarm')
    axes[1].set_title("Value Function")
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig("figures/pomdp/entropy_and_value.png")
    plt.close()

    # 2. Observation model visualization (for a few states)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    states_to_show = [0, grid_size // 2, grid_size * (grid_size // 2) + grid_size // 2, grid_size * grid_size - 1]
    for i, state_idx in enumerate(states_to_show):
        obs_dist = adapter.A[:, state_idx].reshape(grid_size, grid_size).T
        im = axes[i].imshow(obs_dist, cmap='Blues')
        state_loc = adapter.render_state(state_idx)
        axes[i].set_title(f"P(obs | state={state_loc})")
        plt.colorbar(im, ax=axes[i])

    plt.tight_layout()
    plt.savefig("figures/pomdp/observation_model.png")
    plt.close()

    print("Visualizations saved to figures/pomdp/")


if __name__ == "__main__":
    run_pomdp_gridworld_example()
