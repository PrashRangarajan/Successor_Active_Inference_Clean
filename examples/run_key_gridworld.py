"""Example: Running hierarchical SR agent on Key Gridworld.

This demonstrates how to use the unified agent with the Key Gridworld environment,
which has augmented state space (location + has_key).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

# Import the unified framework
from core import HierarchicalSRAgent
from environments.key_gridworld import KeyGridworldAdapter

# Import environment - can use either:
# Option 1: Original environment
# sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'key_env'))
# from env_key import SR_Gridworld
# Option 2: Unified environment (recommended)
from unified_env import KeyGridworld as SR_Gridworld


def run_key_gridworld_example():
    """Run the hierarchical SR agent on Key Gridworld."""

    # Configuration
    grid_size = 5
    n_clusters = 5
    init_loc = (0, 0)
    key_loc = (3, 0)
    goal_loc = (grid_size - 1, grid_size - 1)
    has_pickup_action = True  # Separate pickup action

    # Define walls
    walls = (
        [(1, x) for x in range(grid_size // 2 + 1)] +
        [(4, 2)] +
        [(3, x) for x in range(grid_size // 2 - 1, grid_size) if x != grid_size // 2 + 1]
    )

    # Create the original key environment
    env = SR_Gridworld(grid_size, key_loc=key_loc, pickup=has_pickup_action)
    env.set_walls(walls)

    # Wrap with adapter
    adapter = KeyGridworldAdapter(env, grid_size, has_pickup_action=has_pickup_action)

    # Create unified agent
    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=n_clusters,
        gamma=0.99,
        learning_rate=0.05,
        learn_from_experience=True,
    )

    # Set goal: reach goal location WITH key (has_key=1)
    goal_spec = (goal_loc[0], goal_loc[1], 1)  # (x, y, has_key)
    agent.set_goal(goal_spec, reward=100.0)

    # Learn environment
    print("\n" + "="*50)
    print("LEARNING PHASE")
    print("="*50)
    agent.learn_environment(num_episodes=1500)

    # Run test episodes
    print("\n" + "="*50)
    print("TEST PHASE - Hierarchical Policy")
    print("="*50)

    # Start at init_loc with no key (has_key=0)
    agent.reset_episode(init_state=init_loc + (0,))
    result = agent.run_episode_hierarchical(max_steps=100)

    print(f"\nResults:")
    print(f"  Steps taken: {result['steps']}")
    print(f"  Total reward: {result['reward']:.2f}")
    print(f"  Reached goal: {result['reached_goal']}")
    print(f"  Final state: {result['final_state']}")

    # Compare with flat policy
    print("\n" + "="*50)
    print("TEST PHASE - Flat Policy")
    print("="*50)

    agent.reset_episode(init_state=init_loc + (0,))
    result_flat = agent.run_episode_flat(max_steps=100)

    print(f"\nResults:")
    print(f"  Steps taken: {result_flat['steps']}")
    print(f"  Total reward: {result_flat['reward']:.2f}")
    print(f"  Reached goal: {result_flat['reached_goal']}")
    print(f"  Final state: {result_flat['final_state']}")

    print("\n" + "="*50)
    print("COMPARISON")
    print("="*50)
    print(f"Hierarchical: {result['steps']} steps")
    print(f"Flat: {result_flat['steps']} steps")

    # Visualization
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)

    # Visualize matrices
    agent.view_matrices(save_dir="figures/matrices/key", learned=True)
    print("  Saved matrix visualizations to figures/matrices/key/")

    # Visualize clusters (note: for augmented state space, this shows flattened view)
    agent.visualize_clusters(save_dir="figures/clustering/key")
    print("  Saved cluster visualizations to figures/clustering/key/")

    # Visualize macro action policies
    agent.visualize_policy(save_dir="figures/Macro Action Network/key")
    print("  Saved policy visualizations to figures/Macro Action Network/key/")


if __name__ == '__main__':
    run_key_gridworld_example()
