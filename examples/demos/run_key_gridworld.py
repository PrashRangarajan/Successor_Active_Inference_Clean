"""Example: Running hierarchical SR agent on Key Gridworld.

This demonstrates how to use the unified agent with the Key Gridworld environment,
which has augmented state space (location + has_key).
"""

import os

import numpy as np

# Import the unified framework
from core import HierarchicalSRAgent
from environments.key_gridworld import KeyGridworldAdapter
from examples.configs import KEY_GRIDWORLD

from unified_env import KeyGridworld as SR_Gridworld

def run_key_gridworld_example():
    """Run the hierarchical SR agent on Key Gridworld."""

    # Configuration (from centralized config)
    grid_size = KEY_GRIDWORLD["grid_size"]
    n_clusters = KEY_GRIDWORLD["n_clusters"]
    init_loc = KEY_GRIDWORLD["init_loc"]
    key_loc = KEY_GRIDWORLD["key_loc"]
    goal_loc = (grid_size - 1, grid_size - 1)
    has_pickup_action = KEY_GRIDWORLD["has_pickup_action"]

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
        gamma=KEY_GRIDWORLD["gamma"],
        learning_rate=KEY_GRIDWORLD["learning_rate"],
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

    # Visualize matrices (M from origin uses init_loc by default)
    origin_idx = init_loc[0] * grid_size + init_loc[1]
    agent.view_matrices(save_dir="figures/demos/key_gridworld/matrices", learned=True,
                        origin_state=origin_idx)
    print("  Saved matrix visualizations to figures/demos/key_gridworld/matrices/")

    # Visualize clusters (note: for augmented state space, this shows flattened view)
    agent.visualize_clusters(save_dir="figures/demos/key_gridworld/clustering")
    print("  Saved cluster visualizations to figures/demos/key_gridworld/clustering/")

    # Visualize value function
    agent.plot_value_function(save_path="figures/demos/key_gridworld/value_function.png")
    agent.plot_policy(save_path="figures/demos/key_gridworld/policy.png")
    agent.plot_value_with_policy(save_path="figures/demos/key_gridworld/value_with_policy.png")

    # Run another episode and visualize trajectory
    agent.reset_episode(init_state=init_loc + (0,))
    agent.run_episode_flat(max_steps=100)

    # Visualize actions taken
    agent.show_actions(save_path="figures/demos/key_gridworld/Actions_taken.png",
                       init_loc=init_loc, goal_loc=goal_loc)
    print("  Saved action trajectory to figures/demos/key_gridworld/Actions_taken.png")

    # Generate video of episode trajectory (with key status + backtracking indicators)
    agent.show_video(save_path="figures/demos/key_gridworld/episode_video.mp4",
                     init_loc=init_loc, goal_loc=goal_loc, key_loc=key_loc)
    print("  Saved video to figures/demos/key_gridworld/episode_video.mp4")

    # Visualize macro action policies
    agent.visualize_policy(save_dir="figures/demos/key_gridworld/macro_action_network")
    print("  Saved policy visualizations to figures/demos/key_gridworld/macro_action_network/")

    # Composite figure (grid layout + value function + clusters + spectral embedding)
    agent.visualize_key_gridworld_composite(
        save_path="figures/demos/key_gridworld/key_gridworld_composite.png",
        init_loc=init_loc, goal_loc=goal_loc, key_loc=key_loc,
    )
    print("  Saved composite figure to figures/demos/key_gridworld/key_gridworld_composite.png")

if __name__ == '__main__':
    run_key_gridworld_example()
