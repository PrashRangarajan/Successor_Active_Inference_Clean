"""Example: Running hierarchical SR agent on Gridworld.

This demonstrates how to use the unified agent with the Gridworld environment.
You can use either the original environment or the new unified environment.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np

# Import the unified framework
from core import HierarchicalSRAgent
from environments.gridworld import GridworldAdapter

# Import environment - can use either:
# Option 1: Original environment
# from env import SR_Gridworld
# Option 2: Unified environment (recommended)
from unified_env import StandardGridworld as SR_Gridworld


def run_gridworld_example():
    """Run the hierarchical SR agent on a gridworld environment."""

    # Configuration
    grid_size = 9
    n_clusters = 4
    init_loc = (0, 0)
    goal_loc = (grid_size - 1, grid_size - 1)

    # Define walls
    walls = (
        [(4, x) for x in range(grid_size) if x not in [2, 6]] +
        [(x, 4) for x in range(grid_size) if x not in [2, 6]]
    )

    # Create the original environment
    env = SR_Gridworld(grid_size)
    env.set_walls(walls)

    # Wrap with adapter
    adapter = GridworldAdapter(env, grid_size)

    # Create unified agent
    # Note: For gridworld with known transition dynamics, use learn_from_experience=False
    # for more reliable results. Use learn_from_experience=True for environments
    # where dynamics must be learned (e.g., Mountain Car).
    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=n_clusters,
        gamma=0.99,
        learning_rate=0.05,
        learn_from_experience=False,  # Use analytical M since we know B
    )

    # Set goal
    agent.set_goal(goal_loc, reward=100.0)

    # Learn environment (SR, clustering, adjacency)
    print("\n" + "="*50)
    print("LEARNING PHASE")
    print("="*50)
    agent.learn_environment(num_episodes=1000)

    # Run test episodes
    print("\n" + "="*50)
    print("TEST PHASE - Hierarchical Policy")
    print("="*50)

    agent.reset_episode(init_state=0)  # Start at state 0
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

    agent.reset_episode(init_state=0)
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

    # Visualization (optional - uncomment to generate figures)
    print("\n" + "="*50)
    print("VISUALIZATION")
    print("="*50)

    # Visualize matrices (B, M at micro and macro levels)
    agent.view_matrices(save_dir="figures/gridworld/matrices", learned=False)
    print("  Saved matrix visualizations to figures/gridworld/matrices/")

    # Visualize macro state clusters
    agent.visualize_clusters(save_dir="figures/gridworld/clustering")
    print("  Saved cluster visualizations to figures/gridworld/clustering/")

    # Visualize value function
    agent.visualize_value_function(save_path="figures/gridworld/value_function.png")
    print("  Saved value function to figures/gridworld/value_function.png")

    # Run another episode and visualize trajectory
    agent.reset_episode(init_state=0)
    agent.run_episode_flat(max_steps=100)

    # Visualize actions taken
    agent.show_actions(save_path="figures/gridworld/Actions_taken.png",
                       init_loc=init_loc, goal_loc=goal_loc)
    print("  Saved action trajectory to figures/gridworld/Actions_taken.png")

    # Generate video of episode trajectory
    agent.show_video(save_path="figures/gridworld/episode_video.mp4",
                     init_loc=init_loc, goal_loc=goal_loc)
    print("  Saved video to figures/gridworld/episode_video.mp4")

    # Visualize macro action policies
    agent.visualize_policy(save_dir="figures/gridworld/macro_action_network")
    print("  Saved policy visualizations to figures/gridworld/macro_action_network/")


if __name__ == '__main__':
    run_gridworld_example()
