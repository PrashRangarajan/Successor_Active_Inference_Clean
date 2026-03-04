"""Example: Running hierarchical SR agent on Gridworld.

This demonstrates how to use the unified agent with the Gridworld environment.
You can use either the original environment or the new unified environment.
"""

import os

import argparse
import numpy as np

# Import the unified framework
from core import HierarchicalSRAgent
from environments.gridworld import GridworldAdapter, get_layout, AVAILABLE_LAYOUTS
from examples.configs import GRIDWORLD

from unified_env import StandardGridworld as SR_Gridworld

def run_gridworld_example(layout_name="fourrooms"):
    """Run the hierarchical SR agent on a gridworld environment."""

    # Configuration (from centralized config)
    grid_size = GRIDWORLD["grid_size"]
    init_loc = (0, 0)

    layout = get_layout(layout_name, grid_size)
    n_clusters = layout.n_clusters
    goal_loc = layout.default_goal
    walls = layout.walls

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
        gamma=GRIDWORLD["gamma"],
        learning_rate=GRIDWORLD["learning_rate"],
        learn_from_experience=GRIDWORLD["learn_from_experience"],
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
    agent.view_matrices(save_dir="figures/demos/gridworld/matrices", learned=False)
    print("  Saved matrix visualizations to figures/demos/gridworld/matrices/")

    # Visualize macro state clusters
    agent.visualize_clusters(save_dir="figures/demos/gridworld/clustering")
    print("  Saved cluster visualizations to figures/demos/gridworld/clustering/")

    # Visualize value function
    agent.plot_value_function(save_path="figures/demos/gridworld/value_function.png")
    agent.plot_policy(save_path="figures/demos/gridworld/policy.png")
    agent.plot_value_with_policy(save_path="figures/demos/gridworld/value_with_policy.png")
    print("  Saved value function to figures/demos/gridworld/value_function.png")

    # Run another episode and visualize trajectory
    agent.reset_episode(init_state=0)
    agent.run_episode_flat(max_steps=100)

    # Visualize actions taken
    agent.show_actions(save_path="figures/demos/gridworld/Actions_taken.png",
                       init_loc=init_loc, goal_loc=goal_loc)
    print("  Saved action trajectory to figures/demos/gridworld/Actions_taken.png")

    # Generate video of episode trajectory
    agent.show_video(save_path="figures/demos/gridworld/episode_video.mp4",
                     init_loc=init_loc, goal_loc=goal_loc)
    print("  Saved video to figures/demos/gridworld/episode_video.mp4")

    # Visualize macro action policies
    agent.visualize_policy(save_dir="figures/demos/gridworld/macro_action_network")
    print("  Saved policy visualizations to figures/demos/gridworld/macro_action_network/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Hierarchical SR agent on Gridworld"
    )
    parser.add_argument("--layout", type=str, default="fourrooms",
                        choices=AVAILABLE_LAYOUTS,
                        help="Wall layout (default: fourrooms)")
    args = parser.parse_args()
    run_gridworld_example(layout_name=args.layout)
