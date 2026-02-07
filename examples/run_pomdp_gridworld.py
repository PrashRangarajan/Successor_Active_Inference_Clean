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

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from unified_env import StandardGridworld as SR_Gridworld
from environments.pomdp_gridworld import POMDPGridworldAdapter
from core.hierarchical_agent import HierarchicalSRAgent


def visualize_pomdp_episode_comparison(adapter, agent, grid_size):
    """Compare hierarchical vs flat trajectories on the POMDP environment.

    Shows true path (solid) vs believed path (dashed) overlaid on the
    observation entropy heatmap for both hierarchical and flat episodes.

    Args:
        adapter: POMDP adapter with episode history
        agent: Agent (used for accessing goal_states)
        grid_size: Size of the grid
    """
    os.makedirs("figures/pomdp_gridworld", exist_ok=True)

    true_states = adapter.state_history
    beliefs = adapter.belief_history

    if not true_states or len(true_states) < 2:
        print("No episode history to compare")
        return

    entropy_vals = adapter.get_observation_entropy()
    entropy_grid = entropy_vals.reshape(grid_size, grid_size).T

    # Convert indices to (x, y) coordinates
    true_locs = [adapter.render_state(s) for s in true_states]
    belief_locs = [adapter.render_state(b) for b in beliefs]

    true_x = [loc[0] for loc in true_locs]
    true_y = [loc[1] for loc in true_locs]
    belief_x = [loc[0] for loc in belief_locs]
    belief_y = [loc[1] for loc in belief_locs]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(entropy_grid, cmap='YlOrRd', interpolation='nearest', alpha=0.6)

    # True trajectory
    ax.plot(true_x, true_y, 'b-o', linewidth=2, markersize=3,
            alpha=0.8, label='True Path', zorder=4)
    # Believed trajectory
    ax.plot(belief_x, belief_y, 'r--s', linewidth=1.5, markersize=3,
            alpha=0.7, label='Believed Path', zorder=3)

    # Mark start and goal
    ax.scatter(true_x[0], true_y[0], color='blue', s=200, marker='o',
              zorder=5, edgecolors='white', linewidths=2, label='Start')
    if agent.goal_states:
        goal_loc = adapter.state_space.index_to_state(agent.goal_states[0])
        ax.scatter(goal_loc[0], goal_loc[1], color='lime', s=300, marker='*',
                  zorder=5, edgecolors='black', linewidths=1.5, label='Goal')

    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_title("POMDP Episode: True vs Believed Trajectory", fontsize=15)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig("figures/pomdp_gridworld/episode_trajectory_comparison.png", bbox_inches='tight')
    plt.close()
    print("Episode trajectory comparison saved to figures/pomdp_gridworld/episode_trajectory_comparison.png")


def run_pomdp_gridworld_example():
    """Run hierarchical active inference on a POMDP gridworld."""

    print("=" * 50)
    print("POMDP GRIDWORLD - Hierarchical Active Inference")
    print("=" * 50)

    # ==================== Environment Setup ====================
    grid_size = 9
    n_clusters = 5

    # 4-rooms layout: vertical and horizontal walls with doorways
    mid = grid_size // 2  # 4

    # Vertical wall (x=mid for all y, except doorways)
    vertical_door_y = [1, 7]
    vertical_wall = [(mid, y) for y in range(grid_size) if y not in vertical_door_y]

    # Horizontal wall (y=mid for all x, except doorways)
    horizontal_door_x = [1, 7]
    horizontal_wall = [(x, mid) for x in range(grid_size) if x not in horizontal_door_x]

    walls = vertical_wall + horizontal_wall

    # Noisy states: top-right room is heavily noisy
    # Agent starts at (0,0), goal at (8,8). Direct route goes through top-right room.
    # With high noise there, the EFE-aware agent should prefer the longer
    # route through the bottom-left room (clean path).
    #
    # Room layout:
    #   Top-left (0-3, 0-3):     Agent start - CLEAN
    #   Top-right (5-8, 0-3):    Direct route - NOISY
    #   Bottom-left (0-3, 5-8):  Alternative  - CLEAN
    #   Bottom-right (5-8, 5-8): Goal         - CLEAN
    hallway_states = [
        (x, y)
        for x in range(mid + 1, grid_size)
        for y in range(mid)
        if (x, y) not in walls
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
        noise_level=0.3,  # 30% base noise — harder POMDP
        noisy_states=hallway_indices,  # Extra noise in top-right room
        noise_spread=3.0,  # Noisy room is 3x noisier than base
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
    beta = 1.0  # Weight for observation entropy — strong enough to reroute around noisy room
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

    # ==================== Visualize POMDP Environment & Learning ====================
    print("\n" + "=" * 50)
    print("VISUALIZATIONS")
    print("=" * 50)

    # POMDP environment visualizations
    agent.visualize_observation_entropy()
    agent.visualize_observation_model()
    agent.visualize_noise_zones(init_loc=init_loc, goal_loc=goal_loc)
    agent.visualize_pomdp_value_comparison(beta=beta)

    # Standard agent visualizations
    agent.visualize_clusters(save_dir="figures/pomdp_gridworld/clustering")
    agent.visualize_value_function(save_path="figures/pomdp_gridworld/value_function.png")
    agent.view_matrices(save_dir="figures/pomdp_gridworld/matrices")

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

    # Generate video of hierarchical episode trajectory
    agent.show_video(save_path="figures/pomdp_gridworld/episode_video_hier.mp4",
                     init_loc=init_loc, goal_loc=goal_loc)
    print("  Saved video to figures/pomdp_gridworld/episode_video_hier.mp4")

    # Visualize belief trajectory for hierarchical episode
    agent.visualize_belief_trajectory()

    # Visualize episode trajectory comparison
    visualize_pomdp_episode_comparison(adapter, agent, grid_size)

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
        match = "Y" if true_states[i] == belief_states[i] else "N"
        print(f"{i:>4} | {str(true_loc):>12} | {str(obs_loc):>12} | {str(belief_loc):>12} | {match:>6}")

    # Count belief errors
    n_errors = sum(1 for t, b in zip(true_states, belief_states) if t != b)
    print(f"\nBelief accuracy: {len(true_states) - n_errors}/{len(true_states)} " +
          f"({100 * (len(true_states) - n_errors) / len(true_states):.1f}%)")
    # print(f"Note: Errors occur due to noisy observations - this is expected in a POMDP!")

    return agent, adapter


if __name__ == "__main__":
    run_pomdp_gridworld_example()
