"""Random policy baseline: Success Rate vs. Maximum Steps Allowed.

Runs a random agent (uniform action selection) on the gridworld to measure
how often it stumbles upon the goal under various step budgets.  This serves
as a sanity-check baseline for the learned agents.

Usage:
    python -m examples.demos.run_random_baseline [--layout fourrooms]
                                                  [--n_episodes 1000]
"""

import argparse
import numpy as np

from environments.gridworld import GridworldAdapter, get_layout, AVAILABLE_LAYOUTS
from examples.configs import GRIDWORLD
from unified_env import StandardGridworld as SR_Gridworld


def run_random_baseline(layout_name="fourrooms", n_episodes=1000):
    """Evaluate a purely random policy on the gridworld."""

    grid_size = GRIDWORLD["grid_size"]
    layout = get_layout(layout_name, grid_size)
    goal_loc = layout.default_goal
    walls = layout.walls

    env = SR_Gridworld(grid_size)
    env.set_walls(walls)
    adapter = GridworldAdapter(env, grid_size)

    goal_idx = adapter.state_space.state_to_index(goal_loc)
    n_actions = adapter.n_actions
    max_steps_list = [45, 100, 250, 500, 1000]

    print(f"Layout: {layout_name}  |  Grid: {grid_size}x{grid_size}  |  "
          f"Goal: {goal_loc}  |  Episodes per setting: {n_episodes}\n")
    goal_reward = 100.0
    step_cost = -0.1

    print(f"{'Max Steps Allowed':>20s}  {'Success Rate (%)':>16s}  {'Avg Reward':>12s}")
    print("-" * 56)

    for max_steps in max_steps_list:
        successes = 0
        total_reward = 0.0
        for _ in range(n_episodes):
            adapter.reset(init_state=0)
            ep_reward = 0.0
            for step in range(max_steps):
                action = np.random.randint(n_actions)
                adapter.step(action)
                if adapter.get_current_state_index() == goal_idx:
                    ep_reward += goal_reward
                    successes += 1
                    break
                else:
                    ep_reward += step_cost
            total_reward += ep_reward
        rate = 100.0 * successes / n_episodes
        avg_reward = total_reward / n_episodes
        print(f"{max_steps:>20d}  {rate:>16.1f}  {avg_reward:>12.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Random policy baseline on Gridworld"
    )
    parser.add_argument("--layout", type=str, default="fourrooms",
                        choices=AVAILABLE_LAYOUTS)
    parser.add_argument("--n_episodes", type=int, default=1000)
    args = parser.parse_args()
    run_random_baseline(layout_name=args.layout, n_episodes=args.n_episodes)
