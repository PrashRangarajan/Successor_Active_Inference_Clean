"""Continuing-goal replanning experiment for PointMaze.

Demonstrates the key advantage of hierarchical SR: learn structure once,
replan instantly.  The agent navigates to a sequence of 5 goals across
different rooms of the U-maze.  After reaching each goal, it replans to
the next goal by recomputing only the preference vector C (O(N)) and the
macro preference C_macro (O(N)), while M, M_macro, clusters, adjacency
and bottleneck states are all reused unchanged.

Requires: pip install gymnasium-robotics
"""

import os
from collections import OrderedDict

import numpy as np
import imageio
import matplotlib.pyplot as plt

from core import HierarchicalSRAgent
from environments.point_maze import PointMazeAdapter
from examples.configs import POINTMAZE
from core.eval_utils import AGENT_COLORS, plot_planning_cost_bars

from examples.demos.run_point_maze import run_episode_with_tracking

import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

FIGURES_DIR = "figures/pointmaze_replan"


def run_replan_experiment():
    """Run the continuing-goal replanning experiment."""

    # ==================== Setup ====================
    n_x_bins = POINTMAZE["n_x_bins"]
    n_y_bins = POINTMAZE["n_y_bins"]
    n_clusters = POINTMAZE["n_clusters"]

    env = gym.make(
        POINTMAZE["maze_id"],
        render_mode="rgb_array",
        max_episode_steps=500,
        continuing_task=False,
    )
    adapter = PointMazeAdapter(env, n_x_bins=n_x_bins, n_y_bins=n_y_bins)
    print(f"State space: {adapter.n_states} total bins, "
          f"{adapter.n_states - len(adapter.get_wall_indices())} navigable")

    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=n_clusters,
        gamma=POINTMAZE["gamma"],
        learning_rate=POINTMAZE["learning_rate"],
        learn_from_experience=True,
        train_smooth_steps=POINTMAZE["train_smooth_steps"],
        test_smooth_steps=POINTMAZE["test_smooth_steps"],
    )

    # Initial goal for learning phase (arbitrary — will be overridden)
    goal_cell = np.array([3, 1])
    adapter.reset(reset_options={"goal_cell": goal_cell})
    agent.set_goal(None, reward=POINTMAZE["reward"],
                   default_cost=POINTMAZE["default_cost"])

    # ==================== Learning (once!) ====================
    print("\n" + "=" * 60)
    print("LEARNING PHASE (done once — structure is reusable)")
    print("=" * 60)
    agent.learn_environment(num_episodes=POINTMAZE["train_episodes"])
    print(f"  Clusters: {n_clusters}, Adjacency: {agent.adj_list}")
    n_navigable = adapter.n_states - len(adapter.get_wall_indices())

    # ==================== Goal Sequence ====================
    replan_goals = POINTMAZE["replan_goals"]
    start_position = [-1.375, 1.375]  # Top-left bin center

    print("\n" + "=" * 60)
    print(f"REPLANNING EXPERIMENT — {len(replan_goals)} sequential goals")
    print("=" * 60)
    print(f"  Start position: ({start_position[0]}, {start_position[1]})")
    for i, g in enumerate(replan_goals):
        print(f"  Goal {i + 1}: cell {g['cell']} — {g['label']}")

    results = []
    all_trajectories = []
    all_goal_positions = []
    all_frames = []
    current_pos = list(start_position)

    for gi, goal_info in enumerate(replan_goals):
        cell = np.array(goal_info["cell"])
        goal_options = {"goal_cell": cell}

        print(f"\n--- Goal {gi + 1}/{len(replan_goals)}: {goal_info['label']} "
              f"(cell {goal_info['cell']}) ---")
        print(f"  Start from: ({current_pos[0]:.3f}, {current_pos[1]:.3f})")

        # Helper: reset + re-sync goal so _agent_goal_xy matches the
        # rendered marker (PointMaze randomizes exact position per reset).
        def _reset_with_goal():
            adapter.reset(init_state=current_pos, reset_options=goal_options)
            agent.set_goal(None, reward=POINTMAZE["reward"],
                           default_cost=POINTMAZE["default_cost"])
            agent._compute_macro_preference()
            agent.current_state = agent._get_planning_state()

        # ---- Hierarchical episode ----
        _reset_with_goal()
        goal_xy = adapter._agent_goal_xy.copy()
        all_goal_positions.append({"xy": goal_xy, "label": goal_info["label"]})
        print(f"  Goal position: ({goal_xy[0]:.3f}, {goal_xy[1]:.3f})")
        result_hier = agent.run_episode_hierarchical(
            max_steps=POINTMAZE["test_max_steps"],
        )
        print(f"  Hierarchy: {result_hier['steps']} steps, "
              f"goal={result_hier['reached_goal']}, "
              f"macro_decisions={result_hier['macro_decisions']}")

        # ---- Flat episode (from same start, for comparison) ----
        _reset_with_goal()
        result_flat = agent.run_episode_flat(
            max_steps=POINTMAZE["test_max_steps"],
        )
        print(f"  Flat:      {result_flat['steps']} steps, "
              f"goal={result_flat['reached_goal']}")

        # ---- Tracking episode (for video/trajectory) ----
        # run_episode_with_tracking calls adapter.reset internally.
        # Pre-sync goal, then the tracking's own reset will update
        # _agent_goal_xy via the sync logic in adapter.reset().
        _reset_with_goal()
        frames, x_pos, y_pos, actions = run_episode_with_tracking(
            agent, adapter, max_steps=POINTMAZE["test_max_steps"],
            start_position=current_pos, reset_options=goal_options,
        )
        reached_tracking = agent._is_at_goal()
        print(f"  Tracking:  {len(actions)} actions, {len(frames)} frames, "
              f"goal={reached_tracking}")

        all_trajectories.append({"x": x_pos, "y": y_pos})
        all_frames.extend(frames)

        # Update current_pos for the next goal (continuing task)
        obs = adapter.get_current_obs()
        current_pos = [float(obs[0]), float(obs[1])]

        results.append({
            "goal_label": goal_info["label"],
            "goal_xy": goal_xy,
            "hier_steps": result_hier["steps"],
            "hier_reached": result_hier["reached_goal"],
            "hier_macro": result_hier["macro_decisions"],
            "flat_steps": result_flat["steps"],
            "flat_reached": result_flat["reached_goal"],
            "tracking_actions": len(actions),
        })

    # ==================== Summary ====================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Goal':<16} {'Hier Steps':>10} {'Hier OK':>8} {'Macro':>6} "
          f"{'Flat Steps':>10} {'Flat OK':>8}")
    print("-" * 62)
    for r in results:
        print(f"{r['goal_label']:<16} {r['hier_steps']:>10} "
              f"{'YES' if r['hier_reached'] else 'NO':>8} "
              f"{r['hier_macro']:>6} "
              f"{r['flat_steps']:>10} "
              f"{'YES' if r['flat_reached'] else 'NO':>8}")

    hier_success = sum(1 for r in results if r["hier_reached"])
    flat_success = sum(1 for r in results if r["flat_reached"])
    print("-" * 62)
    print(f"{'Success rate':<16} {hier_success}/{len(results):>9} "
          f"{'':>8} {'':>6} {flat_success}/{len(results):>9}")

    # ==================== Visualizations ====================
    print("\n" + "=" * 60)
    print("VISUALIZATIONS")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Multi-goal trajectory overlay on maze
    adapter.plot_multi_goal_maze(
        agent, all_trajectories, all_goal_positions,
        save_path=f"{FIGURES_DIR}/multi_goal_trajectory.png",
    )

    # 2. Per-goal steps comparison bar chart
    plot_replan_comparison_bars(
        results,
        save_path=f"{FIGURES_DIR}/replan_steps_comparison.png",
    )

    # 3. Replanning cost comparison
    plot_planning_cost_bars(
        n_states=n_navigable,
        n_clusters=n_clusters,
        save_path=f"{FIGURES_DIR}/replanning_cost.png",
    )

    # 4. Cluster map with all goals
    # Reset to first goal for a clean cluster plot, then add all goals manually
    first_goal_options = {"goal_cell": np.array(replan_goals[0]["cell"])}
    adapter.reset(reset_options=first_goal_options)
    agent.set_goal(None, reward=POINTMAZE["reward"],
                   default_cost=POINTMAZE["default_cost"])
    agent._compute_macro_preference()
    adapter.plot_clusters_on_maze(
        agent, save_path=f"{FIGURES_DIR}/maze_clusters_all_goals.png",
        show_goal=False,  # We'll add all goals manually
    )
    # Overwrite with a version that has all goals marked
    _plot_clusters_with_all_goals(
        adapter, agent, all_goal_positions,
        save_path=f"{FIGURES_DIR}/maze_clusters_all_goals.png",
    )

    # 5. Combined video
    if all_frames and len(all_frames) > 10:
        video_path = f"{FIGURES_DIR}/replan_combined.mp4"
        imageio.mimsave(video_path, all_frames, fps=30, macro_block_size=1)
        print(f"  Saved combined video: {video_path} ({len(all_frames)} frames)")

    env.close()
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


def plot_replan_comparison_bars(results, save_path: str):
    """Grouped bar chart: hierarchy vs flat steps per goal."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels = [r["goal_label"] for r in results]
    hier_steps = [r["hier_steps"] for r in results]
    flat_steps = [r["flat_steps"] for r in results]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_hier = ax.bar(x - width / 2, hier_steps, width,
                       label="Hierarchy",
                       color=AGENT_COLORS.get("Hierarchy", "C0"),
                       edgecolor='black', linewidth=0.5)
    bars_flat = ax.bar(x + width / 2, flat_steps, width,
                       label="Flat",
                       color=AGENT_COLORS.get("Flat", "C1"),
                       edgecolor='black', linewidth=0.5)

    # Mark failures with ✗
    for i, r in enumerate(results):
        if not r["hier_reached"]:
            ax.text(x[i] - width / 2, hier_steps[i] + 50, "✗",
                    ha='center', fontsize=14, color='red')
        if not r["flat_reached"]:
            ax.text(x[i] + width / 2, flat_steps[i] + 50, "✗",
                    ha='center', fontsize=14, color='red')

    ax.set_ylabel("Steps to Goal", fontsize=12)
    ax.set_xlabel("Goal", fontsize=12)
    ax.set_title("Replanning: Hierarchy vs Flat per Goal", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved replan comparison to {save_path}")


def _plot_clusters_with_all_goals(adapter, agent, all_goal_positions,
                                   save_path: str):
    """Cluster map with ALL numbered goal stars."""
    import matplotlib.patches as mpatches
    import matplotlib.patheffects as PathEffects

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    x_edges, y_edges = adapter.get_bin_edges()
    x_centers, y_centers = adapter.get_bin_centers()
    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]

    tab = plt.get_cmap("tab10")
    cluster_colors = [tab(i) for i in range(agent.n_clusters)]

    fig, ax = plt.subplots(figsize=(8, 8))

    for micro_idx, macro_idx in agent.micro_to_macro.items():
        xi, yi = adapter.state_space.index_to_state(micro_idx)
        rect = mpatches.Rectangle(
            (x_edges[xi], y_edges[yi]), dx, dy,
            facecolor=cluster_colors[macro_idx], edgecolor='white',
            linewidth=0.3, alpha=0.85, zorder=3,
        )
        ax.add_patch(rect)

    adapter.draw_maze_walls(ax)

    # All goal stars
    traj_cmap = plt.get_cmap("Set1")
    n_goals = len(all_goal_positions)
    for gi, g in enumerate(all_goal_positions):
        color = traj_cmap(gi / max(n_goals - 1, 1))
        gx, gy = g["xy"]
        ax.plot(gx, gy, marker='*', markersize=22, color=color,
                markeredgecolor='black', markeredgewidth=1.5, zorder=15)
        ax.text(
            gx + 0.12, gy + 0.12, str(gi + 1),
            fontsize=12, fontweight='bold', color='white',
            path_effects=[PathEffects.withStroke(linewidth=3, foreground='black')],
            zorder=16,
        )

    ax.set_xlim(adapter._x_range)
    ax.set_ylim(adapter._y_range)
    ax.set_aspect('equal')
    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Y Position", fontsize=12)
    ax.set_title("Macro-State Clusters with All Goals", fontsize=14)

    patches = [mpatches.Patch(color=cluster_colors[i], label=f'Cluster {i}')
               for i in range(agent.n_clusters)]
    patches.append(mpatches.Patch(color='#2d2d2d', label='Wall'))
    ax.legend(handles=patches, loc='upper right', fontsize=10)

    fig.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved cluster+goals plot to {save_path}")


if __name__ == '__main__':
    run_replan_experiment()
