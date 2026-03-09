"""Visualize Q-value landscape for trained Neural SF PointMaze agents.

Generates heatmaps of max Q-values and policy arrows over the maze,
revealing where the agent gets stuck and why.

Usage:
    python scripts/visualize_qvalues.py
    python scripts/visualize_qvalues.py --checkpoint data/neural_point_maze_exp_a/checkpoint.pt
"""

import argparse
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize

import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

from environments.point_maze import PointMazeAdapter
from core.neural.continuous_adapter import ContinuousAdapter
from core.neural.agent import NeuralSRAgent
from examples.configs import NEURAL_POINTMAZE


# 8 directional actions: unit vectors for arrow plotting
ACTION_DIRS = {
    0: (1, 0),    # right
    1: (1, 1),    # up-right
    2: (0, 1),    # up
    3: (-1, 1),   # up-left
    4: (-1, 0),   # left
    5: (-1, -1),  # down-left
    6: (0, -1),   # down
    7: (1, -1),   # down-right
}


def load_agent(checkpoint_path, device='cpu'):
    """Load a trained agent from checkpoint."""
    cfg = NEURAL_POINTMAZE

    env = gym.make(cfg["maze_id"], max_episode_steps=cfg["steps_per_episode"])
    base_adapter = PointMazeAdapter(env, n_x_bins=cfg["n_x_bins"], n_y_bins=cfg["n_y_bins"])
    adapter = ContinuousAdapter(base_adapter)
    adapter.reset()

    her_goal_indices = tuple(cfg.get("her_goal_indices", [4, 6]))

    agent = NeuralSRAgent(
        adapter=adapter,
        sf_dim=cfg["sf_dim"],
        hidden_sizes=cfg["hidden_sizes"],
        gamma=cfg["gamma"],
        lr=cfg["lr"],
        lr_w=cfg["lr_w"],
        batch_size=cfg["batch_size"],
        buffer_size=cfg["buffer_size"],
        target_update_freq=cfg["target_update_freq"],
        tau=cfg["tau"],
        epsilon_start=cfg["epsilon_end"],  # no exploration
        epsilon_end=cfg["epsilon_end"],
        epsilon_decay_steps=1,
        device=device,
        use_per=cfg.get("use_per", False),
        use_episodic_replay=cfg.get("use_episodic_replay", False),
        use_her=cfg.get("use_her", False),
        her_k=cfg.get("her_k", 4),
        her_goal_indices=her_goal_indices,
        train_every=cfg.get("train_every", 1),
    )
    agent.load(checkpoint_path)
    env.close()
    return agent, base_adapter


def compute_qvalue_grid(agent, base_adapter, goal_xy, grid_res=40):
    """Compute Q-values over a spatial grid.

    Args:
        agent: Trained NeuralSRAgent
        base_adapter: PointMazeAdapter (for wall info)
        goal_xy: (goal_x, goal_y) tuple
        grid_res: Number of grid points per axis

    Returns:
        x_coords, y_coords: 1D arrays of grid coordinates
        q_max: (grid_res, grid_res) max Q-value at each position
        best_action: (grid_res, grid_res) best action index
        wall_mask: (grid_res, grid_res) True where walls exist
    """
    x_range = base_adapter._x_range
    y_range = base_adapter._y_range

    x_coords = np.linspace(x_range[0] + 0.05, x_range[1] - 0.05, grid_res)
    y_coords = np.linspace(y_range[0] + 0.05, y_range[1] - 0.05, grid_res)

    q_max = np.full((grid_res, grid_res), np.nan)
    q_all = np.full((grid_res, grid_res, 8), np.nan)
    best_action = np.full((grid_res, grid_res), -1, dtype=int)
    wall_mask = np.zeros((grid_res, grid_res), dtype=bool)

    # Precompute wall check via maze_map
    import math
    maze_map = base_adapter._maze_map
    mx_center = base_adapter._maze_x_center
    my_center = base_adapter._maze_y_center

    gx, gy = goal_xy

    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            # Check if this position is a wall
            col = int(math.floor(x + mx_center))
            row = int(math.floor(my_center - y))
            col = max(0, min(col, len(maze_map[0]) - 1))
            row = max(0, min(row, len(maze_map) - 1))
            if maze_map[row][col] == 1:
                wall_mask[j, i] = True
                continue

            # Build observation: [x, y, vx, vy, goal_x, goal_y]
            obs = np.array([x, y, 0.0, 0.0, gx, gy], dtype=np.float32)
            qvals = agent.get_q_values(obs)
            q_all[j, i, :] = qvals
            q_max[j, i] = qvals.max()
            best_action[j, i] = qvals.argmax()

    return x_coords, y_coords, q_max, q_all, best_action, wall_mask


def plot_qvalue_landscape(x_coords, y_coords, q_max, best_action, wall_mask,
                          goal_xy, start_xy=None, title="Q-value Landscape",
                          save_path=None):
    """Plot Q-value heatmap with policy arrows."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Panel 1: Max Q-value heatmap ---
    ax = axes[0]
    q_display = np.ma.masked_where(wall_mask, q_max)

    im = ax.pcolormesh(x_coords, y_coords, q_display, cmap='viridis', shading='auto')
    # Overlay walls in gray
    wall_display = np.ma.masked_where(~wall_mask, np.ones_like(q_max))
    ax.pcolormesh(x_coords, y_coords, wall_display, cmap='gray_r', shading='auto',
                  vmin=0, vmax=2, alpha=0.7)

    ax.plot(*goal_xy, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=1.5,
            label='Goal')
    if start_xy:
        ax.plot(*start_xy, 'go', markersize=12, markeredgecolor='white',
                markeredgewidth=1.5, label='Start')

    fig.colorbar(im, ax=ax, label='max Q(s, a)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{title} — Max Q-value')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')

    # --- Panel 2: Policy arrows ---
    ax = axes[1]
    # Background: walls in gray
    wall_display = np.ma.masked_where(~wall_mask, np.ones_like(q_max))
    ax.pcolormesh(x_coords, y_coords, wall_display, cmap='gray_r', shading='auto',
                  vmin=0, vmax=2, alpha=0.7)

    # Draw policy arrows (subsample for readability)
    step = max(1, len(x_coords) // 20)
    for i in range(0, len(x_coords), step):
        for j in range(0, len(y_coords), step):
            if wall_mask[j, i] or best_action[j, i] < 0:
                continue
            a = best_action[j, i]
            dx, dy = ACTION_DIRS[a]
            scale = 0.08
            ax.arrow(x_coords[i], y_coords[j], dx * scale, dy * scale,
                     head_width=0.04, head_length=0.02,
                     fc='steelblue', ec='steelblue', alpha=0.8)

    ax.plot(*goal_xy, 'r*', markersize=20, markeredgecolor='white', markeredgewidth=1.5)
    if start_xy:
        ax.plot(*start_xy, 'go', markersize=12, markeredgecolor='white',
                markeredgewidth=1.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'{title} — Greedy Policy')
    ax.set_aspect('equal')
    ax.set_xlim(x_coords[0], x_coords[-1])
    ax.set_ylim(y_coords[0], y_coords[-1])

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def plot_q_comparison(results, goal_xy, save_path=None):
    """Side-by-side Q-value comparison of multiple checkpoints."""
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(7 * n, 12))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, (name, data) in enumerate(results.items()):
        x, y, q_max, best_action, wall_mask = (
            data['x'], data['y'], data['q_max'], data['best_action'], data['wall_mask']
        )

        # Row 0: Q-value heatmap
        ax = axes[0, col]
        q_display = np.ma.masked_where(wall_mask, q_max)
        im = ax.pcolormesh(x, y, q_display, cmap='viridis', shading='auto')
        wall_display = np.ma.masked_where(~wall_mask, np.ones_like(q_max))
        ax.pcolormesh(x, y, wall_display, cmap='gray_r', shading='auto',
                      vmin=0, vmax=2, alpha=0.7)
        ax.plot(*goal_xy, 'r*', markersize=15, markeredgecolor='white', markeredgewidth=1)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f'{name}\nMax Q-value')
        ax.set_aspect('equal')

        # Row 1: Policy arrows
        ax = axes[1, col]
        wall_display = np.ma.masked_where(~wall_mask, np.ones_like(q_max))
        ax.pcolormesh(x, y, wall_display, cmap='gray_r', shading='auto',
                      vmin=0, vmax=2, alpha=0.7)
        # Also show Q-values as faint background
        q_display = np.ma.masked_where(wall_mask, q_max)
        ax.pcolormesh(x, y, q_display, cmap='viridis', shading='auto', alpha=0.3)

        step = max(1, len(x) // 20)
        for i in range(0, len(x), step):
            for j in range(0, len(y), step):
                if wall_mask[j, i] or best_action[j, i] < 0:
                    continue
                a = best_action[j, i]
                dx, dy = ACTION_DIRS[a]
                scale = 0.08
                ax.arrow(x[i], y[j], dx * scale, dy * scale,
                         head_width=0.04, head_length=0.02,
                         fc='steelblue', ec='steelblue', alpha=0.8)
        ax.plot(*goal_xy, 'r*', markersize=15, markeredgecolor='white', markeredgewidth=1)
        ax.set_title(f'{name}\nGreedy Policy')
        ax.set_aspect('equal')
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(y[0], y[-1])

    plt.suptitle('Q-value Landscape Comparison — PointMaze UMaze', fontsize=14, y=1.01)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize Q-value landscape")
    parser.add_argument("--checkpoint", type=str, nargs='+',
                        default=None,
                        help="Checkpoint path(s). Default: all three experiments")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--grid-res", type=int, default=40,
                        help="Grid resolution per axis")
    args = parser.parse_args()

    # Default: compare all three experiments
    if args.checkpoint is None:
        checkpoints = {
            "Original\n(SF frozen)": "data/neural_point_maze/checkpoint.pt",
            "Exp A\n(joint fine-tune)": "data/neural_point_maze_exp_a/checkpoint.pt",
            "Exp B\n(no staging)": "data/neural_point_maze_exp_b/checkpoint.pt",
            "Exp C\n(more Phase 1)": "data/neural_point_maze_exp_c/checkpoint.pt",
        }
    else:
        checkpoints = {os.path.basename(c): c for c in args.checkpoint}

    # Filter to existing checkpoints
    checkpoints = {k: v for k, v in checkpoints.items() if os.path.exists(v)}
    if not checkpoints:
        print("No checkpoints found!")
        return

    print(f"Loading {len(checkpoints)} checkpoint(s)...")

    results = {}
    goal_xy = None

    for name, ckpt_path in checkpoints.items():
        print(f"\n  Loading: {ckpt_path}")
        agent, base_adapter = load_agent(ckpt_path, args.device)

        if goal_xy is None:
            # Use the goal from the adapter
            g = base_adapter._desired_goal
            if g is not None:
                goal_xy = (float(g[0]), float(g[1]))
            else:
                goal_xy = (-1.5, -0.5)  # default UMaze goal

        print(f"  Goal: ({goal_xy[0]:.2f}, {goal_xy[1]:.2f})")
        print(f"  Computing Q-values on {args.grid_res}x{args.grid_res} grid...")

        x, y, q_max, q_all, best_action, wall_mask = compute_qvalue_grid(
            agent, base_adapter, goal_xy, grid_res=args.grid_res
        )

        results[name] = {
            'x': x, 'y': y, 'q_max': q_max, 'q_all': q_all,
            'best_action': best_action, 'wall_mask': wall_mask,
        }

        # Print Q-value stats
        valid = ~wall_mask & ~np.isnan(q_max)
        print(f"  Q-value range: [{q_max[valid].min():.2f}, {q_max[valid].max():.2f}]")
        print(f"  Q-value mean: {q_max[valid].mean():.2f}")

    # Generate comparison plot
    save_path = "figures/neural_point_maze/qvalue_comparison.png"
    plot_q_comparison(results, goal_xy, save_path)

    # Also generate individual plots
    for name, data in results.items():
        safe_name = name.replace('\n', '_').replace(' ', '_').replace('(', '').replace(')', '')
        ind_path = f"figures/neural_point_maze/qvalue_{safe_name}.png"
        plot_qvalue_landscape(
            data['x'], data['y'], data['q_max'], data['best_action'], data['wall_mask'],
            goal_xy, title=name, save_path=ind_path,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
