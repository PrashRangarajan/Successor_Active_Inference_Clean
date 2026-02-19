"""Trajectory, policy, and video visualization methods.

This module provides plotting and video generation capabilities for
visualizing continuous-space trajectories, discrete grid trajectories,
macro action policies, and episode videos.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib import colors
from typing import Optional, List, Tuple


class TrajectoryVizMixin(object):
    """Mixin class providing trajectory, policy, and video visualization methods.

    Requires the agent to have:
    - self.adapter: Environment adapter
    - self.B, self.M: Transition and successor matrices
    - self.macro_state_list, self.micro_to_macro: Clustering results
    - self.state_history, self.action_history: Episode tracking
    - self.adj_list, self.bottleneck_states: Macro action network
    """

    # ==================== Continuous-Space Trajectory Visualization ========

    def _get_macro_action_target(self, s_idx: int) -> Optional[int]:
        """Get the target cluster for a micro state under the macro policy.

        Returns the cluster index the macro policy would navigate toward,
        or None if the state is in a goal cluster or has no adjacency.
        """
        if s_idx not in self.micro_to_macro:
            return None
        s_macro = self.micro_to_macro[s_idx]

        # Check if in goal cluster
        goal_macro_states = set()
        for gs in self.goal_states:
            if gs in self.micro_to_macro:
                goal_macro_states.add(self.micro_to_macro[gs])
        if s_macro in goal_macro_states:
            return None  # Goal cluster — no macro action needed

        if s_macro not in self.adj_list or not self.adj_list[s_macro]:
            return None

        V_macro = self.M_macro @ self.C_macro
        adj_states = self.adj_list[s_macro]
        values = [V_macro[adj] for adj in adj_states]
        sorted_idx = np.argsort(values)[::-1]
        for idx in sorted_idx:
            if adj_states[idx] != s_macro:
                return adj_states[idx]
        return None

    def plot_trajectory_with_macro_states(
        self,
        positions: List[float],
        velocities: List[float],
        save_path: str = None,
        color_by: str = 'macro_state',
        macro_action_targets: List = None,
    ):
        """Plot a 2D phase-space trajectory colored by macro state or macro action.

        Works for any 2D binned continuous environment (Mountain Car, Pendulum).

        Args:
            positions: Continuous dim-0 values (e.g. position / angle) per step.
            velocities: Continuous dim-1 values (e.g. velocity / angular vel) per step.
            save_path: Path to save the figure.
            color_by: 'macro_state' to color by cluster membership (default),
                      'macro_action' to color by target cluster under macro policy.
            macro_action_targets: Optional list of per-step macro action targets
                (cluster index or None for goal/micro phase).  When provided and
                color_by='macro_action', uses these instead of recomputing.
        """
        if save_path is None:
            raise ValueError("save_path is required (e.g. 'figures/mountaincar/trajectory_macro.png')")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if hasattr(self.adapter, 'get_dimension_labels'):
            dim0_label, dim1_label = self.adapter.get_dimension_labels()
        else:
            dim0_label, dim1_label = "Dim 0", "Dim 1"

        num_points = len(positions)
        tab_cmap = plt.get_cmap("tab10")
        macro_colors = [tab_cmap(i) for i in range(self.n_clusters)]
        goal_color = (1.0, 0.84, 0.0, 1.0)  # gold for goal cluster

        plt.figure(figsize=(6, 5))

        # Color each point
        for i in range(num_points):
            if color_by == 'macro_action':
                if macro_action_targets is not None and i < len(macro_action_targets):
                    target = macro_action_targets[i]
                else:
                    obs = np.array([positions[i], velocities[i]])
                    discrete = self.adapter.discretize_obs(obs)
                    s_idx = self.adapter.state_space.state_to_index(discrete)
                    target = self._get_macro_action_target(s_idx)
                if target is None:
                    c = goal_color  # In goal cluster / micro phase
                else:
                    c = macro_colors[target]
            else:
                obs = np.array([positions[i], velocities[i]])
                discrete = self.adapter.discretize_obs(obs)
                s_idx = self.adapter.state_space.state_to_index(discrete)
                macro = self.micro_to_macro.get(s_idx, 0)
                c = macro_colors[macro]

            plt.plot(positions[i], velocities[i],
                     marker="o", markersize=4, color=c)

        # Gray connecting line
        plt.plot(positions, velocities, linewidth=1, color="gray", alpha=0.5)

        # Legend entries
        if color_by == 'macro_action':
            plt.scatter([], [], color=goal_color, label='Goal',
                        marker="o", s=40, edgecolor='black')
            for i in range(self.n_clusters):
                plt.scatter([], [], color=macro_colors[i], label=f'→ {i}',
                            marker="o", s=40, edgecolor='black')
            title = "Trajectory with Macro Actions"
        else:
            for i in range(self.n_clusters):
                plt.scatter([], [], color=macro_colors[i], label=f'{i}',
                            marker="o", s=40, edgecolor='black')
            title = "Trajectory with Macro States"

        # Start / End markers
        plt.scatter(positions[0], velocities[0],
                    color='white', label="Start",
                    marker="o", s=60, edgecolor='black', linewidths=1.5, zorder=5)
        plt.scatter(positions[-1], velocities[-1],
                    color='white', label="End",
                    marker="s", s=60, edgecolor='black', linewidths=1.5, zorder=5)

        # Set axes to full state space extent (not just visited trajectory)
        if hasattr(self.adapter, 'get_bin_edges'):
            edges0, edges1 = self.adapter.get_bin_edges()
            plt.xlim(edges0[0], edges0[-1])
            plt.ylim(edges1[0], edges1[-1])

        plt.xlabel(dim0_label)
        plt.ylabel(dim1_label)
        # plt.title(title)
        plt.legend(loc="best", fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved {color_by} trajectory to {save_path}")

    def plot_trajectory_with_actions(
        self,
        positions: List[float],
        velocities: List[float],
        actions: List[int],
        save_path: str = None,
    ):
        """Plot phase-space trajectory colored by action taken at each step.

        Args:
            positions: List of position values along the trajectory.
            velocities: List of velocity values along the trajectory.
            actions: List of action indices (one per decision step).
            save_path: Output file path (e.g. 'figures/mountaincar/trajectory_actions.png').
        """
        if save_path is None:
            raise ValueError("save_path is required (e.g. 'figures/mountaincar/trajectory_actions.png')")
        import matplotlib.colors as mcolors

        # Get axis labels from adapter
        if hasattr(self.adapter, 'get_dimension_labels'):
            dim0_label, dim1_label = self.adapter.get_dimension_labels()
        else:
            dim0_label, dim1_label = "Dim 0", "Dim 1"

        # Get action labels from adapter (or fall back to numeric)
        if hasattr(self.adapter, 'get_action_labels'):
            action_labels = self.adapter.get_action_labels()
        else:
            n_act = self.adapter.n_actions
            action_labels = [str(i) for i in range(n_act)]

        n_actions = len(action_labels)

        # Trim to matched lengths (actions has one fewer entry than positions)
        n = min(len(positions), len(velocities), len(actions))
        pos = np.asarray(positions[:n], dtype=float)
        vel = np.asarray(velocities[:n], dtype=float)
        act = np.asarray(actions[:n], dtype=int)

        # Discrete colormap: one distinct colour per action
        base_cmap = plt.get_cmap("tab10")
        colors = [base_cmap(i) for i in range(n_actions)]
        cmap = mcolors.ListedColormap(colors[:n_actions])
        bounds = np.arange(-0.5, n_actions, 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        plt.figure(figsize=(8, 6))

        # Gray connecting line
        plt.plot(pos, vel, linewidth=0.8, color="gray", alpha=0.4)

        # Scatter colored by action
        scatter = plt.scatter(pos, vel, c=act, cmap=cmap, norm=norm, s=18,
                              edgecolors='none')

        # Legend with action labels
        handles = [
            plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=colors[i], markersize=8,
                       label=action_labels[i])
            for i in range(n_actions)
        ]
        plt.legend(handles=handles, title="Action", loc="best")

        # Start / End markers
        plt.scatter(pos[0], vel[0], s=80, marker='o', facecolors='none',
                    edgecolors='black', linewidths=2, label='Start', zorder=5)
        plt.scatter(pos[-1], vel[-1], s=80, marker='s', facecolors='none',
                    edgecolors='black', linewidths=2, label='End', zorder=5)

        # Set axes to full state space extent (not just visited trajectory)
        if hasattr(self.adapter, 'get_bin_edges'):
            edges0, edges1 = self.adapter.get_bin_edges()
            plt.xlim(edges0[0], edges0[-1])
            plt.ylim(edges1[0], edges1[-1])

        plt.xlabel(dim0_label, fontsize=12)
        plt.ylabel(dim1_label, fontsize=12)
        # plt.title("Trajectory with Actions")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved action trajectory to {save_path}")

    def plot_macro_action_heatmap(self, save_path: str = None):
        """Plot a 2D heatmap showing the macro action (target cluster) at each state.

        For every micro state, determines the best macro action (which adjacent
        cluster to navigate toward) based on the macro-level value function.
        States within the goal cluster are marked separately.

        Works for 2D binned continuous environments (Mountain Car, Pendulum).

        Args:
            save_path: Path to save the figure (e.g. 'figures/pendulum/macro_actions.png').
        """
        if save_path is None:
            raise ValueError("save_path is required")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if not hasattr(self.adapter, 'state_space') or \
           not hasattr(self.adapter.state_space, 'n_bins_per_dim'):
            print("  Macro action heatmap requires 2D binned environment")
            return

        bins = self.adapter.state_space.n_bins_per_dim
        if len(bins) != 2:
            print("  Macro action heatmap only supports 2D state spaces")
            return

        # Get axis labels and bin edges
        if hasattr(self.adapter, 'get_dimension_labels'):
            dim0_label, dim1_label = self.adapter.get_dimension_labels()
        else:
            dim0_label, dim1_label = "Dim 0", "Dim 1"

        has_edges = hasattr(self.adapter, 'get_bin_edges')
        if has_edges:
            edges0, edges1 = self.adapter.get_bin_edges()
            extent = [edges0[0], edges0[-1], edges1[0], edges1[-1]]
        else:
            extent = None

        has_centers = hasattr(self.adapter, 'get_bin_centers')
        if has_centers:
            centers0, centers1 = self.adapter.get_bin_centers()

        # Compute macro-level values
        V_macro = self.M_macro @ self.C_macro

        # Determine goal macro state(s)
        goal_macro_states = set()
        for gs in self.goal_states:
            if gs in self.micro_to_macro:
                goal_macro_states.add(self.micro_to_macro[gs])

        # Build the macro action grid
        # Values: target cluster index for non-goal states, -1 for unassigned,
        #         -2 for goal cluster states
        action_grid = np.full((bins[0], bins[1]), -1, dtype=int)

        for s_idx in range(self.adapter.n_states):
            if s_idx not in self.micro_to_macro:
                continue
            s_macro = self.micro_to_macro[s_idx]
            state_tuple = self.adapter.state_space.index_to_state(s_idx)

            if s_macro in goal_macro_states:
                action_grid[state_tuple] = -2  # Goal cluster
            elif s_macro in self.adj_list and self.adj_list[s_macro]:
                # Find best adjacent macro state
                adj_states = self.adj_list[s_macro]
                values = [V_macro[adj] for adj in adj_states]
                best_idx = int(np.argmax(values))
                # Skip self-loops
                if adj_states[best_idx] != s_macro:
                    action_grid[state_tuple] = adj_states[best_idx]
                elif len(adj_states) > 1:
                    # Pick second best
                    sorted_idx = np.argsort(values)[::-1]
                    for idx in sorted_idx:
                        if adj_states[idx] != s_macro:
                            action_grid[state_tuple] = adj_states[idx]
                            break

        # Build colormap: one color per cluster + goal + unassigned
        import matplotlib.colors as mcolors
        tab_cmap = plt.get_cmap("tab10")
        # Cluster colors for targets 0..n_clusters-1
        cluster_colors = [tab_cmap(i) for i in range(self.n_clusters)]
        # Goal color (gold) and unassigned (light gray)
        goal_color = (1.0, 0.84, 0.0, 1.0)  # gold
        unassigned_color = (0.85, 0.85, 0.85, 1.0)

        # Map action_grid values to a plottable integer grid
        # -1 → 0 (unassigned), -2 → 1 (goal), cluster_i → i+2
        plot_grid = np.zeros_like(action_grid, dtype=int)
        for i in range(bins[0]):
            for j in range(bins[1]):
                v = action_grid[i, j]
                if v == -1:
                    plot_grid[i, j] = 0
                elif v == -2:
                    plot_grid[i, j] = 1
                else:
                    plot_grid[i, j] = v + 2

        n_categories = self.n_clusters + 2  # unassigned + goal + clusters
        color_list = [unassigned_color, goal_color] + cluster_colors
        cmap = mcolors.ListedColormap(color_list[:n_categories])
        bounds = np.arange(-0.5, n_categories, 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            plot_grid.T,
            aspect='auto',
            origin='lower',
            extent=extent,
            interpolation='nearest',
            cmap=cmap,
            norm=norm,
        )

        # Subsampled tick labels
        if has_centers:
            max_ticks = 7
            tick_pos0 = np.linspace(
                extent[0] if extent else 0,
                extent[1] if extent else bins[0] - 1,
                bins[0],
            )
            tick_pos1 = np.linspace(
                extent[2] if extent else 0,
                extent[3] if extent else bins[1] - 1,
                bins[1],
            )
            step0 = max(1, len(tick_pos0) // max_ticks)
            step1 = max(1, len(tick_pos1) // max_ticks)
            idx0 = np.arange(0, len(tick_pos0), step0)
            idx1 = np.arange(0, len(tick_pos1), step1)
            ax.set_xticks(tick_pos0[idx0])
            ax.set_xticklabels(np.round(centers0[idx0], 2))
            ax.set_yticks(tick_pos1[idx1])
            ax.set_yticklabels(np.round(centers1[idx1], 2))

        ax.set_xlabel(dim0_label, fontsize=12)
        ax.set_ylabel(dim1_label, fontsize=12)
        # ax.set_title("Macro Action Policy (target cluster)", fontsize=14)

        # Legend
        handles = [mpatches.Patch(color=goal_color, label='Goal')]
        for i in range(self.n_clusters):
            handles.append(mpatches.Patch(color=cluster_colors[i],
                                          label=f'→ {i}'))
        ax.legend(handles=handles, loc='best', fontsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved macro action heatmap to {save_path}")

    def plot_stage_state_diagram(
        self,
        frames: List[np.ndarray],
        positions: List[float],
        velocities: List[float],
        stage_idx: Optional[List[int]] = None,
        save_path: str = None,
        annotate_state: bool = True,
    ):
        """Create a composite figure linking environment snapshots to phase-space states.

        Top row:  rendered frames at selected stages labelled (a), (b), (c), ...
        Bottom:   phase plot (position vs velocity) with selected stages highlighted.

        Args:
            frames: RGB frames from env.render(), one per timestep.
            positions: Continuous dim-0 value per timestep.
            velocities: Continuous dim-1 value per timestep.
            stage_idx: Indices into the trajectory to snapshot.
                       If None, auto-selects [start, valley/extremum, goal/end].
            save_path: Output PNG path (e.g. 'figures/mountaincar/stage_diagram.png').
            annotate_state: If True, overlay coordinate text on each snapshot.
        """
        if save_path is None:
            raise ValueError("save_path is required (e.g. 'figures/mountaincar/stage_diagram.png')")
        from matplotlib.gridspec import GridSpec

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if hasattr(self.adapter, 'get_dimension_labels'):
            dim0_label, dim1_label = self.adapter.get_dimension_labels()
        else:
            dim0_label, dim1_label = "Dim 0", "Dim 1"

        # Extract short annotation symbols from dimension labels.
        # e.g. "Angle (θ)" → "θ", "Position" → "x", "Angular Velocity (ω)" → "ω"
        import re
        def _short_symbol(label: str, fallback: str) -> str:
            m = re.search(r'\(([^)]+)\)', label)
            return m.group(1) if m else fallback

        sym0 = _short_symbol(dim0_label, "x")
        sym1 = _short_symbol(dim1_label, "v")

        T = len(positions)
        pos = np.asarray(positions, dtype=float)
        vel = np.asarray(velocities, dtype=float)

        # Auto-select stages: start, key moments, end
        if stage_idx is None:
            stage_idx = [0]

            # 1. Find an interesting mid-trajectory moment:
            #    - extremum of dim-0 (valley for Mountain Car)
            #    - or peak |velocity| (max swing speed for Pendulum)
            extremum_idx = int(np.argmin(pos))
            if extremum_idx not in (0, T - 1):
                stage_idx.append(extremum_idx)
            else:
                # dim-0 min is at start/end — use peak |velocity| instead
                peak_vel_idx = int(np.argmax(np.abs(vel)))
                if peak_vel_idx not in (0, T - 1):
                    stage_idx.append(peak_vel_idx)

            # 2. Find first time the trajectory enters the goal region
            #    (e.g., first upright moment for pendulum, reaching x≥0.5 for
            #    Mountain Car).  Uses the adapter's goal bins when available.
            goal_arrival = None
            if hasattr(self.adapter, 'discretize_obs') and hasattr(self, 'goal_states'):
                for k in range(T):
                    obs_k = np.array([pos[k], vel[k]])
                    try:
                        disc = self.adapter.discretize_obs(obs_k)
                        s_idx = self.adapter.state_space.state_to_index(disc)
                        if s_idx in self.goal_states:
                            goal_arrival = k
                            break
                    except Exception:
                        pass

            if goal_arrival is not None and goal_arrival not in stage_idx and goal_arrival != T - 1:
                stage_idx.append(goal_arrival)

            stage_idx.append(T - 1)
        stage_idx = sorted(set(int(i) for i in stage_idx if 0 <= int(i) < T))
        if len(stage_idx) == 0:
            stage_idx = [0]

        n = len(stage_idx)

        fig = plt.figure(figsize=(4 * n, 7.5))
        gs = GridSpec(2, n, height_ratios=[1, 1.8], hspace=0.25, wspace=0.05)

        # ---- Snapshot frames (top row) ----
        for j, idx in enumerate(stage_idx):
            ax_img = fig.add_subplot(gs[0, j])
            if idx < len(frames) and frames[idx] is not None:
                ax_img.imshow(frames[idx])
            ax_img.axis("off")
            ax_img.set_title(f"({chr(97 + j)}) t={idx}", fontsize=14)

            if annotate_state:
                ax_img.text(
                    0.02, 0.95,
                    f"{sym0}={pos[idx]:.3f}\n{sym1}={vel[idx]:.3f}",
                    transform=ax_img.transAxes, fontsize=12, va="top",
                    bbox=dict(boxstyle="round", alpha=0.6),
                )

        # ---- Phase plot (bottom, centered, ~50% figure width) ----
        # Use explicit figure-level positioning so the plot is truly centered
        phase_frac = 0.45  # fraction of figure width
        phase_left = (1.0 - phase_frac) / 2
        # Bottom row occupies roughly the lower 55% of the figure
        phase_bottom = 0.06
        phase_height = 0.42
        ax_phase = fig.add_axes([phase_left, phase_bottom, phase_frac, phase_height])
        t = np.arange(T)
        ax_phase.scatter(pos, vel, c=t, s=18, cmap="plasma")
        ax_phase.plot(pos, vel, linewidth=1, color="gray", alpha=0.4)

        # Set axes to full state space extent
        if hasattr(self.adapter, 'get_bin_edges'):
            edges0, edges1 = self.adapter.get_bin_edges()
            ax_phase.set_xlim(edges0[0], edges0[-1])
            ax_phase.set_ylim(edges1[0], edges1[-1])

        ax_phase.set_xlabel(dim0_label)
        ax_phase.set_ylabel(dim1_label)
        # ax_phase.set_title("Trajectory in state space")

        # Highlight chosen stages
        xlim = ax_phase.get_xlim()
        x_mid = (xlim[0] + xlim[1]) / 2
        for j, idx in enumerate(stage_idx):
            ax_phase.scatter(pos[idx], vel[idx], s=160, facecolors="none",
                             linewidths=3, edgecolors="black")
            # Place label on the side with more room
            label = f"({chr(97 + j)})"
            if pos[idx] > x_mid:
                ax_phase.text(pos[idx], vel[idx], f"{label}  ",
                              fontsize=14, va="center", ha="right")
            else:
                ax_phase.text(pos[idx], vel[idx], f"  {label}",
                              fontsize=14, va="center", ha="left")

        # Start / end markers
        ax_phase.scatter(pos[0], vel[0], s=70, marker="o", edgecolors="black")
        ax_phase.scatter(pos[-1], vel[-1], s=70, marker="o", edgecolors="black")
        ax_phase.grid(True, alpha=0.3)

        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved stage diagram to {save_path}")
        return save_path

    def generate_combined_video(
        self,
        frames: List[np.ndarray],
        positions: List[float],
        velocities: List[float],
        save_path: str = None,
        fps: int = 30,
        color_by: str = 'macro_action',
        macro_action_targets: List = None,
    ):
        """Generate a vertically stacked video: environment render (top) + animated trajectory (bottom).

        The bottom panel shows a phase-space trajectory that grows over time.

        Args:
            frames: RGB frames from env.render(), one per decision step.
            positions: Continuous dim-0 values per decision step.
            velocities: Continuous dim-1 values per decision step.
            save_path: Output MP4 path (e.g. 'figures/mountaincar/combined_vertical.mp4').
            fps: Frames per second for the output video.
            color_by: 'macro_state' to color by cluster membership,
                      'macro_action' to color by target cluster under macro policy (default).
            macro_action_targets: Optional list of per-step macro action targets
                (cluster index or None for goal/micro phase).
        """
        if save_path is None:
            raise ValueError("save_path is required (e.g. 'figures/mountaincar/combined_vertical.mp4')")
        from matplotlib.collections import LineCollection
        from PIL import Image as PILImage
        from io import BytesIO
        import imageio

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if hasattr(self.adapter, 'get_dimension_labels'):
            dim0_label, dim1_label = self.adapter.get_dimension_labels()
        else:
            dim0_label, dim1_label = "Dim 0", "Dim 1"

        pos = np.asarray(positions, dtype=float)
        vel = np.asarray(velocities, dtype=float)
        T = len(pos)

        # Compute color per point
        tab_cmap = plt.get_cmap("tab10")
        macro_colors = [tab_cmap(i) for i in range(self.n_clusters)]
        goal_color = (1.0, 0.84, 0.0, 1.0)  # gold

        point_color_indices = []  # int cluster index, or -1 for goal
        for i in range(T):
            if color_by == 'macro_action' and macro_action_targets is not None and i < len(macro_action_targets):
                target = macro_action_targets[i]
                point_color_indices.append(target if target is not None else -1)
            elif color_by == 'macro_action':
                obs = np.array([pos[i], vel[i]])
                discrete = self.adapter.discretize_obs(obs)
                s_idx = self.adapter.state_space.state_to_index(discrete)
                target = self._get_macro_action_target(s_idx)
                point_color_indices.append(target if target is not None else -1)
            else:
                obs = np.array([pos[i], vel[i]])
                discrete = self.adapter.discretize_obs(obs)
                s_idx = self.adapter.state_space.state_to_index(discrete)
                point_color_indices.append(self.micro_to_macro.get(s_idx, 0))

        def _idx_to_color(idx):
            if idx == -1:
                return goal_color
            return macro_colors[idx]

        # Build segments and per-segment colours
        points = np.column_stack([pos, vel])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        seg_colors = [_idx_to_color(point_color_indices[i]) for i in range(T - 1)]

        # Determine axis limits with padding
        x_pad = 0.05 * (pos.max() - pos.min() + 1e-6)
        y_pad = 0.05 * (vel.max() - vel.min() + 1e-6)
        xlim = (pos.min() - x_pad, pos.max() + x_pad)
        ylim = (vel.min() - y_pad, vel.max() + y_pad)

        # Use bin edges for limits if available
        if hasattr(self.adapter, 'get_bin_edges'):
            edges0, edges1 = self.adapter.get_bin_edges()
            xlim = (edges0[0], edges0[-1])
            ylim = (edges1[0], edges1[-1])

        # Match frames length to trajectory length (one frame per decision step)
        n_frames = min(len(frames), T)

        # Determine target width from the environment frame
        env_h, env_w = frames[0].shape[:2]
        fig_w_inches = 6.0
        fig_h_inches = fig_w_inches * (env_h / env_w)  # preserve aspect
        dpi = int(np.ceil(env_w / fig_w_inches))

        combined_frames = []
        for t in range(n_frames):
            # Create trajectory plot for this timestep
            fig, ax = plt.subplots(figsize=(fig_w_inches, fig_h_inches))
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(dim0_label, fontsize=10)
            ax.set_ylabel(dim1_label, fontsize=10)
            # if color_by == 'macro_action':
            #     ax.set_title("Trajectory with macro actions", fontsize=12)
            # else:
            #     ax.set_title("Trajectory with macro states", fontsize=12)

            # Add legend for colors seen so far
            seen = set()
            for ci in point_color_indices[:t + 1]:
                seen.add(ci)
            if color_by == 'macro_action':
                if -1 in seen:
                    ax.scatter([], [], color=goal_color, label='Goal',
                               marker='o', s=30, edgecolor='black')
                for m in sorted(c for c in seen if c >= 0):
                    ax.scatter([], [], color=macro_colors[m], label=f'→ {m}',
                               marker='o', s=30, edgecolor='black')
            else:
                for m in sorted(c for c in seen if c >= 0):
                    ax.scatter([], [], color=macro_colors[m], label=f'{m}',
                               marker='o', s=30, edgecolor='black')
            ax.legend(loc='upper right', fontsize=8)

            # Draw trajectory segments up to current time
            if t > 0:
                lc = LineCollection(segments[:t], colors=seg_colors[:t], linewidths=2)
                ax.add_collection(lc)

            # Current position marker
            ax.plot(pos[t], vel[t], 'ro', markersize=5)

            fig.tight_layout()

            # Render figure to numpy array via savefig (portable across backends)
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                        pad_inches=0.1)
            buf.seek(0)
            plot_pil = PILImage.open(buf).convert('RGB')
            plt.close(fig)

            # Resize plot image to match environment frame width
            new_h = int(plot_pil.height * env_w / plot_pil.width)
            plot_pil = plot_pil.resize((env_w, new_h), PILImage.LANCZOS)
            plot_arr = np.array(plot_pil)

            # Stack vertically: environment on top, trajectory on bottom
            env_frame = frames[t]
            if env_frame.shape[1] != env_w:
                env_pil = PILImage.fromarray(env_frame)
                env_pil = env_pil.resize(
                    (env_w, int(env_frame.shape[0] * env_w / env_frame.shape[1])),
                    PILImage.LANCZOS)
                env_frame = np.array(env_pil)

            combined = np.vstack([env_frame, plot_arr])

            # Ensure dimensions are even (required by libx264)
            h, w = combined.shape[:2]
            h = h - (h % 2)
            w = w - (w % 2)
            combined = combined[:h, :w]

            combined_frames.append(combined)

        if combined_frames:
            imageio.mimsave(save_path, combined_frames, fps=fps, macro_block_size=1)
            print(f"  Saved combined video to {save_path} "
                  f"({len(combined_frames)} frames, "
                  f"{combined_frames[0].shape[1]}x{combined_frames[0].shape[0]})")
        else:
            print("  No frames to combine")

    # ==================== Trajectory Visualization ====================

    def show_actions(self, save_path: str = None,
                     init_loc: Tuple[int, int] = None,
                     goal_loc: Tuple[int, int] = None):
        """Visualize actions taken during an episode.

        Args:
            save_path: Path to save the figure (e.g. 'figures/gridworld/Actions_taken.png')
            init_loc: Starting location (default: (0,0))
            goal_loc: Goal location (default: bottom-right corner)
        """
        if save_path is None:
            raise ValueError("save_path is required (e.g. 'figures/gridworld/Actions_taken.png')")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if not hasattr(self.adapter, 'grid_size'):
            print("Action visualization requires grid-based environment")
            return

        if not self.state_history or not self.action_history:
            print("No episode history to visualize")
            return

        grid_size = self.adapter.grid_size

        # Get locations
        if init_loc is None:
            init_loc = (0, 0)
        if goal_loc is None:
            goal_loc = (grid_size - 1, grid_size - 1)

        # Get wall locations using render_state for correct (x, y) coordinates
        walls = self.adapter.get_wall_indices() if hasattr(self.adapter, 'get_wall_indices') else []
        wall_locs = set()
        for w in walls:
            loc = self.adapter.render_state(w)
            wall_locs.add((loc[0], loc[1]))  # take only (x, y), ignore augmented dims

        # Create grid
        grid = np.zeros((grid_size, grid_size))
        grid[init_loc] = 1
        for w in wall_locs:
            grid[w] = 2
        grid[goal_loc] = 0.5

        # Build arrows grid
        arrows = {1: (1, 0), 0: (-1, 0), 3: (0, 1), 2: (0, -1)}
        scale = 0.25
        arrows_grid = np.full((grid_size, grid_size), -1)

        # Get state locations from history
        for i, (state, action) in enumerate(zip(self.state_history[1:], self.action_history)):
            if hasattr(self.adapter, 'onehot_to_index'):
                idx = self.adapter.onehot_to_index(state)
            else:
                idx = np.argmax(state.flatten())

            loc = self.adapter.render_state(idx)
            arrows_grid[loc[0], loc[1]] = action

        arrows_grid = arrows_grid.T

        # Plot
        fig, ax = plt.subplots(figsize=(grid_size, grid_size))

        # Draw arrows
        for r, row in enumerate(arrows_grid):
            for c, cell in enumerate(row):
                if cell in arrows:
                    plt.arrow(c - scale * arrows[cell][0],
                             r - scale * arrows[cell][1],
                             scale * arrows[cell][0],
                             scale * arrows[cell][1],
                             head_width=0.15, color='w')

        cmap = colors.ListedColormap(['black', 'purple', 'yellow', 'white'])
        plt.imshow(grid.T, aspect='equal', cmap=cmap)

        ax.text(goal_loc[0], goal_loc[1], 'Goal', fontsize=18,
                ha="center", va="center", color="w")
        ax.text(init_loc[0], init_loc[1], 'Agent', fontsize=18,
                ha="center", va="center", color="b")

        # Grid setup
        ax.set_xticks(np.arange(0, grid_size, 1))
        ax.set_yticks(np.arange(0, grid_size, 1))
        ax.set_xticklabels(range(0, grid_size))
        ax.set_yticklabels(range(0, grid_size))
        ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        ax.tick_params(which='minor', bottom=False, left=False)

        plt.savefig(save_path, format="png", bbox_inches='tight')
        plt.close()

    # ==================== Policy Visualization ====================

    def visualize_policy(self, save_dir: str = None):
        """Visualize macro action policies.

        For each macro state transition, shows the micro-level actions
        that would be taken to reach the bottleneck state.

        For grid environments: arrow plots on the grid.
        For 2D binned continuous environments (Mountain Car, Pendulum):
        heatmaps showing best micro action at each (dim0, dim1) cell.

        Args:
            save_dir: Directory to save figures (e.g. 'figures/gridworld/macro_action_network')
        """
        if save_dir is None:
            raise ValueError("save_dir is required (e.g. 'figures/gridworld/macro_action_network')")
        os.makedirs(save_dir, exist_ok=True)

        # Route to the appropriate method based on environment type
        if hasattr(self.adapter, 'state_space') and \
           hasattr(self.adapter.state_space, 'n_bins_per_dim') and \
           not hasattr(self.adapter, 'grid_size'):
            self._visualize_policy_binned_2d(save_dir)
            return

        if not hasattr(self.adapter, 'grid_size'):
            print("Policy visualization requires grid-based or 2D binned environment")
            return

        # Clear old images
        for filename in glob.glob(f"{save_dir}/Macro_Action_Network_*"):
            os.remove(filename)

        grid_size = self.adapter.grid_size

        # Check if augmented state space
        base_n_states = grid_size * grid_size
        is_augmented = self.adapter.n_states != base_n_states
        n_augment = self.adapter.n_states // base_n_states if is_augmented else 1

        for macro_state in range(self.n_clusters):
            if macro_state not in self.adj_list:
                continue

            for macro_action, macro_final_state in enumerate(self.adj_list[macro_state]):
                bottleneck = self.bottleneck_states.get((macro_state, macro_final_state), [])
                if not bottleneck:
                    continue

                final_state = bottleneck[0]

                # Create goal at bottleneck
                C_temp = self.adapter.create_goal_prior([final_state], reward=1.0, default_cost=0.0)
                V = self.adapter.multiply_M_C(self.M, C_temp)

                print(f'\nMacro transition: {macro_state} -> {macro_final_state}')
                print(f'Bottleneck state: {final_state}')

                if is_augmented:
                    # Create separate arrows grids for each augment value
                    arrows_grids = [np.full((grid_size, grid_size), -1) for _ in range(n_augment)]

                    for state in self.macro_state_list[macro_state]:
                        state_loc = self.adapter.state_space.index_to_state(state)
                        # state_loc is (base_idx, aug_idx) for augmented spaces
                        if len(state_loc) == 2:
                            base_idx, aug_idx = state_loc
                            # Convert base_idx to (x, y)
                            x, y = divmod(base_idx, grid_size)

                            # Get state one-hot
                            s_onehot = self.adapter.index_to_onehot(state)

                            # Compute values for each action
                            V_adj = []
                            for act in range(self.adapter.n_actions):
                                s_next = self.adapter.multiply_B_s(self.B, s_onehot, act)
                                next_idx = self.adapter.onehot_to_index(s_next)
                                V_adj.append(V[next_idx])

                            best_action = np.argmax(V_adj)
                            arrows_grids[aug_idx][x, y] = best_action

                    # Transpose all grids
                    arrows_grids = [g.T for g in arrows_grids]

                    # Plot with multiple panels
                    self._plot_policy_arrows_augmented(
                        arrows_grids, grid_size, n_augment,
                        f"{save_dir}/Macro_Action_Network_{macro_state}_{macro_final_state}.png",
                        f"Macro Action Policy: {macro_state} -> {macro_final_state}")
                else:
                    # Standard state space
                    arrows_grid = np.full((grid_size, grid_size), -1)

                    for state in self.macro_state_list[macro_state]:
                        state_loc = self.adapter.state_space.index_to_state(state)
                        if len(state_loc) != 2:
                            continue

                        # Get state one-hot
                        s_onehot = self.adapter.index_to_onehot(state)

                        # Compute values for each action
                        V_adj = []
                        for act in range(self.adapter.n_actions):
                            s_next = self.adapter.multiply_B_s(self.B, s_onehot, act)
                            next_idx = self.adapter.onehot_to_index(s_next)
                            V_adj.append(V[next_idx])

                        best_action = np.argmax(V_adj)
                        arrows_grid[state_loc] = best_action

                    arrows_grid = arrows_grid.T

                    # Plot
                    self._plot_policy_arrows(arrows_grid, grid_size,
                                            f"{save_dir}/Macro_Action_Network_{macro_state}_{macro_final_state}.png",
                                            f"Macro Action Policy: {macro_state} -> {macro_final_state}")

    def _visualize_policy_binned_2d(self, save_dir: str):
        """Visualize macro action micro-policies for 2D binned environments.

        For each macro transition (cluster A → cluster B), creates a heatmap
        showing the best micro action at each (dim0, dim1) cell when the agent
        is navigating toward the bottleneck of that transition.  Only cells
        belonging to the source cluster are coloured; other cells are grayed out.
        """
        import matplotlib.colors as mcolors
        import matplotlib.patheffects as PathEffects

        bins = self.adapter.state_space.n_bins_per_dim
        if len(bins) != 2:
            print("  Policy visualization only supports 2D binned state spaces")
            return

        # Clear old images
        for filename in glob.glob(f"{save_dir}/Macro_Action_Network_*"):
            os.remove(filename)

        # Axis labels and extent
        if hasattr(self.adapter, 'get_dimension_labels'):
            dim0_label, dim1_label = self.adapter.get_dimension_labels()
        else:
            dim0_label, dim1_label = "Dim 0", "Dim 1"

        has_edges = hasattr(self.adapter, 'get_bin_edges')
        if has_edges:
            edges0, edges1 = self.adapter.get_bin_edges()
            extent = [edges0[0], edges0[-1], edges1[0], edges1[-1]]
        else:
            extent = None

        has_centers = hasattr(self.adapter, 'get_bin_centers')
        if has_centers:
            centers0, centers1 = self.adapter.get_bin_centers()

        # Action labels and colours
        if hasattr(self.adapter, 'get_action_labels'):
            action_labels = self.adapter.get_action_labels()
        else:
            action_labels = [str(i) for i in range(self.adapter.n_actions)]
        n_actions = self.adapter.n_actions

        # Build cluster background heatmap (same colouring as Macro_s.png)
        tab_cmap = plt.get_cmap("tab10")
        cluster_colors = [tab_cmap(i) for i in range(self.n_clusters)]
        unassigned_color = (0.85, 0.85, 0.85, 1.0)

        labels = np.ones(self.adapter.n_states, dtype=int) * self.n_clusters
        for micro_idx, macro_idx in self.micro_to_macro.items():
            labels[micro_idx] = macro_idx
        labels_grid = labels.reshape(bins[0], bins[1])

        bg_colors = cluster_colors + [unassigned_color]
        bg_cmap = mcolors.ListedColormap(bg_colors)
        bg_bounds = np.arange(-0.5, self.n_clusters + 1.5, 1)
        bg_norm = mcolors.BoundaryNorm(bg_bounds, bg_cmap.N)

        # Arrow symbols mapped by action index.
        # Mountain Car: 0=left, 1=no-op, 2=right
        # Generic fallback uses horizontal arrows for first/last, dot for middle.
        arrow_dx = {}
        noop_actions = set()
        for i in range(n_actions):
            label_lower = action_labels[i].lower()
            if 'left' in label_lower:
                arrow_dx[i] = (-1, 0)
            elif 'right' in label_lower:
                arrow_dx[i] = (1, 0)
            elif 'up' in label_lower:
                arrow_dx[i] = (0, 1)
            elif 'down' in label_lower:
                arrow_dx[i] = (0, -1)
            else:
                noop_actions.add(i)

        # Compute pixel-centre positions that exactly match imshow cell centres.
        # imshow with extent places N pixels so that pixel i is centred at
        #   extent_lo + (i + 0.5) * cell_width
        if extent is not None:
            dx_cell = (extent[1] - extent[0]) / bins[0]
            dy_cell = (extent[3] - extent[2]) / bins[1]
            pix_x = np.array([extent[0] + (i + 0.5) * dx_cell for i in range(bins[0])])
            pix_y = np.array([extent[2] + (j + 0.5) * dy_cell for j in range(bins[1])])
        else:
            dx_cell = 1.0
            dy_cell = 1.0
            pix_x = np.arange(bins[0], dtype=float)
            pix_y = np.arange(bins[1], dtype=float)
        arrow_scale = 0.3  # fraction of cell size

        for macro_state in range(self.n_clusters):
            if macro_state not in self.adj_list:
                continue

            for macro_final_state in self.adj_list[macro_state]:
                bottleneck = self.bottleneck_states.get(
                    (macro_state, macro_final_state), [])
                if not bottleneck:
                    bottleneck = self.macro_state_list[macro_final_state]
                if not bottleneck:
                    continue

                # Create temporary value function toward bottleneck
                C_temp = self.adapter.create_goal_prior(
                    bottleneck, reward=10.0, default_cost=0.0)
                V = self.adapter.multiply_M_C(self.M, C_temp)

                # Compute best action for each source cluster member
                action_grid = np.full((bins[0], bins[1]), -1, dtype=int)
                source_members = set(self.macro_state_list[macro_state])

                for s_idx in source_members:
                    state_tuple = self.adapter.state_space.index_to_state(s_idx)
                    s_onehot = self.adapter.index_to_onehot(s_idx)
                    V_adj = []
                    for act in range(n_actions):
                        s_next = self.adapter.multiply_B_s(self.B, s_onehot, act)
                        next_idx = self.adapter.onehot_to_index(s_next)
                        V_adj.append(V[next_idx])
                    action_grid[state_tuple] = int(np.argmax(V_adj))

                # ---- Plot: cluster background + arrow/symbol overlay ----
                fig, ax = plt.subplots(figsize=(7, 5.5))
                ax.imshow(
                    labels_grid.T,
                    aspect='auto', origin='lower', extent=extent,
                    interpolation='nearest',
                    cmap=bg_cmap, norm=bg_norm, alpha=0.5,
                )

                # Draw arrows / symbols on source cluster cells
                for i in range(bins[0]):
                    for j in range(bins[1]):
                        act = action_grid[i, j]
                        if act == -1:
                            continue
                        cx, cy = pix_x[i], pix_y[j]

                        if act in arrow_dx:
                            adx, ady = arrow_dx[act]
                            ax.annotate(
                                '',
                                xy=(cx + arrow_scale * adx * dx_cell,
                                    cy + arrow_scale * ady * dy_cell),
                                xytext=(cx - arrow_scale * adx * dx_cell,
                                        cy - arrow_scale * ady * dy_cell),
                                arrowprops=dict(arrowstyle='->', color='white',
                                                lw=1.8, mutation_scale=12),
                                zorder=5,
                            )
                        else:
                            # No-op / neutral action → dot
                            ax.plot(cx, cy, marker='.', markersize=6,
                                    color='white', zorder=5)

                # Mark bottleneck states with ★
                for bn in bottleneck:
                    bn_tuple = self.adapter.state_space.index_to_state(bn)
                    bx, by = pix_x[bn_tuple[0]], pix_y[bn_tuple[1]]
                    ax.plot(bx, by, marker='*', markersize=14,
                            color='black', markeredgecolor='white',
                            markeredgewidth=0.8, zorder=10)

                # Subsampled tick labels
                if has_centers:
                    max_ticks = 7
                    tp0 = np.linspace(extent[0], extent[1], bins[0]) if extent else np.arange(bins[0])
                    tp1 = np.linspace(extent[2], extent[3], bins[1]) if extent else np.arange(bins[1])
                    step0 = max(1, len(tp0) // max_ticks)
                    step1 = max(1, len(tp1) // max_ticks)
                    idx0 = np.arange(0, len(tp0), step0)
                    idx1 = np.arange(0, len(tp1), step1)
                    ax.set_xticks(tp0[idx0])
                    ax.set_xticklabels(np.round(centers0[idx0], 2))
                    ax.set_yticks(tp1[idx1])
                    ax.set_yticklabels(np.round(centers1[idx1], 2))

                ax.set_xlabel(dim0_label, fontsize=12)
                ax.set_ylabel(dim1_label, fontsize=12)

                # Legend: cluster colours + action symbols
                legend_handles = []
                # Cluster patches
                for c in range(self.n_clusters):
                    legend_handles.append(mpatches.Patch(
                        color=cluster_colors[c], alpha=0.5,
                        label=f'Cluster {c}'))
                # Action symbols
                for i in range(n_actions):
                    if i in arrow_dx:
                        sym = (r'$\rightarrow$' if arrow_dx[i][0] > 0
                               else (r'$\leftarrow$' if arrow_dx[i][0] < 0
                               else (r'$\uparrow$' if arrow_dx[i][1] > 0
                               else r'$\downarrow$')))
                        legend_handles.append(plt.Line2D(
                            [0], [0], marker=sym,
                            color='w', markerfacecolor='black', markersize=10,
                            label=action_labels[i], linestyle='None'))
                    else:
                        legend_handles.append(plt.Line2D(
                            [0], [0], marker='.', color='w',
                            markerfacecolor='black', markersize=10,
                            label=action_labels[i], linestyle='None'))
                # Bottleneck symbol
                legend_handles.append(plt.Line2D(
                    [0], [0], marker='*', color='w',
                    markerfacecolor='black', markeredgecolor='white',
                    markersize=12, label='Bottleneck', linestyle='None'))
                ax.legend(handles=legend_handles, loc='best', fontsize=8)

                fname = f"Macro_Action_Network_{macro_state}_{macro_final_state}.png"
                plt.tight_layout()
                plt.savefig(f"{save_dir}/{fname}", dpi=150, bbox_inches='tight')
                plt.close()

        print(f"  Saved macro action policy plots to {save_dir}/")

    def _plot_policy_arrows(self, arrows_grid: np.ndarray, grid_size: int,
                            save_path: str, title: str):
        """Plot policy arrows on grid.

        Actions: 0=left, 1=right, 2=up, 3=down, 4=pickup (shown as 'X')
        """
        arrows = {1: (1, 0), 0: (-1, 0), 3: (0, 1), 2: (0, -1)}
        offset = 0.3  # half-length of arrow

        fig, ax = plt.subplots(figsize=(grid_size, grid_size))

        for r, row in enumerate(arrows_grid):
            for c, cell in enumerate(row):
                if cell in arrows:
                    dx, dy = arrows[cell]
                    ax.annotate('',
                                xy=(c + offset * dx, r + offset * dy),
                                xytext=(c - offset * dx, r - offset * dy),
                                arrowprops=dict(arrowstyle='->', color='w',
                                                lw=2, mutation_scale=15))
                elif cell == 4:  # Pickup action
                    ax.text(c, r, 'X', fontsize=14, ha='center', va='center',
                            fontweight='bold', color='w')

        if self.labels_grid is not None:
            im = plt.imshow(self.labels_grid, cmap='gist_heat')
            colours = im.cmap(im.norm(np.unique(self.labels_grid)))
            patches = [mpatches.Patch(color=colours[i], label=f'{i}')
                       for i in range(len(colours) - 1)]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        # plt.title(title, fontsize=16)
        plt.savefig(save_path, format="png", bbox_inches='tight')
        plt.close()

    def _plot_policy_arrows_augmented(self, arrows_grids: List[np.ndarray], grid_size: int,
                                       n_augment: int, save_path: str, title: str):
        """Plot policy arrows for augmented state spaces with multiple panels.

        Actions: 0=left, 1=right, 2=up, 3=down, 4=pickup (shown as 'X')

        Args:
            arrows_grids: List of arrow grids, one per augment value
            grid_size: Size of the grid
            n_augment: Number of augment values (e.g., 2 for key/no-key)
            save_path: Path to save the figure
            title: Main title for the figure
        """
        arrows = {1: (1, 0), 0: (-1, 0), 3: (0, 1), 2: (0, -1)}
        offset = 0.3  # half-length of arrow

        aug_labels = ["✗ Without key", "★ With key"] if n_augment == 2 else [f"Aug {i}" for i in range(n_augment)]

        fig, axes = plt.subplots(1, n_augment, figsize=(grid_size * n_augment + 2, grid_size))
        if n_augment == 1:
            axes = [axes]

        for aug_idx, (arrows_grid, ax) in enumerate(zip(arrows_grids, axes)):
            # Draw arrows or pickup symbol
            for r, row in enumerate(arrows_grid):
                for c, cell in enumerate(row):
                    if cell in arrows:
                        dx, dy = arrows[cell]
                        ax.annotate('',
                                    xy=(c + offset * dx, r + offset * dy),
                                    xytext=(c - offset * dx, r - offset * dy),
                                    arrowprops=dict(arrowstyle='->', color='w',
                                                    lw=2, mutation_scale=15))
                    elif cell == 4:  # Pickup action
                        ax.text(c, r, 'X', fontsize=14, ha='center', va='center',
                                fontweight='bold', color='w')

            # Create background showing cluster labels if available
            if hasattr(self, 'micro_to_macro') and self.micro_to_macro:
                # Build cluster grid for this augment value
                cluster_grid = np.ones((grid_size, grid_size)) * self.n_clusters
                base_n_states = grid_size * grid_size

                for micro_idx, macro_idx in self.micro_to_macro.items():
                    if hasattr(self.adapter, 'state_space'):
                        state = self.adapter.state_space.index_to_state(micro_idx)
                        if len(state) == 2:
                            base_idx, state_aug_idx = state
                            if state_aug_idx == aug_idx:
                                x, y = divmod(base_idx, grid_size)
                                cluster_grid[x, y] = macro_idx

                cmap_d, norm_d = self._cluster_cmap_and_norm()
                ax.imshow(cluster_grid.T, cmap=cmap_d, norm=norm_d)
            else:
                # Just show a blank grid
                ax.imshow(np.zeros((grid_size, grid_size)), cmap='gray')

            ax.set_title(f'{aug_labels[aug_idx]}', fontsize=14)
            ax.set_xticks(np.arange(grid_size))
            ax.set_yticks(np.arange(grid_size))

        # Add legend
        colours = plt.cm.gist_heat(np.linspace(0, 0.8, self.n_clusters))
        patches = [mpatches.Patch(color=colours[i], label=f'{i}')
                   for i in range(self.n_clusters)]
        fig.legend(handles=patches, loc='center right', borderaxespad=0.5)

        # fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, format="png", bbox_inches='tight')
        plt.close()

    # ==================== Video Generation ====================

    def show_video(self, save_path: str = None,
                   init_loc: Tuple[int, int] = None,
                   goal_loc: Tuple[int, int] = None,
                   key_loc: Tuple[int, int] = None):
        """Generate video of episode trajectory.

        For key gridworld environments, shows:
        - Key pickup status indicator (color changes when key is acquired)
        - Pre-key path in one color, post-key path in another
        - Backtracking detection (revisited cells highlighted)

        Args:
            save_path: Path to save the video (e.g. 'figures/gridworld/episode_video.mp4')
            init_loc: Starting location
            goal_loc: Goal location
            key_loc: Key location (optional, auto-detected for key gridworlds)
        """
        if save_path is None:
            raise ValueError("save_path is required (e.g. 'figures/gridworld/episode_video.mp4')")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if not hasattr(self.adapter, 'grid_size'):
            print("Video generation requires grid-based environment")
            return

        if not self.state_history:
            print("No episode history to visualize")
            return

        grid_size = self.adapter.grid_size

        # Get locations
        if init_loc is None:
            init_loc = (0, 0)
        if goal_loc is None:
            goal_loc = (grid_size - 1, grid_size - 1)

        # Detect augmented (key) state space
        is_augmented = hasattr(self.adapter, 'state_space') and hasattr(self.adapter.state_space, 'n_augment')

        # Get wall locations using render_state for correct (x, y) coordinates
        walls = self.adapter.get_wall_indices() if hasattr(self.adapter, 'get_wall_indices') else []
        wall_locs = set()
        for w in walls:
            loc = self.adapter.render_state(w)
            wall_locs.add((loc[0], loc[1]))
        wall_locs = list(wall_locs)

        # Convert state history to locations and key status
        state_locs = []
        has_key_history = []
        for state in self.state_history:
            if hasattr(self.adapter, 'onehot_to_index'):
                idx = self.adapter.onehot_to_index(state)
            else:
                idx = np.argmax(state.flatten())
            loc = self.adapter.render_state(idx)
            state_locs.append((loc[0], loc[1]))

            # Extract key status from augmented state
            if is_augmented and len(loc) >= 3:
                has_key_history.append(int(loc[2]))
            else:
                has_key_history.append(None)

        # Find the step where key was picked up
        key_pickup_step = None
        if is_augmented:
            for i, hk in enumerate(has_key_history):
                if hk == 1:
                    key_pickup_step = i
                    break

        # Detect backtracking: cells visited more than once
        visit_counts = {}
        for i, loc in enumerate(state_locs):
            visit_counts[loc] = visit_counts.get(loc, 0) + 1

        # Setup figure
        fig, ax = plt.subplots(figsize=(8, 8))

        grid = np.zeros((grid_size, grid_size))
        if wall_locs:
            wall_idx = tuple(np.array(wall_locs).T)
            grid[wall_idx] = 0.25
        grid[init_loc] = 1
        grid[goal_loc] = 0.5

        im = ax.imshow(grid.T, aspect='equal', cmap='magma')

        # Title with key status indicator
        title_text = ax.set_title('Gridworld Episode', fontsize=14)

        ax.set_xticks(np.arange(0, grid_size, 1))
        ax.set_yticks(np.arange(0, grid_size, 1))
        ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        ax.tick_params(which='minor', bottom=False, left=False)

        # Colors for pre/post key paths
        pre_key_color = '#FFA500'   # Orange for path before key
        post_key_color = '#00BFFF'  # Deep sky blue for path after key
        backtrack_color = '#FF4444'  # Red for backtracking

        past_locs = []
        past_has_key = []
        scatter_objects = []

        # Build legend up front so it is visible on every frame
        legend_elements = []
        if is_augmented:
            legend_elements.append(mpatches.Patch(color=pre_key_color, label='Path (no key)'))
            legend_elements.append(mpatches.Patch(color=post_key_color, label='Path (has key)'))
        else:
            legend_elements.append(mpatches.Patch(color='yellow', label='Path'))
        # Check if any backtracking will happen across the whole episode
        has_backtracking = any(v > 1 for v in visit_counts.values())
        if has_backtracking:
            legend_elements.append(plt.Line2D([0], [0], marker='x', color='w',
                                   markerfacecolor=backtrack_color,
                                   markeredgecolor=backtrack_color,
                                   markersize=10, label='Backtracking', linestyle='None'))
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
                  framealpha=0.8)

        def init():
            grid = np.zeros((grid_size, grid_size))
            if wall_locs:
                for w in wall_locs:
                    grid[w] = 0.25
            grid[init_loc] = 1
            grid[goal_loc] = 0.5
            im.set_data(grid.T)
            return im,

        def animate(i):
            # Clear previous text and scatter
            for txt in ax.texts[::-1]:
                txt.remove()
            for sc in scatter_objects:
                sc.remove()
            scatter_objects.clear()

            grid = np.zeros((grid_size, grid_size))
            if wall_locs:
                for w in wall_locs:
                    grid[w] = 0.25

            s_loc = state_locs[i]
            grid[s_loc] = 1
            grid[goal_loc] = 0.5
            im.set_data(grid.T)

            past_locs.append(s_loc)
            past_has_key.append(has_key_history[i])

            # Update title with key status and step count
            if is_augmented:
                hk = has_key_history[i]
                if hk == 1:
                    key_str = '★ With key'
                    title_color = '#00BFFF'
                else:
                    key_str = '✗ Without key'
                    title_color = '#FFA500'
                title_text.set_text(f'Step {i}/{len(state_locs)-1}  |  {key_str}')
                title_text.set_color(title_color)
            else:
                title_text.set_text(f'Gridworld Episode  |  Step {i}/{len(state_locs)-1}')

            # Draw past positions with color coding
            if i > 0:
                # Count visits up to current step for backtracking detection
                visits_so_far = {}
                for j in range(i):
                    loc = past_locs[j]
                    visits_so_far[loc] = visits_so_far.get(loc, 0) + 1

                for j in range(i):
                    loc = past_locs[j]
                    is_backtrack = visits_so_far.get(loc, 0) > 1

                    if is_augmented:
                        # Color by key status phase
                        if is_backtrack:
                            color = backtrack_color
                            marker = 'x'
                            size = 60
                        elif past_has_key[j] == 1:
                            color = post_key_color
                            marker = 'o'
                            size = 40
                        else:
                            color = pre_key_color
                            marker = 'o'
                            size = 40
                    else:
                        if is_backtrack:
                            color = backtrack_color
                            marker = 'x'
                            size = 60
                        else:
                            color = 'yellow'
                            marker = 'o'
                            size = 40

                    sc = ax.scatter(loc[0], loc[1], color=color, marker=marker,
                                    s=size, zorder=5)
                    scatter_objects.append(sc)

            # Draw agent and goal labels (avoid overlap when agent is at goal)
            agent_color = '#00BFFF' if (is_augmented and has_key_history[i] == 1) else 'b'
            at_goal = (s_loc == goal_loc)
            if at_goal:
                ax.text(s_loc[0], s_loc[1], 'GOAL!', fontsize=10,
                        ha="center", va="center", color='#00FF00',
                        fontweight='bold', zorder=10)
            else:
                ax.text(s_loc[0], s_loc[1], 'Agent', fontsize=10,
                        ha="center", va="center", color=agent_color,
                        fontweight='bold', zorder=10)
                ax.text(goal_loc[0], goal_loc[1], 'Goal', fontsize=10,
                        ha="center", va="center", color="w")

            # Show key location if available and key not yet picked up
            if key_loc is not None and (not is_augmented or has_key_history[i] == 0):
                ax.text(key_loc[0], key_loc[1], '★', fontsize=14,
                        ha="center", va="center", color='#FFD700',
                        fontweight='bold', zorder=8)

            for w in wall_locs:
                ax.text(w[0], w[1], 'Wall', fontsize=8,
                        ha="center", va="center", color="w")

            if i == len(state_locs) - 1:
                # Save final frame as trajectory image
                traj_path = save_path.replace('.mp4', '_trajectory.png')
                plt.savefig(traj_path)

            return im,

        ani = animation.FuncAnimation(fig, animate, np.arange(len(state_locs)),
                                      init_func=init, interval=500, blit=True)
        ani.save(save_path)
        plt.close()
        print(f"Video saved to {save_path}")
