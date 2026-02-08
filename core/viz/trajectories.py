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

    def plot_trajectory_with_macro_states(
        self,
        positions: List[float],
        velocities: List[float],
        save_path: str = None,
    ):
        """Plot a 2D phase-space trajectory colored by macro state membership.

        Works for any 2D binned continuous environment (Mountain Car, Pendulum).
        Each trajectory point is colored according to its macro state cluster,
        using the same colormap as the cluster heatmap.

        Args:
            positions: Continuous dim-0 values (e.g. position / angle) per step.
            velocities: Continuous dim-1 values (e.g. velocity / angular vel) per step.
            save_path: Path to save the figure (e.g. 'figures/mountaincar/trajectory_macro.png').
        """
        if save_path is None:
            raise ValueError("save_path is required (e.g. 'figures/mountaincar/trajectory_macro.png')")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if hasattr(self.adapter, 'get_dimension_labels'):
            dim0_label, dim1_label = self.adapter.get_dimension_labels()
        else:
            dim0_label, dim1_label = "Dim 0", "Dim 1"

        num_points = len(positions)
        viridis_cmap = plt.get_cmap("viridis")
        macro_colors = [viridis_cmap(i / max(self.n_clusters - 1, 1))
                        for i in range(self.n_clusters)]

        plt.figure(figsize=(6, 5))

        # Color each point by its macro state
        for i in range(num_points):
            obs = np.array([positions[i], velocities[i]])
            discrete = self.adapter.discretize_obs(obs)
            s_idx = self.adapter.state_space.state_to_index(discrete)
            macro = self.micro_to_macro.get(s_idx, 0)
            plt.plot(positions[i], velocities[i],
                     marker="o", markersize=4, color=macro_colors[macro])

        # Gray connecting line
        plt.plot(positions, velocities, linewidth=1, color="gray", alpha=0.5)

        # Legend entries for each cluster
        for i in range(self.n_clusters):
            plt.scatter([], [], color=macro_colors[i], label=f'{i}',
                        marker="o", s=40, edgecolor='black')

        # Start / End markers
        s0_obs = np.array([positions[0], velocities[0]])
        s0_disc = self.adapter.discretize_obs(s0_obs)
        s0_macro = self.micro_to_macro.get(
            self.adapter.state_space.state_to_index(s0_disc), 0)
        plt.scatter(positions[0], velocities[0],
                    color=macro_colors[s0_macro], label="Start",
                    marker="o", s=60, edgecolor='black', linewidths=1.5)

        sN_obs = np.array([positions[-1], velocities[-1]])
        sN_disc = self.adapter.discretize_obs(sN_obs)
        sN_macro = self.micro_to_macro.get(
            self.adapter.state_space.state_to_index(sN_disc), 0)
        plt.scatter(positions[-1], velocities[-1],
                    color=macro_colors[sN_macro], label="End",
                    marker="o", s=60, edgecolor='black', linewidths=1.5)

        plt.xlabel(dim0_label)
        plt.ylabel(dim1_label)
        plt.title("Trajectory with Macro States")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved macro-state trajectory to {save_path}")

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

        plt.xlabel(dim0_label, fontsize=12)
        plt.ylabel(dim1_label, fontsize=12)
        plt.title("Trajectory with Actions")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved action trajectory to {save_path}")

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

        T = len(positions)
        pos = np.asarray(positions, dtype=float)
        vel = np.asarray(velocities, dtype=float)

        # Auto-select stages: start, extremum of dim-0, end
        if stage_idx is None:
            stage_idx = [0]
            # Find the extremum (min for Mountain Car, max or min for others)
            extremum_idx = int(np.argmin(pos))
            if extremum_idx not in (0, T - 1):
                stage_idx.append(extremum_idx)
            stage_idx.append(T - 1)
        stage_idx = [int(i) for i in stage_idx if 0 <= int(i) < T]
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
                    f"x={pos[idx]:.3f}\nv={vel[idx]:.3f}",
                    transform=ax_img.transAxes, fontsize=12, va="top",
                    bbox=dict(boxstyle="round", alpha=0.6),
                )

        # ---- Phase plot (bottom, spanning all columns) ----
        ax_phase = fig.add_subplot(gs[1, :])
        t = np.arange(T)
        ax_phase.scatter(pos, vel, c=t, s=18, cmap="plasma")
        ax_phase.plot(pos, vel, linewidth=1, color="gray", alpha=0.4)

        ax_phase.set_xlabel(dim0_label)
        ax_phase.set_ylabel(dim1_label)
        ax_phase.set_title("Trajectory in state space")

        # Highlight chosen stages
        for j, idx in enumerate(stage_idx):
            ax_phase.scatter(pos[idx], vel[idx], s=160, facecolors="none",
                             linewidths=3, edgecolors="black")
            ax_phase.text(pos[idx], vel[idx], f"  ({chr(97 + j)})",
                          fontsize=14, va="center")

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
    ):
        """Generate a vertically stacked video: environment render (top) + animated trajectory (bottom).

        The bottom panel shows a phase-space trajectory that grows over time,
        colored by macro state membership.  This produces the same style as the
        legacy ``mountain_car_combined_vertical.mp4``.

        Args:
            frames: RGB frames from env.render(), one per decision step.
            positions: Continuous dim-0 values per decision step.
            velocities: Continuous dim-1 values per decision step.
            save_path: Output MP4 path (e.g. 'figures/mountaincar/combined_vertical.mp4').
            fps: Frames per second for the output video.
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

        # Compute macro state per point
        viridis_cmap = plt.get_cmap("viridis")
        macro_colors = [viridis_cmap(i / max(self.n_clusters - 1, 1))
                        for i in range(self.n_clusters)]

        macro_states = []
        for i in range(T):
            obs = np.array([pos[i], vel[i]])
            discrete = self.adapter.discretize_obs(obs)
            s_idx = self.adapter.state_space.state_to_index(discrete)
            macro_states.append(self.micro_to_macro.get(s_idx, 0))
        macro_states = np.array(macro_states)

        # Build segments and per-segment colours
        points = np.column_stack([pos, vel])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        seg_colors = [macro_colors[m] for m in macro_states[:-1]]

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
            ax.set_title("Trajectory with macro actions", fontsize=12)

            # Add legend for macro states visited so far
            seen = set()
            for m in macro_states[:t + 1]:
                seen.add(m)
            for m in sorted(seen):
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

        For augmented state spaces (e.g., key gridworld), creates separate
        panels for each augment value (No Key / Has Key).

        Args:
            save_dir: Directory to save figures (e.g. 'figures/gridworld/macro_action_network')
        """
        if save_dir is None:
            raise ValueError("save_dir is required (e.g. 'figures/gridworld/macro_action_network')")
        os.makedirs(save_dir, exist_ok=True)

        if not hasattr(self.adapter, 'grid_size'):
            print("Policy visualization requires grid-based environment")
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

    def _plot_policy_arrows(self, arrows_grid: np.ndarray, grid_size: int,
                            save_path: str, title: str):
        """Plot policy arrows on grid.

        Actions: 0=left, 1=right, 2=up, 3=down, 4=pickup (shown as 'X')
        """
        arrows = {1: (1, 0), 0: (-1, 0), 3: (0, 1), 2: (0, -1)}
        scale = 0.25

        fig, ax = plt.subplots(figsize=(grid_size, grid_size))

        for r, row in enumerate(arrows_grid):
            for c, cell in enumerate(row):
                if cell in arrows:
                    ax.arrow(c - scale * arrows[cell][0],
                             r - scale * arrows[cell][1],
                             scale * arrows[cell][0],
                             scale * arrows[cell][1],
                             head_width=0.15, color='w')
                elif cell == 4:  # Pickup action
                    ax.text(c, r, 'X', fontsize=14, ha='center', va='center',
                            fontweight='bold', color='w')

        if self.labels_grid is not None:
            im = plt.imshow(self.labels_grid, cmap='gist_heat')
            colours = im.cmap(im.norm(np.unique(self.labels_grid)))
            patches = [mpatches.Patch(color=colours[i], label=f'{i}')
                       for i in range(len(colours) - 1)]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.title(title, fontsize=16)
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
        scale = 0.25

        aug_labels = ["✗ No Key", "★ Has Key"] if n_augment == 2 else [f"Aug {i}" for i in range(n_augment)]

        fig, axes = plt.subplots(1, n_augment, figsize=(grid_size * n_augment + 2, grid_size))
        if n_augment == 1:
            axes = [axes]

        for aug_idx, (arrows_grid, ax) in enumerate(zip(arrows_grids, axes)):
            # Draw arrows or pickup symbol
            for r, row in enumerate(arrows_grid):
                for c, cell in enumerate(row):
                    if cell in arrows:
                        ax.arrow(c - scale * arrows[cell][0],
                                r - scale * arrows[cell][1],
                                scale * arrows[cell][0],
                                scale * arrows[cell][1],
                                head_width=0.15, color='w')
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

                ax.imshow(cluster_grid.T, cmap='gist_heat')
            else:
                # Just show a blank grid
                ax.imshow(np.zeros((grid_size, grid_size)), cmap='gray')

            ax.set_title(f'{aug_labels[aug_idx]}', fontsize=14)
            ax.set_xticks(np.arange(grid_size))
            ax.set_yticks(np.arange(grid_size))

        # Add legend
        colours = plt.cm.gist_heat(np.linspace(0, 0.8, self.n_clusters))
        patches = [mpatches.Patch(color=colours[i], label=f'Cluster {i}')
                   for i in range(self.n_clusters)]
        fig.legend(handles=patches, loc='center right', borderaxespad=0.5)

        fig.suptitle(title, fontsize=16)
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
                    key_str = '★ Has Key'
                    title_color = '#00BFFF'
                else:
                    key_str = '✗ No Key'
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
