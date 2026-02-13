"""Matrix and clustering visualization methods.

This module provides plotting capabilities for transition matrices,
successor matrices, and macro state cluster visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import Optional, List


class MatrixVizMixin(object):
    """Mixin class providing matrix and clustering visualization methods.

    Requires the agent to have:
    - self.adapter: Environment adapter
    - self.B, self.M: Transition and successor matrices
    - self.B_macro, self.M_macro: Macro-level matrices
    - self.macro_state_list, self.micro_to_macro: Clustering results
    """

    # ==================== Matrix Visualization ====================

    def view_matrices(self, save_dir: str = None, learned: bool = True,
                      origin_state: Optional[int] = None):
        """Visualize transition and successor matrices.

        Args:
            save_dir: Directory to save figures (e.g. 'figures/gridworld/matrices')
            learned: Whether matrices were learned (vs analytical)
            origin_state: Origin state index for M-from-origin plot.
                         For grid environments this is the base location index
                         (row * grid_size + col). If None, defaults to 0.
        """
        if save_dir is None:
            raise ValueError("save_dir is required (e.g. 'figures/gridworld/matrices')")
        os.makedirs(save_dir, exist_ok=True)
        learn_str = 'estimated' if learned else 'actual'

        # Get dimensions
        n_states = self.adapter.n_states
        n_actions = self.adapter.n_actions

        # Plot averaged B matrix
        if self.B is not None:
            self._plot_B_matrix(self.B, n_states, n_actions,
                              f"{save_dir}/B_matrix_micro.png",
                              "Default Policy B Matrix")

        # Plot M matrix
        if self.M is not None:
            self._plot_M_matrix(self.M, n_states,
                              f"{save_dir}/m_{learn_str}_micro.png",
                              f"Successor Matrix M ({learn_str})")

            # Plot M from origin state
            self._plot_M_from_origin(self.M, n_states,
                                    f"{save_dir}/m_origin_{learn_str}.png",
                                    origin_state=origin_state)

            # Save M matrix
            os.makedirs("data", exist_ok=True)
            np.save("data/M.npy", self.M)

        # Plot macro-level matrices
        if self.B_macro is not None and self.M_macro is not None:
            self._plot_macro_matrices(save_dir, learn_str)

    def _plot_B_matrix(self, B: np.ndarray, n_states: int, n_actions: int,
                       save_path: str, title: str):
        """Plot averaged transition matrix."""
        if B.ndim == 3:
            B_avg = np.sum(B, axis=2) / n_actions
        elif B.ndim == 5:
            # Augmented state space - flatten first
            B_flat = self._flatten_B_for_viz(B)
            B_avg = np.sum(B_flat, axis=2) / n_actions
            n_states = B_avg.shape[0]
        else:
            return

        plt.figure(figsize=(10, 8))
        plt.title(title, fontsize=20)
        plt.imshow(B_avg)
        plt.colorbar()
        plt.savefig(save_path, format="png", bbox_inches='tight')
        plt.close()

    def _plot_M_matrix(self, M: np.ndarray, n_states: int,
                       save_path: str, title: str):
        """Plot successor matrix."""
        if M.ndim == 4:
            # Augmented state space - flatten
            M_flat = self._flatten_M_for_viz(M)
        else:
            M_flat = M

        plt.figure(figsize=(10, 8))
        plt.imshow(M_flat, aspect='equal', cmap='cividis')
        plt.colorbar()
        plt.title(title, fontsize=20)
        plt.savefig(save_path, format="png", bbox_inches='tight')
        plt.close()

    def _plot_M_from_origin(self, M: np.ndarray, n_states: int, save_path: str,
                            origin_state: Optional[int] = None):
        """Plot successor representation from a given origin state.

        Args:
            M: Successor matrix (2D or 4D for augmented state spaces)
            n_states: Number of states
            save_path: Path to save the figure
            origin_state: Origin state index (base location index for grid envs).
                         If None, defaults to 0 (top-left corner).

        For augmented state spaces (e.g., key gridworld), creates a 2x2 grid showing
        M from origin for all combinations of (origin_key_state, target_key_state):
        - Top-left: No Key -> No Key
        - Top-right: No Key -> Has Key
        - Bottom-left: Has Key -> No Key
        - Bottom-right: Has Key -> Has Key
        """
        if not hasattr(self.adapter, 'grid_size'):
            return

        grid_size = self.adapter.grid_size

        if origin_state is None:
            origin_state = 0

        # Convert origin index to (x, y) for title
        origin_x, origin_y = divmod(origin_state, grid_size)

        if M.ndim == 4:
            # Augmented state space (e.g., key gridworld)
            # M shape: (N, 2, N, 2) where 2 is n_augment (has_key in {0, 1})
            n_augment = M.shape[1]

            fig, axes = plt.subplots(n_augment, n_augment, figsize=(12, 10))
            fig.suptitle(f'Successor Representation from ({origin_x},{origin_y})', fontsize=16)

            aug_labels = ["No Key", "Has Key"] if n_augment == 2 else [f"Aug {i}" for i in range(n_augment)]

            # Find global max for consistent color scaling
            global_max = max(M[origin_state, i, :, j].max()
                           for i in range(n_augment) for j in range(n_augment))
            global_max = global_max if global_max > 0 else 1

            for origin_aug in range(n_augment):
                for target_aug in range(n_augment):
                    ax = axes[origin_aug, target_aug] if n_augment > 1 else axes

                    # M[origin_loc, origin_aug, target_loc, target_aug]
                    M_slice = M[origin_state, origin_aug, :, target_aug].reshape(grid_size, grid_size).T

                    im = ax.imshow(M_slice, aspect='equal', cmap='copper',
                                   norm=LogNorm(vmin=0.01, vmax=global_max))
                    ax.set_title(f'{aug_labels[origin_aug]} \u2192 {aug_labels[target_aug]}', fontsize=12)
                    ax.set_xticks(np.arange(grid_size))
                    ax.set_yticks(np.arange(grid_size))

                    # Mark origin state
                    ax.plot(origin_y, origin_x, marker='*', color='cyan',
                            markersize=15, markeredgecolor='white', markeredgewidth=1)

            # Add colorbar
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
            plt.savefig(save_path, format="png", bbox_inches='tight')
            plt.close()
        else:
            # Standard state space
            M_orig = M[origin_state, :].reshape(grid_size, grid_size).T

            plt.figure(figsize=(8, 8))
            vmax = M_orig.max() if M_orig.max() > 0 else 1
            plt.imshow(M_orig, aspect='equal', cmap='copper',
                       norm=LogNorm(vmin=0.01, vmax=vmax))
            plt.colorbar()
            plt.title(f'Successor Representation from ({origin_x},{origin_y})', fontsize=14)
            plt.xticks(np.arange(grid_size), fontsize=12)
            plt.yticks(np.arange(grid_size), fontsize=12)

            # Mark origin state
            plt.plot(origin_y, origin_x, marker='*', color='cyan',
                     markersize=15, markeredgecolor='white', markeredgewidth=1)

            plt.savefig(save_path, format="png", bbox_inches='tight')
            plt.close()

    def _plot_macro_matrices(self, save_dir: str, learn_str: str):
        """Plot macro-level transition and successor matrices."""
        n_clusters = self.n_clusters
        n_macro_actions = self.n_macro_actions

        # B_macro
        B_macro_avg = np.sum(self.B_macro, axis=2) / n_macro_actions
        plt.figure(figsize=(8, 6))
        plt.title("Default Policy B Matrix (Macro)", fontsize=20)
        plt.imshow(B_macro_avg)
        plt.xticks(range(n_clusters), fontsize=12)
        plt.yticks(range(n_clusters), fontsize=12)
        for i in range(n_clusters):
            for j in range(n_clusters):
                plt.text(j, i, f'{B_macro_avg[i, j]:.2f}',
                        ha="center", va="center", color="w")
        plt.colorbar()
        plt.savefig(f"{save_dir}/B_matrix_macro.png", format="png", bbox_inches='tight')
        plt.close()

        # M_macro
        plt.figure(figsize=(8, 6))
        plt.imshow(self.M_macro, aspect='equal', cmap='cividis')
        plt.xticks(range(n_clusters), fontsize=12)
        plt.yticks(range(n_clusters), fontsize=12)
        plt.title(f"Successor Matrix M Macro ({learn_str})", fontsize=20)
        for i in range(n_clusters):
            for j in range(n_clusters):
                plt.text(j, i, f'{self.M_macro[i, j]:.2f}',
                        ha="center", va="center", color="w")
        plt.colorbar()
        plt.savefig(f"{save_dir}/m_{learn_str}_macro.png", format="png", bbox_inches='tight')
        plt.close()

    def _flatten_B_for_viz(self, B: np.ndarray) -> np.ndarray:
        """Flatten augmented B matrix for visualization."""
        if hasattr(self.adapter, '_flatten_transition'):
            return self.adapter._flatten_transition(B)
        # Manual flattening
        N = B.shape[0]
        n_actions = B.shape[-1]
        result = np.zeros((2 * N, 2 * N, n_actions))
        for a in range(n_actions):
            result[:, :, a] = np.hstack([
                np.vstack([B[:, j, :, i, a] for j in [0, 1]])
                for i in [0, 1]
            ])
        return result

    def _flatten_M_for_viz(self, M: np.ndarray) -> np.ndarray:
        """Flatten augmented M matrix for visualization."""
        if hasattr(self.adapter, 'flatten_successor_for_clustering'):
            return self.adapter.flatten_successor_for_clustering(M)
        # Manual flattening
        N = M.shape[0]
        return np.hstack([
            np.vstack([M[:, j, :, i] for j in [0, 1]])
            for i in [0, 1]
        ])

    # ==================== Clustering Visualization ====================

    def visualize_clusters(self, save_dir: str = None):
        """Visualize macro state clusters.

        Args:
            save_dir: Directory to save figures (e.g. 'figures/gridworld/clustering')
        """
        if save_dir is None:
            raise ValueError("save_dir is required (e.g. 'figures/gridworld/clustering')")
        os.makedirs(save_dir, exist_ok=True)

        if not hasattr(self.adapter, 'grid_size'):
            # Check for 2D binned continuous state space (Mountain Car, Pendulum)
            if (hasattr(self.adapter, 'state_space') and
                    hasattr(self.adapter.state_space, 'n_bins_per_dim') and
                    len(self.adapter.state_space.n_bins_per_dim) == 2):
                self._visualize_clusters_2d_binned(save_dir)
                return
            print("Cluster visualization requires grid-based or 2D binned environment")
            return

        grid_size = self.adapter.grid_size
        n_states = self.adapter.n_states

        # Build labels array
        labels = np.ones(n_states) * self.n_clusters  # Default to wall label
        for micro_idx, macro_idx in self.micro_to_macro.items():
            labels[micro_idx] = macro_idx

        # Check if this is an augmented state space (e.g., key gridworld)
        base_n_states = grid_size * grid_size
        is_augmented = n_states != base_n_states

        if is_augmented:
            # For augmented state spaces, create separate grids for each augment value
            n_augment = n_states // base_n_states

            fig, axes = plt.subplots(1, n_augment, figsize=(6 * n_augment, 6))
            if n_augment == 1:
                axes = [axes]

            for aug_idx in range(n_augment):
                # Extract labels for this augment value
                aug_labels = np.ones(base_n_states) * self.n_clusters
                for micro_idx, macro_idx in self.micro_to_macro.items():
                    if hasattr(self.adapter, 'state_space'):
                        state = self.adapter.state_space.index_to_state(micro_idx)
                        if len(state) == 2:
                            base_idx, state_aug_idx = state
                            if state_aug_idx == aug_idx:
                                aug_labels[base_idx] = macro_idx

                labels_grid = aug_labels.reshape(grid_size, grid_size).T

                ax = axes[aug_idx]
                im = ax.imshow(labels_grid, cmap='gist_heat')
                aug_label = "No Key" if aug_idx == 0 else "Has Key"
                ax.set_title(f"{aug_label}", fontsize=16)
                ax.set_xticks(np.arange(grid_size))
                ax.set_yticks(np.arange(grid_size))

            # Store for policy visualization
            self.labels_grid = None  # Not applicable for augmented

            # Add legend
            colours = plt.cm.gist_heat(np.linspace(0, 0.8, self.n_clusters))
            patches = [mpatches.Patch(color=colours[i], label=f'Cluster {i}')
                       for i in range(self.n_clusters)]
            fig.legend(handles=patches, loc='center right', borderaxespad=0.5)

            plt.suptitle("Macro State Clusters (Augmented State Space)", fontsize=20)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/Macro_s.png", format="png", bbox_inches='tight')
            plt.close()

        else:
            # Standard grid
            labels_grid = labels.reshape(grid_size, grid_size).T
            self.labels_grid = labels_grid

            plt.figure(figsize=(10, 8))
            im = plt.imshow(labels_grid, cmap='gist_heat')
            colours = im.cmap(im.norm(np.unique(labels_grid)))
            plt.xticks(np.arange(grid_size), fontsize=12)
            plt.yticks(np.arange(grid_size), fontsize=12)
            plt.title("Macro State Clusters", fontsize=20)

            n_colors = len(colours) - 1 if len(colours) > self.n_clusters else len(colours)
            patches = [mpatches.Patch(color=colours[i], label=f'{i}')
                       for i in range(n_colors)]
            plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            plt.savefig(f"{save_dir}/Macro_s.png", format="png", bbox_inches='tight')
            plt.close()

            # Plot spectral embedding if available (only for standard)
            if self.spectral_positions is not None:
                self._plot_spectral_embedding(save_dir, colours)

        # Plot spectral embedding for any case
        if self.spectral_positions is not None and not is_augmented:
            pass  # Already done above
        elif self.spectral_positions is not None:
            self._plot_spectral_embedding_augmented(save_dir)
            self._plot_spectral_embedding_with_ellipses(save_dir)

    def _plot_spectral_embedding(self, save_dir: str, colours: np.ndarray):
        """Plot spectral embedding of states."""
        plt.figure(figsize=(10, 8))
        for i, states in enumerate(self.macro_state_list):
            positions = self.spectral_positions[states]
            plt.scatter(positions[:, 0], positions[:, 1],
                       label=f'{i}', color=colours[i], s=75)
        plt.legend()
        plt.title('Micro States Spectral Embedding')
        plt.savefig(f'{save_dir}/macro_state_viz.png', bbox_inches='tight')
        plt.close()

    def _plot_spectral_embedding_augmented(self, save_dir: str):
        """Plot spectral embedding for augmented state spaces."""
        plt.figure(figsize=(10, 8))
        colours = plt.cm.gist_heat(np.linspace(0, 0.8, self.n_clusters))
        for i, states in enumerate(self.macro_state_list):
            if len(states) > 0:
                positions = self.spectral_positions[states]
                plt.scatter(positions[:, 0], positions[:, 1],
                           label=f'Cluster {i}', color=colours[i], s=75)
        plt.legend()
        plt.title('Micro States Spectral Embedding (Augmented)')
        plt.savefig(f'{save_dir}/macro_state_viz.png', bbox_inches='tight')
        plt.close()

    def _plot_spectral_embedding_with_ellipses(self, save_dir: str):
        """Plot spectral embedding with fitted 2σ covariance ellipses per cluster.

        Produces a scatter plot colored by cluster with fitted Gaussian
        ellipses overlaid and an inset thumbnail showing the cluster map
        on the grid.  Saved alongside the existing scatter as a separate file.
        """
        if self.spectral_positions is None:
            return

        colours = plt.cm.gist_heat(np.linspace(0, 0.8, self.n_clusters))
        fig, ax = plt.subplots(figsize=(10, 8))

        # Scatter points by cluster
        for i, states in enumerate(self.macro_state_list):
            if len(states) == 0:
                continue
            positions = self.spectral_positions[states]
            ax.scatter(positions[:, 0], positions[:, 1],
                       label=f'Cluster {i}', color=colours[i], s=60,
                       alpha=0.7, zorder=3)

            # Fit 2σ covariance ellipse
            if len(positions) >= 3:
                mean = positions.mean(axis=0)
                cov = np.cov(positions, rowvar=False)
                # Eigendecomposition for ellipse orientation and axes
                eigvals, eigvecs = np.linalg.eigh(cov)
                # Ensure positive eigenvalues (numerical safety)
                eigvals = np.maximum(eigvals, 1e-10)
                # Angle from first eigenvector
                angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                # 2-sigma width and height
                width = 2 * 2 * np.sqrt(eigvals[0])
                height = 2 * 2 * np.sqrt(eigvals[1])
                ell = Ellipse(xy=mean, width=width, height=height,
                              angle=angle, facecolor=colours[i],
                              edgecolor='black', linewidth=1.5,
                              alpha=0.2, zorder=2)
                ax.add_patch(ell)

        ax.legend(fontsize=9, loc='upper right')
        ax.set_title('Spectral Embedding with Cluster Ellipses', fontsize=14)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')

        # Inset: cluster map thumbnail (if grid-based)
        if hasattr(self.adapter, 'grid_size'):
            grid_size = self.adapter.grid_size
            n_states = self.adapter.n_states
            base_n_states = grid_size * grid_size
            is_augmented = n_states != base_n_states

            inset = inset_axes(ax, width="25%", height="25%", loc='lower right',
                               borderpad=1.5)

            if is_augmented:
                # Show has_key=0 panel as thumbnail
                aug_labels = np.ones(base_n_states) * self.n_clusters
                for micro_idx, macro_idx in self.micro_to_macro.items():
                    state = self.adapter.state_space.index_to_state(micro_idx)
                    if len(state) == 2:
                        base_idx, aug_idx = state
                        if aug_idx == 0:
                            aug_labels[base_idx] = macro_idx
                labels_grid = aug_labels.reshape(grid_size, grid_size).T
            else:
                labels_arr = np.ones(n_states) * self.n_clusters
                for micro_idx, macro_idx in self.micro_to_macro.items():
                    labels_arr[micro_idx] = macro_idx
                labels_grid = labels_arr.reshape(grid_size, grid_size).T

            inset.imshow(labels_grid, cmap='gist_heat')
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title('Clusters', fontsize=8)

        plt.savefig(f'{save_dir}/macro_state_viz_ellipses.png', bbox_inches='tight')
        plt.close()

    def _visualize_clusters_2d_binned(self, save_dir: str):
        """Visualize clusters for 2D binned continuous state spaces.

        Creates a heatmap showing cluster assignments over the 2D discretized
        state space (e.g., position x velocity for Mountain Car, theta x omega for
        Pendulum).  When the adapter provides ``get_bin_edges()``, the axes
        are labelled with physical coordinates; otherwise plain bin indices.
        """
        bins = self.adapter.state_space.n_bins_per_dim
        n_states = self.adapter.n_states

        # Build labels array
        labels = np.ones(n_states, dtype=int) * self.n_clusters  # default = "unassigned"
        for micro_idx, macro_idx in self.micro_to_macro.items():
            labels[micro_idx] = macro_idx

        # Reshape to 2D grid (row-major ordering matches BinnedContinuousStateSpace)
        labels_grid = labels.reshape(bins[0], bins[1])

        # Get axis labels from adapter if available
        if hasattr(self.adapter, 'get_dimension_labels'):
            dim0_label, dim1_label = self.adapter.get_dimension_labels()
        else:
            dim0_label, dim1_label = "Dim 0", "Dim 1"

        # Get physical extent from bin edges (if available)
        has_edges = hasattr(self.adapter, 'get_bin_edges')
        if has_edges:
            edges0, edges1 = self.adapter.get_bin_edges()
            extent = [edges0[0], edges0[-1], edges1[0], edges1[-1]]
        else:
            extent = None

        # Get bin center values for tick labels (if available)
        has_centers = hasattr(self.adapter, 'get_bin_centers')
        if has_centers:
            centers0, centers1 = self.adapter.get_bin_centers()

        plt.figure()
        im = plt.imshow(
            labels_grid.T,
            aspect='auto',
            origin='lower',
            extent=extent,
            interpolation='nearest',
        )

        # Build legend from the colormap used by imshow
        colours = im.cmap(im.norm(np.unique(labels_grid)))
        n_actual = len(np.unique(labels_grid))
        # Exclude unassigned label from the legend
        n_legend = n_actual - 1 if self.n_clusters in labels else n_actual
        patches = [mpatches.Patch(color=colours[i], label=f'{i}')
                   for i in range(n_legend)]
        plt.legend(handles=patches)

        # Custom tick labels showing physical bin centers (subsampled to avoid clutter)
        if has_centers:
            max_ticks = 7  # Keep axes readable
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
            # Subsample to at most max_ticks evenly spaced ticks
            step0 = max(1, len(tick_pos0) // max_ticks)
            step1 = max(1, len(tick_pos1) // max_ticks)
            idx0 = np.arange(0, len(tick_pos0), step0)
            idx1 = np.arange(0, len(tick_pos1), step1)
            plt.xticks(ticks=tick_pos0[idx0], labels=np.round(centers0[idx0], 2), fontsize=10)
            plt.yticks(ticks=tick_pos1[idx1], labels=np.round(centers1[idx1], 2), fontsize=10)

        plt.xlabel(dim0_label, fontsize=12)
        plt.ylabel(dim1_label, fontsize=12)
        plt.title("Macro state clusters", fontsize=20)
        plt.savefig(f"{save_dir}/Macro_s.png", format="png", bbox_inches='tight')
        plt.close()
        print(f"  Saved cluster heatmap to {save_dir}/Macro_s.png")

        # Also plot spectral embedding if available
        if self.spectral_positions is not None:
            colours_arr = np.array(colours[:n_legend])
            self._plot_spectral_embedding(save_dir, colours_arr)

    # ==================== Composite Key Gridworld Figure ====================

    def visualize_key_gridworld_composite(self, save_path: str,
                                           init_loc: tuple, goal_loc: tuple,
                                           key_loc: tuple):
        """Generate a 2×2 composite figure for the Key Gridworld.

        Panels:
            (a) Grid layout — Agent / Key / Goal / walls
            (b) Split value function — heatmaps per key state
            (c) Split cluster maps — heatmaps per key state
            (d) Spectral embedding with 2σ ellipses + inset

        Args:
            save_path: Where to save the composite figure.
            init_loc: (x, y) agent start.
            goal_loc: (x, y) goal position.
            key_loc: (x, y) key position.
        """
        from matplotlib import colors as mcolors

        if not hasattr(self.adapter, 'grid_size'):
            print("Composite figure requires grid-based key gridworld adapter")
            return
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        grid_size = self.adapter.grid_size
        n_base = grid_size ** 2
        n_states = self.adapter.n_states

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # ---- (a) Grid layout ----
        ax_a = axes[0, 0]
        grid = np.zeros((grid_size, grid_size))
        walls = self.adapter.env.get_walls() if hasattr(self.adapter.env, 'get_walls') else []
        if len(walls) > 0:
            wall_idx = tuple(np.array(walls).T)[::-1]
            grid[wall_idx] = 1

        kx, ky = key_loc
        grid[ky, kx] = 0.75
        sx, sy = init_loc
        grid[sy, sx] = 0.25
        gx, gy = goal_loc
        grid[gy, gx] = 0.5

        cmap_grid = mcolors.ListedColormap(['black', 'yellow', 'purple', 'orange', 'white'])
        ax_a.imshow(grid, aspect='equal', cmap=cmap_grid)
        ax_a.text(kx, ky, 'Key', fontsize=11, weight='bold', ha='center', va='center', color='k')
        ax_a.text(sx, sy, 'Agent', fontsize=11, weight='bold', ha='center', va='center', color='k')
        ax_a.text(gx, gy, 'Goal', fontsize=11, weight='bold', ha='center', va='center', color='w')
        ax_a.set_xticks(np.arange(grid_size))
        ax_a.set_yticks(np.arange(grid_size))
        ax_a.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax_a.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax_a.grid(which='minor', color='w', linestyle='-', linewidth=2)
        ax_a.tick_params(which='minor', bottom=False, left=False)
        ax_a.set_title('(a) Grid Layout', fontsize=14)

        # ---- (b) Split value function ----
        ax_b = axes[0, 1]
        if self.M is not None and self.C is not None:
            V = self.adapter.multiply_M_C(self.M, self.C)
            is_augmented_v = V.shape[0] == 2 * n_base

            wall_mask = np.zeros(n_base, dtype=bool)
            for wi in self.adapter.get_wall_indices():
                if is_augmented_v:
                    base_idx, _ = self.adapter.state_space.index_to_state(wi)
                    wall_mask[base_idx] = True
                else:
                    wall_mask[wi] = True

            if is_augmented_v:
                V_no = V[:n_base].copy()
                V_has = V[n_base:].copy()
                wmask_grid = wall_mask.reshape(grid_size, grid_size).T

                V_no_grid = np.ma.masked_where(wmask_grid, V_no.reshape(grid_size, grid_size).T)
                V_has_grid = np.ma.masked_where(wmask_grid, V_has.reshape(grid_size, grid_size).T)

                vmin = min(V_no_grid.min(), V_has_grid.min())
                vmax = max(V_no_grid.max(), V_has_grid.max())

                # Split the axis into two sub-axes
                ax_b.set_visible(False)
                gs_inner = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.4)
                ax_b1 = fig.add_subplot(gs_inner[0, 2])
                ax_b2 = fig.add_subplot(gs_inner[0, 3])

                ax_b1.set_facecolor('white')
                ax_b1.imshow(V_no_grid, cmap='copper', vmin=vmin, vmax=vmax)
                ax_b1.set_title('No Key', fontsize=11)
                ax_b1.set_xticks(np.arange(grid_size))
                ax_b1.set_yticks(np.arange(grid_size))

                ax_b2.set_facecolor('white')
                im_v = ax_b2.imshow(V_has_grid, cmap='copper', vmin=vmin, vmax=vmax)
                ax_b2.set_title('Has Key', fontsize=11)
                ax_b2.set_xticks(np.arange(grid_size))
                ax_b2.set_yticks(np.arange(grid_size))

                fig.colorbar(im_v, ax=[ax_b1, ax_b2], shrink=0.7, label='Value')
                fig.text(0.73, 0.93, '(b) Value Function', ha='center', fontsize=14)
            else:
                V_grid = np.ma.masked_where(
                    wall_mask.reshape(grid_size, grid_size).T,
                    V.reshape(grid_size, grid_size).T,
                )
                ax_b.set_facecolor('white')
                ax_b.imshow(V_grid, cmap='viridis')
                ax_b.set_title('(b) Value Function', fontsize=14)
                ax_b.set_xticks(np.arange(grid_size))
                ax_b.set_yticks(np.arange(grid_size))
        else:
            ax_b.text(0.5, 0.5, 'No M/C', ha='center', va='center',
                      transform=ax_b.transAxes, fontsize=14)
            ax_b.set_title('(b) Value Function', fontsize=14)

        # ---- (c) Split cluster maps ----
        ax_c = axes[1, 0]
        is_augmented_c = n_states != n_base
        cluster_colours = plt.cm.gist_heat(np.linspace(0, 0.8, self.n_clusters))

        if is_augmented_c:
            ax_c.set_visible(False)
            gs_inner_c = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.4)
            ax_c1 = fig.add_subplot(gs_inner_c[1, 0])
            ax_c2 = fig.add_subplot(gs_inner_c[1, 1])

            for aug_idx, (ax_sub, title) in enumerate(
                    [(ax_c1, 'No Key'), (ax_c2, 'Has Key')]):
                aug_labels = np.ones(n_base) * self.n_clusters
                for micro_idx, macro_idx in self.micro_to_macro.items():
                    state = self.adapter.state_space.index_to_state(micro_idx)
                    if len(state) == 2:
                        base_idx, st_aug = state
                        if st_aug == aug_idx:
                            aug_labels[base_idx] = macro_idx
                labels_grid = aug_labels.reshape(grid_size, grid_size).T
                ax_sub.imshow(labels_grid, cmap='gist_heat')
                ax_sub.set_title(title, fontsize=11)
                ax_sub.set_xticks(np.arange(grid_size))
                ax_sub.set_yticks(np.arange(grid_size))

            # Cluster legend
            patches_c = [mpatches.Patch(color=cluster_colours[i], label=f'Cluster {i}')
                         for i in range(self.n_clusters)]
            fig.legend(handles=patches_c, loc='lower left',
                       bbox_to_anchor=(0.02, 0.02), fontsize=8, ncol=2)
            fig.text(0.27, 0.47, '(c) Macro State Clusters', ha='center', fontsize=14)
        else:
            labels_arr = np.ones(n_states) * self.n_clusters
            for micro_idx, macro_idx in self.micro_to_macro.items():
                labels_arr[micro_idx] = macro_idx
            labels_grid = labels_arr.reshape(grid_size, grid_size).T
            ax_c.imshow(labels_grid, cmap='gist_heat')
            ax_c.set_title('(c) Macro State Clusters', fontsize=14)
            ax_c.set_xticks(np.arange(grid_size))
            ax_c.set_yticks(np.arange(grid_size))

        # ---- (d) Spectral embedding with ellipses ----
        ax_d = axes[1, 1]
        if self.spectral_positions is not None:
            for i, states in enumerate(self.macro_state_list):
                if len(states) == 0:
                    continue
                positions = self.spectral_positions[states]
                ax_d.scatter(positions[:, 0], positions[:, 1],
                             label=f'Cluster {i}', color=cluster_colours[i],
                             s=50, alpha=0.7, zorder=3)
                # 2σ covariance ellipse
                if len(positions) >= 3:
                    mean = positions.mean(axis=0)
                    cov = np.cov(positions, rowvar=False)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    eigvals = np.maximum(eigvals, 1e-10)
                    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
                    width = 2 * 2 * np.sqrt(eigvals[0])
                    height = 2 * 2 * np.sqrt(eigvals[1])
                    ell = Ellipse(xy=mean, width=width, height=height,
                                  angle=angle, facecolor=cluster_colours[i],
                                  edgecolor='black', linewidth=1.2,
                                  alpha=0.2, zorder=2)
                    ax_d.add_patch(ell)
            ax_d.legend(fontsize=8, loc='upper right')

            # Inset cluster thumbnail
            if hasattr(self.adapter, 'grid_size'):
                inset = inset_axes(ax_d, width="22%", height="22%", loc='lower right',
                                   borderpad=1.0)
                if is_augmented_c:
                    aug_labels_in = np.ones(n_base) * self.n_clusters
                    for micro_idx, macro_idx in self.micro_to_macro.items():
                        state = self.adapter.state_space.index_to_state(micro_idx)
                        if len(state) == 2:
                            base_idx, aug_idx = state
                            if aug_idx == 0:
                                aug_labels_in[base_idx] = macro_idx
                    inset.imshow(aug_labels_in.reshape(grid_size, grid_size).T, cmap='gist_heat')
                else:
                    inset.imshow(labels_grid, cmap='gist_heat')
                inset.set_xticks([])
                inset.set_yticks([])
                inset.set_title('Clusters', fontsize=7)
        else:
            ax_d.text(0.5, 0.5, 'No spectral data', ha='center', va='center',
                      transform=ax_d.transAxes, fontsize=14)
        ax_d.set_title('(d) Spectral Embedding', fontsize=14)

        fig.suptitle('Key Gridworld — Hierarchical SR Agent', fontsize=16, y=0.98)
        try:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        except Exception:
            pass  # Mixed gridspec axes may not be compatible with tight_layout
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Composite figure saved to {save_path}")
