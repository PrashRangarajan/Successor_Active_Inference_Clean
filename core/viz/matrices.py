"""Matrix and clustering visualization methods.

This module provides plotting capabilities for transition matrices,
successor matrices, and macro state cluster visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as PathEffects
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm, PowerNorm, ListedColormap, BoundaryNorm
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

    # ==================== Colormap Helpers ====================

    def _cluster_colours(self):
        """Return a list of RGBA tuples for cluster indices 0..n_clusters-1."""
        src = plt.get_cmap('tab10') if self.n_clusters <= 10 else plt.get_cmap('tab20')
        return [src(i) for i in range(self.n_clusters)]

    def _cluster_cmap_and_norm(self):
        """Return a discrete (ListedColormap, BoundaryNorm) for cluster plots.

        Maps cluster indices 0..n_clusters-1 to distinct ``tab10`` colours
        and reserves **white** for wall / invalid cells (which carry the
        sentinel value ``n_clusters``).
        """
        base_colours = self._cluster_colours()
        # Append white for the wall sentinel
        all_colours = list(base_colours) + [(1, 1, 1, 1)]  # white for walls
        cmap = ListedColormap(all_colours)
        # Boundaries: [−0.5, 0.5, 1.5, …, n_clusters−0.5, n_clusters+0.5]
        bounds = np.arange(-0.5, self.n_clusters + 1.5, 1)
        norm = BoundaryNorm(bounds, cmap.N)
        return cmap, norm

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
        # plt.title(title, fontsize=20)
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
        # plt.title(title, fontsize=20)
        plt.savefig(save_path, format="png", bbox_inches='tight')
        plt.close()

    def _plot_M_from_origin(self, M: np.ndarray, n_states: int, save_path: str,
                            origin_state: Optional[int] = None):
        """Plot successor representation from a given origin state.

        For augmented state spaces (4-D M of shape ``(N, n_augment, N, n_augment)``),
        produces a **n_augment × n_augment** subplot grid (typically 2×2 for key
        gridworld) showing M slices for every ``(origin_aug, target_aug)`` pair:

        +-----------------------+-----------------------+
        | No Key → No Key       | No Key → Has Key      |
        +-----------------------+-----------------------+
        | Has Key → No Key      | Has Key → Has Key     |
        +-----------------------+-----------------------+

        Each cell is ``M[origin_loc, origin_aug, :, target_aug]`` reshaped to
        the base grid.

        For standard (2-D) M, a single heatmap is shown.

        Args:
            M: Successor matrix (2D or 4D for augmented state spaces)
            n_states: Number of states
            save_path: Path to save the figure
            origin_state: Origin state index (base location index for grid envs).
                         If None, defaults to 0 (top-left corner).
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
            # fig.suptitle(f'Successor Representation from ({origin_x},{origin_y})', fontsize=16)

            aug_labels = ["✗ Without key", "★ With key"] if n_augment == 2 else [f"Aug {i}" for i in range(n_augment)]

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
                    # ax.set_title(f'{aug_labels[origin_aug]} \u2192 {aug_labels[target_aug]}', fontsize=12)
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
            # plt.title(f'Successor Representation from ({origin_x},{origin_y})', fontsize=14)
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
        # plt.title("Default Policy B Matrix (Macro)", fontsize=20)
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
        # plt.title(f"Successor Matrix M Macro ({learn_str})", fontsize=20)
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

        For augmented state spaces (e.g., key gridworld with ``n_states == 2 * grid_size²``),
        produces a **1×n_augment** subplot layout — one panel per augment value
        ("Without key" / "With key").  Each panel maps micro-to-macro assignments
        back onto the base grid by decomposing the flat index via
        ``state_space.index_to_state()``.

        For standard (non-augmented) grids, a single heatmap is shown.

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
            cmap_d, norm_d = self._cluster_cmap_and_norm()

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
                im = ax.imshow(labels_grid, cmap=cmap_d, norm=norm_d)
                aug_label = "✗ Without key" if aug_idx == 0 else "★ With key"
                ax.set_title(f"{aug_label}", fontsize=16)
                ax.set_xticks(np.arange(grid_size))
                ax.set_yticks(np.arange(grid_size))

            # Store for policy visualization
            self.labels_grid = None  # Not applicable for augmented

            # Add legend
            colours = self._cluster_colours()
            patches = [mpatches.Patch(color=colours[i], label=f'{i}')
                       for i in range(self.n_clusters)]
            fig.legend(handles=patches, loc='center right', borderaxespad=0.5)

            # plt.suptitle("Macro State Clusters (Augmented State Space)", fontsize=20)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/Macro_s.png", format="png", bbox_inches='tight')
            plt.close()

        else:
            # Standard grid
            labels_grid = labels.reshape(grid_size, grid_size).T
            self.labels_grid = labels_grid

            plt.figure(figsize=(10, 8))
            cmap_d, norm_d = self._cluster_cmap_and_norm()
            im = plt.imshow(labels_grid, cmap=cmap_d, norm=norm_d)
            colours = np.array(self._cluster_colours())
            plt.xticks(np.arange(grid_size), fontsize=12)
            plt.yticks(np.arange(grid_size), fontsize=12)
            # plt.title("Macro State Clusters", fontsize=20)

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
        # plt.title('Micro States Spectral Embedding')
        plt.savefig(f'{save_dir}/macro_state_viz.png', bbox_inches='tight')
        plt.close()

    def _plot_spectral_embedding_augmented(self, save_dir: str):
        """Plot spectral embedding for augmented state spaces."""
        plt.figure(figsize=(10, 8))
        colours = self._cluster_colours()
        for i, states in enumerate(self.macro_state_list):
            if len(states) > 0:
                positions = self.spectral_positions[states]
                plt.scatter(positions[:, 0], positions[:, 1],
                           label=f'{i}', color=colours[i], s=75)
        plt.legend()
        # plt.title('Micro States Spectral Embedding (Augmented)')
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

        colours = self._cluster_colours()
        fig, ax = plt.subplots(figsize=(10, 8))

        # Compute minimum visible ellipse size from overall spectral spread
        all_idx = np.concatenate([s for s in self.macro_state_list if len(s) > 0])
        overall_var = np.var(self.spectral_positions[all_idx], axis=0)
        spread_floor = max(0.01 * np.mean(overall_var), 1e-6)

        # Scatter points by cluster
        for i, states in enumerate(self.macro_state_list):
            if len(states) == 0:
                continue
            positions = self.spectral_positions[states]
            ax.scatter(positions[:, 0], positions[:, 1],
                       label=f'{i}', color=colours[i], s=60,
                       alpha=0.7, zorder=3)

            # Fit 2σ covariance ellipse
            if len(positions) >= 3:
                mean = positions.mean(axis=0)
                cov = np.cov(positions, rowvar=False)
                # Eigendecomposition for ellipse orientation and axes
                eigvals, eigvecs = np.linalg.eigh(cov)
                # Floor at 1% of overall spread so degenerate clusters stay visible
                eigvals = np.maximum(eigvals, spread_floor)
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
        # ax.set_title('Spectral Embedding with Cluster Ellipses', fontsize=14)
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

            cmap_in, norm_in = self._cluster_cmap_and_norm()
            inset.imshow(labels_grid, cmap=cmap_in, norm=norm_in)
            inset.set_xticks([])
            inset.set_yticks([])

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

        # Use tab10 discrete colormap to match trajectory / macro action plots
        tab_cmap = plt.get_cmap("tab10")
        cluster_colors = [tab_cmap(i) for i in range(self.n_clusters)]
        unassigned_color = (0.85, 0.85, 0.85, 1.0)  # light gray
        all_colors = cluster_colors + [unassigned_color]
        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap = ListedColormap(all_colors)
        bounds = np.arange(-0.5, self.n_clusters + 1.5, 1)
        norm = BoundaryNorm(bounds, cmap.N)

        plt.figure()
        im = plt.imshow(
            labels_grid.T,
            aspect='auto',
            origin='lower',
            extent=extent,
            interpolation='nearest',
            cmap=cmap,
            norm=norm,
        )

        # Build legend from tab10 colours (exclude unassigned)
        patches = [mpatches.Patch(color=cluster_colors[i], label=f'{i}')
                   for i in range(self.n_clusters)]
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
        plt.savefig(f"{save_dir}/Macro_s.png", format="png", bbox_inches='tight')

        # ---- Overlay macro action arrows and save as separate figure ----
        if has_centers and hasattr(self, 'adj_list') and self.adj_list:
            # Compute cluster centroids in physical coordinates
            centroids = {}
            for c in range(self.n_clusters):
                member_indices = [s for s, m in self.micro_to_macro.items() if m == c]
                if not member_indices:
                    continue
                coords = np.array([self.adapter.state_space.index_to_state(s) for s in member_indices])
                # Map bin indices to physical centres
                cx = np.mean(centers0[coords[:, 0]])
                cy = np.mean(centers1[coords[:, 1]])
                centroids[c] = (cx, cy)

            # Determine goal macro states
            goal_macro_states = set()
            for gs in self.goal_states:
                if gs in self.micro_to_macro:
                    goal_macro_states.add(self.micro_to_macro[gs])

            # Compute best macro action per cluster
            V_macro = self.M_macro @ self.C_macro
            ax = plt.gca()
            for c in range(self.n_clusters):
                if c not in centroids or c in goal_macro_states:
                    continue
                if c not in self.adj_list or not self.adj_list[c]:
                    continue
                adj = self.adj_list[c]
                values = [V_macro[a] for a in adj]
                sorted_idx = np.argsort(values)[::-1]
                target = None
                for idx in sorted_idx:
                    if adj[idx] != c:
                        target = adj[idx]
                        break
                if target is None or target not in centroids:
                    continue
                x0, y0 = centroids[c]
                x1, y1 = centroids[target]
                dx, dy = x1 - x0, y1 - y0
                ax.annotate(
                    '', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle='->', lw=2.5,
                        color='black', shrinkA=8, shrinkB=8,
                    ),
                    zorder=10,
                )

            # Mark centroids with cluster number
            for c, (cx, cy) in centroids.items():
                star = '★' if c in goal_macro_states else ''
                ax.text(cx, cy, f'{c}{star}', fontsize=11, fontweight='bold',
                        ha='center', va='center',
                        color='white',
                        path_effects=[
                            PathEffects.withStroke(linewidth=3, foreground='black')
                        ],
                        zorder=11)

            plt.savefig(f"{save_dir}/Macro_s_actions.png", format="png", bbox_inches='tight')
            print(f"  Saved cluster+action heatmap to {save_dir}/Macro_s_actions.png")

        plt.close()
        print(f"  Saved cluster heatmap to {save_dir}/Macro_s.png")

        # Also plot spectral embedding if available
        if self.spectral_positions is not None:
            colours_arr = np.array(cluster_colors)
            self._plot_spectral_embedding(save_dir, colours_arr)

    # ==================== Composite Key Gridworld Figure ====================

    def visualize_key_gridworld_composite(self, save_path: str,
                                           init_loc: tuple, goal_loc: tuple,
                                           key_loc: tuple):
        """Generate a 2×2 composite figure for the Key Gridworld.

        The outer layout is a ``plt.subplots(2, 2)`` grid.  Panels (b) and (c)
        need to display two side-by-side heatmaps (one per key state), so their
        original axes are hidden and replaced by inner sub-axes created via
        ``fig.add_gridspec(2, 4)``.  This gridspec-inside-subplots approach is
        why ``plt.tight_layout`` emits a warning (suppressed here) — the mixed
        axes are not fully compatible with the tight-layout engine, but
        ``bbox_inches='tight'`` on ``savefig`` still produces correct output.

        Panels:
            (a) Grid layout — Agent / Key / Goal / walls
            (b) Split value function — 1×2 heatmaps per key state
            (c) Split cluster maps  — 1×2 heatmaps per key state
            (d) Spectral embedding with 2σ ellipses + inset thumbnail

        Args:
            save_path: Where to save the composite figure.
            init_loc: (x, y) agent start.
            goal_loc: (x, y) goal position.
            key_loc: (x, y) key position.
        """
        from matplotlib import colors as mcolors
        from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

        if not hasattr(self.adapter, 'grid_size'):
            print("Composite figure requires grid-based key gridworld adapter")
            return
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        grid_size = self.adapter.grid_size
        n_base = grid_size ** 2
        n_states = self.adapter.n_states
        is_augmented_c = n_states != n_base

        # Build the entire layout from a single GridSpec to avoid the
        # tight_layout incompatibility that arose from mixing subplots()
        # with fig.add_gridspec().
        fig = plt.figure(figsize=(14, 12))
        outer = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.4)

        # ---- (a) Grid layout ----
        ax_a = fig.add_subplot(outer[0, 0])
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

        # ---- (b) Split value function ----
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
                inner_b = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 1],
                                                 wspace=0.3)
                ax_b1 = fig.add_subplot(inner_b[0, 0])
                ax_b2 = fig.add_subplot(inner_b[0, 1])

                V_no = V[:n_base].copy()
                V_has = V[n_base:].copy()
                wmask_grid = wall_mask.reshape(grid_size, grid_size).T

                V_no_grid = np.ma.masked_where(wmask_grid, V_no.reshape(grid_size, grid_size).T)
                V_has_grid = np.ma.masked_where(wmask_grid, V_has.reshape(grid_size, grid_size).T)

                vmin = min(V_no_grid.min(), V_has_grid.min())
                vmax = max(V_no_grid.max(), V_has_grid.max())

                ax_b1.set_facecolor('white')
                ax_b1.imshow(V_no_grid, cmap='copper', vmin=vmin, vmax=vmax)
                ax_b1.set_title('✗ Without key', fontsize=11)
                ax_b1.set_xticks(np.arange(grid_size))
                ax_b1.set_yticks(np.arange(grid_size))

                ax_b2.set_facecolor('white')
                im_v = ax_b2.imshow(V_has_grid, cmap='copper', vmin=vmin, vmax=vmax)
                ax_b2.set_title('★ With key', fontsize=11)
                ax_b2.set_xticks(np.arange(grid_size))
                ax_b2.set_yticks(np.arange(grid_size))

                fig.colorbar(im_v, ax=[ax_b1, ax_b2], shrink=0.7, label='Value')
            else:
                ax_b = fig.add_subplot(outer[0, 1])
                V_grid = np.ma.masked_where(
                    wall_mask.reshape(grid_size, grid_size).T,
                    V.reshape(grid_size, grid_size).T,
                )
                ax_b.set_facecolor('white')
                ax_b.imshow(V_grid, cmap='viridis')
                ax_b.set_xticks(np.arange(grid_size))
                ax_b.set_yticks(np.arange(grid_size))
        else:
            ax_b = fig.add_subplot(outer[0, 1])
            ax_b.text(0.5, 0.5, 'No M/C', ha='center', va='center',
                      transform=ax_b.transAxes, fontsize=14)

        # ---- (c) Split cluster maps ----
        cluster_colours = self._cluster_colours()

        if is_augmented_c:
            inner_c = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1, 0],
                                             wspace=0.3)
            ax_c1 = fig.add_subplot(inner_c[0, 0])
            ax_c2 = fig.add_subplot(inner_c[0, 1])

            cmap_d, norm_d = self._cluster_cmap_and_norm()
            for aug_idx, (ax_sub, title) in enumerate(
                    [(ax_c1, '✗ Without key'), (ax_c2, '★ With key')]):
                aug_labels = np.ones(n_base) * self.n_clusters
                for micro_idx, macro_idx in self.micro_to_macro.items():
                    state = self.adapter.state_space.index_to_state(micro_idx)
                    if len(state) == 2:
                        base_idx, st_aug = state
                        if st_aug == aug_idx:
                            aug_labels[base_idx] = macro_idx
                labels_grid = aug_labels.reshape(grid_size, grid_size).T
                ax_sub.imshow(labels_grid, cmap=cmap_d, norm=norm_d)
                ax_sub.set_title(title, fontsize=11)
                ax_sub.set_xticks(np.arange(grid_size))
                ax_sub.set_yticks(np.arange(grid_size))

            # Cluster legend
            patches_c = [mpatches.Patch(color=cluster_colours[i], label=f'{i}')
                         for i in range(self.n_clusters)]
            fig.legend(handles=patches_c, loc='lower left',
                       bbox_to_anchor=(0.02, 0.02), fontsize=8, ncol=2)
        else:
            ax_c = fig.add_subplot(outer[1, 0])
            labels_arr = np.ones(n_states) * self.n_clusters
            for micro_idx, macro_idx in self.micro_to_macro.items():
                labels_arr[micro_idx] = macro_idx
            labels_grid = labels_arr.reshape(grid_size, grid_size).T
            cmap_d_c, norm_d_c = self._cluster_cmap_and_norm()
            ax_c.imshow(labels_grid, cmap=cmap_d_c, norm=norm_d_c)
            ax_c.set_xticks(np.arange(grid_size))
            ax_c.set_yticks(np.arange(grid_size))

        # ---- (d) Spectral embedding with ellipses ----
        ax_d = fig.add_subplot(outer[1, 1])
        if self.spectral_positions is not None:
            # Compute minimum visible ellipse size from overall spread
            all_idx = np.concatenate([s for s in self.macro_state_list if len(s) > 0])
            overall_var = np.var(self.spectral_positions[all_idx], axis=0)
            spread_floor = max(0.01 * np.mean(overall_var), 1e-6)

            for i, states in enumerate(self.macro_state_list):
                if len(states) == 0:
                    continue
                positions = self.spectral_positions[states]
                ax_d.scatter(positions[:, 0], positions[:, 1],
                             label=f'{i}', color=cluster_colours[i],
                             s=50, alpha=0.7, zorder=3)
                # 2σ covariance ellipse
                if len(positions) >= 3:
                    mean = positions.mean(axis=0)
                    cov = np.cov(positions, rowvar=False)
                    eigvals, eigvecs = np.linalg.eigh(cov)
                    eigvals = np.maximum(eigvals, spread_floor)
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
                cmap_in_d, norm_in_d = self._cluster_cmap_and_norm()
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
                    inset.imshow(aug_labels_in.reshape(grid_size, grid_size).T,
                                 cmap=cmap_in_d, norm=norm_in_d)
                else:
                    inset.imshow(labels_grid, cmap=cmap_in_d, norm=norm_in_d)
                inset.set_xticks([])
                inset.set_yticks([])
        else:
            ax_d.text(0.5, 0.5, 'No spectral data', ha='center', va='center',
                      transform=ax_d.transAxes, fontsize=14)

        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Composite figure saved to {save_path}")

    # ==================== Value / Policy / Overlay Plots ====================

    def _build_value_grid(self):
        """Build a 2D value grid for any environment type.

        Returns:
            dict with keys:
                V_2d       – 2D numpy array (dim0 × dim1)
                extent     – [x0, x1, y0, y1] for imshow (or None for grid envs)
                label0     – x-axis label string
                label1     – y-axis label string
                centers0   – 1D array of dim0 bin/cell centers (or None)
                centers1   – 1D array of dim1 bin/cell centers (or None)
                wall_mask  – 2D bool array for grid envs (or None)
                is_grid    – True for grid-based environments
                is_augmented – True for key-gridworld style augmented grids
                V_panels   – list of (V_2d, title) for augmented envs
                avg_note   – subtitle note if averaging was applied (or None)
        """
        V = self.adapter.multiply_M_C(self.M, self.C)
        result = {
            'wall_mask': None, 'is_grid': False, 'is_augmented': False,
            'V_panels': None, 'avg_note': None, 'centers0': None, 'centers1': None,
        }

        # --- Grid-based environments ---
        if hasattr(self.adapter, 'grid_size'):
            grid_size = self.adapter.grid_size
            n_base = grid_size ** 2
            is_augmented = V.shape[0] != n_base and V.shape[0] == 2 * n_base

            wall_mask_base = np.zeros(n_base, dtype=bool)
            if hasattr(self.adapter, 'get_wall_indices'):
                for wi in self.adapter.get_wall_indices():
                    if is_augmented:
                        base_idx, _ = self.adapter.state_space.index_to_state(wi)
                        wall_mask_base[base_idx] = True
                    else:
                        wall_mask_base[wi] = True

            result['is_grid'] = True
            result['label0'] = 'x'
            result['label1'] = 'y'
            result['extent'] = None

            if is_augmented:
                result['is_augmented'] = True
                V_no = V[:n_base].copy()
                V_has = V[n_base:].copy()
                wmask = wall_mask_base.reshape(grid_size, grid_size).T
                result['V_panels'] = [
                    (np.ma.masked_where(wmask, V_no.reshape(grid_size, grid_size).T),
                     '\u2717 Without key'),
                    (np.ma.masked_where(wmask, V_has.reshape(grid_size, grid_size).T),
                     '\u2605 With key'),
                ]
                result['wall_mask'] = wmask
                result['V_2d'] = result['V_panels'][0][0]
            else:
                wmask = wall_mask_base.reshape(grid_size, grid_size).T
                result['V_2d'] = np.ma.masked_where(wmask, V.reshape(grid_size, grid_size).T)
                result['wall_mask'] = wmask

            return result

        # --- Binned continuous environments ---
        bins = self.adapter.state_space.n_bins_per_dim
        n_dims = len(bins)

        if hasattr(self.adapter, 'get_dimension_labels'):
            label0, label1 = self.adapter.get_dimension_labels()
        else:
            label0, label1 = 'Dim 0', 'Dim 1'

        if n_dims == 2:
            # 2D (Mountain Car, Pendulum)
            V_2d = np.zeros((bins[0], bins[1]))
            for i in range(bins[0]):
                for j in range(bins[1]):
                    idx = self.adapter.state_space.state_to_index((i, j))
                    V_2d[i, j] = V[idx]

            centers = self.adapter.get_bin_centers()
            centers0, centers1 = centers[0], centers[1]

            if hasattr(self.adapter, 'get_bin_edges'):
                edges0, edges1 = self.adapter.get_bin_edges()
                extent = [edges0[0], edges0[-1], edges1[0], edges1[-1]]
            else:
                extent = [centers0[0], centers0[-1], centers1[0], centers1[-1]]

            result['V_2d'] = V_2d
            result['extent'] = extent
            result['centers0'] = centers0
            result['centers1'] = centers1

        elif n_dims == 4:
            # 4D (Acrobot, CartPole) — average over velocity dims
            V_4d = V.reshape(bins[0], bins[1], bins[2], bins[3])
            V_2d = np.mean(V_4d, axis=(1, 3))  # average over velocity dims

            centers = self.adapter.get_bin_centers()
            centers0, centers1 = centers[0], centers[2]

            extent = [centers0[0], centers0[-1], centers1[0], centers1[-1]]
            result['V_2d'] = V_2d
            result['extent'] = extent
            result['centers0'] = centers0
            result['centers1'] = centers1
            result['avg_note'] = '(averaged over velocities)'
        else:
            print(f"  Value plot not supported for {n_dims}D state spaces")
            return None

        result['label0'] = label0
        result['label1'] = label1
        return result

    def _build_policy_grid(self):
        """Build a 2D greedy policy grid for any environment type.

        Returns:
            dict with keys:
                policy_2d  – 2D numpy int array (dim0 × dim1) of greedy action indices
                n_actions  – number of actions
                (plus same geometry keys as _build_value_grid)
        """
        V = self.adapter.multiply_M_C(self.M, self.C)
        n_states = self.adapter.n_states
        n_actions = self.adapter.n_actions
        eye = np.eye(n_states)

        # --- Grid-based environments ---
        if hasattr(self.adapter, 'grid_size'):
            grid_size = self.adapter.grid_size
            n_base = grid_size ** 2
            is_augmented = n_states != n_base

            if is_augmented:
                # For augmented grids, compute policy for base grid (has_key=0)
                policy_2d = np.full((grid_size, grid_size), -1, dtype=int)
                # Collect wall base indices
                wall_base = set()
                if hasattr(self.adapter, 'get_wall_indices'):
                    for wi in self.adapter.get_wall_indices():
                        base_idx, _ = self.adapter.state_space.index_to_state(wi)
                        wall_base.add(base_idx)
                for s in range(n_base):
                    if s in wall_base:
                        continue
                    # Proper (N, 2) one-hot for augmented state (s, has_key=0)
                    flat_idx = self.adapter.state_space.state_to_index((s, 0))
                    s_onehot = self.adapter.state_space.index_to_onehot(flat_idx)
                    action_vals = []
                    for a in range(n_actions):
                        next_dist = self.adapter.multiply_B_s(self.B, s_onehot, a)
                        # Flatten in Fortran order to match multiply_M_C output
                        action_vals.append(next_dist.flatten('F') @ V)
                    policy_2d[s // grid_size, s % grid_size] = int(np.argmax(action_vals))
                policy_2d = policy_2d.T
            else:
                policy_2d = np.full((grid_size, grid_size), -1, dtype=int)
                wall_indices = set()
                if hasattr(self.adapter, 'get_wall_indices'):
                    wall_indices = set(self.adapter.get_wall_indices())
                for s in range(n_base):
                    if s in wall_indices:
                        continue
                    action_vals = []
                    for a in range(n_actions):
                        next_dist = self.adapter.multiply_B_s(self.B, eye[s], a)
                        action_vals.append(next_dist @ V)
                    x, y = divmod(s, grid_size)
                    policy_2d[x, y] = int(np.argmax(action_vals))
                policy_2d = policy_2d.T

            return {'policy_2d': policy_2d, 'n_actions': n_actions}

        # --- Binned continuous environments ---
        bins = self.adapter.state_space.n_bins_per_dim
        n_dims = len(bins)

        if n_dims == 2:
            policy_2d = np.full((bins[0], bins[1]), -1, dtype=int)
            for i in range(bins[0]):
                for j in range(bins[1]):
                    idx = self.adapter.state_space.state_to_index((i, j))
                    action_vals = []
                    for a in range(n_actions):
                        next_dist = self.adapter.multiply_B_s(self.B, eye[idx], a)
                        action_vals.append(next_dist @ V)
                    policy_2d[i, j] = int(np.argmax(action_vals))

        elif n_dims == 4:
            # For 4D, pick dominant action per (dim0, dim2) cell
            # by majority vote across velocity dims
            from collections import Counter
            policy_2d = np.full((bins[0], bins[2]), -1, dtype=int)
            for i in range(bins[0]):
                for k in range(bins[2]):
                    votes = []
                    for j in range(bins[1]):
                        for l in range(bins[3]):
                            idx = self.adapter.state_space.state_to_index((i, j, k, l))
                            action_vals = []
                            for a in range(n_actions):
                                next_dist = self.adapter.multiply_B_s(self.B, eye[idx], a)
                                action_vals.append(next_dist @ V)
                            votes.append(int(np.argmax(action_vals)))
                    policy_2d[i, k] = Counter(votes).most_common(1)[0][0]
        else:
            return None

        return {'policy_2d': policy_2d, 'n_actions': n_actions}

    @staticmethod
    def _pixel_centers(extent, n0, n1):
        """Compute imshow pixel centers for each axis.

        When ``imshow`` renders an array with a given ``extent``, pixels
        are distributed **uniformly** across the extent regardless of
        whether the underlying bins have uniform width.  Arrow and tick
        positions must therefore use these pixel centres — not the
        physical bin centres — to stay aligned with the heatmap cells.
        """
        px0 = np.array([extent[0] + (i + 0.5) * (extent[1] - extent[0]) / n0
                         for i in range(n0)])
        px1 = np.array([extent[2] + (j + 0.5) * (extent[3] - extent[2]) / n1
                         for j in range(n1)])
        return px0, px1

    @staticmethod
    def _value_norm(data):
        """Return a PowerNorm that compresses dynamic range for value heatmaps.

        V values typically span several orders of magnitude (e.g. -0.1 to 9000)
        so a linear colourscale makes everything look black.  PowerNorm(gamma=0.4)
        stretches the low end and compresses the high end, revealing structure.

        Returns:
            (norm, vmin, vmax) tuple.  *norm* is None if the data is constant.
        """
        arr = np.ma.asarray(data) if isinstance(data, np.ma.MaskedArray) else np.asarray(data)
        vals = arr.compressed() if isinstance(arr, np.ma.MaskedArray) and arr.mask is not np.ma.nomask else arr.ravel()
        if vals.size == 0:
            return None, 0, 1
        vmin = float(max(0.0, np.nanmin(vals)))
        vmax = float(np.nanmax(vals))
        if vmax <= vmin:
            return None, vmin, vmin + 1
        return PowerNorm(gamma=0.4, vmin=vmin, vmax=vmax), vmin, vmax

    def _draw_landmarks(self, ax, aug_idx=None):
        """Draw goal and key markers on a grid-based plot.

        Args:
            ax: Matplotlib axes (imshow with grid coordinates).
            aug_idx: For augmented panels, which panel this is
                     (0 = without key, 1 = with key).  If None, always draw.
        """
        # Goal marker — red star
        if hasattr(self, 'goal_states') and self.goal_states:
            for gs_idx in self.goal_states:
                rendered = self.adapter.render_state(gs_idx)
                if len(rendered) == 3:
                    gx, gy, g_key = rendered
                    # In augmented mode, only show on the matching panel
                    if aug_idx is not None and g_key != aug_idx:
                        continue
                elif len(rendered) == 2:
                    gx, gy = rendered
                else:
                    continue
                ax.plot(gx, gy, marker='*', color='red', markersize=18,
                        markeredgecolor='darkred', markeredgewidth=0.8,
                        zorder=10)

        # Key marker — cyan diamond (only for key gridworld)
        key_loc = getattr(getattr(self.adapter, 'env', None), 'key_loc', None)
        if key_loc is not None:
            kx, ky = key_loc
            # Key only exists when agent doesn't have it (aug_idx=0)
            if aug_idx is None or aug_idx == 0:
                ax.plot(kx, ky, marker='D', color='cyan', markersize=14,
                        markeredgecolor='darkblue', markeredgewidth=1.0,
                        zorder=10)

    def _draw_value_heatmap(self, ax, vg, fig=None):
        """Draw value function heatmap on the given axes.

        Uses PowerNorm(gamma=0.4) to compress the wide dynamic range typical
        of SR value functions, making spatial gradients visible.

        Args:
            ax: Matplotlib axes
            vg: dict from _build_value_grid()
            fig: Figure (needed for colorbar placement)
        Returns:
            The imshow image object
        """
        norm, vmin, vmax = self._value_norm(vg['V_2d'])

        if vg['is_grid']:
            ax.set_facecolor('white')
            im = ax.imshow(vg['V_2d'], cmap='copper', norm=norm)
            if hasattr(self.adapter, 'grid_size'):
                gs = self.adapter.grid_size
                ax.set_xticks(np.arange(gs))
                ax.set_yticks(np.arange(gs))
        else:
            im = ax.imshow(
                vg['V_2d'].T, origin='lower', aspect='auto',
                extent=vg['extent'], cmap='copper', norm=norm,
            )
            # Subsampled tick labels — placed at *pixel* centres so they
            # align with the heatmap cells (important for non-uniform bins).
            if vg['centers0'] is not None:
                n0 = len(vg['centers0'])
                n1 = len(vg['centers1'])
                px0, px1 = self._pixel_centers(vg['extent'], n0, n1)

                max_ticks = 7
                step0 = max(1, n0 // max_ticks)
                step1 = max(1, n1 // max_ticks)
                idx0 = np.arange(0, n0, step0)
                idx1 = np.arange(0, n1, step1)
                ax.set_xticks(px0[idx0])
                ax.set_xticklabels(np.round(vg['centers0'][idx0], 3))
                ax.set_yticks(px1[idx1])
                ax.set_yticklabels(np.round(vg['centers1'][idx1], 3))

        ax.set_xlabel(vg['label0'], fontsize=12)
        ax.set_ylabel(vg['label1'], fontsize=12)
        return im

    def _draw_policy_arrows(self, ax, vg, pg):
        """Draw policy arrows on the given axes.

        For continuous environments the arrows are placed at the **imshow
        pixel centres** (not physical bin centres) so they stay aligned
        with the heatmap cells even when bins have non-uniform widths
        (e.g. Mountain Car position bins).

        Args:
            ax: Matplotlib axes
            vg: dict from _build_value_grid() (for geometry)
            pg: dict from _build_policy_grid()
        """
        policy_2d = pg['policy_2d']
        n_actions = pg['n_actions']
        rows, cols = policy_2d.shape

        if vg['is_grid']:
            # Grid: black arrows with white stroke for visibility on any bg
            grid_arrows = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
            for r in range(rows):
                for c in range(cols):
                    a = policy_2d[r, c]
                    if a < 0 or a not in grid_arrows:
                        continue
                    dx, dy = grid_arrows[a]
                    scale = 0.3
                    arrow = ax.arrow(c - scale * dx, r - scale * dy,
                                     scale * dx, scale * dy,
                                     head_width=0.18, color='black',
                                     length_includes_head=True, zorder=5)
                    arrow.set_path_effects([
                        PathEffects.withStroke(linewidth=3, foreground='white'),
                    ])
        else:
            # Continuous: colored arrows placed at pixel centres
            extent = vg['extent']
            if extent is None:
                return
            px0, px1 = self._pixel_centers(extent, rows, cols)

            cell_w = (extent[1] - extent[0]) / rows
            arrow_scale = cell_w * 0.35

            if n_actions == 2:
                # CartPole: 0=left, 1=right
                colors = {0: 'green', 1: 'red'}
                dirs = {0: -1, 1: 1}
                for i in range(rows):
                    for j in range(cols):
                        cx, cy = px0[i], px1[j]
                        a = policy_2d[i, j]
                        if a < 0:
                            continue
                        d = dirs[a]
                        ax.annotate('', xy=(cx + d * arrow_scale, cy),
                                    xytext=(cx - d * arrow_scale * 0.3, cy),
                                    arrowprops=dict(arrowstyle='->', color=colors[a],
                                                    lw=1.5, mutation_scale=12))

            elif n_actions == 3:
                # Mountain Car / Acrobot: 0=left, 1=neutral, 2=right
                for i in range(rows):
                    for j in range(cols):
                        cx, cy = px0[i], px1[j]
                        a = policy_2d[i, j]
                        if a < 0:
                            continue
                        if a == 0:
                            ax.annotate('', xy=(cx - arrow_scale, cy),
                                        xytext=(cx + arrow_scale * 0.3, cy),
                                        arrowprops=dict(arrowstyle='->', color='green',
                                                        lw=1.5, mutation_scale=12))
                        elif a == 2:
                            ax.annotate('', xy=(cx + arrow_scale, cy),
                                        xytext=(cx - arrow_scale * 0.3, cy),
                                        arrowprops=dict(arrowstyle='->', color='red',
                                                        lw=1.5, mutation_scale=12))
                        else:
                            ax.plot(cx, cy, 'o', color='white', markersize=3)

            else:
                # Pendulum (N torque bins): arrow direction from action index
                mid = n_actions // 2
                for i in range(rows):
                    for j in range(cols):
                        cx, cy = px0[i], px1[j]
                        a = policy_2d[i, j]
                        if a < 0:
                            continue
                        if a < mid:
                            # Negative torque → left (green)
                            strength = (mid - a) / mid
                            ax.annotate('', xy=(cx - arrow_scale * strength, cy),
                                        xytext=(cx + arrow_scale * 0.2, cy),
                                        arrowprops=dict(arrowstyle='->', color='green',
                                                        lw=1.5, mutation_scale=12))
                        elif a > mid:
                            # Positive torque → right (red)
                            strength = (a - mid) / mid
                            ax.annotate('', xy=(cx + arrow_scale * strength, cy),
                                        xytext=(cx - arrow_scale * 0.2, cy),
                                        arrowprops=dict(arrowstyle='->', color='red',
                                                        lw=1.5, mutation_scale=12))
                        else:
                            ax.plot(cx, cy, 'o', color='white', markersize=3)

    # ---- Public API ----

    def plot_value_function(self, save_path: str = None):
        """Plot value function V = M·C as a 2D heatmap with colorbar.

        Works for all environment types: grid, 2D binned, and 4D binned.
        For 4D environments (Acrobot, CartPole), averages over velocity dims.

        Args:
            save_path: Path to save the figure.
        """
        if save_path is None:
            raise ValueError("save_path is required")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if self.M is None or self.C is None:
            print("  No M or C to visualize")
            return

        vg = self._build_value_grid()
        if vg is None:
            return

        # Augmented grid: multi-panel
        if vg.get('is_augmented') and vg.get('V_panels'):
            panels = vg['V_panels']
            # Shared PowerNorm across panels for consistent colour mapping
            all_vals = np.concatenate([p[0].ravel() for p in panels])
            norm, _, _ = self._value_norm(all_vals)

            fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5))
            if len(panels) == 1:
                axes = [axes]
            for idx, (ax, (V_panel, title)) in enumerate(zip(axes, panels)):
                ax.set_facecolor('white')
                im = ax.imshow(V_panel, cmap='copper', norm=norm)
                ax.set_title(title, fontsize=12)
                if hasattr(self.adapter, 'grid_size'):
                    gs = self.adapter.grid_size
                    ax.set_xticks(np.arange(gs))
                    ax.set_yticks(np.arange(gs))
                self._draw_landmarks(ax, aug_idx=idx)
            fig.colorbar(im, ax=list(axes), shrink=0.8, label='Value')
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            im = self._draw_value_heatmap(ax, vg, fig)
            self._draw_landmarks(ax)
            fig.colorbar(im, ax=ax, label='Value (V = M\u00b7C)')
            if vg['avg_note']:
                ax.set_title(vg['avg_note'], fontsize=10, style='italic')

        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  Value function saved to {save_path}")

    def plot_policy(self, save_path: str = None):
        """Plot greedy policy as arrows on a white/light background.

        Works for all environment types.

        Args:
            save_path: Path to save the figure.
        """
        if save_path is None:
            raise ValueError("save_path is required")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if self.M is None or self.C is None or self.B is None:
            print("  No M, C, or B to compute policy")
            return

        vg = self._build_value_grid()
        pg = self._build_policy_grid()
        if vg is None or pg is None:
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        if vg['is_grid']:
            # White background with light grid
            gs = self.adapter.grid_size
            bg = np.ones((gs, gs)) * 0.95
            if vg['wall_mask'] is not None:
                bg[vg['wall_mask']] = 0.3
            ax.imshow(bg, cmap='gray', vmin=0, vmax=1)
            ax.set_xticks(np.arange(gs))
            ax.set_yticks(np.arange(gs))
            ax.set_xticks(np.arange(-0.5, gs, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, gs, 1), minor=True)
            ax.grid(which='minor', color='lightgray', linewidth=0.5)
            ax.tick_params(which='minor', bottom=False, left=False)
        else:
            # Light background for continuous envs
            rows, cols = pg['policy_2d'].shape
            bg = np.ones((rows, cols)) * 0.95
            ax.imshow(bg.T, origin='lower', aspect='auto',
                      extent=vg['extent'], cmap='gray', vmin=0, vmax=1)
            if vg['centers0'] is not None:
                n0 = len(vg['centers0'])
                n1 = len(vg['centers1'])
                px0, px1 = self._pixel_centers(vg['extent'], n0, n1)

                max_ticks = 7
                step0 = max(1, n0 // max_ticks)
                step1 = max(1, n1 // max_ticks)
                idx0 = np.arange(0, n0, step0)
                idx1 = np.arange(0, n1, step1)
                ax.set_xticks(px0[idx0])
                ax.set_xticklabels(np.round(vg['centers0'][idx0], 3))
                ax.set_yticks(px1[idx1])
                ax.set_yticklabels(np.round(vg['centers1'][idx1], 3))

        self._draw_policy_arrows(ax, vg, pg)
        self._draw_landmarks(ax)
        ax.set_xlabel(vg['label0'], fontsize=12)
        ax.set_ylabel(vg['label1'], fontsize=12)

        if vg.get('avg_note'):
            ax.set_title(vg['avg_note'], fontsize=10, style='italic')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  Policy plot saved to {save_path}")

    def plot_value_with_policy(self, save_path: str = None):
        """Plot value function heatmap with greedy policy arrows overlaid.

        Combines the value heatmap and policy arrows into a single figure.

        Args:
            save_path: Path to save the figure.
        """
        if save_path is None:
            raise ValueError("save_path is required")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if self.M is None or self.C is None or self.B is None:
            print("  No M, C, or B to visualize")
            return

        vg = self._build_value_grid()
        pg = self._build_policy_grid()
        if vg is None or pg is None:
            return

        fig, ax = plt.subplots(figsize=(8, 6))
        im = self._draw_value_heatmap(ax, vg, fig)
        self._draw_policy_arrows(ax, vg, pg)
        self._draw_landmarks(ax)
        fig.colorbar(im, ax=ax, label='Value (V = M\u00b7C)')

        if vg.get('avg_note'):
            ax.set_title(vg['avg_note'], fontsize=10, style='italic')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  Value + policy overlay saved to {save_path}")
