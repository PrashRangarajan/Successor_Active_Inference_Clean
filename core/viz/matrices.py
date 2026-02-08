"""Matrix and clustering visualization methods.

This module provides plotting capabilities for transition matrices,
successor matrices, and macro state cluster visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
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

        # Custom tick labels showing physical bin centers
        if has_centers:
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
            plt.xticks(ticks=tick_pos0, labels=np.round(centers0, 3), fontsize=10)
            plt.yticks(ticks=tick_pos1, labels=np.round(centers1, 3), fontsize=10)

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
