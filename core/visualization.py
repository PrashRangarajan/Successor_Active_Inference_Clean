"""Visualization methods for Hierarchical SR Agent.

This module provides plotting and video generation capabilities for
visualizing the agent's learned representations and behavior.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib import colors
from matplotlib.colors import LogNorm
from typing import Optional, List, Tuple, Dict, Any


class VisualizationMixin:
    """Mixin class providing visualization methods for HierarchicalSRAgent.

    This mixin adds plotting capabilities for:
    - Transition and successor matrices (B, M)
    - Macro state clusters
    - Action trajectories
    - Policy visualization
    - Episode videos

    Requires the agent to have:
    - self.adapter: Environment adapter
    - self.B, self.M: Transition and successor matrices
    - self.B_macro, self.M_macro: Macro-level matrices
    - self.macro_state_list, self.micro_to_macro: Clustering results
    - self.state_history, self.action_history: Episode tracking
    - self.C, self.goal_states: Goal information
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional attributes for visualization
        self.labels_grid = None  # Grid of macro state labels
        self.spectral_positions = None  # Spectral embedding positions

    # ==================== Matrix Visualization ====================

    def view_matrices(self, save_dir: str = "figures/matrices", learned: bool = True):
        """Visualize transition and successor matrices.

        Args:
            save_dir: Directory to save figures
            learned: Whether matrices were learned (vs analytical)
        """
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
                                    f"{save_dir}/m_origin_{learn_str}.png")

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

    def _plot_M_from_origin(self, M: np.ndarray, n_states: int, save_path: str):
        """Plot successor representation from origin state.

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

        if M.ndim == 4:
            # Augmented state space (e.g., key gridworld)
            # M shape: (N, 2, N, 2) where 2 is n_augment (has_key in {0, 1})
            n_augment = M.shape[1]

            fig, axes = plt.subplots(n_augment, n_augment, figsize=(12, 10))
            fig.suptitle('Successor Representation from Origin (0,0)', fontsize=16)

            aug_labels = ["No Key", "Has Key"] if n_augment == 2 else [f"Aug {i}" for i in range(n_augment)]

            # Find global max for consistent color scaling
            global_max = max(M[0, i, :, j].max() for i in range(n_augment) for j in range(n_augment))
            global_max = global_max if global_max > 0 else 1

            for origin_aug in range(n_augment):
                for target_aug in range(n_augment):
                    ax = axes[origin_aug, target_aug] if n_augment > 1 else axes

                    # M[origin_loc, origin_aug, target_loc, target_aug]
                    # Get M from origin (0,0) with origin_aug to all locations with target_aug
                    M_slice = M[0, origin_aug, :, target_aug].reshape(grid_size, grid_size).T

                    im = ax.imshow(M_slice, aspect='equal', cmap='copper',
                                   norm=LogNorm(vmin=0.01, vmax=global_max))
                    ax.set_title(f'{aug_labels[origin_aug]} → {aug_labels[target_aug]}', fontsize=12)
                    ax.set_xticks(np.arange(grid_size))
                    ax.set_yticks(np.arange(grid_size))

            # Add colorbar
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
            plt.tight_layout()
            plt.savefig(save_path, format="png", bbox_inches='tight')
            plt.close()
        else:
            # Standard state space
            M_orig = M[0, :].reshape(grid_size, grid_size).T

            plt.figure(figsize=(8, 8))
            vmax = M_orig.max() if M_orig.max() > 0 else 1
            plt.imshow(M_orig, aspect='equal', cmap='copper',
                       norm=LogNorm(vmin=0.01, vmax=vmax))
            plt.colorbar()
            plt.title('Successor Representation from Origin (0,0)', fontsize=14)
            plt.xticks(np.arange(grid_size), fontsize=12)
            plt.yticks(np.arange(grid_size), fontsize=12)
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

    def visualize_clusters(self, save_dir: str = "figures/clustering"):
        """Visualize macro state clusters.

        Args:
            save_dir: Directory to save figures
        """
        os.makedirs(save_dir, exist_ok=True)

        if not hasattr(self.adapter, 'grid_size'):
            print("Cluster visualization requires grid-based environment")
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

    # ==================== Trajectory Visualization ====================

    def show_actions(self, save_path: str = "figures/Actions_taken.png",
                     init_loc: Tuple[int, int] = None,
                     goal_loc: Tuple[int, int] = None):
        """Visualize actions taken during an episode.

        Args:
            save_path: Path to save the figure
            init_loc: Starting location (default: (0,0))
            goal_loc: Goal location (default: bottom-right corner)
        """
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

        # Get wall locations
        walls = self.adapter.get_wall_indices() if hasattr(self.adapter, 'get_wall_indices') else []
        wall_locs = [self.adapter.state_space.index_to_state(w) for w in walls] if walls else []

        # Create grid
        grid = np.zeros((grid_size, grid_size))
        grid[init_loc] = 1
        if wall_locs:
            for w in wall_locs:
                if len(w) == 2:
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

            loc = self.adapter.state_space.index_to_state(idx)
            if len(loc) == 2:
                arrows_grid[loc] = action

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

    def visualize_policy(self, save_dir: str = "figures/Macro Action Network"):
        """Visualize macro action policies.

        For each macro state transition, shows the micro-level actions
        that would be taken to reach the bottleneck state.

        For augmented state spaces (e.g., key gridworld), creates separate
        panels for each augment value (No Key / Has Key).

        Args:
            save_dir: Directory to save figures
        """
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
                            fontweight='bold')

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

        aug_labels = ["No Key", "Has Key"] if n_augment == 2 else [f"Aug {i}" for i in range(n_augment)]

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
                                fontweight='bold')

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

    def show_video(self, save_path: str = "videos/env_micro.mp4",
                   init_loc: Tuple[int, int] = None,
                   goal_loc: Tuple[int, int] = None):
        """Generate video of episode trajectory.

        Args:
            save_path: Path to save the video
            init_loc: Starting location
            goal_loc: Goal location
        """
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

        # Get wall locations
        walls = self.adapter.get_wall_indices() if hasattr(self.adapter, 'get_wall_indices') else []
        wall_locs = []
        for w in walls:
            loc = self.adapter.state_space.index_to_state(w)
            if len(loc) == 2:
                wall_locs.append(loc)

        # Convert state history to locations
        state_locs = []
        for state in self.state_history:
            if hasattr(self.adapter, 'onehot_to_index'):
                idx = self.adapter.onehot_to_index(state)
            else:
                idx = np.argmax(state.flatten())
            loc = self.adapter.state_space.index_to_state(idx)
            if len(loc) >= 2:
                state_locs.append((loc[0], loc[1]))

        # Setup figure
        fig = plt.figure(figsize=(8, 8))
        grid = np.zeros((grid_size, grid_size))

        if wall_locs:
            wall_idx = tuple(np.array(wall_locs).T)
            grid[wall_idx] = 0.25

        grid[init_loc] = 1
        grid[goal_loc] = 0.5

        im = plt.imshow(grid.T, aspect='equal', cmap='magma')
        plt.title(f'Gridworld Episode')

        ax = plt.gca()
        ax.set_xticks(np.arange(0, grid_size, 1))
        ax.set_yticks(np.arange(0, grid_size, 1))
        ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        ax.tick_params(which='minor', bottom=False, left=False)

        past_idx_array = []

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
            for txt in ax.texts[::-1]:
                txt.remove()

            grid = np.zeros((grid_size, grid_size))
            if wall_locs:
                for w in wall_locs:
                    grid[w] = 0.25

            s_idx = state_locs[i]
            grid[s_idx] = 1
            grid[goal_loc] = 0.5
            im.set_data(grid.T)

            past_idx_array.append(s_idx)

            ax.text(s_idx[0], s_idx[1], 'Agent', fontsize=10,
                    ha="center", va="center", color="b")
            ax.text(goal_loc[0], goal_loc[1], 'Goal', fontsize=10,
                    ha="center", va="center", color="w")

            for w in wall_locs:
                ax.text(w[0], w[1], 'Wall', fontsize=8,
                        ha="center", va="center", color="w")

            if i > 0:
                ax.scatter(past_idx_array[i-1][0], past_idx_array[i-1][1], color='y')

            if i == len(state_locs) - 1:
                # Save final frame
                traj_path = save_path.replace('.mp4', '_trajectory.png')
                plt.savefig(traj_path)

            return im,

        ani = animation.FuncAnimation(fig, animate, np.arange(len(state_locs)),
                                      init_func=init, interval=500, blit=True)
        ani.save(save_path)
        plt.close()
        print(f"Video saved to {save_path}")

    # ==================== Value Function Visualization ====================

    def visualize_value_function(self, save_path: str = "figures/value_function.png"):
        """Visualize the value function on the grid.

        Args:
            save_path: Path to save the figure
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if not hasattr(self.adapter, 'grid_size'):
            print("Value function visualization requires grid-based environment")
            return

        if self.M is None or self.C is None:
            print("No learned M or C to visualize")
            return

        grid_size = self.adapter.grid_size
        V = self.adapter.multiply_M_C(self.M, self.C)

        # Handle augmented state space - take max over key states
        if V.shape[0] != grid_size ** 2:
            # Probably augmented - reshape and take max
            n_base = grid_size ** 2
            if V.shape[0] == 2 * n_base:
                V = np.maximum(V[:n_base], V[n_base:])

        V_grid = V.reshape(grid_size, grid_size).T

        plt.figure(figsize=(10, 8))
        plt.imshow(V_grid, cmap='viridis')
        plt.colorbar(label='Value')
        plt.title('Value Function V = M @ C', fontsize=16)
        plt.xticks(np.arange(grid_size))
        plt.yticks(np.arange(grid_size))

        # Mark goal
        if self.goal_states:
            for gs in self.goal_states:
                loc = self.adapter.state_space.index_to_state(gs)
                if len(loc) >= 2:
                    plt.scatter(loc[0], loc[1], color='red', s=200, marker='*',
                               label='Goal' if gs == self.goal_states[0] else '')

        plt.legend()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Value function saved to {save_path}")

    # ==================== POMDP Visualization ====================

    def _is_pomdp(self) -> bool:
        """Check if the adapter is a POMDP adapter with observation model."""
        return hasattr(self.adapter, 'observation_model')

    def visualize_observation_entropy(self, save_path: str = "figures/pomdp/observation_entropy.png"):
        """Visualize per-state observation entropy on the grid.

        High entropy states have noisier observations (harder to localize).

        Args:
            save_path: Path to save the figure
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if not self._is_pomdp():
            print("Observation entropy visualization requires POMDP adapter")
            return
        if not hasattr(self.adapter, 'grid_size'):
            print("Observation entropy visualization requires grid-based environment")
            return

        grid_size = self.adapter.grid_size
        entropy_vals = self.adapter.get_observation_entropy()
        entropy_grid = entropy_vals.reshape(grid_size, grid_size).T

        # Mask walls
        walls = self.adapter.get_wall_indices() if hasattr(self.adapter, 'get_wall_indices') else []

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(entropy_grid, cmap='hot', interpolation='nearest')

        # Mark walls
        for w in walls:
            loc = self.adapter.state_space.index_to_state(w)
            if len(loc) >= 2:
                ax.add_patch(plt.Rectangle((loc[0] - 0.5, loc[1] - 0.5), 1, 1,
                                           facecolor='black', edgecolor='white', linewidth=1))

        ax.set_xticks(np.arange(grid_size))
        ax.set_yticks(np.arange(grid_size))
        ax.set_title("Observation Entropy (Higher = Noisier)", fontsize=16)
        plt.colorbar(im, ax=ax, label="Entropy (nats)")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Observation entropy saved to {save_path}")

    def visualize_observation_model(self, save_dir: str = "figures/pomdp",
                                     states_to_show: Optional[List[int]] = None):
        """Visualize P(obs|state) for selected states.

        Shows how observations are distributed given the true state,
        revealing which states have clean vs noisy observation profiles.

        Args:
            save_dir: Directory to save figures
            states_to_show: List of state indices to visualize.
                If None, auto-selects representative states.
        """
        os.makedirs(save_dir, exist_ok=True)

        if not self._is_pomdp():
            print("Observation model visualization requires POMDP adapter")
            return
        if not hasattr(self.adapter, 'grid_size'):
            print("Observation model visualization requires grid-based environment")
            return

        grid_size = self.adapter.grid_size
        A = self.adapter.A
        entropy_vals = self.adapter.get_observation_entropy()

        if states_to_show is None:
            # Auto-select: corner, highest entropy, lowest entropy, goal, center
            candidates = []
            candidates.append(0)  # top-left corner
            candidates.append(np.argmax(entropy_vals))  # noisiest
            candidates.append(np.argmin(entropy_vals))  # cleanest
            candidates.append(grid_size * (grid_size // 2) + grid_size // 2)  # center
            if self.goal_states:
                candidates.append(self.goal_states[0])
            candidates.append(grid_size * grid_size - 1)  # bottom-right corner
            # Deduplicate while preserving order
            seen = set()
            states_to_show = []
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    states_to_show.append(c)
            states_to_show = states_to_show[:6]

        n_states = len(states_to_show)
        n_cols = min(3, n_states)
        n_rows = (n_states + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
        if n_states == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        for i, state_idx in enumerate(states_to_show):
            row, col = divmod(i, n_cols)
            ax = axes[row, col]

            obs_dist = A[:, state_idx].reshape(grid_size, grid_size).T
            im = ax.imshow(obs_dist, cmap='Blues', interpolation='nearest')

            # Mark true state position
            state_loc = self.adapter.render_state(state_idx)
            ax.scatter(state_loc[0], state_loc[1], color='red', s=100,
                      marker='o', zorder=5, edgecolors='white', linewidths=1.5)

            ax.set_title(f"P(obs | state={state_loc})\nH={entropy_vals[state_idx]:.3f}",
                        fontsize=11)
            plt.colorbar(im, ax=ax, shrink=0.8)

        # Hide unused axes
        for i in range(n_states, n_rows * n_cols):
            row, col = divmod(i, n_cols)
            axes[row, col].set_visible(False)

        fig.suptitle("Observation Model: P(observation | true state)", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/observation_model.png", bbox_inches='tight')
        plt.close()
        print(f"Observation model saved to {save_dir}/observation_model.png")

    def visualize_noise_zones(self, save_path: str = "figures/pomdp/noise_zones.png",
                               init_loc: Optional[Tuple[int, int]] = None,
                               goal_loc: Optional[Tuple[int, int]] = None):
        """Visualize POMDP environment overview with noise zones highlighted.

        Shows entropy heatmap with noisy zones outlined, goal, start, and walls.

        Args:
            save_path: Path to save the figure
            init_loc: Agent start location (default: (0,0))
            goal_loc: Goal location (default: bottom-right corner)
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if not self._is_pomdp():
            print("Noise zone visualization requires POMDP adapter")
            return
        if not hasattr(self.adapter, 'grid_size'):
            print("Noise zone visualization requires grid-based environment")
            return

        grid_size = self.adapter.grid_size
        entropy_vals = self.adapter.get_observation_entropy()
        entropy_grid = entropy_vals.reshape(grid_size, grid_size).T

        if init_loc is None:
            init_loc = (0, 0)
        if goal_loc is None:
            if self.goal_states:
                goal_loc = self.adapter.state_space.index_to_state(self.goal_states[0])
            else:
                goal_loc = (grid_size - 1, grid_size - 1)

        walls = self.adapter.get_wall_indices() if hasattr(self.adapter, 'get_wall_indices') else []

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(entropy_grid, cmap='YlOrRd', interpolation='nearest', alpha=0.85)

        # Outline noisy zones (entropy > mean + 1 std)
        threshold = entropy_vals.mean() + entropy_vals.std()
        for idx in range(len(entropy_vals)):
            if entropy_vals[idx] > threshold:
                loc = self.adapter.state_space.index_to_state(idx)
                if len(loc) >= 2:
                    ax.add_patch(plt.Rectangle((loc[0] - 0.5, loc[1] - 0.5), 1, 1,
                                               facecolor='none', edgecolor='red',
                                               linewidth=2.5, linestyle='--'))

        # Mark walls
        for w in walls:
            loc = self.adapter.state_space.index_to_state(w)
            if len(loc) >= 2:
                ax.add_patch(plt.Rectangle((loc[0] - 0.5, loc[1] - 0.5), 1, 1,
                                           facecolor='black', edgecolor='white', linewidth=1))

        # Mark start and goal
        ax.scatter(init_loc[0], init_loc[1], color='blue', s=250, marker='o',
                  zorder=5, edgecolors='white', linewidths=2, label='Start')
        ax.scatter(goal_loc[0], goal_loc[1], color='lime', s=350, marker='*',
                  zorder=5, edgecolors='black', linewidths=1.5, label='Goal')

        ax.set_xticks(np.arange(grid_size))
        ax.set_yticks(np.arange(grid_size))
        ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.tick_params(which='minor', bottom=False, left=False)

        ax.set_title("POMDP Environment: Noise Zones", fontsize=16)
        plt.colorbar(im, ax=ax, label="Observation Entropy")
        ax.legend(loc='upper right', fontsize=11)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Noise zones saved to {save_path}")

    def visualize_pomdp_value_comparison(self, save_path: str = "figures/pomdp/value_comparison.png",
                                          beta: float = 1.0):
        """Compare pragmatic, epistemic, and combined value functions.

        Shows how observation noise modifies the value landscape:
        - Pragmatic value: reward-seeking (M @ C_reward)
        - Epistemic value: information-seeking (M @ entropy(A))
        - Combined: expected free energy used by the POMDP agent

        Args:
            save_path: Path to save the figure
            beta: Weight for epistemic value (should match experiment setting)
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if not self._is_pomdp():
            print("Value comparison visualization requires POMDP adapter")
            return
        if not hasattr(self.adapter, 'grid_size'):
            print("Value comparison visualization requires grid-based environment")
            return
        if self.M is None or self.C is None:
            print("No learned M or C to visualize")
            return

        grid_size = self.adapter.grid_size
        entropy_vals = self.adapter.get_observation_entropy()

        # Pragmatic: M @ C (standard value)
        V_pragmatic = self.M @ self.C

        # Epistemic: M @ entropy(A) (expected observation uncertainty)
        V_epistemic = self.M @ entropy_vals

        # Combined: pragmatic - beta * epistemic
        V_combined = V_pragmatic - beta * V_epistemic

        grids = [
            V_pragmatic.reshape(grid_size, grid_size).T,
            V_epistemic.reshape(grid_size, grid_size).T,
            V_combined.reshape(grid_size, grid_size).T,
        ]
        titles = [
            "Pragmatic Value\n(M @ C)",
            "Epistemic Value\n(M @ entropy(A))",
            f"Combined EFE\n(pragmatic - {beta} * epistemic)",
        ]
        cmaps = ['viridis', 'magma', 'coolwarm']

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        for ax, grid, title, cmap in zip(axes, grids, titles, cmaps):
            im = ax.imshow(grid, cmap=cmap, interpolation='nearest')
            ax.set_title(title, fontsize=13)
            ax.set_xticks(np.arange(grid_size))
            ax.set_yticks(np.arange(grid_size))
            plt.colorbar(im, ax=ax, shrink=0.8)

            # Mark goal
            if self.goal_states:
                for gs in self.goal_states:
                    loc = self.adapter.state_space.index_to_state(gs)
                    if len(loc) >= 2:
                        ax.scatter(loc[0], loc[1], color='red', s=200, marker='*',
                                  zorder=5, edgecolors='white', linewidths=1)

        fig.suptitle("POMDP Value Decomposition", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Value comparison saved to {save_path}")

    def visualize_belief_trajectory(self, save_dir: str = "figures/pomdp",
                                     max_steps: Optional[int] = None):
        """Visualize belief vs true state over an episode trajectory.

        Shows three panels:
        1. Time series of true state, observation, and belief indices
        2. Grid heatmap of where belief errors occurred
        3. Accuracy timeline (green=correct, red=mismatch)

        Args:
            save_dir: Directory to save figures
            max_steps: Maximum number of steps to show (None for all)
        """
        os.makedirs(save_dir, exist_ok=True)

        if not self._is_pomdp():
            print("Belief trajectory visualization requires POMDP adapter")
            return
        if not hasattr(self.adapter, 'grid_size'):
            print("Belief trajectory visualization requires grid-based environment")
            return

        true_states = self.adapter.state_history
        observations = self.adapter.observation_history
        beliefs = self.adapter.belief_history

        if not true_states or len(true_states) < 2:
            print("No episode history to visualize (run an episode first)")
            return

        grid_size = self.adapter.grid_size

        if max_steps is not None:
            true_states = true_states[:max_steps]
            observations = observations[:max_steps]
            beliefs = beliefs[:max_steps]

        n_steps = len(true_states)
        steps = np.arange(n_steps)
        matches = [t == b for t, b in zip(true_states, beliefs)]
        accuracy = sum(matches) / len(matches)

        fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                                  gridspec_kw={'height_ratios': [3, 3, 1]})

        # Row 1: Time series
        ax1 = axes[0]
        ax1.plot(steps, true_states, 'b-', linewidth=1.5, alpha=0.8, label='True State')
        ax1.plot(steps, observations, 'orange', linewidth=1, alpha=0.5, label='Observation')
        ax1.plot(steps, beliefs, 'r--', linewidth=1.5, alpha=0.8, label='Belief')
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("State Index")
        ax1.set_title(f"True State vs Belief vs Observation (Accuracy: {accuracy:.1%})", fontsize=14)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Row 2: Error heatmap on grid
        ax2 = axes[1]
        error_grid = np.zeros((grid_size, grid_size))
        visit_grid = np.zeros((grid_size, grid_size))
        for i, (true_s, match) in enumerate(zip(true_states, matches)):
            loc = self.adapter.render_state(true_s)
            if len(loc) >= 2:
                visit_grid[loc[0], loc[1]] += 1
                if not match:
                    error_grid[loc[0], loc[1]] += 1

        # Compute error rate per cell (avoid div by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            error_rate = np.where(visit_grid > 0, error_grid / visit_grid, 0)

        im2 = ax2.imshow(error_rate.T, cmap='Reds', interpolation='nearest',
                         vmin=0, vmax=1)
        ax2.set_title("Belief Error Rate by Grid Cell", fontsize=14)
        ax2.set_xticks(np.arange(grid_size))
        ax2.set_yticks(np.arange(grid_size))
        plt.colorbar(im2, ax=ax2, label="Error Rate")

        # Row 3: Accuracy strip
        ax3 = axes[2]
        colors_strip = ['#2ecc71' if m else '#e74c3c' for m in matches]
        ax3.bar(steps, np.ones(n_steps), color=colors_strip, width=1.0, edgecolor='none')
        ax3.set_xlim(-0.5, n_steps - 0.5)
        ax3.set_ylim(0, 1)
        ax3.set_yticks([])
        ax3.set_xlabel("Time Step")
        ax3.set_title("Belief Accuracy Timeline (Green=Correct, Red=Mismatch)", fontsize=12)

        plt.tight_layout()
        plt.savefig(f"{save_dir}/belief_trajectory.png", bbox_inches='tight')
        plt.close()
        print(f"Belief trajectory saved to {save_dir}/belief_trajectory.png")
