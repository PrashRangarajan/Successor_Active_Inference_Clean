"""Value function and POMDP visualization methods.

This module provides plotting capabilities for value functions,
observation entropy, observation models, noise zones, POMDP value
decomposition, and belief trajectory analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple


class POMDPVizMixin(object):
    """Mixin class providing value function and POMDP visualization methods.

    Requires the agent to have:
    - self.adapter: Environment adapter
    - self.M, self.C: Successor matrix and goal prior
    - self.goal_states: Goal state indices
    """

    # ==================== Value Function Visualization ====================

    def visualize_value_function(self, save_path: str = None):
        """Visualize the value function on the grid.

        For augmented state spaces (e.g., key gridworld) this produces two
        side-by-side heatmaps — one per augment value ("Without key"
        / "With key").

        For standard state spaces, a single heatmap is shown.

        Args:
            save_path: Path to save the figure (e.g. 'figures/gridworld/value_function.png')
        """
        if save_path is None:
            raise ValueError("save_path is required (e.g. 'figures/gridworld/value_function.png')")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        if not hasattr(self.adapter, 'grid_size'):
            print("Value function visualization requires grid-based environment")
            return

        if self.M is None or self.C is None:
            print("No learned M or C to visualize")
            return

        grid_size = self.adapter.grid_size
        n_base = grid_size ** 2
        V = self.adapter.multiply_M_C(self.M, self.C)

        # Detect augmented state space
        is_augmented = V.shape[0] != n_base and V.shape[0] == 2 * n_base

        # Get wall mask for the base grid
        wall_mask_base = np.zeros(n_base, dtype=bool)
        if hasattr(self.adapter, 'get_wall_indices'):
            for wi in self.adapter.get_wall_indices():
                # For augmented envs, wall indices cover both key states;
                # map back to base location index.
                if is_augmented:
                    base_idx, _ = self.adapter.state_space.index_to_state(wi)
                    wall_mask_base[base_idx] = True
                else:
                    wall_mask_base[wi] = True

        if is_augmented:
            # Split into per-augment-value panels
            V_no_key = V[:n_base].copy()
            V_has_key = V[n_base:].copy()

            # Mask walls
            V_no_key_grid = np.ma.masked_where(
                wall_mask_base.reshape(grid_size, grid_size).T,
                V_no_key.reshape(grid_size, grid_size).T,
            )
            V_has_key_grid = np.ma.masked_where(
                wall_mask_base.reshape(grid_size, grid_size).T,
                V_has_key.reshape(grid_size, grid_size).T,
            )

            vmin = min(V_no_key_grid.min(), V_has_key_grid.min())
            vmax = max(V_no_key_grid.max(), V_has_key_grid.max())

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            # fig.suptitle('Value Function V = M @ C', fontsize=16)

            for ax, V_grid, title in [
                (axes[0], V_no_key_grid, '✗ Without key'),
                (axes[1], V_has_key_grid, '★ With key'),
            ]:
                ax.set_facecolor('white')
                im = ax.imshow(V_grid, cmap='copper', vmin=vmin, vmax=vmax)
                ax.set_title(title, fontsize=14)
                ax.set_xticks(np.arange(grid_size))
                ax.set_yticks(np.arange(grid_size))

            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label='Value')

            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            # Standard (non-augmented) state space
            V_grid = np.ma.masked_where(
                wall_mask_base.reshape(grid_size, grid_size).T,
                V.reshape(grid_size, grid_size).T,
            )

            plt.figure(figsize=(10, 8))
            ax = plt.gca()
            ax.set_facecolor('white')
            plt.imshow(V_grid, cmap='viridis')
            plt.colorbar(label='Value')
            # plt.title('Value Function V = M @ C', fontsize=16)
            plt.xticks(np.arange(grid_size))
            plt.yticks(np.arange(grid_size))

            # Mark goal
            if self.goal_states:
                for gs in self.goal_states:
                    loc = self.adapter.render_state(gs)
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

    def visualize_observation_entropy(self, save_path: str = None):
        """Visualize per-state observation entropy on the grid.

        High entropy states have noisier observations (harder to localize).

        Args:
            save_path: Path to save the figure (e.g. 'figures/pomdp_gridworld/observation_entropy.png')
        """
        if save_path is None:
            raise ValueError("save_path is required (e.g. 'figures/pomdp_gridworld/observation_entropy.png')")
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
            loc = self.adapter.render_state(w)
            ax.add_patch(plt.Rectangle((loc[0] - 0.5, loc[1] - 0.5), 1, 1,
                                       facecolor='black', edgecolor='white', linewidth=1))

        ax.set_xticks(np.arange(grid_size))
        ax.set_yticks(np.arange(grid_size))
        ax.set_title("Observation Entropy (Higher = Noisier)", fontsize=16)
        plt.colorbar(im, ax=ax, label="Entropy (nats)")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Observation entropy saved to {save_path}")

    def visualize_observation_model(self, save_dir: str = None,
                                     states_to_show: Optional[List[int]] = None):
        """Visualize P(obs|state) for selected states.

        Shows how observations are distributed given the true state,
        revealing which states have clean vs noisy observation profiles.

        Args:
            save_dir: Directory to save figures (e.g. 'figures/pomdp_gridworld')
            states_to_show: List of state indices to visualize.
                If None, auto-selects representative states.
        """
        if save_dir is None:
            raise ValueError("save_dir is required (e.g. 'figures/pomdp_gridworld')")
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

    def visualize_noise_zones(self, save_path: str = None,
                               init_loc: Optional[Tuple[int, int]] = None,
                               goal_loc: Optional[Tuple[int, int]] = None):
        """Visualize POMDP environment overview with noise zones highlighted.

        Shows entropy heatmap with noisy zones outlined, goal, start, and walls.

        Args:
            save_path: Path to save the figure (e.g. 'figures/pomdp_gridworld/noise_zones.png')
            init_loc: Agent start location (default: (0,0))
            goal_loc: Goal location (default: bottom-right corner)
        """
        if save_path is None:
            raise ValueError("save_path is required (e.g. 'figures/pomdp_gridworld/noise_zones.png')")
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
                loc = self.adapter.render_state(self.goal_states[0])
                goal_loc = (loc[0], loc[1])
            else:
                goal_loc = (grid_size - 1, grid_size - 1)

        walls = self.adapter.get_wall_indices() if hasattr(self.adapter, 'get_wall_indices') else []

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(entropy_grid, cmap='YlOrRd', interpolation='nearest', alpha=0.85)

        # Outline noisy zones (entropy > mean + 1 std)
        threshold = entropy_vals.mean() + entropy_vals.std()
        for idx in range(len(entropy_vals)):
            if entropy_vals[idx] > threshold:
                loc = self.adapter.render_state(idx)
                ax.add_patch(plt.Rectangle((loc[0] - 0.5, loc[1] - 0.5), 1, 1,
                                           facecolor='none', edgecolor='red',
                                           linewidth=2.5, linestyle='--'))

        # Mark walls
        for w in walls:
            loc = self.adapter.render_state(w)
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

    def visualize_pomdp_value_comparison(self, save_path: str = None,
                                          beta: float = 1.0):
        """Compare pragmatic, epistemic, and combined value functions.

        Shows how observation noise modifies the value landscape:
        - Pragmatic value: reward-seeking (M @ C_reward)
        - Epistemic value: information-seeking (M @ entropy(A))
        - Combined: expected free energy used by the POMDP agent

        Args:
            save_path: Path to save the figure (e.g. 'figures/pomdp_gridworld/value_comparison.png')
            beta: Weight for epistemic value (should match experiment setting)
        """
        if save_path is None:
            raise ValueError("save_path is required (e.g. 'figures/pomdp_gridworld/value_comparison.png')")
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
                    loc = self.adapter.render_state(gs)
                    ax.scatter(loc[0], loc[1], color='red', s=200, marker='*',
                              zorder=5, edgecolors='white', linewidths=1)

        fig.suptitle("POMDP Value Decomposition", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"Value comparison saved to {save_path}")

    def visualize_belief_trajectory(self, save_dir: str = None,
                                     max_steps: Optional[int] = None):
        """Visualize belief vs true state over an episode trajectory.

        Shows three panels:
        1. Time series of true state, observation, and belief indices
        2. Grid heatmap of where belief errors occurred
        3. Accuracy timeline (green=correct, red=mismatch)

        Args:
            save_dir: Directory to save figures (e.g. 'figures/pomdp_gridworld')
            max_steps: Maximum number of steps to show (None for all)
        """
        if save_dir is None:
            raise ValueError("save_dir is required (e.g. 'figures/pomdp_gridworld')")
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
