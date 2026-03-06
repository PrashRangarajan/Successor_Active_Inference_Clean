"""PointMaze environment adapter for Hierarchical SR Active Inference.

PointMaze is a 2D continuous-control maze navigation task from
gymnasium-robotics. The agent controls a point mass with 2D force
in a maze with walls and corridors.

Observation (Gym): Dict with 'observation' (4D: x, y, vx, vy),
                   'achieved_goal' (2D), 'desired_goal' (2D).
Action (Gym):      Box(-1, 1, (2,)) -- continuous 2D force.

State space:  2D position-only (x, y) discretized into bins.
              Velocity is NOT included -- bottleneck structure is
              purely spatial, and the SR captures temporal dynamics
              implicitly.

Actions:      8 discrete directions (4 cardinal + 4 diagonal),
              mapped to 2D force vectors via _process_action().
"""

from typing import Any, List, Optional, Tuple

import math
import numpy as np

from environments.binned_continuous_adapter import BinnedContinuousAdapter, clamp
from core.state_space import BinnedContinuousStateSpace


# 8 discrete force directions: cardinal + diagonal
_FORCE_DIRECTIONS = np.array([
    [ 1.0,  0.0],        # 0: East  (→)
    [-1.0,  0.0],        # 1: West  (←)
    [ 0.0,  1.0],        # 2: North (↑)
    [ 0.0, -1.0],        # 3: South (↓)
    [ 0.707,  0.707],    # 4: NE (↗)
    [-0.707,  0.707],    # 5: NW (↖)
    [ 0.707, -0.707],    # 6: SE (↘)
    [-0.707, -0.707],    # 7: SW (↙)
], dtype=np.float32)


class PointMazeAdapter(BinnedContinuousAdapter):
    """Adapter for gymnasium-robotics PointMaze environments.

    State:   (x_bin, y_bin) -- 2D position discretized into bins.
    Actions: 8 discrete directions (4 cardinal + 4 diagonal).
    """

    def __init__(self, env, n_x_bins: int = 20, n_y_bins: int = 20):
        """
        Args:
            env: Gymnasium PointMaze environment
                 (e.g., ``gym.make("PointMaze_UMaze-v3")``).
            n_x_bins: Number of bins along the x axis.
            n_y_bins: Number of bins along the y axis.
        """
        self._env = env
        self.n_x_bins = n_x_bins
        self.n_y_bins = n_y_bins

        # Set top-down camera so the full maze is visible in renders.
        try:
            renderer = env.unwrapped.point_env.mujoco_renderer
            viewer = renderer._get_viewer(env.render_mode)
            viewer.cam.elevation = -90
            viewer.cam.azimuth = 90
            viewer.cam.distance = 12
            viewer.cam.lookat[:] = [0, 0, 0]
        except Exception:
            pass  # non-render envs or API changes
        self._state_space = BinnedContinuousStateSpace([n_x_bins, n_y_bins])
        self._n_actions = len(_FORCE_DIRECTIONS)  # 8

        # Extract maze structure
        self._maze = env.unwrapped.maze
        self._maze_map = self._maze.maze_map

        # Compute physical coordinate bounds from maze layout.
        # Each cell is 1 unit; the cell (row, col) center is at
        #   x = col - x_center,  y = y_center - row
        # where x_center = map_width / 2, y_center = map_length / 2.
        n_rows = self._maze.map_length
        n_cols = self._maze.map_width
        x_center = n_cols / 2.0
        y_center = n_rows / 2.0
        self._x_range = (-x_center, x_center)
        self._y_range = (-y_center, y_center)
        self._maze_x_center = x_center
        self._maze_y_center = y_center

        # Create discretization bins (interior edges)
        self.x_space, self.y_space = self._create_bins()

        # Precompute wall bin indices
        self._wall_indices = self._compute_wall_indices()
        self._wall_set = set(self._wall_indices)

        # Cached desired goal from most recent reset
        self._desired_goal = None

        # The goal XY the agent was configured with (set once by get_goal_states,
        # stable across later resets that change _desired_goal).
        self._agent_goal_xy = None

        # State tracking — position only (2D ndarray)
        self._current_obs = None
        self._current_state = None

    # ==================== Binning ====================

    def _create_bins(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create interior bin edges (n-1 edges for n bins).

        Uses the CartPole/Acrobot interior-edge pattern so that
        ``np.digitize`` naturally produces [0, n_bins-1] after clamping.
        """
        x_space = np.linspace(
            self._x_range[0], self._x_range[1], self.n_x_bins + 1
        )[1:-1]
        y_space = np.linspace(
            self._y_range[0], self._y_range[1], self.n_y_bins + 1
        )[1:-1]
        return x_space, y_space

    def _compute_wall_indices(self) -> List[int]:
        """Identify bins whose centers lie inside maze walls.

        The maze_map is a grid: ``1`` = wall, ``0`` / ``'R'`` / ``'G'``
        = open.  Each cell is 1 unit wide.
        """
        x_centers, y_centers = self.get_bin_centers()
        wall_indices = []

        n_rows = len(self._maze_map)
        n_cols = len(self._maze_map[0]) if n_rows > 0 else 0

        for xi in range(self.n_x_bins):
            for yi in range(self.n_y_bins):
                cx, cy = x_centers[xi], y_centers[yi]
                # Map continuous (x, y) to maze cell (row, col).
                # row = floor(y_center - y), col = floor(x + x_center)
                col = int(math.floor(cx + self._maze_x_center))
                row = int(math.floor(self._maze_y_center - cy))
                col = max(0, min(col, n_cols - 1))
                row = max(0, min(row, n_rows - 1))

                if self._maze_map[row][col] == 1:
                    idx = self._state_space.state_to_index((xi, yi))
                    wall_indices.append(idx)

        return wall_indices

    # ==================== Abstract method implementations ====================

    def discretize_obs(self, obs: np.ndarray) -> Tuple[int, int]:
        """Convert (x, y) position to discrete bin indices.

        Args:
            obs: 2D ndarray ``[x, y]`` (NOT the full Gym dict).
        """
        x, y = float(obs[0]), float(obs[1])
        i = clamp(np.digitize(x, self.x_space), 0, self.n_x_bins - 1)
        j = clamp(np.digitize(y, self.y_space), 0, self.n_y_bins - 1)
        return (i, j)

    def get_goal_states(self, goal_spec: Any = None) -> List[int]:
        """Return state indices near the goal location.

        Args:
            goal_spec: ``None`` → use ``desired_goal`` from last reset,
                       or ``[x, y]`` position.

        Returns:
            List of bin indices within 0.5 units of the goal centre.
        """
        if goal_spec is None:
            if self._desired_goal is None:
                raise ValueError("No goal available. Call reset() first.")
            goal_xy = self._desired_goal
        elif isinstance(goal_spec, (list, np.ndarray)) and len(goal_spec) >= 2:
            goal_xy = np.asarray(goal_spec[:2], dtype=np.float64)
        else:
            raise ValueError(f"Invalid goal spec: {goal_spec}")

        # Store the goal XY the agent will navigate toward — used by
        # is_terminal() for continuous distance checks.
        self._agent_goal_xy = goal_xy.copy()

        goal_radius = 0.5
        x_centers, y_centers = self.get_bin_centers()
        goal_states = []

        for xi in range(self.n_x_bins):
            for yi in range(self.n_y_bins):
                dist = math.sqrt(
                    (x_centers[xi] - goal_xy[0]) ** 2
                    + (y_centers[yi] - goal_xy[1]) ** 2
                )
                if dist <= goal_radius:
                    idx = self._state_space.state_to_index((xi, yi))
                    if idx not in self._wall_set:
                        goal_states.append(idx)

        return goal_states

    def sample_random_state(self) -> np.ndarray:
        """Reset to a random navigable position for diverse training starts.

        Uses the environment's built-in reset (which randomizes the start
        position among open cells) and optionally injects a uniform random
        position via MuJoCo state setting.
        """
        obs_dict, _ = self._env.reset()
        self._desired_goal = obs_dict['desired_goal'].copy()

        # Try injecting a uniformly random navigable position
        x_centers, y_centers = self.get_bin_centers()
        injected = False

        for _attempt in range(50):
            x = np.random.uniform(self._x_range[0], self._x_range[1])
            y = np.random.uniform(self._y_range[0], self._y_range[1])
            disc = self.discretize_obs(np.array([x, y]))
            idx = self._state_space.state_to_index(disc)
            if idx not in self._wall_set:
                # Inject position via MuJoCo
                try:
                    point_env = self._env.unwrapped.point_env
                    qpos = np.array([x, y])
                    qvel = np.array([0.0, 0.0])
                    point_env.set_state(qpos, qvel)
                    self._current_obs = np.array([x, y])
                    injected = True
                except Exception:
                    pass
                break

        if not injected:
            # Fallback: use whatever position env.reset() gave us
            self._current_obs = obs_dict['observation'][:2].copy()

        discrete_state = self.discretize_obs(self._current_obs)
        state_idx = self._state_space.state_to_index(discrete_state)
        self._current_state = self._state_space.index_to_onehot(state_idx)
        return self._current_state

    # ==================== Terminal check ====================

    def is_terminal(self) -> Optional[bool]:
        """Check if the agent is within continuous distance of the goal.

        Uses a threshold of 0.45 units, matching PointMaze's native
        ``compute_terminated`` success criterion.  This is checked both
        in the agent's ``_is_at_goal()`` method and inside
        ``_step_with_smooth`` to stop the ball the instant it enters
        the goal region, preventing overshoot.

        Returns:
            True/False if a goal has been set, None otherwise.
        """
        if self._agent_goal_xy is None:
            return None
        dist = math.sqrt(
            (self._current_obs[0] - self._agent_goal_xy[0]) ** 2
            + (self._current_obs[1] - self._agent_goal_xy[1]) ** 2
        )
        return dist < 0.45

    # ==================== Walls ====================

    def get_wall_indices(self) -> List[int]:
        """Return precomputed list of wall bin indices."""
        return self._wall_indices

    # ==================== Step / Reset (dict obs handling) ====================

    def _extract_position(self, obs_dict: dict) -> np.ndarray:
        """Extract (x, y) position from PointMaze dict observation."""
        return obs_dict['observation'][:2].copy()

    def step(self, action: int) -> np.ndarray:
        """Take action, handling dict observation from PointMaze."""
        env_action = self._process_action(action)
        obs_dict, reward, terminated, truncated, info = self._env.step(env_action)
        self._current_obs = self._extract_position(obs_dict)
        discrete_state = self.discretize_obs(self._current_obs)
        state_idx = self.state_space.state_to_index(discrete_state)
        self._current_state = self.state_space.index_to_onehot(state_idx)
        return self._current_state

    def step_with_info(self, action: int):
        """Take action and return (state, reward, terminated, truncated, info)."""
        env_action = self._process_action(action)
        obs_dict, reward, terminated, truncated, info = self._env.step(env_action)
        self._current_obs = self._extract_position(obs_dict)
        discrete_state = self.discretize_obs(self._current_obs)
        state_idx = self.state_space.state_to_index(discrete_state)
        self._current_state = self.state_space.index_to_onehot(state_idx)
        return self._current_state, reward, terminated, truncated, info

    def _process_action(self, action: int):
        """Convert discrete direction index to continuous 2D force array."""
        return _FORCE_DIRECTIONS[action].copy()

    def reset(self, init_state: Optional[Any] = None,
              reset_options: Optional[dict] = None) -> np.ndarray:
        """Reset environment.

        Args:
            init_state: Optional ``[x, y]`` continuous position.
                        If None, uses Gym's randomized initialization.
            reset_options: Optional dict forwarded to ``env.reset(options=...)``.
                          E.g. ``{'goal_cell': np.array([3, 1])}`` to force
                          the goal to a specific maze cell.

        Returns:
            One-hot encoded discretized state.
        """
        obs_dict, _ = self._env.reset(options=reset_options)

        if init_state is not None:
            xy = np.array(init_state[:2], dtype=np.float64)
            try:
                point_env = self._env.unwrapped.point_env
                point_env.set_state(xy, np.array([0.0, 0.0]))
                self._current_obs = xy.copy()
            except Exception:
                self._current_obs = self._extract_position(obs_dict)
        else:
            self._current_obs = self._extract_position(obs_dict)

        # Cache desired goal for get_goal_states
        self._desired_goal = obs_dict['desired_goal'].copy()

        # Keep the continuous goal target (_agent_goal_xy) in sync with
        # the rendered goal marker.  PointMaze randomizes the exact
        # continuous position within a cell on each reset, so a stale
        # _agent_goal_xy would make is_terminal() check distance to a
        # slightly different point than what's rendered.
        # Only sync when reset_options is explicitly provided (i.e. the
        # caller is setting a specific goal cell for testing), NOT during
        # learning where sample_random_state calls env.reset randomly.
        if self._agent_goal_xy is not None and reset_options is not None:
            self._agent_goal_xy = self._desired_goal.copy()

        discrete_state = self.discretize_obs(self._current_obs)
        state_idx = self.state_space.state_to_index(discrete_state)
        self._current_state = self.state_space.index_to_onehot(state_idx)
        return self._current_state

    # ==================== Clustering ====================

    def get_clustering_affinity(self, M: np.ndarray,
                                sigma: float = 0.5,
                                blend: float = 0.15) -> np.ndarray:
        """Build clustering affinity with light spatial smoothing.

        The SR in a maze already captures wall-induced disconnections well,
        so a lower blend weight is used compared to MountainCar. The spatial
        kernel only helps smooth under-visited regions.

        This method receives the *valid-states-only* M (walls already
        removed by ``_learn_macro_clusters``), so only navigable bins are
        present.

        Args:
            M: Successor matrix (N_valid x N_valid), already symmetrised.
            sigma: Bandwidth for spatial RBF kernel.
            blend: Maximum weight for the spatial kernel.

        Returns:
            Blended affinity matrix.
        """
        from scipy.spatial.distance import squareform, pdist

        # Identify which valid states we have.  M is indexed from
        # 0..N_valid-1.  We need to map back to (x, y) coordinates.
        # The caller passes only valid (non-wall) rows/cols of M.
        # We rebuild coordinates for each valid state in order.
        valid_mask = self.get_valid_state_mask()
        valid_indices = np.where(valid_mask)[0]
        N_valid = len(valid_indices)

        x_centers, y_centers = self.get_bin_centers()
        x_range = x_centers[-1] - x_centers[0]
        y_range = y_centers[-1] - y_centers[0]

        coords = np.zeros((N_valid, 2))
        for i, idx in enumerate(valid_indices):
            xi, yi = self.state_space.index_to_state(idx)
            coords[i, 0] = (x_centers[xi] - x_centers[0]) / max(x_range, 1e-8)
            coords[i, 1] = (y_centers[yi] - y_centers[0]) / max(y_range, 1e-8)

        # Spatial RBF kernel
        D2 = squareform(pdist(coords, 'sqeuclidean'))
        K_spatial = np.exp(-D2 / (2.0 * sigma ** 2))

        # Normalize M to [0, 1]
        M_min, M_max = M.min(), M.max()
        if M_max > M_min:
            M_norm = (M - M_min) / (M_max - M_min)
        else:
            M_norm = np.zeros_like(M)

        # Adaptive blend based on row-sum confidence
        row_sums = M.sum(axis=1)
        rs_max = row_sums.max()
        if rs_max > 0:
            confidence = row_sums / rs_max
        else:
            confidence = np.zeros(N_valid)

        conf_i = confidence[:, None]
        conf_j = confidence[None, :]
        pair_conf = np.minimum(conf_i, conf_j)

        alpha = blend * (1.0 - pair_conf)

        A = (1.0 - alpha) * M_norm + alpha * K_spatial
        return A

    # ==================== Visualization ====================

    def get_state_label(self, state_index: int) -> str:
        """Get human-readable state label."""
        x_bin, y_bin = self.state_space.index_to_state(state_index)
        return f"(x{x_bin},y{y_bin})"

    def get_bin_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get center values of x and y bins."""
        x_edges = np.linspace(
            self._x_range[0], self._x_range[1], self.n_x_bins + 1
        )
        y_edges = np.linspace(
            self._y_range[0], self._y_range[1], self.n_y_bins + 1
        )
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        return x_centers, y_centers

    def get_bin_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return bin edge arrays for each dimension."""
        x_edges = np.linspace(
            self._x_range[0], self._x_range[1], self.n_x_bins + 1
        )
        y_edges = np.linspace(
            self._y_range[0], self._y_range[1], self.n_y_bins + 1
        )
        return x_edges, y_edges

    def get_dimension_labels(self) -> Tuple[str, str]:
        """Return axis labels for visualization."""
        return ("Position (x)", "Position (y)")

    def obs_to_continuous(self, obs: np.ndarray) -> Tuple[float, float]:
        """Extract (x, y) from stored observation."""
        return float(obs[0]), float(obs[1])

    def get_action_labels(self) -> List[str]:
        """Return human-readable labels for each action."""
        return ["→ E", "← W", "↑ N", "↓ S", "↗ NE", "↖ NW", "↘ SE", "↙ SW"]

    def print_maze_layout(self):
        """Print the maze map with wall/open indicators."""
        print("Maze layout (1=wall, 0=open):")
        for row in self._maze_map:
            print("  ", row)

    def print_bin_wall_map(self):
        """Print a grid showing which bins are walls vs navigable."""
        x_centers, y_centers = self.get_bin_centers()
        print(f"Bin wall map ({self.n_x_bins}x{self.n_y_bins}):")
        print(f"  ({len(self._wall_indices)} wall bins, "
              f"{self.n_states - len(self._wall_indices)} navigable)")
        # Print top-down: y decreasing
        for yi in range(self.n_y_bins - 1, -1, -1):
            row_str = ""
            for xi in range(self.n_x_bins):
                idx = self._state_space.state_to_index((xi, yi))
                row_str += "█" if idx in self._wall_set else "·"
            print(f"  y{yi:2d} {row_str}")

    # ==================== Maze Visualization ====================

    def draw_maze_walls(self, ax):
        """Draw maze wall cells as filled dark rectangles on a matplotlib axis.

        Draws one filled rectangle per wall cell in the maze map, using the
        physical coordinate system. Assumes the axis already has the correct
        x/y limits matching the maze bounds.

        Args:
            ax: matplotlib Axes object.
        """
        import matplotlib.patches as mpatches

        n_rows = len(self._maze_map)
        n_cols = len(self._maze_map[0]) if n_rows > 0 else 0

        for row in range(n_rows):
            for col in range(n_cols):
                if self._maze_map[row][col] == 1:
                    # Cell physical bounds
                    x_lo = col - self._maze_x_center
                    y_hi = self._maze_y_center - row
                    rect = mpatches.Rectangle(
                        (x_lo, y_hi - 1), 1.0, 1.0,
                        facecolor='#2d2d2d', edgecolor='#444444',
                        linewidth=0.5, zorder=2,
                    )
                    ax.add_patch(rect)

    def plot_clusters_on_maze(self, agent, save_path: str,
                              show_arrows: bool = True,
                              show_goal: bool = True,
                              goal_xy=None,
                              start_xy=None):
        """Visualize macro-state clusters overlaid on the physical maze.

        Creates a figure with:
        - Dark rectangles for wall cells
        - Colored tiles (one per navigable bin) showing cluster assignment
        - Macro-action arrows between cluster centroids
        - Start and goal location markers

        Args:
            agent: ``HierarchicalSRAgent`` (for cluster data).
            save_path: Where to save the figure.
            show_arrows: Draw macro-action arrows between clusters.
            show_goal: Mark the goal location.
            goal_xy: Explicit goal coordinates (overrides adapter cache).
            start_xy: Start coordinates to show (green circle marker).
        """
        import os
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.patheffects as PathEffects

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        x_edges, y_edges = self.get_bin_edges()
        x_centers, y_centers = self.get_bin_centers()
        dx = x_edges[1] - x_edges[0]
        dy = y_edges[1] - y_edges[0]

        # Cluster colors (tab10)
        tab = plt.get_cmap("tab10")
        cluster_colors = [tab(i) for i in range(agent.n_clusters)]

        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw navigable bins colored by cluster
        for micro_idx, macro_idx in agent.micro_to_macro.items():
            xi, yi = self.state_space.index_to_state(micro_idx)
            rect = mpatches.Rectangle(
                (x_edges[xi], y_edges[yi]), dx, dy,
                facecolor=cluster_colors[macro_idx], edgecolor='white',
                linewidth=0.3, alpha=0.85, zorder=3,
            )
            ax.add_patch(rect)

        # Draw walls on top
        self.draw_maze_walls(ax)

        # Compute cluster centroids (used by arrows and markers)
        centroids = {}
        for c in range(agent.n_clusters):
            members = agent.macro_state_list[c]
            if not members:
                continue
            coords = np.array([
                self.state_space.index_to_state(s) for s in members
            ])
            cx = np.mean(x_centers[coords[:, 0]])
            cy = np.mean(y_centers[coords[:, 1]])
            centroids[c] = (cx, cy)

        # Macro-action arrows
        if show_arrows and hasattr(agent, 'adj_list') and agent.adj_list:
            # Goal macro states
            goal_macros = set()
            for gs in agent.goal_states:
                if gs in agent.micro_to_macro:
                    goal_macros.add(agent.micro_to_macro[gs])

            V_macro = agent.M_macro @ agent.C_macro
            for c in range(agent.n_clusters):
                if c not in centroids or c in goal_macros:
                    continue
                if c not in agent.adj_list or not agent.adj_list[c]:
                    continue
                adj = agent.adj_list[c]
                values = [V_macro[a] for a in adj]
                target = adj[int(np.argmax(values))]
                if target == c or target not in centroids:
                    continue
                x0, y0 = centroids[c]
                x1, y1 = centroids[target]
                ax.annotate(
                    '', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(
                        arrowstyle='->', lw=2.5, color='black',
                        shrinkA=10, shrinkB=10,
                    ),
                    zorder=12,
                )

            # Cluster labels at centroids
            for c, (cx, cy) in centroids.items():
                label = f'{c}' + (' *' if c in goal_macros else '')
                ax.text(
                    cx, cy, label, fontsize=13, fontweight='bold',
                    ha='center', va='center', color='white',
                    path_effects=[
                        PathEffects.withStroke(linewidth=3, foreground='black')
                    ],
                    zorder=13,
                )

        # Helper: check if a marker position clashes with any centroid
        def _needs_offset(marker_xy, centroids_dict, threshold=0.4):
            for _, (cx, cy) in centroids_dict.items():
                dist = np.sqrt((marker_xy[0] - cx)**2 + (marker_xy[1] - cy)**2)
                if dist < threshold:
                    return True
            return False

        legend_handles = []

        # Start marker
        if start_xy is not None:
            sx, sy = float(start_xy[0]), float(start_xy[1])
            if _needs_offset(start_xy, centroids):
                sx += 0.45
            ax.plot(sx, sy, 'o', markersize=16, color='limegreen',
                    markeredgecolor='darkgreen', markeredgewidth=2, zorder=15)
            legend_handles.append(
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen',
                           markeredgecolor='darkgreen', markersize=10, label='Start'))

        # Goal marker — use explicit goal_xy if provided, else fall back to
        # adapter's cached desired_goal (which may have changed on later resets).
        _goal = goal_xy if goal_xy is not None else self._desired_goal
        if show_goal and _goal is not None:
            gx, gy = float(_goal[0]), float(_goal[1])
            if _needs_offset(_goal, centroids):
                gx += 0.45
            ax.plot(gx, gy, marker='*', markersize=22, color='gold',
                    markeredgecolor='black', markeredgewidth=1.5, zorder=15)
            legend_handles.append(
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                           markeredgecolor='black', markersize=12, label='Goal'))

        if legend_handles:
            ax.legend(handles=legend_handles, loc='upper right', fontsize=11,
                      framealpha=0.9)

        ax.set_xlim(self._x_range)
        ax.set_ylim(self._y_range)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        fig.savefig(save_path, bbox_inches='tight', dpi=150, pad_inches=0.02)
        plt.close(fig)
        print(f"  Saved maze cluster plot to {save_path}")

    def plot_multi_goal_maze(self, agent, trajectories, goals, save_path: str):
        """Overlay multiple goal-reaching trajectories on the maze cluster map.

        Args:
            agent: Trained ``HierarchicalSRAgent``.
            trajectories: List of dicts, each with ``'x'`` and ``'y'`` lists.
            goals: List of dicts, each with ``'xy'`` (physical [x, y]) and
                   ``'label'`` (short name like "Bottom-left").
            save_path: Output path for the figure.
        """
        import os
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import matplotlib.patheffects as PathEffects

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        x_edges, y_edges = self.get_bin_edges()
        x_centers, y_centers = self.get_bin_centers()
        dx = x_edges[1] - x_edges[0]
        dy = y_edges[1] - y_edges[0]

        tab = plt.get_cmap("tab10")
        cluster_colors = [tab(i) for i in range(agent.n_clusters)]

        fig, ax = plt.subplots(figsize=(9, 9))

        # Navigable bins coloured by cluster (faded for backdrop)
        for micro_idx, macro_idx in agent.micro_to_macro.items():
            xi, yi = self.state_space.index_to_state(micro_idx)
            rect = mpatches.Rectangle(
                (x_edges[xi], y_edges[yi]), dx, dy,
                facecolor=cluster_colors[macro_idx], edgecolor='white',
                linewidth=0.3, alpha=0.35, zorder=3,
            )
            ax.add_patch(rect)

        self.draw_maze_walls(ax)

        # Trajectory colours — distinct from cluster colours
        traj_cmap = plt.get_cmap("Set1")
        n_goals = len(goals)

        for gi, (traj, goal_info) in enumerate(zip(trajectories, goals)):
            color = traj_cmap(gi / max(n_goals - 1, 1))
            xs, ys = traj['x'], traj['y']
            if len(xs) < 2:
                continue
            ax.plot(xs, ys, '-', color=color, linewidth=2.0, alpha=0.85,
                    zorder=6, label=f"Goal {gi + 1}: {goal_info['label']}")
            # Start marker for first trajectory only
            if gi == 0:
                ax.plot(xs[0], ys[0], 'o', color='white', markersize=10,
                        markeredgecolor='black', markeredgewidth=2, zorder=8)

            # Goal star with number
            gx, gy = goal_info['xy']
            ax.plot(gx, gy, marker='*', markersize=20, color=color,
                    markeredgecolor='black', markeredgewidth=1.2, zorder=10)
            ax.text(
                gx + 0.12, gy + 0.12, str(gi + 1),
                fontsize=11, fontweight='bold', color='white',
                path_effects=[PathEffects.withStroke(linewidth=3, foreground='black')],
                zorder=11,
            )

        ax.set_xlim(self._x_range)
        ax.set_ylim(self._y_range)
        ax.set_aspect('equal')
        # ax.set_xlabel("X Position", fontsize=12)
        # ax.set_ylabel("Y Position", fontsize=12)
        # ax.set_title("Multi-Goal Replanning Trajectories", fontsize=14)
        # ax.legend(loc='upper right', fontsize=9)

        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  Saved multi-goal trajectory plot to {save_path}")
