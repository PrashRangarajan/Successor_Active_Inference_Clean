"""Mountain Car environment adapter for Hierarchical SR Active Inference."""

from typing import Any, List, Optional, Tuple
import numpy as np

from environments.binned_continuous_adapter import BinnedContinuousAdapter, clamp
from core.state_space import BinnedContinuousStateSpace

class MountainCarAdapter(BinnedContinuousAdapter):
    """Adapter for Mountain Car continuous control environment.

    State representation: (pos_bin, vel_bin) discretized position and velocity
    Transition matrix shape: (N, N, n_actions) where N = n_pos_bins * n_vel_bins
    Actions: 0=push left, 1=no push, 2=push right
    """

    def __init__(self, env, n_pos_bins: int = 10, n_vel_bins: int = 10):
        """
        Args:
            env: Gymnasium MountainCar-v0 environment
            n_pos_bins: Number of position bins
            n_vel_bins: Number of velocity bins
        """
        self._env = env
        self.n_pos_bins = n_pos_bins
        self.n_vel_bins = n_vel_bins
        self._state_space = BinnedContinuousStateSpace([n_pos_bins, n_vel_bins])
        self._n_actions = env.action_space.n

        # Get observation space bounds
        self.low = env.observation_space.low
        self.high = env.observation_space.high

        # Create discretization bins
        self.pos_space, self.vel_space = self._create_bins()

        self._current_obs = None
        self._current_state = None

    def _create_bins(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create position and velocity bin edges."""
        # Position bins with special handling for goal region
        pos_space = np.concatenate([
            np.linspace(self.low[0], 0.5, self.n_pos_bins),
            [self.high[0]]
        ])
        vel_space = np.linspace(self.low[1], self.high[1], self.n_vel_bins + 1)
        return pos_space, vel_space

    def discretize_obs(self, obs: np.ndarray) -> Tuple[int, int]:
        """Convert continuous observation to discrete bin indices."""
        pos, vel = obs
        i = clamp(np.digitize(pos, self.pos_space), 1, self.n_pos_bins) - 1
        j = clamp(np.digitize(vel, self.vel_space), 1, self.n_vel_bins) - 1
        return (i, j)

    # ==================== Environment Interaction ====================

    def reset(self, init_state: Optional[Any] = None) -> np.ndarray:
        """Reset environment.

        Args:
            init_state: Initial state as [position, velocity] continuous values

        Returns:
            One-hot encoded discretized state
        """
        obs, _ = self._env.reset()

        if init_state is not None:
            # Set specific initial state
            self._env.unwrapped.state = np.array(init_state)
            obs = np.array(init_state)

        self._current_obs = obs
        discrete_state = self.discretize_obs(obs)
        state_idx = self.state_space.state_to_index(discrete_state)
        self._current_state = self.state_space.index_to_onehot(state_idx)
        return self._current_state

    def get_state_for_reset(self) -> np.ndarray:
        """Get current state in a format suitable for reset().

        MountainCar reset() expects continuous [position, velocity],
        not discrete bin indices.
        """
        return self._current_obs.copy()

    def sample_random_state(self) -> np.ndarray:
        """Reset to a uniformly random state across the full state space.

        Ensures every region of the discretized state space has a chance of
        being visited during training, which is critical for building a
        complete transition model B.  Without this, every episode starts
        from the default position (-0.5) and Gym's 200-step truncation
        prevents the agent from ever visiting the goal region.

        Returns:
            One-hot encoded discretized state.
        """
        # Normal reset so Gym's bookkeeping is valid
        obs, _ = self._env.reset()

        # Sample random continuous state within environment bounds
        pos = np.random.uniform(self.low[0], self.high[0])
        vel = np.random.uniform(self.low[1], self.high[1])

        # Inject into the physics engine
        self._env.unwrapped.state = np.array([pos, vel])
        obs = np.array([pos, vel])

        self._current_obs = obs
        discrete_state = self.discretize_obs(obs)
        state_idx = self.state_space.state_to_index(discrete_state)
        self._current_state = self.state_space.index_to_onehot(state_idx)
        return self._current_state

    # ==================== Goal/Reward ====================

    def get_goal_states(self, goal_spec: Any) -> List[int]:
        """Convert goal specification to state indices.

        Args:
            goal_spec: None (default: rightmost position), float (position),
                      or [pos, vel] (specific state)

        Returns:
            List of goal state indices
        """
        if goal_spec is None:
            # Default: all states at maximum position (rightmost column)
            pos = self.n_pos_bins - 1
            return [
                self.state_space.state_to_index((pos, v))
                for v in range(self.n_vel_bins)
            ]
        elif isinstance(goal_spec, (int, float)):
            # Position only - all velocities at that position
            pos_bin = self.discretize_obs((goal_spec, 0))[0]
            return [
                self.state_space.state_to_index((pos_bin, v))
                for v in range(self.n_vel_bins)
            ]
        elif isinstance(goal_spec, (list, np.ndarray)) and len(goal_spec) == 2:
            # Specific (pos, vel)
            discrete = self.discretize_obs(goal_spec)
            return [self.state_space.state_to_index(discrete)]
        else:
            raise ValueError(f"Invalid goal specification: {goal_spec}")

    # ==================== Clustering ====================

    def get_clustering_affinity(self, M: np.ndarray,
                                sigma: float = 0.5,
                                blend: float = 0.3) -> np.ndarray:
        """Build a clustering affinity that smooths sparse goal-region states.

        Mountain Car's goal region (position > 0.5) is rarely visited during
        random exploration, so the SR matrix M has noisy/sparse rows there.
        Pure M-based spectral clustering splits these states into tiny
        outlier clusters.

        The blend weight is **adaptive**: the spatial kernel contributes
        more for state pairs where M is poorly estimated (low row sums,
        indicating sparse visitation) and less where M is reliable.
        This avoids artificially connecting states that the dynamics
        show are truly disconnected (e.g. in environments with walls),
        while still rescuing sparse boundary states from outlier clusters.

        Args:
            M: Successor matrix (N × N), already symmetrised.
            sigma: Bandwidth for the spatial RBF kernel.
            blend: Maximum weight for the spatial kernel.  The per-pair
                   effective weight is ``blend * (1 - confidence_ij)``
                   where confidence is derived from M's row sums.

        Returns:
            Blended affinity matrix (N × N), non-negative and symmetric.
        """
        pos_centers, vel_centers = self.get_bin_centers()

        # Normalise both dimensions to [0, 1]
        pos_range = pos_centers[-1] - pos_centers[0]
        vel_range = vel_centers[-1] - vel_centers[0]

        N = self.n_states
        coords = np.zeros((N, 2))

        for p in range(self.n_pos_bins):
            for v in range(self.n_vel_bins):
                idx = self.state_space.state_to_index((p, v))
                coords[idx, 0] = (pos_centers[p] - pos_centers[0]) / pos_range
                coords[idx, 1] = (vel_centers[v] - vel_centers[0]) / vel_range

        # Pairwise squared Euclidean distance in normalised space
        from scipy.spatial.distance import squareform, pdist
        D2 = squareform(pdist(coords, 'sqeuclidean'))
        K_spatial = np.exp(-D2 / (2.0 * sigma ** 2))

        # Normalise M to [0, 1] for fair blending
        M_min, M_max = M.min(), M.max()
        if M_max > M_min:
            M_norm = (M - M_min) / (M_max - M_min)
        else:
            M_norm = np.zeros_like(M)

        # Adaptive blend: per-state confidence from M row sums.
        # Well-visited states have large row sums → trust M more.
        # Poorly-visited states have small row sums → lean on spatial.
        row_sums = M.sum(axis=1)
        rs_max = row_sums.max()
        if rs_max > 0:
            confidence = row_sums / rs_max          # 0..1 per state
        else:
            confidence = np.zeros(N)

        # Per-pair confidence = min of the two states' confidences
        conf_i = confidence[:, None]                # (N, 1)
        conf_j = confidence[None, :]                # (1, N)
        pair_conf = np.minimum(conf_i, conf_j)      # (N, N)

        # Effective spatial weight: high where confidence is low
        alpha = blend * (1.0 - pair_conf)            # (N, N) in [0, blend]

        A = (1.0 - alpha) * M_norm + alpha * K_spatial
        return A

    # ==================== Visualization ====================

    def get_state_label(self, state_index: int) -> str:
        """Get human-readable state label."""
        pos_bin, vel_bin = self.state_space.index_to_state(state_index)
        return f"(p{pos_bin},v{vel_bin})"

    def get_bin_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get center values of position and velocity bins."""
        pos_centers = 0.5 * (self.pos_space[:-1] + self.pos_space[1:])
        vel_centers = 0.5 * (self.vel_space[:-1] + self.vel_space[1:])
        return pos_centers, vel_centers

    def get_dimension_labels(self) -> Tuple[str, str]:
        """Return human-readable axis labels for the 2D state space."""
        return ("Position", "Velocity")

    def get_bin_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return bin edge arrays for each dimension.

        Used by cluster and trajectory visualizations to map bin indices
        to physical coordinates.
        """
        return self.pos_space, self.vel_space

    def obs_to_continuous(self, obs: np.ndarray) -> Tuple[float, float]:
        """Extract (position, velocity) from raw observation."""
        return float(obs[0]), float(obs[1])

    def get_action_labels(self) -> List[str]:
        """Return human-readable labels for each action index."""
        return ["\u2190 Push Left", "\u00b7 No Push", "\u2192 Push Right"]
