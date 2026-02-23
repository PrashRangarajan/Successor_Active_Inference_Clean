"""Pendulum environment adapter for Hierarchical SR Active Inference.

Pendulum-v1 has a continuous action space (torque in [-2, 2]) which is
discretized into a fixed number of bins.  The observation [cos(θ), sin(θ), ω]
is converted to (θ, ω) for binning.

Goal: swing up to the upright position (θ ≈ 0, ω ≈ 0).
"""

import math
from typing import Any, List, Optional, Tuple

import numpy as np

from environments.binned_continuous_adapter import BinnedContinuousAdapter, clamp
from core.state_space import BinnedContinuousStateSpace

def angle_from_obs(obs: np.ndarray) -> float:
    """Extract angle from Pendulum observation [cos(θ), sin(θ), ω]."""
    cos_t, sin_t = obs[0], obs[1]
    return math.atan2(sin_t, cos_t)

class PendulumAdapter(BinnedContinuousAdapter):
    """Adapter for Pendulum continuous control environment.

    State representation: (theta_bin, omega_bin) discretized angle and
    angular velocity.
    Actions: Discretized torque values from [-2.0, 2.0].
    Goal: Upright position (θ ≈ 0) with low angular velocity.
    """

    def __init__(self, env, n_theta_bins: int = 20, n_omega_bins: int = 20,
                 n_torque_bins: int = 5):
        """
        Args:
            env: Gymnasium Pendulum-v1 environment
            n_theta_bins: Number of angle bins (covers [-π, π])
            n_omega_bins: Number of angular velocity bins (covers [-8, 8])
            n_torque_bins: Number of discrete torque actions (covers [-2, 2])
        """
        self._env = env
        self.n_theta_bins = n_theta_bins
        self.n_omega_bins = n_omega_bins
        self.n_torque_bins = n_torque_bins
        self._state_space = BinnedContinuousStateSpace([n_theta_bins, n_omega_bins])
        self._n_actions = n_torque_bins

        # Discrete torque values
        self._discrete_torques = np.linspace(-2.0, 2.0, n_torque_bins)

        # Create discretization bins
        self.theta_space, self.omega_space = self._create_bins()

        self._current_obs = None
        self._current_state = None

    def _create_bins(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create angle and angular velocity bin edges.

        Uses interior edges only (n-1 edges for n bins) following the
        Acrobot pattern so that ``np.digitize`` naturally produces bin
        indices in [0, n_bins-1] after clamping.
        """
        theta_space = np.linspace(-np.pi, np.pi, self.n_theta_bins + 1)[1:-1]
        omega_space = np.linspace(-8.0, 8.0, self.n_omega_bins + 1)[1:-1]
        return theta_space, omega_space

    def discretize_obs(self, obs: np.ndarray) -> Tuple[int, int]:
        """Convert continuous observation to discrete bin indices.

        Accepts either:
        - Gym observation [cos(θ), sin(θ), ω]  (length 3)
        - Direct (θ, ω) tuple/array            (length 2)
        """
        if len(obs) == 3:
            theta = angle_from_obs(obs)
            omega = float(obs[2])
        elif len(obs) == 2:
            theta, omega = float(obs[0]), float(obs[1])
        else:
            raise ValueError(f"Expected obs of length 2 or 3, got {len(obs)}")

        # Wrap angle to [-π, π]
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi

        i = clamp(np.digitize(theta, self.theta_space), 1, self.n_theta_bins) - 1
        j = clamp(np.digitize(omega, self.omega_space), 1, self.n_omega_bins) - 1
        return (i, j)

    def _process_action(self, action: int):
        """Convert discrete action index to continuous torque array.

        Pendulum-v1 expects a numpy array [torque] as the action.
        """
        return np.array([self._discrete_torques[action]])

    # ==================== Environment Interaction ====================

    def reset(self, init_state: Optional[Any] = None) -> np.ndarray:
        """Reset environment.

        Args:
            init_state: Initial state as [theta, omega] continuous values.
                        If None, uses Gym's default random initialization.

        Returns:
            One-hot encoded discretized state
        """
        obs, _ = self._env.reset()

        if init_state is not None:
            theta, omega = float(init_state[0]), float(init_state[1])
            self._env.unwrapped.state = np.array([theta, omega])
            obs = np.array([np.cos(theta), np.sin(theta), omega])

        self._current_obs = obs
        discrete_state = self.discretize_obs(obs)
        state_idx = self.state_space.state_to_index(discrete_state)
        self._current_state = self.state_space.index_to_onehot(state_idx)
        return self._current_state

    def sample_random_state(self) -> np.ndarray:
        """Reset to a uniformly random state across the full state space.

        Ensures every region of the discretized state space has a chance of
        being visited during training, which is critical for building a
        complete transition model B.

        Returns:
            One-hot encoded discretized state.
        """
        # Normal reset so Gym's bookkeeping is valid
        obs, _ = self._env.reset()

        # Sample random continuous state
        theta = np.random.uniform(-np.pi, np.pi)
        omega = np.random.uniform(-8.0, 8.0)

        # Inject into the physics engine
        self._env.unwrapped.state = np.array([theta, omega])

        # Build corresponding observation
        obs = np.array([np.cos(theta), np.sin(theta), omega])

        self._current_obs = obs
        discrete_state = self.discretize_obs(obs)
        state_idx = self.state_space.state_to_index(discrete_state)
        self._current_state = self.state_space.index_to_onehot(state_idx)
        return self._current_state

    # ==================== Goal / Reward ====================

    def get_goal_states(self, goal_spec: Any) -> List[int]:
        """Convert goal specification to state indices.

        Args:
            goal_spec: None (default: upright position),
                       (theta, omega) tuple for specific state,
                       or list of state indices.

        Returns:
            List of goal state indices
        """
        if goal_spec is None:
            # Default: upright position (θ ≈ 0, ω ≈ 0)
            # 3×3 neighborhood around center bins
            center_theta = self.n_theta_bins // 2
            center_omega = self.n_omega_bins // 2

            goal_states = []
            for dt in range(-1, 2):
                for dw in range(-1, 2):
                    t = center_theta + dt
                    w = center_omega + dw
                    if 0 <= t < self.n_theta_bins and 0 <= w < self.n_omega_bins:
                        goal_states.append(
                            self.state_space.state_to_index((t, w))
                        )
            return goal_states

        elif isinstance(goal_spec, (tuple, list, np.ndarray)) and len(goal_spec) == 2:
            # Specific (theta, omega)
            discrete = self.discretize_obs(goal_spec)
            return [self.state_space.state_to_index(discrete)]

        else:
            raise ValueError(f"Invalid goal specification: {goal_spec}")

    # ==================== Shaped Reward ====================

    def create_shaped_prior(self, scale: float = 10.0) -> np.ndarray:
        """Create a continuous shaped reward vector C for all states.

        Mirrors the Pendulum-v1 reward structure:
            r = -(θ² + 0.1·ω² + 0.001·τ²)
        Since C is per-state (torque is per-action), we use the state part:
            C(s) = -(θ² + 0.1·ω²)

        The raw values are shifted and rescaled so that the best state
        (θ=0, ω=0) has C = +scale and the worst state has C = -scale.
        This positive/negative split is essential for the SR framework:
        V = M @ C must be positive at desirable states so the agent is
        attracted toward them.  An all-negative C would make V most
        positive at poorly-connected edge states (small M rows), producing
        the opposite of the intended gradient.

        Args:
            scale: Half-amplitude.  C ∈ [-scale, +scale].

        Returns:
            C vector of shape (n_states,)
        """
        theta_centers, omega_centers = self.get_bin_centers()
        C_raw = np.zeros(self.n_states)
        for t in range(self.n_theta_bins):
            for w in range(self.n_omega_bins):
                idx = self.state_space.state_to_index((t, w))
                theta = theta_centers[t]
                omega = omega_centers[w]
                C_raw[idx] = -(theta ** 2 + 0.1 * omega ** 2)

        # Shift and rescale to [-scale, +scale]
        # Best state (θ=0, ω=0) → +scale, worst → -scale
        c_min, c_max = C_raw.min(), C_raw.max()
        c_range = c_max - c_min
        if c_range > 0:
            C = 2.0 * scale * (C_raw - c_min) / c_range - scale
        else:
            C = np.zeros_like(C_raw)
        return C

    def create_sparse_prior(self, radius: float = 1.0,
                            reward: float = 10.0,
                            default_cost: float = -0.1) -> np.ndarray:
        """Create a sparse reward vector C for all states.

        Gives ``+reward`` to states within a quadratic ball of the
        upright position and ``default_cost`` everywhere else::

            C(s) = reward      if  θ² + 0.1·ω² < radius²
                   default_cost   otherwise

        This uses the same quadratic metric as :meth:`create_shaped_prior`
        but applies a hard threshold, producing a step-function reward
        landscape.  With sparse C the value function ``V = M @ C`` has a
        much weaker gradient far from goal, making flat planning harder.
        Hierarchical planning compensates because ``C_macro`` concentrates
        the sparse signal at the cluster level.

        Args:
            radius: Threshold in the ``(θ, √0.1·ω)`` metric.
                A value of ~1.0 yields roughly 15-25 goal states
                at 21×21 resolution (~4 % of state space).
            reward: Reward value for states inside the ball.
            default_cost: Cost for states outside the ball
                (small negative encourages reaching goal faster).

        Returns:
            C vector of shape ``(n_states,)``
        """
        theta_centers, omega_centers = self.get_bin_centers()
        C = np.full(self.n_states, default_cost)

        n_goal = 0
        for t in range(self.n_theta_bins):
            for w in range(self.n_omega_bins):
                theta = theta_centers[t]
                omega = omega_centers[w]
                if theta ** 2 + 0.1 * omega ** 2 < radius ** 2:
                    idx = self.state_space.state_to_index((t, w))
                    C[idx] = reward
                    n_goal += 1

        print(f"Sparse prior: {n_goal}/{self.n_states} goal states "
              f"(radius={radius}, reward={reward}, cost={default_cost})")
        return C

    # ==================== Clustering ====================

    def get_clustering_affinity(self, M: np.ndarray,
                                sigma: float = 1.0,
                                blend: float = 0.5) -> np.ndarray:
        """Build a clustering affinity that respects angle periodicity.

        The standard spectral clustering uses M directly as a precomputed
        affinity.  For the pendulum, θ is periodic: θ = -π and θ = +π are
        the *same* physical state, but they sit at opposite grid edges.
        Pure M-based clustering can split these dynamically-connected
        boundary states into different clusters.

        This method builds a **cylindrical distance kernel** — embedding
        each state as (cos θ, sin θ, ω) — and blends it with M.  The
        blend weight is **adaptive**: the spatial kernel contributes more
        for state pairs where M is poorly estimated (low row sums,
        indicating sparse visitation) and less where M is reliable.
        This avoids artificially connecting states that the dynamics
        show are truly disconnected (e.g. wall-separated regions).

        Args:
            M: Successor matrix (N × N), already symmetrised.
            sigma: Bandwidth for the cylindrical RBF kernel.
            blend: Maximum weight for the spatial kernel.  The per-pair
                   effective weight is ``blend * (1 - confidence_ij)``
                   where confidence is derived from M's row sums.

        Returns:
            Blended affinity matrix (N × N), non-negative and symmetric.
        """
        theta_centers, omega_centers = self.get_bin_centers()

        # Normalise ω to similar scale as the unit-circle coords
        omega_scale = 8.0  # max |ω|
        N = self.n_states
        coords = np.zeros((N, 3))  # (cos θ, sin θ, ω_norm)

        for t in range(self.n_theta_bins):
            for w in range(self.n_omega_bins):
                idx = self.state_space.state_to_index((t, w))
                coords[idx, 0] = np.cos(theta_centers[t])
                coords[idx, 1] = np.sin(theta_centers[t])
                coords[idx, 2] = omega_centers[w] / omega_scale

        # Pairwise squared Euclidean distance in cylindrical space
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
        theta_bin, omega_bin = self.state_space.index_to_state(state_index)
        return f"(θ{theta_bin},ω{omega_bin})"

    def get_bin_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get center values of theta and omega bins.

        Useful for value function heatmaps.
        """
        theta_edges = np.linspace(-np.pi, np.pi, self.n_theta_bins + 1)
        omega_edges = np.linspace(-8.0, 8.0, self.n_omega_bins + 1)
        theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
        omega_centers = 0.5 * (omega_edges[:-1] + omega_edges[1:])
        return theta_centers, omega_centers

    def get_dimension_labels(self) -> Tuple[str, str]:
        """Return human-readable axis labels for the 2D state space."""
        return ("Angle (θ)", "Angular Velocity (ω)")

    def get_bin_edges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return bin edge arrays for each dimension.

        Used by cluster and trajectory visualizations to map bin indices
        to physical coordinates.
        """
        theta_edges = np.linspace(-np.pi, np.pi, self.n_theta_bins + 1)
        omega_edges = np.linspace(-8.0, 8.0, self.n_omega_bins + 1)
        return theta_edges, omega_edges

    def obs_to_continuous(self, obs: np.ndarray) -> Tuple[float, float]:
        """Extract (theta, omega) from raw observation.

        Accepts either Gym obs [cos(θ), sin(θ), ω] or direct (θ, ω).
        """
        if len(obs) == 3:
            theta = angle_from_obs(obs)
            omega = float(obs[2])
        else:
            theta, omega = float(obs[0]), float(obs[1])
        return theta, omega

    def get_action_labels(self) -> List[str]:
        """Return human-readable labels for each action index."""
        torque_edges = np.linspace(-2.0, 2.0, self._n_actions + 1)
        centers = 0.5 * (torque_edges[:-1] + torque_edges[1:])
        return [f"τ={c:.1f}" for c in centers]

    def render(self) -> Optional[np.ndarray]:
        """Render the current frame (for video capture)."""
        return self._env.render()
