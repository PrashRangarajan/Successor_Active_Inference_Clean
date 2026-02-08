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

    def render(self) -> Optional[np.ndarray]:
        """Render the current frame (for video capture)."""
        return self._env.render()
