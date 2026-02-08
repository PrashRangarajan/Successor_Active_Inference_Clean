"""Acrobot environment adapter for Hierarchical SR Active Inference.

Acrobot is a 2-link planar robot that must swing the end-effector above a threshold.
The observation space is 6D: [cos(theta1), sin(theta1), cos(theta2), sin(theta2), dtheta1, dtheta2]
Actions: 0 = torque -1, 1 = torque 0, 2 = torque +1

We discretize into a 4D state space: (theta1_bin, theta2_bin, dtheta1_bin, dtheta2_bin)
"""

from typing import Any, List, Optional, Tuple
import numpy as np
import math

from environments.binned_continuous_adapter import BinnedContinuousAdapter, clamp
from core.state_space import BinnedContinuousStateSpace

def angles_from_obs(obs) -> Tuple[float, float]:
    """Convert Acrobot observation to angles (theta1, theta2).

    Standard Gym/Gymnasium Acrobot observation:
        [cos(theta1), sin(theta1), cos(theta2), sin(theta2), thetaDot1, thetaDot2]
    """
    c1, s1, c2, s2 = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
    theta1 = math.atan2(s1, c1)
    theta2 = math.atan2(s2, c2)
    return theta1, theta2

def angles_and_vels_from_obs(obs) -> Tuple[float, float, float, float]:
    """Convert Acrobot observation to (theta1, theta2, thetaDot1, thetaDot2)."""
    theta1, theta2 = angles_from_obs(obs)
    dtheta1 = float(obs[4])
    dtheta2 = float(obs[5])
    return theta1, theta2, dtheta1, dtheta2

class AcrobotAdapter(BinnedContinuousAdapter):
    """Adapter for Gymnasium Acrobot-v1 environment.

    State representation: (theta1_bin, theta2_bin, dtheta1_bin, dtheta2_bin)
    Discretizes the continuous 4D state into bins.

    Transition matrix shape: (N, N, n_actions) where N = product of all bins
    Actions: 0=torque -1, 1=torque 0, 2=torque +1
    """

    def __init__(self, env, n_theta_bins: int = 15, n_dtheta_bins: int = 15,
                 goal_velocity_filter: bool = False):
        """
        Args:
            env: Gymnasium Acrobot-v1 environment
            n_theta_bins: Number of bins for each angle (theta1, theta2)
            n_dtheta_bins: Number of bins for each angular velocity (dtheta1, dtheta2)
            goal_velocity_filter: If True, only include goal states where the
                first link has upward angular velocity (dt1 >= n_dtheta_bins // 2).
                This reduces goal state dilution from ~19% to ~6-10%, concentrating
                goals in fewer macro clusters for better hierarchical planning.
        """
        self._env = env
        self.n_theta_bins = n_theta_bins
        self.n_dtheta_bins = n_dtheta_bins
        self.goal_velocity_filter = goal_velocity_filter

        # 4D state space
        self.bin_sizes = (n_theta_bins, n_theta_bins, n_dtheta_bins, n_dtheta_bins)
        self._state_space = BinnedContinuousStateSpace(list(self.bin_sizes))
        self._n_actions = env.action_space.n

        # Get observation space bounds
        self.low = env.observation_space.low
        self.high = env.observation_space.high

        # Create discretization bins
        (
            self.theta1_space,
            self.theta2_space,
            self.dtheta1_space,
            self.dtheta2_space,
        ) = self._create_bins()

        self._current_obs = None
        self._current_state = None

    def _create_bins(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create bin edges for Acrobot angles and angular velocities.

        Angles are in [-pi, pi]. Angular velocity ranges are set to common
        Acrobot limits (dtheta1 in [-4*pi, 4*pi], dtheta2 in [-9*pi, 9*pi]).
        """
        # Angles
        theta1_low, theta1_high = -np.pi, np.pi
        theta2_low, theta2_high = -np.pi, np.pi

        theta1_space = np.linspace(theta1_low, theta1_high, self.n_theta_bins + 1)[1:-1]
        theta2_space = np.linspace(theta2_low, theta2_high, self.n_theta_bins + 1)[1:-1]

        # Angular velocities (typical Acrobot ranges)
        dtheta1_low, dtheta1_high = -4.0 * np.pi, 4.0 * np.pi
        dtheta2_low, dtheta2_high = -9.0 * np.pi, 9.0 * np.pi

        dtheta1_space = np.linspace(dtheta1_low, dtheta1_high, self.n_dtheta_bins + 1)[1:-1]
        dtheta2_space = np.linspace(dtheta2_low, dtheta2_high, self.n_dtheta_bins + 1)[1:-1]

        return theta1_space, theta2_space, dtheta1_space, dtheta2_space

    def discretize_obs(self, obs: np.ndarray) -> Tuple[int, int, int, int]:
        """Discretize an observation into (theta1_bin, theta2_bin, dtheta1_bin, dtheta2_bin).

        Accepts either:
          - full Acrobot obs: [cos(t1), sin(t1), cos(t2), sin(t2), dt1, dt2]
          - a 4-tuple/list: (theta1, theta2, dtheta1, dtheta2)
        """
        if isinstance(obs, (list, tuple, np.ndarray)) and len(obs) == 4:
            theta1, theta2, dtheta1, dtheta2 = map(float, obs)
        else:
            theta1, theta2, dtheta1, dtheta2 = angles_and_vels_from_obs(obs)

        # Wrap angles to [-pi, pi] for stable binning
        theta1 = ((theta1 + np.pi) % (2 * np.pi)) - np.pi
        theta2 = ((theta2 + np.pi) % (2 * np.pi)) - np.pi

        i = clamp(np.digitize(theta1, self.theta1_space), 1, self.n_theta_bins) - 1
        j = clamp(np.digitize(theta2, self.theta2_space), 1, self.n_theta_bins) - 1
        k = clamp(np.digitize(dtheta1, self.dtheta1_space), 1, self.n_dtheta_bins) - 1
        l = clamp(np.digitize(dtheta2, self.dtheta2_space), 1, self.n_dtheta_bins) - 1
        return (i, j, k, l)

    # ==================== Environment Interaction ====================

    def reset(self, init_state: Optional[Any] = None) -> np.ndarray:
        """Reset environment.

        Args:
            init_state: Initial state as [theta1, theta2, dtheta1, dtheta2] continuous values
                       or None for random initialization

        Returns:
            One-hot encoded discretized state
        """
        obs, _ = self._env.reset()

        if init_state is not None:
            # Set specific initial state
            self._env.unwrapped.state = np.array(init_state)
            # Need to re-get observation after setting state
            # Acrobot stores state as [theta1, theta2, dtheta1, dtheta2]
            state = np.array(init_state)
            # Convert to observation format [cos, sin, cos, sin, dtheta1, dtheta2]
            obs = np.array([
                np.cos(state[0]), np.sin(state[0]),
                np.cos(state[1]), np.sin(state[1]),
                state[2], state[3]
            ])

        self._current_obs = obs
        discrete_state = self.discretize_obs(obs)
        state_idx = self.state_space.state_to_index(discrete_state)
        self._current_state = self.state_space.index_to_onehot(state_idx)
        return self._current_state

    def get_state_for_reset(self) -> Any:
        """Get current state in a format suitable for reset().

        Acrobot reset() expects continuous [theta1, theta2, dtheta1, dtheta2],
        not discrete bin indices. Extract angles from observation.
        """
        return list(angles_and_vels_from_obs(self._current_obs))

    def sample_random_state(self) -> np.ndarray:
        """Reset to a uniformly random state across the full state space.

        Samples random angles in [-pi, pi] and angular velocities within
        the Acrobot's physical limits, then sets the underlying Gym state
        directly.  This ensures every region of the discretised state space
        has a chance of being visited during training — critical for building
        a complete transition model B.

        Returns:
            One-hot encoded discretised state (same as ``reset``).
        """
        # First do a normal reset so Gym's internal bookkeeping is valid
        obs, _ = self._env.reset()

        # Sample random continuous state
        theta1 = np.random.uniform(-np.pi, np.pi)
        theta2 = np.random.uniform(-np.pi, np.pi)
        dtheta1 = np.random.uniform(-4 * np.pi, 4 * np.pi)
        dtheta2 = np.random.uniform(-9 * np.pi, 9 * np.pi)

        # Inject into the physics engine
        self._env.unwrapped.state = np.array([theta1, theta2, dtheta1, dtheta2])

        # Build the corresponding observation
        obs = np.array([
            np.cos(theta1), np.sin(theta1),
            np.cos(theta2), np.sin(theta2),
            dtheta1, dtheta2,
        ])

        self._current_obs = obs
        discrete_state = self.discretize_obs(obs)
        state_idx = self.state_space.state_to_index(discrete_state)
        self._current_state = self.state_space.index_to_onehot(state_idx)
        return self._current_state

    # ==================== Goal/Reward ====================

    def get_goal_states(self, goal_spec: Any = None) -> List[int]:
        """Convert goal specification to state indices.

        For Acrobot, the default goal is the terminal condition used by the environment:
            -cos(theta1) - cos(theta1 + theta2) > 1.0

        A bin is marked as a goal if the bin CENTER satisfies the terminal
        condition.  With 6 angle bins this gives ~17% of angle bins as goals,
        closely matching the true continuous fraction (~18.5%).

        Args:
            goal_spec: None (default terminal condition),
                      or (theta1, theta2) continuous angles for a target bin

        Returns:
            List of goal state indices
        """
        if goal_spec is None:
            goal_states = []

            # Bin edges and centers
            theta1_edges = np.linspace(-np.pi, np.pi, self.n_theta_bins + 1)
            theta2_edges = np.linspace(-np.pi, np.pi, self.n_theta_bins + 1)
            theta1_centers = (theta1_edges[:-1] + theta1_edges[1:]) / 2.0
            theta2_centers = (theta2_edges[:-1] + theta2_edges[1:]) / 2.0

            for i in range(self.n_theta_bins):
                for j in range(self.n_theta_bins):
                    t1 = theta1_centers[i]
                    t2 = theta2_centers[j]
                    if (-np.cos(t1) - np.cos(t1 + t2)) > 1.0:
                        for dt1 in range(self.n_dtheta_bins):
                            # When filtering, only include states where the first
                            # link has upward angular velocity.  This concentrates
                            # goals in a smaller, more navigable region.
                            if self.goal_velocity_filter and dt1 < self.n_dtheta_bins // 2:
                                continue
                            for dt2 in range(self.n_dtheta_bins):
                                state_idx = self.state_space.state_to_index((i, j, dt1, dt2))
                                goal_states.append(state_idx)

            if not goal_states:
                # Fallback if coarse bins miss threshold
                print("Warning: No goal states found with terminal condition, using fallback")
                # Pick high-elevation states
                for dt1 in range(self.n_dtheta_bins):
                    for dt2 in range(self.n_dtheta_bins):
                        state_idx = self.state_space.state_to_index(
                            (self.n_theta_bins - 1, self.n_theta_bins // 2, dt1, dt2)
                        )
                        goal_states.append(state_idx)

            return list(set(goal_states))

        elif isinstance(goal_spec, (list, tuple, np.ndarray)) and len(goal_spec) == 2:
            # Specific (theta1, theta2) target
            theta1, theta2 = goal_spec
            i = clamp(np.digitize(theta1, self.theta1_space), 1, self.n_theta_bins) - 1
            j = clamp(np.digitize(theta2, self.theta2_space), 1, self.n_theta_bins) - 1
            # All velocity states at this angle
            goal_states = []
            for dt1 in range(self.n_dtheta_bins):
                for dt2 in range(self.n_dtheta_bins):
                    state_idx = self.state_space.state_to_index((i, j, dt1, dt2))
                    goal_states.append(state_idx)
            return goal_states

        else:
            raise ValueError(f"Invalid goal specification for Acrobot: {goal_spec}")

    def is_terminal(self, obs: np.ndarray = None) -> bool:
        """Check if current or given observation is terminal.

        Uses the continuous state directly to check the Acrobot terminal
        condition: -cos(theta1) - cos(theta1 + theta2) > 1.0

        When goal_velocity_filter is enabled (coarse-bin mode), the discrete
        goal-bin check in _is_at_goal() is already curated to be physically
        meaningful, so we return None to let the agent rely on discrete bins
        alone.  This avoids false negatives where a coarse goal bin's center
        satisfies the threshold but the actual continuous state does not.
        """
        if self.goal_velocity_filter:
            return None  # Fall back to discrete goal-bin check only

        if obs is None:
            obs = self._current_obs
        theta1, theta2 = angles_from_obs(obs)
        return (-np.cos(theta1) - np.cos(theta1 + theta2)) > 1.0

    def is_in_goal_bin(self, goal_states: List[int], obs: np.ndarray = None) -> bool:
        """Check if the current discretized state is in the goal state set.

        This matches the agent's goal check exactly, avoiding mismatches
        between the continuous terminal condition and the discretized goal bins.
        """
        if obs is None:
            obs = self._current_obs
        discrete_state = self.discretize_obs(obs)
        state_idx = self.state_space.state_to_index(discrete_state)
        return state_idx in goal_states

    # ==================== Visualization ====================

    def get_state_label(self, state_index: int) -> str:
        """Get human-readable state label."""
        t1, t2, dt1, dt2 = self.state_space.index_to_state(state_index)
        return f"(t1:{t1},t2:{t2},dt1:{dt1},dt2:{dt2})"

    def get_angle_bin_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get center values of angle bins."""
        theta1_centers = np.linspace(-np.pi, np.pi, self.n_theta_bins)
        theta2_centers = np.linspace(-np.pi, np.pi, self.n_theta_bins)
        return theta1_centers, theta2_centers

    def get_vel_bin_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get center values of angular velocity bins."""
        dtheta1_centers = np.linspace(-4 * np.pi, 4 * np.pi, self.n_dtheta_bins)
        dtheta2_centers = np.linspace(-9 * np.pi, 9 * np.pi, self.n_dtheta_bins)
        return dtheta1_centers, dtheta2_centers
