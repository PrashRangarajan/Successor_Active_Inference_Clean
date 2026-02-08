"""Mountain Car environment adapter for Hierarchical SR Active Inference."""

from typing import Any, List, Optional, Tuple
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.base_environment import BaseEnvironmentAdapter
from core.state_space import BinnedContinuousStateSpace


def clamp(x: int, minimum: int, maximum: int) -> int:
    """Clamp integer to range."""
    return max(minimum, min(maximum, x))


class MountainCarAdapter(BaseEnvironmentAdapter):
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

    @property
    def state_space(self) -> BinnedContinuousStateSpace:
        return self._state_space

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def env(self) -> Any:
        return self._env

    @property
    def transition_matrix_shape(self) -> Tuple[int, ...]:
        N = self.n_states
        return (N, N, self._n_actions)

    @property
    def successor_matrix_shape(self) -> Tuple[int, ...]:
        N = self.n_states
        return (N, N)

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

    def step(self, action: int) -> np.ndarray:
        """Take action and return new discretized state."""
        step_result = self._env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False

        self._current_obs = obs
        discrete_state = self.discretize_obs(obs)
        state_idx = self.state_space.state_to_index(discrete_state)
        self._current_state = self.state_space.index_to_onehot(state_idx)
        return self._current_state

    def step_with_info(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take action and return full step information."""
        step_result = self._env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False

        self._current_obs = obs
        discrete_state = self.discretize_obs(obs)
        state_idx = self.state_space.state_to_index(discrete_state)
        self._current_state = self.state_space.index_to_onehot(state_idx)
        return self._current_state, reward, terminated, truncated, info

    def get_current_state(self) -> Tuple[int, int]:
        """Get current discrete state (pos_bin, vel_bin)."""
        return self.discretize_obs(self._current_obs)

    def get_current_state_index(self) -> int:
        """Get current state as flat index."""
        return self.state_space.state_to_index(self.get_current_state())

    def get_current_obs(self) -> np.ndarray:
        """Get current continuous observation [pos, vel]."""
        return self._current_obs

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

    # ==================== Matrix Operations ====================

    def multiply_B_s(self, B: np.ndarray, state: np.ndarray, action: Optional[int]) -> np.ndarray:
        """Multiply transition matrix with state vector.

        B shape: (N, N, n_actions)
        state shape: (N,)
        """
        if action is not None:
            return B[:, :, action] @ state
        else:
            result = np.zeros_like(state)
            for a in range(self._n_actions):
                result += B[:, :, a] @ state
            return result / self._n_actions

    def multiply_M_C(self, M: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Multiply successor matrix with preference vector.

        M shape: (N, N)
        C shape: (N,)
        Returns: (N,)
        """
        return M @ C

    # ==================== Transition Matrix ====================

    def get_transition_matrix(self) -> np.ndarray:
        """Mountain Car doesn't have a known transition matrix.

        Returns empty matrix - must be learned from experience.
        """
        return self.create_empty_transition_matrix()

    def normalize_transition_matrix(self, B: np.ndarray, goal_states: List[int] = None) -> np.ndarray:
        """Normalize transition matrix."""
        N = self.n_states

        # Handle zero columns
        for col in range(N):
            for action in range(self._n_actions):
                col_sum = np.sum(B[:, col, action])
                if col_sum == 0:
                    B[col, col, action] = 1

        # Normalize
        B = B / B.sum(axis=0, keepdims=True)

        # Make goal states absorbing
        if goal_states:
            for gs in goal_states:
                B[:, gs, :] = 0
                B[gs, gs, :] = 1

        return B

    # ==================== Goal/Reward ====================

    def create_goal_prior(self, goal_states: List[int], reward: float = 10.0,
                          default_cost: float = -1.0) -> np.ndarray:
        """Create goal preference vector."""
        C = np.ones(self.n_states) * default_cost
        for gs in goal_states:
            C[gs] = reward
        return C

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

    # ==================== Visualization ====================

    def render_state(self, state_index: int) -> Tuple[int, int]:
        """Get (pos_bin, vel_bin) for visualization."""
        return self.state_space.index_to_state(state_index)

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
