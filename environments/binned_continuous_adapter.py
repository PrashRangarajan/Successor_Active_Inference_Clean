"""Base adapter for continuous environments with binned state spaces.

Extracts shared logic from MountainCar, Acrobot, Pendulum, and CartPole
adapters. Subclasses must implement environment-specific methods:
- __init__: Create env, bins, state space
- discretize_obs: Convert continuous observation to discrete bins
- get_goal_states: Define goal region
- sample_random_state: Generate diverse training starts
"""

import numpy as np
from typing import Any, List, Optional, Tuple
from core.base_environment import BaseEnvironmentAdapter
from core.state_space import BinnedContinuousStateSpace


def clamp(x: int, minimum: int, maximum: int) -> int:
    """Clamp integer to range [minimum, maximum]."""
    return max(minimum, min(maximum, x))


class BinnedContinuousAdapter(BaseEnvironmentAdapter):
    """Base adapter for Gymnasium continuous environments with discretized state.

    Subclasses must set these attributes in __init__:
        self._env: Gymnasium environment
        self._state_space: BinnedContinuousStateSpace instance
        self._n_actions: int
        self._current_obs: np.ndarray (continuous observation)
        self._current_state: np.ndarray (one-hot discrete state)

    Subclasses must implement:
        discretize_obs(obs) -> Tuple[int, ...]
        get_goal_states(goal_spec) -> List[int]
        sample_random_state() -> np.ndarray  (for diverse training starts)
    """

    # --- Properties (identical across all 4 adapters) ---

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

    # --- State access (identical across all 4) ---

    def get_current_state(self) -> Tuple[int, ...]:
        """Get current discrete state as bin indices."""
        return self.discretize_obs(self._current_obs)

    def get_current_state_index(self) -> int:
        """Get current state as flat index."""
        return self.state_space.state_to_index(self.get_current_state())

    def get_current_obs(self) -> np.ndarray:
        """Get current continuous observation."""
        return self._current_obs

    # --- Actions (identical across all 4) ---

    def step(self, action: int) -> np.ndarray:
        """Take action and return new one-hot state.

        Subclasses with continuous action spaces should override
        _process_action() instead of this method.
        """
        env_action = self._process_action(action)
        obs, reward, terminated, truncated, info = self._env.step(env_action)
        self._current_obs = obs
        discrete_state = self.discretize_obs(obs)
        state_idx = self.state_space.state_to_index(discrete_state)
        self._current_state = self.state_space.index_to_onehot(state_idx)
        return self._current_state

    def step_with_info(self, action: int):
        """Take action and return (state, reward, terminated, truncated, info)."""
        env_action = self._process_action(action)
        obs, reward, terminated, truncated, info = self._env.step(env_action)
        self._current_obs = obs
        discrete_state = self.discretize_obs(obs)
        state_idx = self.state_space.state_to_index(discrete_state)
        self._current_state = self.state_space.index_to_onehot(state_idx)
        return self._current_state, reward, terminated, truncated, info

    def _process_action(self, action: int):
        """Convert discrete action index to environment action.

        Override in subclasses with continuous action spaces (e.g. Pendulum).
        Default: pass through as-is (for discrete action envs).
        """
        return action

    # --- Matrix operations (identical across all 4) ---

    def multiply_B_s(self, B: np.ndarray, state: np.ndarray,
                     action: Optional[int]) -> np.ndarray:
        """Multiply transition matrix with state vector.

        B shape: (N, N, n_actions), state shape: (N,).
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

        M shape: (N, N), C shape: (N,). Returns: (N,).
        """
        return M @ C

    def get_transition_matrix(self) -> np.ndarray:
        """Continuous environments don't have a known transition matrix.

        Returns empty matrix — must be learned from experience.
        """
        return self.create_empty_transition_matrix()

    def normalize_transition_matrix(self, B: np.ndarray,
                                     goal_states: List[int] = None) -> np.ndarray:
        """Normalize transition matrix columns and make goal states absorbing."""
        N = self.n_states

        # Handle zero columns (unvisited state-action pairs -> self-loop)
        for col in range(N):
            for action in range(self._n_actions):
                col_sum = np.sum(B[:, col, action])
                if col_sum == 0:
                    B[col, col, action] = 1

        # Column-normalize
        B = B / B.sum(axis=0, keepdims=True)

        # Make goal states absorbing
        if goal_states:
            for gs in goal_states:
                B[:, gs, :] = 0
                B[gs, gs, :] = 1

        return B

    def create_goal_prior(self, goal_states: List[int], reward: float = 10.0,
                          default_cost: float = -1.0) -> np.ndarray:
        """Create goal preference vector C."""
        C = np.ones(self.n_states) * default_cost
        for gs in goal_states:
            C[gs] = reward
        return C

    # --- Visualization (identical across all 4) ---

    def render_state(self, state_index: int) -> Tuple[int, ...]:
        """Get state tuple for visualization."""
        return self.state_space.index_to_state(state_index)

    def render(self):
        """Render current frame (for video recording)."""
        return self._env.render()

    # --- Abstract methods (subclasses MUST implement) ---

    def discretize_obs(self, obs: np.ndarray) -> Tuple[int, ...]:
        """Convert continuous observation to discrete bin indices.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def get_goal_states(self, goal_spec=None) -> List[int]:
        """Return list of goal state indices.

        Must be implemented by subclasses.
        """
        raise NotImplementedError
