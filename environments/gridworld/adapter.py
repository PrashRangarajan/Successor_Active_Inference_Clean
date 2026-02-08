"""Gridworld environment adapter for Hierarchical SR Active Inference."""

from typing import Any, List, Optional, Tuple
import numpy as np

from core.base_environment import BaseEnvironmentAdapter
from core.state_space import GridStateSpace

class GridworldAdapter(BaseEnvironmentAdapter):
    """Adapter for standard gridworld MDP environment.

    State representation: (x, y) grid coordinates
    Transition matrix shape: (N, N, n_actions) where N = grid_size^2
    Actions: 0=left, 1=right, 2=up, 3=down
    """

    def __init__(self, env, grid_size: int):
        """
        Args:
            env: SR_Gridworld environment instance
            grid_size: Size of the grid
        """
        self._env = env
        self.grid_size = grid_size
        self._state_space = GridStateSpace(grid_size)
        self._n_actions = 4
        self._current_state = None

    @property
    def state_space(self) -> GridStateSpace:
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
            init_state: Initial state as flat index or (x, y) tuple

        Returns:
            One-hot encoded initial state
        """
        if init_state is not None:
            if isinstance(init_state, int):
                # Flat index
                self._current_state = self._env.reset(init_state)
            else:
                # (x, y) tuple - convert to index
                idx = self.state_space.state_to_index(init_state)
                self._current_state = self._env.reset(idx)
        else:
            self._current_state = self._env.reset()
        return self._current_state

    def step(self, action: int) -> np.ndarray:
        """Take action and return new state."""
        self._current_state = self._env.step(action)
        return self._current_state

    def get_current_state(self) -> Tuple[int, int]:
        """Get current (x, y) position."""
        return self._env.get_state_loc()

    def get_current_state_index(self) -> int:
        """Get current state as flat index."""
        return self._env.get_state_idx()

    # ==================== Matrix Operations ====================

    def multiply_B_s(self, B: np.ndarray, state: np.ndarray, action: Optional[int]) -> np.ndarray:
        """Multiply transition matrix with state vector.

        B shape: (N, N, n_actions)
        state shape: (N,)
        """
        if action is not None:
            return B[:, :, action] @ state
        else:
            # Sum over all actions
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
        """Get true transition matrix from environment."""
        return self._env.get_transition_dist()

    def normalize_transition_matrix(self, B: np.ndarray, goal_states: List[int] = None) -> np.ndarray:
        """Normalize transition matrix.

        Args:
            B: Transition matrix to normalize
            goal_states: If provided, make these states absorbing (self-loops)
        """
        N = self.n_states

        # Handle zero columns (no transitions observed)
        for col in range(N):
            for action in range(self._n_actions):
                col_sum = np.sum(B[:, col, action])
                if col_sum == 0:
                    B[col, col, action] = 1  # Self-loop

        # Normalize
        B = B / B.sum(axis=0, keepdims=True)

        # Make goal states absorbing (they transition to themselves)
        if goal_states:
            for gs in goal_states:
                B[:, gs, :] = 0
                B[gs, gs, :] = 1

        return B

    # ==================== Goal/Reward ====================

    def create_goal_prior(self, goal_states: List[int], reward: float = 100.0,
                          default_cost: float = -0.1) -> np.ndarray:
        """Create goal preference vector."""
        C = np.ones(self.n_states) * default_cost
        for gs in goal_states:
            C[gs] = reward
        return C

    def get_goal_states(self, goal_spec: Any) -> List[int]:
        """Convert goal specification to state indices.

        Args:
            goal_spec: (x, y) tuple or flat index

        Returns:
            List containing the goal state index
        """
        if isinstance(goal_spec, int):
            return [goal_spec]
        elif isinstance(goal_spec, (tuple, list)):
            return [self.state_space.state_to_index(goal_spec)]
        else:
            raise ValueError(f"Invalid goal specification: {goal_spec}")

    # ==================== Walls/Obstacles ====================

    def get_wall_indices(self) -> List[int]:
        """Get wall state indices."""
        walls = self._env.get_walls()
        return [self.state_space.state_to_index(w) for w in walls]

    # ==================== Visualization ====================

    def render_state(self, state_index: int) -> Tuple[int, int]:
        """Get (x, y) coordinates for visualization."""
        return self.state_space.index_to_state(state_index)

    def get_state_label(self, state_index: int) -> str:
        """Get human-readable state label."""
        x, y = self.state_space.index_to_state(state_index)
        return f"({x},{y})"
