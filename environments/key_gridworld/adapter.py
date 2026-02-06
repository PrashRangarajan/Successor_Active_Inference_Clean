"""Key Gridworld environment adapter for Hierarchical SR Active Inference.

This environment has augmented state: (location, has_key) where has_key in {0, 1}.
The agent must pick up a key before reaching the goal.
"""

from typing import Any, List, Optional, Tuple
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from core.base_environment import BaseEnvironmentAdapter
from core.state_space import AugmentedStateSpace


class KeyGridworldAdapter(BaseEnvironmentAdapter):
    """Adapter for gridworld with key pickup.

    State representation: (location_idx, has_key) where has_key in {0, 1}
    Transition matrix shape: (N, 2, N, 2, n_actions)
    Successor matrix shape: (N, 2, N, 2)
    Actions: 0=left, 1=right, 2=up, 3=down, [4=pickup if enabled]
    """

    def __init__(self, env, grid_size: int, has_pickup_action: bool = False):
        """
        Args:
            env: SR_Gridworld environment with key
            grid_size: Size of the grid
            has_pickup_action: Whether there's a separate pickup action (5 actions vs 4)
        """
        self._env = env
        self.grid_size = grid_size
        self.base_n_states = grid_size ** 2
        self._state_space = AugmentedStateSpace(self.base_n_states, n_augment=2)
        self._n_actions = 5 if has_pickup_action else 4
        self._current_state = None

    @property
    def state_space(self) -> AugmentedStateSpace:
        return self._state_space

    @property
    def n_actions(self) -> int:
        return self._n_actions

    @property
    def env(self) -> Any:
        return self._env

    @property
    def transition_matrix_shape(self) -> Tuple[int, ...]:
        N = self.base_n_states
        return (N, 2, N, 2, self._n_actions)

    @property
    def successor_matrix_shape(self) -> Tuple[int, ...]:
        N = self.base_n_states
        return (N, 2, N, 2)

    # ==================== State Conversions ====================

    def loc_to_base_idx(self, loc: Tuple[int, int]) -> int:
        """Convert (x, y) to base location index."""
        return loc[0] * self.grid_size + loc[1]

    def base_idx_to_loc(self, idx: int) -> Tuple[int, int]:
        """Convert base location index to (x, y)."""
        return divmod(idx, self.grid_size)

    def full_state_to_tuple(self, state: Tuple[int, int]) -> Tuple[int, int, int]:
        """Convert (base_idx, has_key) to (x, y, has_key)."""
        base_idx, has_key = state
        x, y = self.base_idx_to_loc(base_idx)
        return (x, y, has_key)

    def tuple_to_full_state(self, state_tuple: Tuple[int, int, int]) -> Tuple[int, int]:
        """Convert (x, y, has_key) to (base_idx, has_key)."""
        x, y, has_key = state_tuple
        base_idx = self.loc_to_base_idx((x, y))
        return (base_idx, has_key)

    # ==================== Environment Interaction ====================

    def reset(self, init_state: Optional[Any] = None) -> np.ndarray:
        """Reset environment.

        Args:
            init_state: Initial state as (x, y, has_key) tuple

        Returns:
            One-hot encoded state of shape (N, 2)
        """
        if init_state is not None:
            self._current_state = self._env.reset(init_state)
        else:
            self._current_state = self._env.reset()
        return self._current_state

    def step(self, action: int) -> np.ndarray:
        """Take action and return new state."""
        self._current_state = self._env.step(action)
        return self._current_state

    def get_current_state(self) -> Tuple[int, int, int]:
        """Get current (x, y, has_key) state."""
        return self._env.get_state()

    def get_current_state_index(self) -> int:
        """Get current state as flat index."""
        x, y, has_key = self.get_current_state()
        base_idx = self.loc_to_base_idx((x, y))
        return self.state_space.state_to_index((base_idx, has_key))

    def get_current_state_val(self) -> int:
        """Get state as single value (for compatibility with original code)."""
        return self.get_current_state_index()

    # ==================== Matrix Operations ====================

    def multiply_B_s(self, B: np.ndarray, state: np.ndarray, action: Optional[int]) -> np.ndarray:
        """Multiply transition matrix with state vector.

        B shape: (N, 2, N, 2, n_actions)
        state shape: (N, 2)

        Uses tensordot for the contraction over (N, 2) dimensions.
        """
        if action is not None:
            # Contract over last two axes of B[:,:,:,:,action] with state axes
            return np.tensordot(B[:, :, :, :, action], state, axes=([2, 3], [0, 1]))
        else:
            # Sum over all actions
            return np.tensordot(B, state, axes=([2, 3], [0, 1]))

    def multiply_M_C(self, M: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Multiply successor matrix with preference vector.

        M shape: (N, 2, N, 2)
        C shape: (N, 2)
        Returns: flattened (2N,) array

        The contraction is over the last two indices of M.
        Result is flattened in Fortran order for compatibility.
        """
        result = np.tensordot(M, C, axes=([2, 3], [0, 1]))
        return result.flatten('F')

    # ==================== Transition Matrix ====================

    def get_transition_matrix(self) -> np.ndarray:
        """Get true transition matrix from environment."""
        return self._env.get_transition_dist()

    def normalize_transition_matrix(self, B: np.ndarray, goal_states: List[int] = None) -> np.ndarray:
        """Normalize transition matrix.

        For augmented state space, normalization is over first two dimensions.

        Args:
            B: Transition matrix to normalize
            goal_states: If provided, make these states absorbing (self-loops)
        """
        N = self.base_n_states

        # Handle zero transitions with self-loops
        for init_state in range(N):
            for init_key in [0, 1]:
                for action in range(self._n_actions):
                    col_sum = np.sum(B[:, :, init_state, init_key, action])
                    if col_sum == 0:
                        B[init_state, init_key, init_state, init_key, action] = 1

        # Normalize: sum over first two axes should equal 1
        B = B / B.sum(axis=0, keepdims=True).sum(axis=1, keepdims=True)

        # Make goal states absorbing
        if goal_states:
            for gs in goal_states:
                base_idx, has_key = self.state_space.index_to_state(gs)
                B[:, :, base_idx, has_key, :] = 0
                B[base_idx, has_key, base_idx, has_key, :] = 1

        return B

    # ==================== Successor Matrix ====================

    def compute_successor_from_transition(self, B: np.ndarray, gamma: float = 0.95) -> np.ndarray:
        """Compute successor matrix for augmented state space.

        Need to flatten to 2D, compute inverse, then reshape.
        """
        # Flatten B to (2N, 2N, n_actions)
        B_flat = self._flatten_transition(B)

        # Average over actions
        B_avg = np.sum(B_flat, axis=2) / self._n_actions

        # Compute successor
        N_flat = 2 * self.base_n_states
        I = np.eye(N_flat)
        M_flat = np.linalg.pinv(I - gamma * B_avg)

        # Reshape back to (N, 2, N, 2)
        return M_flat.reshape((self.base_n_states, 2, self.base_n_states, 2), order='F')

    def _flatten_transition(self, B: np.ndarray) -> np.ndarray:
        """Flatten transition matrix from (N, 2, N, 2, A) to (2N, 2N, A).

        Uses the same ordering as the original code.
        """
        N = self.base_n_states
        result = np.zeros((2 * N, 2 * N, self._n_actions))

        for a in range(self._n_actions):
            # Stack vertically for j=0,1, then horizontally for i=0,1
            result[:, :, a] = np.hstack([
                np.vstack([B[:, j, :, i, a] for j in [0, 1]])
                for i in [0, 1]
            ])

        return result

    def flatten_successor_for_clustering(self, M: np.ndarray) -> np.ndarray:
        """Flatten successor matrix for clustering.

        M shape: (N, 2, N, 2) -> (2N, 2N)
        """
        N = self.base_n_states
        return np.hstack([
            np.vstack([M[:, j, :, i] for j in [0, 1]])
            for i in [0, 1]
        ])

    # ==================== Goal/Reward ====================

    def create_goal_prior(self, goal_states: List[int], reward: float = 100.0,
                          default_cost: float = -0.1) -> np.ndarray:
        """Create goal preference vector.

        Returns array of shape (N, 2).
        """
        C = np.ones((self.base_n_states, 2)) * default_cost

        for gs in goal_states:
            base_idx, has_key = self.state_space.index_to_state(gs)
            C[base_idx, has_key] = reward

        return C

    def get_goal_states(self, goal_spec: Any) -> List[int]:
        """Convert goal specification to state indices.

        For key environment, goal typically requires has_key=1.

        Args:
            goal_spec: (x, y) location tuple - assumes goal requires key

        Returns:
            List of goal state indices (with has_key=1)
        """
        if isinstance(goal_spec, (tuple, list)) and len(goal_spec) == 2:
            # (x, y) -> goal with key
            base_idx = self.loc_to_base_idx(goal_spec)
            return [self.state_space.state_to_index((base_idx, 1))]  # has_key=1
        elif isinstance(goal_spec, (tuple, list)) and len(goal_spec) == 3:
            # (x, y, has_key) explicit
            x, y, has_key = goal_spec
            base_idx = self.loc_to_base_idx((x, y))
            return [self.state_space.state_to_index((base_idx, has_key))]
        else:
            raise ValueError(f"Invalid goal specification: {goal_spec}")

    # ==================== Walls/Obstacles ====================

    def get_wall_indices(self) -> List[int]:
        """Get wall state indices (both has_key=0 and has_key=1)."""
        walls = self._env.get_walls()
        indices = []
        for w in walls:
            base_idx = self.loc_to_base_idx(w)
            # Walls exist in both key states
            indices.append(self.state_space.state_to_index((base_idx, 0)))
            indices.append(self.state_space.state_to_index((base_idx, 1)))
        return indices

    # ==================== Visualization ====================

    def render_state(self, state_index: int) -> Tuple[int, int, int]:
        """Get (x, y, has_key) for visualization."""
        base_idx, has_key = self.state_space.index_to_state(state_index)
        x, y = self.base_idx_to_loc(base_idx)
        return (x, y, has_key)

    def get_state_label(self, state_index: int) -> str:
        """Get human-readable state label."""
        x, y, has_key = self.render_state(state_index)
        key_str = "K" if has_key else ""
        return f"({x},{y}){key_str}"

    # ==================== Val conversions (for compatibility) ====================

    def idx_to_val(self, idx: Tuple[int, int]) -> int:
        """Convert (base_idx, has_key) to flat value (for compatibility)."""
        return self.state_space.state_to_index(idx)

    def val_to_idx(self, val: int) -> Tuple[int, int]:
        """Convert flat value to (base_idx, has_key)."""
        return self.state_space.index_to_state(val)

    def val_to_state(self, val: int) -> Tuple[int, int, int]:
        """Convert flat value to (x, y, has_key)."""
        base_idx, has_key = self.val_to_idx(val)
        x, y = self.base_idx_to_loc(base_idx)
        return (x, y, has_key)

    def state_to_val(self, state: Tuple[int, int, int]) -> int:
        """Convert (x, y, has_key) to flat value."""
        x, y, has_key = state
        base_idx = self.loc_to_base_idx((x, y))
        return self.idx_to_val((base_idx, has_key))

    def val_to_onehot(self, val: int) -> np.ndarray:
        """Convert flat value to one-hot array."""
        return self.state_space.index_to_onehot(val)

    def onehot_to_val(self, onehot: np.ndarray) -> int:
        """Convert one-hot array to flat value."""
        return self.state_space.onehot_to_index(onehot)
