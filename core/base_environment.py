"""Abstract base class for environment adapters.

This module defines the interface that all environment-specific adapters must implement.
The adapter bridges the gap between the specific environment and the generic
Hierarchical SR Active Inference algorithm.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from .state_space import StateSpace


class BaseEnvironmentAdapter(ABC):
    """Abstract base class for environment adapters.

    An adapter wraps a specific environment (gridworld, mountain car, etc.) and provides
    a unified interface for the Hierarchical SR agent. This includes:
    - State space management
    - Transition matrix operations
    - Successor matrix operations
    - Environment interaction (reset, step)
    """

    @property
    @abstractmethod
    def state_space(self) -> StateSpace:
        """Return the StateSpace object for this environment."""
        pass

    @property
    @abstractmethod
    def n_actions(self) -> int:
        """Number of available actions."""
        pass

    @property
    @abstractmethod
    def env(self) -> Any:
        """The underlying environment object."""
        pass

    # ==================== State Space Properties ====================

    @property
    def n_states(self) -> int:
        """Total number of discrete states."""
        return self.state_space.n_states

    @property
    def state_shape(self) -> Tuple[int, ...]:
        """Shape of state representation."""
        return self.state_space.state_shape

    # ==================== Matrix Shapes ====================

    @property
    @abstractmethod
    def transition_matrix_shape(self) -> Tuple[int, ...]:
        """Shape of transition matrix B.

        For simple envs: (N, N, n_actions)
        For augmented envs: (N, K, N, K, n_actions) where K is augment dimension
        """
        pass

    @property
    @abstractmethod
    def successor_matrix_shape(self) -> Tuple[int, ...]:
        """Shape of successor matrix M.

        For simple envs: (N, N)
        For augmented envs: (N, K, N, K)
        """
        pass

    # ==================== State Conversions ====================

    def state_to_index(self, state: Any) -> int:
        """Convert environment state to flat index."""
        return self.state_space.state_to_index(state)

    def index_to_state(self, index: int) -> Any:
        """Convert flat index to environment state."""
        return self.state_space.index_to_state(index)

    def index_to_onehot(self, index: int) -> np.ndarray:
        """Convert flat index to one-hot vector."""
        return self.state_space.index_to_onehot(index)

    def onehot_to_index(self, onehot: np.ndarray) -> int:
        """Convert one-hot vector to flat index."""
        return self.state_space.onehot_to_index(onehot)

    # ==================== Environment Interaction ====================

    @abstractmethod
    def reset(self, init_state: Optional[Any] = None) -> np.ndarray:
        """Reset environment and return initial state as one-hot.

        Args:
            init_state: Optional initial state (environment-specific format)

        Returns:
            One-hot encoded initial state
        """
        pass

    @abstractmethod
    def step(self, action: int) -> np.ndarray:
        """Take action and return new state as one-hot.

        Args:
            action: Action index

        Returns:
            One-hot encoded new state
        """
        pass

    @abstractmethod
    def get_current_state(self) -> Any:
        """Get current state in environment-specific format."""
        pass

    @abstractmethod
    def get_current_state_index(self) -> int:
        """Get current state as flat index."""
        pass

    # ==================== Matrix Operations ====================

    @abstractmethod
    def multiply_B_s(self, B: np.ndarray, state: np.ndarray, action: Optional[int]) -> np.ndarray:
        """Multiply transition matrix B with state vector.

        This handles environment-specific tensor contractions.

        Args:
            B: Transition matrix
            state: One-hot state vector
            action: Action index, or None to sum over all actions

        Returns:
            Resulting state distribution (same shape as state)
        """
        pass

    @abstractmethod
    def multiply_M_C(self, M: np.ndarray, C: np.ndarray) -> np.ndarray:
        """Multiply successor matrix M with reward/preference vector C.

        Args:
            M: Successor matrix
            C: Reward/preference vector

        Returns:
            Value vector (flat 1D array of length n_states)
        """
        pass

    # ==================== Transition Matrix ====================

    @abstractmethod
    def get_transition_matrix(self) -> np.ndarray:
        """Get the true transition matrix B (if known).

        Returns:
            Transition matrix of shape transition_matrix_shape
        """
        pass

    def create_empty_transition_matrix(self) -> np.ndarray:
        """Create an empty transition matrix for learning."""
        return np.zeros(self.transition_matrix_shape)

    def create_empty_successor_matrix(self) -> np.ndarray:
        """Create an empty successor matrix for learning."""
        return np.zeros(self.successor_matrix_shape)

    # ==================== Normalization ====================

    @abstractmethod
    def normalize_transition_matrix(self, B: np.ndarray) -> np.ndarray:
        """Normalize transition matrix to be proper probability distribution.

        Handles environment-specific normalization (different for simple vs augmented).

        Args:
            B: Unnormalized transition counts

        Returns:
            Normalized transition probabilities
        """
        pass

    # ==================== Goal/Reward ====================

    @abstractmethod
    def create_goal_prior(self, goal_states: List[int], reward: float = 10.0,
                          default_cost: float = -0.1) -> np.ndarray:
        """Create goal/preference vector C.

        Args:
            goal_states: List of goal state indices
            reward: Reward value at goal states
            default_cost: Default cost for non-goal states

        Returns:
            Preference vector C of appropriate shape
        """
        pass

    @abstractmethod
    def get_goal_states(self, goal_spec: Any) -> List[int]:
        """Convert goal specification to list of goal state indices.

        Args:
            goal_spec: Environment-specific goal specification
                       (e.g., (x,y) tuple, position value, etc.)

        Returns:
            List of flat state indices that are goal states
        """
        pass

    # ==================== Walls/Obstacles ====================

    def get_wall_indices(self) -> List[int]:
        """Get list of wall/obstacle state indices (if applicable).

        Returns:
            List of state indices that are walls, or empty list
        """
        return []

    def get_valid_state_mask(self) -> np.ndarray:
        """Get boolean mask of valid (non-wall) states.

        Returns:
            Boolean array of shape (n_states,)
        """
        mask = np.ones(self.n_states, dtype=bool)
        for wall_idx in self.get_wall_indices():
            mask[wall_idx] = False
        return mask

    # ==================== Successor Matrix Computation ====================

    def compute_successor_from_transition(self, B: np.ndarray, gamma: float = 0.95) -> np.ndarray:
        """Compute analytical successor matrix from transition matrix.

        M = (I - gamma * B_avg)^{-1}

        This default implementation works for simple state spaces.
        Override for augmented state spaces.

        Args:
            B: Transition matrix
            gamma: Discount factor

        Returns:
            Successor matrix M
        """
        # Average over actions
        B_avg = np.sum(B, axis=-1) / self.n_actions

        # For simple case: B_avg is (N, N)
        I = np.eye(self.n_states)
        M = np.linalg.pinv(I - gamma * B_avg)
        return M

    # ==================== Flattening for Clustering ====================

    def flatten_successor_for_clustering(self, M: np.ndarray) -> np.ndarray:
        """Flatten successor matrix for spectral clustering.

        For simple envs, this is identity.
        For augmented envs, this reshapes to 2D.

        Args:
            M: Successor matrix

        Returns:
            2D matrix suitable for clustering
        """
        if M.ndim == 2:
            return M
        else:
            # Override in subclass for specific flattening
            raise NotImplementedError("Override flatten_successor_for_clustering for non-2D M")

    # ==================== Visualization Helpers ====================

    @abstractmethod
    def render_state(self, state_index: int) -> Any:
        """Get renderable representation of a state (for visualization).

        Returns environment-specific visualization data.
        """
        pass

    def get_state_label(self, state_index: int) -> str:
        """Get human-readable label for a state."""
        return str(self.index_to_state(state_index))
