"""Abstract StateSpace interface for different environment state representations.

This module defines the interface that all environment-specific state spaces must implement.
The state space handles conversions between different state representations:
- Internal state (environment-specific, e.g., (x,y) for gridworld, (pos,vel) for mountain car)
- Flat index (single integer for matrix indexing)
- One-hot vector (for matrix multiplication)
"""

from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union
import numpy as np


class StateSpace(ABC):
    """Abstract base class defining the state space interface.

    Each environment adapter must provide a StateSpace implementation that handles
    the specific state representation for that environment.
    """

    @property
    @abstractmethod
    def n_states(self) -> int:
        """Total number of discrete states."""
        pass

    @property
    @abstractmethod
    def state_shape(self) -> Tuple[int, ...]:
        """Shape of the state representation (e.g., (N,) for simple, (N,2) for key env)."""
        pass

    @abstractmethod
    def state_to_index(self, state: Any) -> int:
        """Convert internal state representation to flat index.

        Args:
            state: Environment-specific state (e.g., (x,y) tuple, (pos_bin, vel_bin), etc.)

        Returns:
            Integer index in range [0, n_states)
        """
        pass

    @abstractmethod
    def index_to_state(self, index: int) -> Any:
        """Convert flat index back to internal state representation.

        Args:
            index: Integer index in range [0, n_states)

        Returns:
            Environment-specific state representation
        """
        pass

    @abstractmethod
    def index_to_onehot(self, index: int) -> np.ndarray:
        """Convert flat index to one-hot vector.

        Args:
            index: Integer index in range [0, n_states)

        Returns:
            One-hot numpy array of shape state_shape
        """
        pass

    @abstractmethod
    def onehot_to_index(self, onehot: np.ndarray) -> int:
        """Convert one-hot vector to flat index.

        Args:
            onehot: One-hot numpy array of shape state_shape

        Returns:
            Integer index in range [0, n_states)
        """
        pass

    def state_to_onehot(self, state: Any) -> np.ndarray:
        """Convert internal state to one-hot vector."""
        return self.index_to_onehot(self.state_to_index(state))

    def onehot_to_state(self, onehot: np.ndarray) -> Any:
        """Convert one-hot vector to internal state."""
        return self.index_to_state(self.onehot_to_index(onehot))


class SimpleStateSpace(StateSpace):
    """Simple 1D state space (e.g., for standard gridworld).

    States are represented as flat indices, one-hot vectors are 1D arrays.
    """

    def __init__(self, n_states: int):
        self._n_states = n_states

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (self._n_states,)

    def state_to_index(self, state: int) -> int:
        return int(state)

    def index_to_state(self, index: int) -> int:
        return int(index)

    def index_to_onehot(self, index: int) -> np.ndarray:
        onehot = np.zeros(self._n_states)
        onehot[index] = 1.0
        return onehot

    def onehot_to_index(self, onehot: np.ndarray) -> int:
        return int(np.argmax(onehot))


class GridStateSpace(StateSpace):
    """2D grid state space where states are (x, y) tuples.

    Used for standard gridworld environments.
    """

    def __init__(self, grid_size: int):
        self.grid_size = grid_size
        self._n_states = grid_size ** 2

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (self._n_states,)

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (x, y) to flat index."""
        return state[0] * self.grid_size + state[1]

    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert flat index to (x, y)."""
        return divmod(index, self.grid_size)

    def index_to_onehot(self, index: int) -> np.ndarray:
        onehot = np.zeros(self._n_states)
        onehot[index] = 1.0
        return onehot

    def onehot_to_index(self, onehot: np.ndarray) -> int:
        return int(np.argmax(onehot))


class BinnedContinuousStateSpace(StateSpace):
    """Discretized continuous state space (e.g., for Mountain Car).

    States are tuples of bin indices for each dimension.
    """

    def __init__(self, n_bins_per_dim: List[int]):
        """
        Args:
            n_bins_per_dim: Number of bins for each dimension, e.g., [n_pos_bins, n_vel_bins]
        """
        self.n_bins_per_dim = n_bins_per_dim
        self.n_dims = len(n_bins_per_dim)
        self._n_states = int(np.prod(n_bins_per_dim))

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (self._n_states,)

    def state_to_index(self, state: Tuple[int, ...]) -> int:
        """Convert tuple of bin indices to flat index.

        Uses row-major (C-style) ordering: last index varies fastest.
        """
        index = 0
        multiplier = 1
        for dim in reversed(range(self.n_dims)):
            index += state[dim] * multiplier
            multiplier *= self.n_bins_per_dim[dim]
        return index

    def index_to_state(self, index: int) -> Tuple[int, ...]:
        """Convert flat index to tuple of bin indices."""
        state = []
        for dim in reversed(range(self.n_dims)):
            state.append(index % self.n_bins_per_dim[dim])
            index //= self.n_bins_per_dim[dim]
        return tuple(reversed(state))

    def index_to_onehot(self, index: int) -> np.ndarray:
        onehot = np.zeros(self._n_states)
        onehot[index] = 1.0
        return onehot

    def onehot_to_index(self, onehot: np.ndarray) -> int:
        return int(np.argmax(onehot))


class AugmentedStateSpace(StateSpace):
    """State space with additional discrete dimension (e.g., has_key flag).

    Used for environments like key-gridworld where state = (location, has_key).
    The state is represented as a 2D array of shape (n_locations, n_augment).
    """

    def __init__(self, base_n_states: int, n_augment: int = 2):
        """
        Args:
            base_n_states: Number of base states (e.g., grid locations)
            n_augment: Number of augmented states (e.g., 2 for has_key in {0, 1})
        """
        self.base_n_states = base_n_states
        self.n_augment = n_augment
        self._n_states = base_n_states * n_augment

    @property
    def n_states(self) -> int:
        return self._n_states

    @property
    def state_shape(self) -> Tuple[int, ...]:
        return (self.base_n_states, self.n_augment)

    def state_to_index(self, state: Tuple[int, int]) -> int:
        """Convert (base_idx, augment_idx) to flat index.

        Flat index = augment_idx * base_n_states + base_idx
        """
        base_idx, augment_idx = state
        return augment_idx * self.base_n_states + base_idx

    def index_to_state(self, index: int) -> Tuple[int, int]:
        """Convert flat index to (base_idx, augment_idx)."""
        augment_idx, base_idx = divmod(index, self.base_n_states)
        return (base_idx, augment_idx)

    def index_to_onehot(self, index: int) -> np.ndarray:
        """Convert to 2D one-hot array of shape (base_n_states, n_augment)."""
        base_idx, augment_idx = self.index_to_state(index)
        onehot = np.zeros((self.base_n_states, self.n_augment))
        onehot[base_idx, augment_idx] = 1.0
        return onehot

    def onehot_to_index(self, onehot: np.ndarray) -> int:
        """Convert 2D one-hot array to flat index."""
        # Flatten in Fortran order to match original code convention
        flat = onehot.flatten('F')
        idx = int(np.argmax(flat))
        # Convert back: idx = augment * base_n + base
        augment_idx, base_idx = divmod(idx, self.base_n_states)
        return self.state_to_index((base_idx, augment_idx))

    def base_index_to_state(self, base_idx: int, grid_size: int) -> Tuple[int, int]:
        """Convert base index to (x, y) location (for grid-based environments)."""
        return divmod(base_idx, grid_size)

    def full_state_to_tuple(self, state: Tuple[int, int], grid_size: int) -> Tuple[int, int, int]:
        """Convert (base_idx, augment_idx) to (x, y, augment)."""
        base_idx, augment_idx = state
        x, y = self.base_index_to_state(base_idx, grid_size)
        return (x, y, augment_idx)
