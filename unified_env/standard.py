"""Standard gridworld environment.

This module implements the standard gridworld with simple state space
(location only, no augmentation).
"""

import numpy as np

from .base import BaseGridworld
from .transitions import generate_standard_B
from .utils import (
    loc_to_idx,
    idx_to_loc,
    loc_to_onehot,
    onehot_to_idx,
    onehot_to_loc,
)


class StandardGridworld(BaseGridworld):
    """Standard gridworld with location-only state space.

    State representation:
    - One-hot vector of shape (N,) where N = grid_size^2
    - Can also be accessed as (x, y) location tuple

    Actions:
    - 0: left, 1: right, 2: up, 3: down

    Transition matrix shape: (N, N, 4)
    """

    def __init__(self, grid_size, env_type='micro', walls=None, noise=None):
        """Initialize standard gridworld.

        Args:
            grid_size: Side length of the grid
            env_type: 'micro' for walls in B, 'macro' for no walls
            walls: List of (x, y) wall positions
            noise: Observation noise level (0-1), None for perfect
        """
        super().__init__(grid_size, env_type, walls, noise)

        # Initialize likelihood matrix
        N = self.grid_size**2
        self.A = np.identity(N)

        if self.noise is not None:
            self.A = self._generate_noisy_likelihood()

        # Generate transition matrix
        self.B = self._generate_transition_matrix()

        # Initialize state
        self.state = None
        self.obs = None
        self.obs_idx = None

    def _generate_transition_matrix(self):
        """Generate the transition matrix B.

        Returns:
            B: Transition matrix of shape (N, N, 4)
        """
        if self.env_type == 'macro':
            return generate_standard_B(self.grid_size, walls=[])
        else:
            return generate_standard_B(self.grid_size, walls=self._walls)

    def reset(self, init=None):
        """Reset the environment.

        Args:
            init: Initial state as:
                - int: flat index
                - tuple (x, y): location
                - None: random valid location

        Returns:
            One-hot state vector
        """
        N = self.grid_size**2
        self.state = np.zeros(N)

        if init is None:
            # Random valid starting position
            wall_idx = [loc_to_idx(wall, self.grid_size) for wall in self._walls]
            allowed_idx = set(range(N)) - set(wall_idx)
            init_idx = np.random.choice(list(allowed_idx))
        elif isinstance(init, tuple):
            init_idx = loc_to_idx(init, self.grid_size)
        else:
            init_idx = init

        self.state[init_idx] = 1
        self._update_obs()

        return self.state

    def step(self, action):
        """Take an action in the environment.

        Args:
            action: Action index (0=left, 1=right, 2=up, 3=down)

        Returns:
            New one-hot state vector
        """
        self.state = self.B[:, :, action] @ self.state
        self._update_obs()
        return self.state

    def _update_obs(self):
        """Update observation using likelihood matrix."""
        self.obs = self.A @ self.state
        self.obs_idx = onehot_to_idx(self.obs)

    def set_state(self, state):
        """Set the current state.

        Args:
            state: State as (x, y) tuple or flat index
        """
        if isinstance(state, tuple):
            self.state = loc_to_onehot(state, self.grid_size)
        else:
            self.state = np.zeros(self.grid_size**2)
            self.state[state] = 1
        self._update_obs()

    def get_state(self):
        """Get current state as one-hot vector."""
        return self.state

    def get_state_loc(self):
        """Get current state as (x, y) location."""
        return onehot_to_loc(self.state, self.grid_size)

    def get_state_idx(self):
        """Get current state as flat index."""
        return onehot_to_idx(self.state)


# Alias for backward compatibility
SR_Gridworld = StandardGridworld
