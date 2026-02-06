"""Key gridworld environment.

This module implements the key gridworld where the state space is
augmented with a binary key indicator (location, has_key).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from .base import BaseGridworld
from .transitions import generate_key_B
from .utils import (
    loc_to_idloc,
    idloc_to_loc,
    augmented_onehot_to_idx,
    augmented_onehot_to_state,
    augmented_state_to_onehot,
    multiply_B_s,
)


class KeyGridworld(BaseGridworld):
    """Gridworld with key pickup mechanics.

    State representation:
    - Matrix of shape (N, 2) where N = grid_size^2
    - First dimension: location, second dimension: has_key (0 or 1)
    - Can also be accessed as (x, y, has_key) tuple

    Actions (without pickup):
    - 0: left, 1: right, 2: up, 3: down
    - Key is automatically picked up when entering key location

    Actions (with pickup):
    - 0: left, 1: right, 2: up, 3: down, 4: pickup
    - Key must be explicitly picked up with action 4

    Transition matrix shape: (N, 2, N, 2, n_actions)
    """

    def __init__(self, grid_size, env_type='micro', walls=None, noise=None,
                 key_loc=None, has_pickup_action=False, pickup=None):
        """Initialize key gridworld.

        Args:
            grid_size: Side length of the grid
            env_type: 'micro' for walls in B, 'macro' for no walls
            walls: List of (x, y) wall positions
            noise: Observation noise level (0-1), None for perfect
            key_loc: (x, y) location of the key, or None for random
            has_pickup_action: If True, add separate pickup action
            pickup: Alias for has_pickup_action (for backward compatibility)
        """
        self.key_loc = key_loc
        # Support both parameter names for backward compatibility
        self.has_pickup_action = pickup if pickup is not None else has_pickup_action
        self.noise_val = 0.7  # Default noise value

        super().__init__(grid_size, env_type, walls, noise)

        # Set key location if not provided
        if self.key_loc is None:
            self._set_random_key_location()
        print(f'Key location: {self.key_loc}')

        # Initialize likelihood matrix
        N = self.grid_size**2
        self.A = np.identity(N)

        if noise is not None:
            self.noise_val = noise
            self.A = self._generate_noisy_likelihood()

        # Generate transition matrix
        self.B = self._generate_transition_matrix()

        # Initialize state (N, 2) matrix
        self.state = np.zeros((N, 2))

    def _set_random_key_location(self):
        """Set key to a random valid location."""
        wall_idx = [loc_to_idloc(wall, self.grid_size) for wall in self._walls]
        allowed_idx = set(range(self.grid_size**2)) - set(wall_idx)
        key_idx = np.random.choice(list(allowed_idx))
        self.key_loc = idloc_to_loc(key_idx, self.grid_size)

    def _generate_transition_matrix(self):
        """Generate the transition matrix B.

        Returns:
            B: Transition matrix of shape (N, 2, N, 2, n_actions)
        """
        if self.env_type == 'macro':
            return generate_key_B(
                self.grid_size,
                self.key_loc,
                walls=[],
                has_pickup_action=self.has_pickup_action
            )
        else:
            return generate_key_B(
                self.grid_size,
                self.key_loc,
                walls=self._walls,
                has_pickup_action=self.has_pickup_action
            )

    def reset(self, init=None):
        """Reset the environment.

        Args:
            init: Initial state as:
                - tuple (x, y, has_key): full state
                - None: random valid state

        Returns:
            State matrix of shape (N, 2)
        """
        N = self.grid_size**2
        self.state = np.zeros((N, 2))

        if init is None:
            # Random valid starting position
            wall_idx = [loc_to_idloc(wall, self.grid_size) for wall in self._walls]
            allowed_idx = set(range(N)) - set(wall_idx)
            init_idx = np.random.choice(list(allowed_idx))
            init_key = np.random.choice([0, 1])
        else:
            init_x, init_y, init_key = init
            init_idx = loc_to_idloc((init_x, init_y), self.grid_size)

        self.state[init_idx, init_key] = 1
        return self.state

    def step(self, action):
        """Take an action in the environment.

        Args:
            action: Action index

        Returns:
            New state matrix of shape (N, 2)
        """
        self.state = multiply_B_s(self.B, self.state, action)
        return self.state

    def set_state(self, state):
        """Set the current state.

        Args:
            state: State as (x, y, has_key) tuple or (N, 2) matrix
        """
        if isinstance(state, tuple):
            self.state = augmented_state_to_onehot(state, self.grid_size)
        else:
            self.state = state

    def get_state(self):
        """Get current state as (x, y, has_key) tuple."""
        return augmented_onehot_to_state(self.state, self.grid_size)

    def get_state_idx(self):
        """Get current state as (loc_idx, key_idx) tuple."""
        return augmented_onehot_to_idx(self.state)

    def get_state_matrix(self):
        """Get current state as (N, 2) matrix."""
        return self.state

    @property
    def n_actions(self):
        """Get number of available actions."""
        return 5 if self.has_pickup_action else 4

    def get_image(self, start_loc, goal_loc, save_path=None):
        """Visualize the key gridworld.

        Args:
            start_loc: (x, y) start location
            goal_loc: (x, y) goal location
            save_path: Path to save image, or None
        """
        gs = self.grid_size
        grid = np.zeros((gs, gs))

        # Mark walls
        if len(self._walls) > 0:
            wall_idx = tuple(np.array(self._walls).T)[::-1]
            grid[wall_idx] = 1

        # Mark key
        key_x, key_y = self.key_loc
        grid[key_y, key_x] = 0.75

        # Mark start and goal
        sx, sy = start_loc
        gx, gy = goal_loc
        grid[sx, sy] = 0.25
        grid[gx, gy] = 0.5

        # Create figure
        plt.figure()
        cmap = colors.ListedColormap(['black', 'yellow', 'purple', 'orange', 'white'])
        plt.imshow(grid, aspect='equal', cmap=cmap)

        ax = plt.gca()
        ax.text(sx, sy, 'Agent', fontsize=12, weight='bold',
                ha="center", va="center", color="k")
        ax.text(key_x, key_y, 'Key', fontsize=12, weight='bold',
                ha="center", va="center", color="k")
        ax.text(gx, gy, 'Goal', fontsize=12, weight='bold',
                ha="center", va="center", color="w")

        # Set up grid lines
        ax.set_xticks(np.arange(0, gs, 1))
        ax.set_yticks(np.arange(0, gs, 1))
        ax.set_xticklabels(range(0, gs))
        ax.set_yticklabels(range(0, gs))
        ax.set_xticks(np.arange(-0.5, gs, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, gs, 1), minor=True)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
        ax.tick_params(which='minor', bottom=False, left=False)

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


# Alias for backward compatibility
SR_Gridworld_Key = KeyGridworld
