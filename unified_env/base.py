"""Base class for gridworld environments.

This module defines the abstract base class that all gridworld
variants inherit from, providing common functionality.
"""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from .utils import loc_to_idx, create_walls


class BaseGridworld(ABC):
    """Abstract base class for gridworld environments.

    This class provides common functionality for all gridworld variants:
    - Wall handling
    - Noisy likelihood generation
    - Visualization
    - Common getters/setters

    Subclasses must implement:
    - reset(): Reset the environment
    - step(): Take an action
    - set_state(): Set the current state
    - get_state(): Get current state as tuple
    - get_state_idx(): Get current state as index
    - _generate_transition_matrix(): Generate the B matrix
    """

    def __init__(self, grid_size, env_type='micro', walls=None, noise=None):
        """Initialize the gridworld.

        Args:
            grid_size: Side length of the grid
            env_type: 'micro' for full walls, 'macro' for no walls in B
            walls: List of (x, y) wall positions, or None for random
            noise: Noise level for observations (0-1), or None for perfect
        """
        self.grid_size = grid_size
        self.env_type = env_type
        self.noise = noise

        # Initialize walls
        self._walls = walls if walls is not None else []
        self.wall_size = len(self._walls)

        # State variables (set by subclass)
        self.state = None
        self.obs = None
        self.obs_idx = None

        # Will be set by subclass
        self.A = None  # Likelihood matrix
        self.B = None  # Transition matrix

    @property
    def walls(self):
        """Get wall positions."""
        return self._walls

    @walls.setter
    def walls(self, value):
        """Set wall positions and regenerate matrices."""
        self.set_walls(value)

    def set_walls(self, walls):
        """Set wall positions and regenerate transition/likelihood matrices.

        Args:
            walls: List of (x, y) wall positions, or None for random
        """
        if walls is None:
            self._walls = create_walls(self.grid_size, self.wall_size)
        else:
            self._walls = walls
            self.wall_size = len(walls)

        # Regenerate matrices
        self.B = self._generate_transition_matrix()

        if self.noise is not None:
            self.A = self._generate_noisy_likelihood()

    def get_walls(self):
        """Get wall positions."""
        return self._walls

    @abstractmethod
    def _generate_transition_matrix(self):
        """Generate the transition matrix B.

        Returns:
            B: Transition matrix (shape depends on subclass)
        """
        pass

    @abstractmethod
    def reset(self, init=None):
        """Reset the environment.

        Args:
            init: Initial state specification (format depends on subclass)

        Returns:
            Initial state
        """
        pass

    @abstractmethod
    def step(self, action):
        """Take an action in the environment.

        Args:
            action: Action index

        Returns:
            New state
        """
        pass

    @abstractmethod
    def set_state(self, state):
        """Set the current state.

        Args:
            state: State specification (format depends on subclass)
        """
        pass

    @abstractmethod
    def get_state(self):
        """Get current state as a tuple.

        Returns:
            State tuple (format depends on subclass)
        """
        pass

    @abstractmethod
    def get_state_idx(self):
        """Get current state as index/indices.

        Returns:
            State index (format depends on subclass)
        """
        pass

    def _generate_noisy_likelihood(self):
        """Generate noisy observation likelihood matrix.

        Creates a matrix A where observations may be confused with
        neighboring states based on the noise level.

        Returns:
            A: Likelihood matrix of shape (N, N)
        """
        N = self.grid_size**2
        A = np.identity(N)
        wall_idx = [loc_to_idx(wall, self.grid_size) for wall in self._walls]

        for i in range(N):
            if i not in wall_idx:
                A[i, i] = self.noise

                # Check neighboring states for noise distribution
                left_valid = (i - 1 not in wall_idx) and (i % self.grid_size != 0)
                right_valid = (i + 1 not in wall_idx) and (i % self.grid_size != self.grid_size - 1)

                if left_valid:
                    if right_valid:
                        A[i - 1, i] = (1 - self.noise) / 2
                        A[i + 1, i] = (1 - self.noise) / 2
                    else:
                        A[i - 1, i] = (1 - self.noise)
                else:
                    if right_valid:
                        A[i + 1, i] = (1 - self.noise)
                    else:
                        A[i, i] = 1.00

        return A

    def get_likelihood_dist(self):
        """Get the observation likelihood matrix A."""
        return self.A

    def set_likelihood_dist(self, A):
        """Set the observation likelihood matrix A."""
        self.A = A

    def get_transition_dist(self):
        """Get the transition matrix B."""
        return self.B

    def get_obs(self):
        """Get current observation."""
        return self.obs

    def get_obs_idx(self):
        """Get current observation index."""
        return self.obs_idx

    def _update_obs(self):
        """Update observation based on current state.

        For standard gridworld, applies likelihood matrix.
        Subclasses may override for different observation models.
        """
        if self.A is not None and self.state is not None:
            # Flatten state if needed for matrix multiplication
            state_flat = self.state.flatten() if hasattr(self.state, 'flatten') else self.state
            if len(state_flat) == self.A.shape[1]:
                self.obs = self.A @ state_flat
            else:
                self.obs = self.state.copy()
        else:
            self.obs = self.state.copy() if self.state is not None else None

    def get_image(self, start_loc, goal_loc, save_path=None):
        """Visualize the gridworld.

        Args:
            start_loc: (x, y) start location
            goal_loc: (x, y) goal location or list of locations
            save_path: Path to save image, or None to not save
        """
        gs = self.grid_size
        grid = np.zeros((gs, gs))

        # Mark walls
        if len(self._walls) > 0:
            wall_idx = tuple(np.array(self._walls).T)
            grid[wall_idx] = 2

        # Mark start
        grid[start_loc] = 1

        # Mark goal(s)
        if isinstance(goal_loc, list):
            grid[tuple(zip(*goal_loc))] = 0.5
        else:
            grid[goal_loc] = 0.5

        # Create figure
        fig = plt.figure()
        cmap = colors.ListedColormap(['black', 'purple', 'yellow', 'white'])
        plt.imshow(grid.T, aspect='equal', cmap=cmap)

        ax = plt.gca()
        ax.text(start_loc[0], start_loc[1], 'Agent', fontsize=10,
                ha="center", va="center", color="k")

        if isinstance(goal_loc, list):
            for loc in goal_loc:
                ax.text(loc[0], loc[1], 'Goal', fontsize=10,
                        ha="center", va="center", color="w")
        else:
            ax.text(goal_loc[0], goal_loc[1], 'Goal', fontsize=10,
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

        return fig
