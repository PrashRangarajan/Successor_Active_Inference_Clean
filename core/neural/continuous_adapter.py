"""Continuous adapter wrapper for neural SR agents.

Wraps an existing BinnedContinuousAdapter to provide raw continuous
observations instead of one-hot encoded discrete states. Reuses the
base adapter's goal specification, terminal checking, and diverse
start logic to ensure evaluation consistency with the tabular agent.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from environments.binned_continuous_adapter import BinnedContinuousAdapter


class ContinuousAdapter:
    """Wrapper providing raw observations from a discretized adapter.

    The neural agent needs continuous observations as network inputs.
    Rather than rewriting the domain-specific adapter logic (angle wrapping,
    goal conditions, terminal checks, etc.), this wrapper delegates to the
    existing adapter while exposing raw observations.

    Args:
        base_adapter: A BinnedContinuousAdapter instance (e.g., AcrobotAdapter).
    """

    def __init__(self, base_adapter: BinnedContinuousAdapter):
        self.base = base_adapter

    @property
    def obs_dim(self) -> int:
        """Dimensionality of raw observations from the environment.

        If the base adapter defines ``continuous_obs_dim`` (e.g., for Dict
        observation spaces like PointMaze), uses that. Otherwise falls back
        to the standard Box observation_space shape.
        """
        if hasattr(self.base, 'continuous_obs_dim'):
            return self.base.continuous_obs_dim
        return self.base.env.observation_space.shape[0]

    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
        return self.base.n_actions

    @property
    def env(self):
        """Underlying Gymnasium environment."""
        return self.base.env

    def reset(self, init_state: Optional[Any] = None) -> np.ndarray:
        """Reset environment and return raw continuous observation.

        Args:
            init_state: Initial state in environment-specific format
                (e.g., [theta1, theta2, dtheta1, dtheta2] for Acrobot).

        Returns:
            Raw observation array.
        """
        self.base.reset(init_state)
        return self.base.get_current_obs().copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take action and return (obs, reward, terminated, truncated, info).

        Args:
            action: Discrete action index.

        Returns:
            Tuple of (raw_obs, reward, terminated, truncated, info).
        """
        _, reward, terminated, truncated, info = self.base.step_with_info(action)
        obs = self.base.get_current_obs().copy()
        return obs, reward, terminated, truncated, info

    def get_current_obs(self) -> np.ndarray:
        """Get current raw continuous observation."""
        return self.base.get_current_obs().copy()

    def sample_random_state(self) -> np.ndarray:
        """Reset to a uniformly random state and return raw observation.

        Delegates to the base adapter's sample_random_state() which handles
        domain-specific sampling (e.g., angle ranges for Acrobot).
        """
        self.base.sample_random_state()
        return self.base.get_current_obs().copy()

    def get_state_for_reset(self) -> Any:
        """Get current state in a format suitable for reset().

        Delegates to base adapter.
        """
        return self.base.get_state_for_reset()

    def get_goal_states(self, goal_spec: Any = None) -> List[int]:
        """Get goal state indices (delegates to base adapter)."""
        return self.base.get_goal_states(goal_spec)

    def is_terminal(self, obs: Optional[np.ndarray] = None) -> Optional[bool]:
        """Check if current state is terminal.

        Delegates to the base adapter's is_terminal(), which uses the
        domain-specific terminal condition (e.g., Acrobot height threshold).

        Returns:
            True/False if adapter can evaluate, or None for fallback.
        """
        return self.base.is_terminal(obs)

    def is_in_goal_bin(self, goal_states: List[int],
                       obs: Optional[np.ndarray] = None) -> bool:
        """Check if current discretized state is in the goal set.

        Uses the base adapter's discretization to check goal membership,
        ensuring identical evaluation with the tabular agent.
        """
        if hasattr(self.base, 'is_in_goal_bin'):
            return self.base.is_in_goal_bin(goal_states, obs)
        # Fallback: discretize and check
        if obs is None:
            obs = self.base.get_current_obs()
        discrete = self.base.discretize_obs(obs)
        idx = self.base.state_space.state_to_index(discrete)
        return idx in goal_states

    def create_goal_reward_fn(self, goal_spec: Any = None,
                              reward: float = 1.0,
                              default_cost: float = 0.0) -> Callable[[np.ndarray], float]:
        """Create a reward function from goal specification.

        Returns a function mapping raw observations to scalar rewards.
        Uses the base adapter's discretization to check goal membership,
        ensuring identical goal conditions as the tabular agent.

        Args:
            goal_spec: Goal specification (environment-specific).
            reward: Reward for reaching goal states.
            default_cost: Cost for non-goal states.

        Returns:
            Function: obs (np.ndarray) -> reward (float).
        """
        goal_states = set(self.base.get_goal_states(goal_spec))

        def reward_fn(obs: np.ndarray) -> float:
            discrete = self.base.discretize_obs(obs)
            idx = self.base.state_space.state_to_index(discrete)
            return reward if idx in goal_states else default_cost

        return reward_fn

    def render(self):
        """Render current frame (for video recording)."""
        return self.base.render()
