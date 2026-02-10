"""Base MuJoCo adapter for neural SR agents.

Provides the same interface as ContinuousAdapter (used by NeuralSRAgent
and HierarchicalNeuralSRAgent) but without the BinnedContinuousAdapter
layer underneath. Goal checking uses continuous-state conditions instead
of discrete bins. Action discretization maps discrete action indices to
continuous MuJoCo actions via a linspace grid.

The neural agent never uses bins — it works entirely with raw continuous
observations. For MuJoCo environments, this direct adapter is the clean
pattern: no vestigial discretization.

Task types:
    - Reach tasks (e.g., Acrobot): goal IS a terminal state. Env terminates
      when the agent reaches the goal. is_terminal() returns is_goal_state().
    - Survival tasks (e.g., InvertedPendulum): goal is to STAY balanced.
      Env terminates on failure (pole fell). is_terminal() returns None
      to let the agent rely on the env's terminated flag.

Subclasses set `self._is_survival_task = True/False` to select behavior.
"""

import abc
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


class MuJoCoAdapter(abc.ABC):
    """Base adapter for MuJoCo environments with discrete action mapping.

    Subclasses must implement:
        - _make_env(): Create and return the gymnasium MuJoCo environment.
        - _discretize_actions(): Return array of discrete action vectors.
        - is_goal_state(obs): Continuous goal condition.
        - sample_random_state(): Reset to a diverse random state.

    Args:
        render_mode: Gymnasium render mode ('rgb_array' for video, None for speed).
    """

    # Set to True for survival tasks where env termination = failure.
    # Set to False for reach tasks where env termination = goal reached.
    _is_survival_task: bool = False

    def __init__(self, render_mode: Optional[str] = None):
        self._env = self._make_env(render_mode=render_mode)
        self._discrete_actions = self._discretize_actions()
        self._current_obs: Optional[np.ndarray] = None

    # ==================== Abstract Methods ====================

    @abc.abstractmethod
    def _make_env(self, render_mode: Optional[str] = None) -> gym.Env:
        """Create and return the gymnasium MuJoCo environment.

        Args:
            render_mode: 'rgb_array' for video, None for speed.

        Returns:
            Gymnasium environment instance.
        """
        ...

    @abc.abstractmethod
    def _discretize_actions(self) -> np.ndarray:
        """Create the discrete-to-continuous action mapping.

        Returns:
            Array of shape (n_discrete_actions, action_dim) where each row
            is a continuous action vector sent to the MuJoCo environment.
        """
        ...

    @abc.abstractmethod
    def is_goal_state(self, obs: np.ndarray) -> bool:
        """Check if observation satisfies the continuous goal condition.

        For reach tasks: "have we reached the target?"
        For survival tasks: "are we still in the desired region?"

        Args:
            obs: Raw observation array.

        Returns:
            True if the goal condition is met.
        """
        ...

    @abc.abstractmethod
    def sample_random_state(self) -> np.ndarray:
        """Reset to a diverse random state and return the observation.

        Uses env.unwrapped.set_state(qpos, qvel) to inject random
        configurations for diverse exploration during training.

        Returns:
            Raw observation array from the new random state.
        """
        ...

    # ==================== Properties ====================

    @property
    def obs_dim(self) -> int:
        """Dimensionality of raw observations."""
        return self._env.observation_space.shape[0]

    @property
    def n_actions(self) -> int:
        """Number of discrete actions."""
        return len(self._discrete_actions)

    @property
    def env(self) -> gym.Env:
        """Underlying gymnasium environment."""
        return self._env

    # ==================== Core Interface ====================

    def reset(self, init_state: Optional[Any] = None) -> np.ndarray:
        """Reset environment and return raw observation.

        Args:
            init_state: Optional initial state as [qpos..., qvel...] array.
                If provided, resets then injects this state via set_state.

        Returns:
            Raw observation array.
        """
        obs, _ = self._env.reset()

        if init_state is not None:
            init_state = np.asarray(init_state, dtype=np.float64)
            nq = self._env.unwrapped.model.nq
            qpos = init_state[:nq]
            qvel = init_state[nq:]
            self._env.unwrapped.set_state(qpos, qvel)
            obs = self._env.unwrapped._get_obs()

        self._current_obs = obs.astype(np.float32)
        return self._current_obs.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a discrete action and return (obs, reward, term, trunc, info).

        Maps the discrete action index to a continuous action vector
        and steps the MuJoCo environment.

        Args:
            action: Discrete action index.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        continuous_action = self._discrete_actions[action]
        obs, reward, terminated, truncated, info = self._env.step(
            continuous_action
        )
        self._current_obs = obs.astype(np.float32)
        return self._current_obs.copy(), float(reward), terminated, truncated, info

    def get_current_obs(self) -> np.ndarray:
        """Get current raw continuous observation."""
        if self._current_obs is None:
            raise RuntimeError("Call reset() before get_current_obs()")
        return self._current_obs.copy()

    # ==================== Goal / Terminal Interface ====================

    def is_terminal(self, obs: Optional[np.ndarray] = None) -> Optional[bool]:
        """Check if the observation represents a terminal goal state.

        Behavior depends on task type:
        - Reach tasks (Acrobot-like): Returns is_goal_state(obs).
          The goal IS the terminal state — reaching it ends the episode.
        - Survival tasks (InvertedPendulum-like): Returns None.
          The agent should keep running while balanced. Termination
          comes from the env's terminated flag (pole fell = failure),
          not from reaching a "goal."

        Args:
            obs: Observation to check. Uses current obs if None.

        Returns:
            True/False for reach tasks, None for survival tasks.
        """
        if self._is_survival_task:
            # For survival tasks, don't short-circuit the episode.
            # Let run_episode() rely on env's terminated/truncated flags.
            return None

        if obs is None:
            obs = self._current_obs
        if obs is None:
            return None
        return self.is_goal_state(obs)

    def is_in_goal_bin(self, goal_states: List[int],
                       obs: Optional[np.ndarray] = None) -> bool:
        """Compatibility shim — delegates to is_goal_state().

        The neural agent's hierarchical code calls this to check if an
        observation is in a goal cluster. For MuJoCo adapters, we just
        check the continuous goal condition directly.

        For survival tasks, this still checks is_goal_state() because the
        hierarchical agent uses it to identify which clusters are "good"
        (for macro-level planning), not to terminate episodes.

        Args:
            goal_states: Ignored (no bins in MuJoCo adapter).
            obs: Observation to check.

        Returns:
            True if goal condition met.
        """
        if obs is None:
            obs = self._current_obs
        if obs is None:
            return False
        return self.is_goal_state(obs)

    def get_goal_states(self, goal_spec: Any = None) -> List[int]:
        """Return goal state indices (empty for MuJoCo — no bins).

        Returns an empty list. For reach tasks, the agent will rely on
        is_terminal() which checks is_goal_state(). For survival tasks,
        the agent will rely on the env's terminated flag.

        Returns:
            Empty list.
        """
        return []

    def create_goal_reward_fn(self, goal_spec: Any = None,
                              reward: float = 1.0,
                              default_cost: float = 0.0
                              ) -> Callable[[np.ndarray], float]:
        """Create a reward function based on the continuous goal condition.

        Args:
            goal_spec: Ignored (goal is defined by is_goal_state).
            reward: Reward for reaching goal.
            default_cost: Cost for non-goal states.

        Returns:
            Function mapping observation to scalar reward.
        """
        def reward_fn(obs: np.ndarray) -> float:
            return reward if self.is_goal_state(obs) else default_cost

        return reward_fn

    # ==================== State Injection Helpers ====================

    def get_state_for_reset(self) -> np.ndarray:
        """Get current state as [qpos..., qvel...] for later reset().

        Returns:
            Concatenated qpos and qvel arrays.
        """
        qpos = self._env.unwrapped.data.qpos.copy()
        qvel = self._env.unwrapped.data.qvel.copy()
        return np.concatenate([qpos, qvel])

    # ==================== Rendering ====================

    def render(self) -> np.ndarray:
        """Render current frame (for video recording).

        Returns:
            RGB array of the current frame.
        """
        return self._env.render()
