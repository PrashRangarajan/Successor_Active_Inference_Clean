"""HalfCheetah-v4 adapter for neural SR agents.

The HalfCheetah is a locomotion task: a 2D cheetah robot runs forward as
fast as possible. The reward decomposes into:
    reward = reward_run (x_velocity) + reward_ctrl (-0.1 * action²)

Observation layout (from gymnasium _get_obs = [qpos[1:], qvel]):
    obs[0:8]  : qpos[1:] — joint angles (root x-position excluded)
    obs[8:17] : qvel     — joint velocities (including root)
    Total: 17 dimensions.

Action: 6D continuous torques in [-1, 1], one per joint.
    Discretized to n_bins_per_joint^6 discrete actions via grid.
    With 3 bins: {-1, 0, +1}^6 = 729 discrete actions.

This is a locomotion task (neither reach nor survival): there is no
terminal state and no failure condition. The env never terminates early.
Episodes end only by truncation (max steps). The goal is to maximize
forward velocity.

State injection: set_state(qpos, qvel) with nq=9, nv=9.
    qpos[0] is the root x-position (not in obs but tracked for reward).
"""

from typing import Optional

import gymnasium as gym
import numpy as np

from .base_adapter import MuJoCoAdapter


class HalfCheetahAdapter(MuJoCoAdapter):
    """Adapter for HalfCheetah-v4 with discretized joint torques.

    Discretizes the 6D continuous action space into a grid:
    each joint gets n_bins_per_joint evenly spaced torques in [-1, 1].
    Total discrete actions = n_bins_per_joint^6.

    Args:
        n_bins_per_joint: Number of torque levels per joint (default 3).
            3 bins → 729 actions, 5 bins → 15625 actions.
        render_mode: 'rgb_array' for video, None for speed.
        max_episode_steps: Override env's default max steps.
    """

    # Locomotion task: no termination, no goal state to reach or maintain
    _is_survival_task = False  # Not survival either — locomotion

    def __init__(self, n_bins_per_joint: int = 3,
                 render_mode: Optional[str] = None,
                 max_episode_steps: Optional[int] = None):
        self._n_bins_per_joint = n_bins_per_joint
        self._n_joints = 6
        self._max_episode_steps = max_episode_steps
        super().__init__(render_mode=render_mode)

    def _make_env(self, render_mode: Optional[str] = None) -> gym.Env:
        """Create HalfCheetah-v4 environment."""
        kwargs = {}
        if render_mode is not None:
            kwargs['render_mode'] = render_mode
        if self._max_episode_steps is not None:
            kwargs['max_episode_steps'] = self._max_episode_steps
        return gym.make('HalfCheetah-v4', **kwargs)

    def _discretize_actions(self) -> np.ndarray:
        """Create grid of discrete 6D actions from per-joint bins.

        For n_bins=3: each joint gets {-1, 0, +1}.
        Total actions = 3^6 = 729.

        Returns:
            Array of shape (n_bins^6, 6) — each row is a 6D torque vector.
        """
        # Per-joint torques
        per_joint = np.linspace(-1.0, 1.0, self._n_bins_per_joint)

        # Cartesian product via meshgrid
        grids = np.meshgrid(*([per_joint] * self._n_joints), indexing='ij')
        # Stack and reshape: (n_bins, n_bins, ..., 6) → (n_bins^6, 6)
        actions = np.stack(grids, axis=-1).reshape(-1, self._n_joints)
        return actions.astype(np.float32)

    def is_goal_state(self, obs: np.ndarray) -> bool:
        """Check if the cheetah has reached a target velocity.

        For locomotion, "goal" is achieving a high forward velocity.
        This is used by the hierarchical agent for cluster preferences.
        We consider the agent in a "good" state if x_velocity > 2.0.

        Note: x_velocity is obs[8] (first element of qvel, which is
        the root x-velocity).

        Args:
            obs: Observation array (17D).

        Returns:
            True if forward velocity exceeds threshold.
        """
        # obs = [qpos[1:], qvel]
        # qvel[0] = root x velocity = obs[8]
        x_velocity = obs[8]
        return bool(x_velocity > 2.0)

    def is_terminal(self, obs: Optional[np.ndarray] = None) -> Optional[bool]:
        """HalfCheetah never terminates early — always returns None.

        Episodes end only by truncation (max_episode_steps).

        Returns:
            None (let run_episode rely on truncation).
        """
        return None

    def sample_random_state(self) -> np.ndarray:
        """Reset to a diverse random state for exploration.

        Samples random joint positions and velocities to provide diverse
        training starts. Ranges are chosen to cover the reachable state
        space without creating physically implausible configurations.

        Returns:
            Raw observation from the random state.
        """
        self._env.reset()

        nq = self._env.unwrapped.model.nq  # 9
        nv = self._env.unwrapped.model.nv  # 9

        # Random joint positions (small perturbations from default)
        qpos = np.random.uniform(-0.1, 0.1, size=nq)
        qpos[0] = np.random.uniform(-1.0, 1.0)  # root x can vary more

        # Random joint velocities (small)
        qvel = np.random.uniform(-0.5, 0.5, size=nv)
        # Give the root some forward velocity sometimes
        qvel[0] = np.random.uniform(-1.0, 3.0)  # root x velocity

        self._env.unwrapped.set_state(qpos, qvel)
        obs = self._env.unwrapped._get_obs()
        self._current_obs = obs.astype(np.float32)
        return self._current_obs.copy()
