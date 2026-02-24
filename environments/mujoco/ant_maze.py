"""AntMaze adapter for neural SR agents (gymnasium-robotics).

AntMaze is a quadruped locomotion + maze navigation task. An 8-DOF Ant
robot must walk through corridors and walls to reach a goal position.
Much harder than PointMaze: the agent must learn to walk AND navigate.

Observation (Gym): Dict with 'observation' (105D: qpos + qvel + contact),
                   'achieved_goal' (2D: ant x,y), 'desired_goal' (2D).
Action (Gym):      Box(-1, 1, (8,)) -- 8 joint torques (4 hips + 4 ankles).

For the neural agent, the Dict observation is flattened to 109D:
    [observation(105), achieved_goal(2), desired_goal(2)]

The ant's x,y position is NOT in the 'observation' key -- it's only
available via 'achieved_goal'. This is critical for navigation.

Action discretization: {-1, 0, +1}^8 = 6561 discrete actions.
Uses ActionConditionedSFNetwork with subset sampling (from HalfCheetah pattern).

State injection: nq=15 (x, y, z, quaternion, 8 joints), nv=14.
"""

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np

from .base_adapter import MuJoCoAdapter


class AntMazeAdapter(MuJoCoAdapter):
    """Adapter for gymnasium-robotics AntMaze with discretized joint torques.

    Discretizes the 8D continuous action space into a grid:
    each joint gets n_bins_per_joint evenly spaced torques in [-1, 1].
    Total discrete actions = n_bins_per_joint^8.

    Handles the Dict observation space by flattening to a 109D array.

    Args:
        maze_id: Gymnasium environment ID (e.g., 'AntMaze_UMaze-v5').
        n_bins_per_joint: Number of torque levels per joint (default 3).
            3 bins -> 6561 actions, 2 bins -> 256 actions.
        render_mode: 'rgb_array' for video, None for speed.
        max_episode_steps: Override env's default max steps.
    """

    # Reach task: goal IS terminal state
    _is_survival_task = False

    def __init__(self, maze_id: str = 'AntMaze_UMaze-v5',
                 n_bins_per_joint: int = 3,
                 render_mode: Optional[str] = None,
                 max_episode_steps: Optional[int] = None):
        self._maze_id = maze_id
        self._n_bins_per_joint = n_bins_per_joint
        self._n_joints = 8  # Ant has 8 joints (4 hips + 4 ankles)
        self._max_episode_steps = max_episode_steps

        # Cache for flattened obs dimension (computed after env creation)
        self._obs_dim_cached = None

        super().__init__(render_mode=render_mode)

    def _make_env(self, render_mode: Optional[str] = None) -> gym.Env:
        """Create AntMaze environment."""
        kwargs = {}
        if render_mode is not None:
            kwargs['render_mode'] = render_mode
        if self._max_episode_steps is not None:
            kwargs['max_episode_steps'] = self._max_episode_steps
        return gym.make(self._maze_id, **kwargs)

    def _discretize_actions(self) -> np.ndarray:
        """Create grid of discrete 8D joint torques.

        For n_bins=3: each joint gets {-1, 0, +1}.
        Total actions = 3^8 = 6561.

        Returns:
            Array of shape (n_bins^8, 8) -- each row is an 8D torque vector.
        """
        per_joint = np.linspace(-1.0, 1.0, self._n_bins_per_joint)
        grids = np.meshgrid(*([per_joint] * self._n_joints), indexing='ij')
        actions = np.stack(grids, axis=-1).reshape(-1, self._n_joints)
        return actions.astype(np.float32)

    # ==================== Dict Obs Handling ====================

    def _flatten_obs(self, obs_dict: dict) -> np.ndarray:
        """Flatten Dict observation to a single array.

        Layout: [observation(105), achieved_goal(2), desired_goal(2)] = 109D

        The ant's x,y position is in achieved_goal, NOT in observation.
        The goal position is in desired_goal.
        """
        return np.concatenate([
            obs_dict['observation'],
            obs_dict['achieved_goal'],
            obs_dict['desired_goal'],
        ]).astype(np.float32)

    # ==================== Overrides for Dict Obs ====================

    @property
    def obs_dim(self) -> int:
        """Dimensionality of flattened observations (109D)."""
        if self._obs_dim_cached is None:
            # Compute from a test reset
            obs_dict, _ = self._env.reset()
            flat = self._flatten_obs(obs_dict)
            self._obs_dim_cached = len(flat)
            self._current_obs = flat
        return self._obs_dim_cached

    def reset(self, init_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Reset environment and return flattened observation.

        Args:
            init_state: Optional [qpos(15), qvel(14)] = 29D array.
                If provided, injects this MuJoCo state after reset.

        Returns:
            Flattened observation (109D).
        """
        obs_dict, _ = self._env.reset()

        if init_state is not None:
            init_state = np.asarray(init_state, dtype=np.float64)
            try:
                ant_env = self._env.unwrapped.ant_env
                nq = ant_env.model.nq  # 15
                qpos = init_state[:nq]
                qvel = init_state[nq:]
                ant_env.set_state(qpos, qvel)
                # Re-read observation after state injection
                obs_dict = {
                    'observation': ant_env._get_obs(),
                    'achieved_goal': qpos[:2].copy(),
                    'desired_goal': obs_dict['desired_goal'],
                }
            except Exception:
                pass  # Fall through to default obs_dict

        self._current_obs = self._flatten_obs(obs_dict)
        return self._current_obs.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take a discrete action and return (obs, reward, term, trunc, info).

        Maps discrete action index to continuous 8D torque vector.
        Flattens the Dict observation.
        """
        continuous_action = self._discrete_actions[action]
        obs_dict, reward, terminated, truncated, info = self._env.step(
            continuous_action
        )
        self._current_obs = self._flatten_obs(obs_dict)
        return self._current_obs.copy(), float(reward), terminated, truncated, info

    def get_current_obs(self) -> np.ndarray:
        """Get current flattened observation."""
        if self._current_obs is None:
            raise RuntimeError("Call reset() before get_current_obs()")
        return self._current_obs.copy()

    # ==================== Goal / Terminal ====================

    def is_goal_state(self, obs: np.ndarray) -> bool:
        """Check if the ant has reached the goal position.

        Observation layout (109D):
            obs[0:105]   = body state (qpos, qvel, contact forces)
            obs[105:107] = achieved_goal (ant x, y)
            obs[107:109] = desired_goal (goal x, y)

        Returns:
            True if Euclidean distance < 0.5 units.
        """
        ant_pos = obs[105:107]
        goal_pos = obs[107:109]
        dist = np.linalg.norm(ant_pos - goal_pos)
        return bool(dist < 0.5)

    # ==================== Diverse Starts ====================

    def sample_random_state(self) -> np.ndarray:
        """Reset to a random state for diverse exploration.

        AntMaze environments handle randomization internally --
        each reset() picks a valid start position (not in walls)
        with some randomization of joint angles.

        Returns:
            Flattened observation from the new random state.
        """
        obs_dict, _ = self._env.reset()
        self._current_obs = self._flatten_obs(obs_dict)
        return self._current_obs.copy()

    # ==================== State Injection Helpers ====================

    def get_state_for_reset(self) -> np.ndarray:
        """Get current MuJoCo state as [qpos(15), qvel(14)] for later reset().

        Returns:
            Concatenated qpos and qvel arrays (29D).
        """
        try:
            ant_env = self._env.unwrapped.ant_env
            qpos = ant_env.data.qpos.copy()
            qvel = ant_env.data.qvel.copy()
            return np.concatenate([qpos, qvel])
        except Exception:
            # Fallback: use base class method
            return super().get_state_for_reset()
