"""InvertedPendulum-v4 adapter for neural SR agents.

The InvertedPendulum task: balance a pole on a cart by applying horizontal
forces. The env gives +1 reward per step (survival reward), terminating
when the pole angle exceeds ~0.2 radians.

Observation layout (from gymnasium _get_obs = [qpos, qvel]):
    obs[0] : x       — cart position
    obs[1] : theta   — pole angle (0 = upright)
    obs[2] : x_dot   — cart velocity
    obs[3] : omega   — pole angular velocity

Action: 1D continuous force in [-3, 3] discretized to n_force_bins
    evenly spaced forces from -3 to +3.

This is a SURVIVAL task: the agent succeeds by keeping the pole balanced
for the full episode. Env termination means failure (pole fell), not
success. The adapter sets _is_survival_task = True so that is_terminal()
returns None (lets run_episode rely on env's terminated flag).

Goal condition (is_goal_state): pole nearly upright AND cart near center.
    Used by the hierarchical agent to identify "good" clusters, NOT to
    terminate episodes.

State injection: set_state(qpos=[x, theta], qvel=[x_dot, omega])
"""

from typing import Optional

import gymnasium as gym
import numpy as np

from .base_adapter import MuJoCoAdapter


class InvertedPendulumAdapter(MuJoCoAdapter):
    """Adapter for InvertedPendulum-v4 with discrete force actions.

    This is a survival task: env termination = failure (pole fell).
    The agent succeeds by balancing for the full episode duration.

    Args:
        n_force_bins: Number of discrete force levels (default 7).
            Maps to linspace(-3, 3, n_force_bins).
        render_mode: 'rgb_array' for video, None for speed.
        max_episode_steps: Override env's default max steps.
    """

    _is_survival_task = True  # Termination = failure, not goal

    def __init__(self, n_force_bins: int = 7,
                 render_mode: Optional[str] = None,
                 max_episode_steps: Optional[int] = None):
        self._n_force_bins = n_force_bins
        self._max_episode_steps = max_episode_steps
        super().__init__(render_mode=render_mode)

    def _make_env(self, render_mode: Optional[str] = None) -> gym.Env:
        """Create InvertedPendulum-v4 environment."""
        kwargs = {}
        if render_mode is not None:
            kwargs['render_mode'] = render_mode
        if self._max_episode_steps is not None:
            kwargs['max_episode_steps'] = self._max_episode_steps
        return gym.make('InvertedPendulum-v4', **kwargs)

    def _discretize_actions(self) -> np.ndarray:
        """Create evenly spaced force actions in [-3, 3].

        Returns:
            Array of shape (n_force_bins, 1) — each row is a 1D force.
        """
        forces = np.linspace(-3.0, 3.0, self._n_force_bins)
        return forces.reshape(-1, 1).astype(np.float32)

    def is_goal_state(self, obs: np.ndarray) -> bool:
        """Check if the pendulum is balanced and cart is centered.

        For a survival task, "goal" means the agent is in the desired
        region — pole nearly upright, cart near center. This is used
        by the hierarchical agent for cluster preference, NOT to
        terminate episodes.

        Observation layout: [x, theta, x_dot, omega]

        Args:
            obs: Observation array.

        Returns:
            True if theta and x are within tight thresholds.
        """
        x = obs[0]          # cart position
        theta = obs[1]      # pole angle (0 = upright)
        return bool(abs(theta) < 0.1 and abs(x) < 0.5)

    def sample_random_state(self) -> np.ndarray:
        """Reset to a diverse random state for exploration.

        Samples random cart positions, pole angles, and small velocities.
        The ranges are chosen to be wider than the env's default reset
        (which uses +-0.01) but safely inside the termination boundary
        (|theta| > 0.2).

        Returns:
            Raw observation from the random state.
        """
        # Reset first to get a clean env state
        self._env.reset()

        # Sample diverse initial conditions
        # Keep theta well inside the +-0.2 termination threshold
        x = np.random.uniform(-0.5, 0.5)              # cart position
        theta = np.random.uniform(-0.08, 0.08)         # pole angle (safe margin)
        x_dot = np.random.uniform(-0.3, 0.3)           # cart velocity
        omega = np.random.uniform(-0.3, 0.3)           # angular velocity

        qpos = np.array([x, theta])
        qvel = np.array([x_dot, omega])

        self._env.unwrapped.set_state(qpos, qvel)
        obs = self._env.unwrapped._get_obs()
        self._current_obs = obs.astype(np.float32)
        return self._current_obs.copy()
