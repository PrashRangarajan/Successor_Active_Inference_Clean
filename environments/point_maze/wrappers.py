"""Gymnasium wrappers for vectorized PointMaze training.

These wrappers decompose PointMazeAdapter's responsibilities into composable,
picklable wrappers compatible with gymnasium.vector.AsyncVectorEnv.

Wrapper chain (innermost to outermost):
    ContinuousObsWrapper(DiscreteActionWrapper(DiverseStartWrapper(gym.make(...))))
"""

import numpy as np
import gymnasium as gym


# Same force directions as adapter.py
_FORCE_DIRECTIONS = np.array([
    [ 1.0,  0.0],        # 0: East
    [-1.0,  0.0],        # 1: West
    [ 0.0,  1.0],        # 2: North
    [ 0.0, -1.0],        # 3: South
    [ 0.707,  0.707],    # 4: NE
    [-0.707,  0.707],    # 5: NW
    [ 0.707, -0.707],    # 6: SE
    [-0.707, -0.707],    # 7: SW
], dtype=np.float32)


class DiverseStartWrapper(gym.Wrapper):
    """Inject a random navigable position on every reset().

    Replaces PointMazeAdapter.sample_random_state(). All constructor args
    are plain Python/numpy types so the wrapper is picklable for subprocesses.
    """

    def __init__(self, env: gym.Env, wall_set: set, x_range: tuple,
                 y_range: tuple, x_bin_edges: np.ndarray,
                 y_bin_edges: np.ndarray, n_x_bins: int, n_y_bins: int):
        super().__init__(env)
        self.wall_set = wall_set
        self.x_range = x_range
        self.y_range = y_range
        self.x_bin_edges = x_bin_edges
        self.y_bin_edges = y_bin_edges
        self.n_x_bins = n_x_bins
        self.n_y_bins = n_y_bins

    def _discretize(self, x: float, y: float):
        """Discretize (x, y) to bin indices, matching adapter.discretize_obs()."""
        xi = int(np.clip(np.digitize(x, self.x_bin_edges), 0, self.n_x_bins - 1))
        yi = int(np.clip(np.digitize(y, self.y_bin_edges), 0, self.n_y_bins - 1))
        return xi, yi

    def _is_navigable(self, x: float, y: float) -> bool:
        """Check if position is not in a wall."""
        xi, yi = self._discretize(x, y)
        flat_idx = xi * self.n_y_bins + yi  # row-major, matches BinnedContinuousStateSpace
        return flat_idx not in self.wall_set

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)

        # Try injecting a uniformly random navigable position
        for _ in range(50):
            x = np.random.uniform(self.x_range[0], self.x_range[1])
            y = np.random.uniform(self.y_range[0], self.y_range[1])
            if self._is_navigable(x, y):
                try:
                    point_env = self.env.unwrapped.point_env
                    qpos = np.array([x, y])
                    qvel = np.array([0.0, 0.0])
                    point_env.set_state(qpos, qvel)
                    # Update the dict observation in place
                    obs['observation'][:2] = [x, y]
                    obs['observation'][2:4] = [0.0, 0.0]
                except Exception:
                    pass
                break

        return obs, info


class DiscreteActionWrapper(gym.ActionWrapper):
    """Convert integer action (0-7) to 2D continuous force vector.

    Replaces PointMazeAdapter._process_action().
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(8)

    def action(self, act: int) -> np.ndarray:
        return _FORCE_DIRECTIONS[int(act)].copy()


class ContinuousObsWrapper(gym.ObservationWrapper):
    """Convert dict obs to 6D array [x, y, vx, vy, goal_x, goal_y].

    Replaces PointMazeAdapter._extract_continuous_obs().
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )

    def observation(self, obs_dict: dict) -> np.ndarray:
        kinematic = obs_dict['observation'][:4]  # [x, y, vx, vy]
        goal = obs_dict['desired_goal'][:2]      # [goal_x, goal_y]
        return np.concatenate([kinematic, goal]).astype(np.float32)


def make_vec_env(
    num_envs: int,
    maze_id: str,
    max_episode_steps: int,
    wall_set: set,
    x_range: tuple,
    y_range: tuple,
    x_bin_edges: np.ndarray,
    y_bin_edges: np.ndarray,
    n_x_bins: int,
    n_y_bins: int,
) -> gym.vector.AsyncVectorEnv:
    """Create a vectorized PointMaze environment with N parallel workers.

    Each subprocess runs the wrapper chain:
        ContinuousObsWrapper(DiscreteActionWrapper(DiverseStartWrapper(gym.make(...))))

    All arguments are picklable for subprocess transport.
    """
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)

    def _make_env(env_idx):
        def _init():
            env = gym.make(maze_id, max_episode_steps=max_episode_steps)
            env = DiverseStartWrapper(
                env, wall_set=wall_set, x_range=x_range, y_range=y_range,
                x_bin_edges=x_bin_edges, y_bin_edges=y_bin_edges,
                n_x_bins=n_x_bins, n_y_bins=n_y_bins,
            )
            env = DiscreteActionWrapper(env)
            env = ContinuousObsWrapper(env)
            return env
        return _init

    return gym.vector.AsyncVectorEnv(
        [_make_env(i) for i in range(num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
