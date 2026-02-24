"""CartPole environment adapter for Hierarchical SR Active Inference.

CartPole-v1 is a *survival* task — the agent must keep the pole balanced for
as long as possible.  This is an experimental fit with the SR framework, which
assumes goal-reaching with absorbing terminal states.

To make it work we reframe the task: the "goal" is the balanced central region
(small cart position, small pole angle, low velocities).  The agent receives
reward for staying in this region.  Episode termination (pole falling or cart
leaving bounds) is handled by the Gym environment, NOT by the SR goal check.

Observation space (Gym): [cart_pos, cart_vel, pole_angle, pole_ang_vel]
Actions: 0 = push left, 1 = push right (discrete — no discretization needed)
"""

from typing import Any, List, Optional, Tuple

import numpy as np

from environments.binned_continuous_adapter import BinnedContinuousAdapter, clamp
from core.state_space import BinnedContinuousStateSpace

class CartPoleAdapter(BinnedContinuousAdapter):
    """Adapter for Gymnasium CartPole-v1 environment.

    State representation: (pos_bin, vel_bin, angle_bin, ang_vel_bin)
    Discretizes the continuous 4D state into bins.

    Transition matrix shape: (N, N, n_actions) where N = product of all bins
    Actions: 0 = push left, 1 = push right
    """

    def __init__(self, env, n_pos_bins: int = 6, n_vel_bins: int = 6,
                 n_angle_bins: int = 8, n_ang_vel_bins: int = 6):
        """
        Args:
            env: Gymnasium CartPole-v1 environment
            n_pos_bins: Number of bins for cart position [-2.4, 2.4]
            n_vel_bins: Number of bins for cart velocity [-3, 3]
            n_angle_bins: Number of bins for pole angle [-0.418, 0.418] rad (~24 deg)
            n_ang_vel_bins: Number of bins for pole angular velocity [-3, 3]
        """
        self._env = env
        self.n_pos_bins = n_pos_bins
        self.n_vel_bins = n_vel_bins
        self.n_angle_bins = n_angle_bins
        self.n_ang_vel_bins = n_ang_vel_bins

        # 4D state space
        self.bin_sizes = (n_pos_bins, n_vel_bins, n_angle_bins, n_ang_vel_bins)
        self._state_space = BinnedContinuousStateSpace(list(self.bin_sizes))
        self._n_actions = env.action_space.n  # 2

        # Physical ranges for discretization
        self._pos_range = (-2.4, 2.4)
        self._vel_range = (-3.0, 3.0)
        self._angle_range = (-0.418, 0.418)  # ~24 degrees in radians
        self._ang_vel_range = (-3.0, 3.0)

        # Create discretization bins
        (
            self.pos_space,
            self.vel_space,
            self.angle_space,
            self.ang_vel_space,
        ) = self._create_bins()

        self._current_obs = None
        self._current_state = None

    def _create_bins(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create bin edges for CartPole state dimensions.

        Uses interior edges only (n-1 edges for n bins) following the
        Acrobot pattern so that ``np.digitize`` naturally produces bin
        indices in [0, n_bins-1] after clamping.
        """
        pos_space = np.linspace(*self._pos_range, self.n_pos_bins + 1)[1:-1]
        vel_space = np.linspace(*self._vel_range, self.n_vel_bins + 1)[1:-1]
        angle_space = np.linspace(*self._angle_range, self.n_angle_bins + 1)[1:-1]
        ang_vel_space = np.linspace(*self._ang_vel_range, self.n_ang_vel_bins + 1)[1:-1]
        return pos_space, vel_space, angle_space, ang_vel_space

    def discretize_obs(self, obs: np.ndarray) -> Tuple[int, int, int, int]:
        """Discretize a continuous observation into bin indices.

        Args:
            obs: Either 4D Gym observation [pos, vel, angle, ang_vel]
                 or a 4-tuple of the same values.

        Returns:
            (pos_bin, vel_bin, angle_bin, ang_vel_bin)
        """
        pos, vel, angle, ang_vel = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])

        i = clamp(np.digitize(pos, self.pos_space), 1, self.n_pos_bins) - 1
        j = clamp(np.digitize(vel, self.vel_space), 1, self.n_vel_bins) - 1
        k = clamp(np.digitize(angle, self.angle_space), 1, self.n_angle_bins) - 1
        l = clamp(np.digitize(ang_vel, self.ang_vel_space), 1, self.n_ang_vel_bins) - 1
        return (i, j, k, l)

    # ==================== Environment Interaction ====================

    def reset(self, init_state: Optional[Any] = None) -> np.ndarray:
        """Reset environment.

        Args:
            init_state: Initial state as [pos, vel, angle, ang_vel] continuous values
                        or None for random initialization.

        Returns:
            One-hot encoded discretized state
        """
        obs, _ = self._env.reset()

        if init_state is not None:
            self._env.unwrapped.state = np.array(init_state, dtype=np.float64)
            obs = np.array(init_state, dtype=np.float64)

        self._current_obs = obs
        discrete_state = self.discretize_obs(obs)
        state_idx = self.state_space.state_to_index(discrete_state)
        self._current_state = self.state_space.index_to_onehot(state_idx)
        return self._current_state

    def sample_random_state(self) -> np.ndarray:
        """Reset to a uniformly random state across the valid state space.

        Samples random values within the CartPole's physical ranges.
        Uses slightly narrower ranges than the termination bounds to
        ensure the sampled state doesn't immediately terminate.

        Returns:
            One-hot encoded discretized state.
        """
        # Normal reset so Gym's bookkeeping is valid
        obs, _ = self._env.reset()

        # Sample within safe ranges (narrower than termination bounds)
        pos = np.random.uniform(-2.0, 2.0)       # termination at ±2.4
        vel = np.random.uniform(-2.5, 2.5)
        angle = np.random.uniform(-0.35, 0.35)    # termination at ±0.2095 rad (12 deg)
        ang_vel = np.random.uniform(-2.5, 2.5)

        # Inject into the physics engine
        self._env.unwrapped.state = np.array([pos, vel, angle, ang_vel])

        obs = np.array([pos, vel, angle, ang_vel])
        self._current_obs = obs
        discrete_state = self.discretize_obs(obs)
        state_idx = self.state_space.state_to_index(discrete_state)
        self._current_state = self.state_space.index_to_onehot(state_idx)
        return self._current_state

    # ==================== Goal / Reward ====================

    def get_goal_states(self, goal_spec: Any = None) -> List[int]:
        """Convert goal specification to state indices.

        For CartPole (a survival task), the "goal" is the balanced central
        region: center third of position × center third of angle × all velocities.

        Args:
            goal_spec: None (default: balanced central region),
                       or list of state indices.

        Returns:
            List of goal state indices
        """
        if goal_spec is None:
            # Goal = center third of position and angle bins, all velocities
            pos_lo = self.n_pos_bins // 3
            pos_hi = 2 * self.n_pos_bins // 3
            angle_lo = self.n_angle_bins // 3
            angle_hi = 2 * self.n_angle_bins // 3

            goal_states = []
            for p in range(pos_lo, pos_hi + 1):
                for v in range(self.n_vel_bins):
                    for a in range(angle_lo, angle_hi + 1):
                        for av in range(self.n_ang_vel_bins):
                            if (0 <= p < self.n_pos_bins and
                                0 <= a < self.n_angle_bins):
                                state_idx = self.state_space.state_to_index(
                                    (p, v, a, av)
                                )
                                goal_states.append(state_idx)

            return list(set(goal_states))

        elif isinstance(goal_spec, list):
            return goal_spec

        else:
            raise ValueError(f"Invalid goal specification for CartPole: {goal_spec}")

    def create_sparse_prior(self, radius: float = 0.5,
                            reward: float = 10.0,
                            default_cost: float = -0.1) -> np.ndarray:
        """Create a sparse reward vector C for all states.

        Gives ``+reward`` to states within a normalised quadratic ball of
        the balanced position and ``default_cost`` everywhere else::

            C(s) = reward        if  (pos/2.4)² + (angle/0.418)² < radius²
                   default_cost  otherwise

        Position and angle are normalised by their respective physical
        ranges so that both contribute equally to the metric.  Velocities
        are excluded (same philosophy as Pendulum's sparse prior).

        Args:
            radius: Threshold in the normalised ``(pos/2.4, angle/0.418)``
                metric.  A value of ~0.5 yields a moderate goal region.
            reward: Reward value for states inside the ball.
            default_cost: Cost for states outside the ball.

        Returns:
            C vector of shape ``(n_states,)``
        """
        pos_centers, vel_centers, angle_centers, ang_vel_centers = (
            self.get_bin_centers()
        )
        C = np.full(self.n_states, default_cost)

        pos_max = self._pos_range[1]      # 2.4
        angle_max = self._angle_range[1]  # 0.418

        n_goal = 0
        for p in range(self.n_pos_bins):
            for v in range(self.n_vel_bins):
                for a in range(self.n_angle_bins):
                    for av in range(self.n_ang_vel_bins):
                        pos_n = pos_centers[p] / pos_max
                        ang_n = angle_centers[a] / angle_max
                        if pos_n ** 2 + ang_n ** 2 < radius ** 2:
                            idx = self.state_space.state_to_index(
                                (p, v, a, av)
                            )
                            C[idx] = reward
                            n_goal += 1

        print(f"Sparse prior: {n_goal}/{self.n_states} goal states "
              f"(radius={radius}, reward={reward}, cost={default_cost})")
        return C

    # ==================== Visualization ====================

    def get_state_label(self, state_index: int) -> str:
        """Get human-readable state label."""
        p, v, a, av = self.state_space.index_to_state(state_index)
        return f"(p:{p},v:{v},a:{a},av:{av})"

    def get_bin_centers(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get center values of all 4 state dimensions."""
        pos_edges = np.linspace(*self._pos_range, self.n_pos_bins + 1)
        vel_edges = np.linspace(*self._vel_range, self.n_vel_bins + 1)
        angle_edges = np.linspace(*self._angle_range, self.n_angle_bins + 1)
        ang_vel_edges = np.linspace(*self._ang_vel_range, self.n_ang_vel_bins + 1)

        pos_centers = 0.5 * (pos_edges[:-1] + pos_edges[1:])
        vel_centers = 0.5 * (vel_edges[:-1] + vel_edges[1:])
        angle_centers = 0.5 * (angle_edges[:-1] + angle_edges[1:])
        ang_vel_centers = 0.5 * (ang_vel_edges[:-1] + ang_vel_edges[1:])

        return pos_centers, vel_centers, angle_centers, ang_vel_centers

    def get_dimension_labels(self) -> Tuple[str, str]:
        """Return human-readable axis labels for the 2D projected state space."""
        return ("Cart Position", "Pole Angle")

    def get_action_labels(self) -> List[str]:
        """Return human-readable labels for each action index."""
        return ["\u2190 Push Left", "\u2192 Push Right"]
