"""Experience replay buffer for neural SR training.

Supports three sampling modes:
- Uniform: random transitions for standard DRL training.
- Prioritized: transitions weighted by TD error (PER) for faster convergence.
- Episodic: complete episodes for hippocampal-style sequential replay,
  preserving the biological plausibility of the tabular agent's replay.

Supports Hindsight Experience Replay (HER):
- After each episode, relabels failed trajectories with achieved goals.
- For each transition, picks k future states as substitute goals.
- Recomputes observations (replacing goal coordinates) and rewards.
- Multiplies effective data ~k× without extra environment steps.
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


# ==================== SumTree ====================

class SumTree:
    """Binary sum-tree for O(log n) prioritized sampling.

    Stores priorities as leaf values and maintains prefix sums in
    internal nodes for efficient proportional sampling.  Also
    maintains a parallel min-tree for O(1) minimum-priority lookup
    (needed for importance-sampling weight normalization).

    Args:
        capacity: Maximum number of leaf entries (must match buffer capacity).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self._min_tree = np.full(2 * capacity - 1, float('inf'),
                                 dtype=np.float64)

    def update(self, leaf_idx: int, priority: float):
        """Update priority for a specific leaf and propagate to root."""
        tree_idx = leaf_idx + self.capacity - 1
        self._tree[tree_idx] = priority
        self._min_tree[tree_idx] = priority

        # Propagate up to root
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            left = 2 * tree_idx + 1
            right = left + 1
            self._tree[tree_idx] = (
                self._tree[left] + self._tree[right]
            )
            self._min_tree[tree_idx] = min(
                self._min_tree[left], self._min_tree[right]
            )

    def sample(self, value: float) -> int:
        """Sample a leaf index proportional to priorities.

        Args:
            value: Random value in [0, total_priority).

        Returns:
            Leaf index (0-based, corresponds to buffer index).
        """
        tree_idx = 0
        while True:
            left = 2 * tree_idx + 1
            if left >= len(self._tree):
                break
            if value <= self._tree[left]:
                tree_idx = left
            else:
                value -= self._tree[left]
                tree_idx = left + 1
        return tree_idx - (self.capacity - 1)

    @property
    def total(self) -> float:
        """Sum of all priorities."""
        return float(self._tree[0])

    @property
    def min_priority(self) -> float:
        """Minimum priority across all leaves."""
        return float(self._min_tree[0])

    def reset(self):
        """Zero out all priorities."""
        self._tree[:] = 0.0
        self._min_tree[:] = float('inf')


# ==================== Replay Buffer ====================

class ReplayBuffer:
    """Fixed-capacity replay buffer with pre-allocated numpy arrays.

    Stores transitions as (obs, action, reward, next_obs, done) and
    tracks episode boundaries for optional episodic sampling.

    Optionally supports Prioritized Experience Replay (PER) via a
    SumTree for O(log n) prioritized sampling.

    Args:
        capacity: Maximum number of transitions to store.
        obs_dim: Dimensionality of observations.
        use_per: If True, enable prioritized sampling.
        per_alpha: Priority exponent (0 = uniform, 1 = full prioritization).
        per_eps: Small constant added to TD errors to avoid zero priorities.
    """

    def __init__(self, capacity: int, obs_dim: int,
                 use_per: bool = False, per_alpha: float = 0.6,
                 per_eps: float = 1e-6):
        self.capacity = capacity
        self.obs_dim = obs_dim

        # Pre-allocate storage
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        # Episode boundary tracking
        self._episode_start_idx = 0  # start of current episode in buffer
        self._episode_starts: List[int] = []  # start indices of stored episodes
        self._episode_lengths: List[int] = []  # length of each stored episode

        self.size = 0
        self._ptr = 0

        # Current-episode accumulator for HER relabeling.
        # Tracks the raw transitions of the episode being collected so that
        # end_episode() can generate hindsight-relabeled copies.
        self._current_ep_obs: List[np.ndarray] = []
        self._current_ep_actions: List[int] = []
        self._current_ep_next_obs: List[np.ndarray] = []
        self._current_ep_dones: List[bool] = []

        # Prioritized experience replay
        self._use_per = use_per
        self._per_alpha = per_alpha
        self._per_eps = per_eps
        self._max_priority = 1.0
        if use_per:
            self._sum_tree = SumTree(capacity)
        else:
            self._sum_tree = None

    def add(self, obs: np.ndarray, action: int, reward: float,
            next_obs: np.ndarray, done: bool):
        """Add a single transition to the buffer.

        Args:
            obs: Current observation.
            action: Action taken.
            reward: Reward received.
            next_obs: Next observation.
            done: Whether episode terminated.
        """
        idx = self._ptr

        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = float(done)

        # Accumulate for HER relabeling
        self._current_ep_obs.append(obs.copy())
        self._current_ep_actions.append(action)
        self._current_ep_next_obs.append(next_obs.copy())
        self._current_ep_dones.append(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

        # New transitions get max priority so they're sampled at least once
        if self._use_per:
            priority = self._max_priority ** self._per_alpha
            self._sum_tree.update(idx, priority)

    def end_episode(self):
        """Mark the end of the current episode for episodic sampling.

        Call this after the last transition of each episode.
        Stores the episode data snapshot for HER relabeling access.
        """
        if self.size == 0:
            self._clear_episode_accumulator()
            return

        # Compute episode length
        if self._ptr > self._episode_start_idx:
            ep_len = self._ptr - self._episode_start_idx
        elif self._ptr == self._episode_start_idx:
            self._clear_episode_accumulator()
            return  # empty episode
        else:
            # Wrapped around the buffer
            ep_len = (self.capacity - self._episode_start_idx) + self._ptr

        if ep_len > 0:
            self._episode_starts.append(self._episode_start_idx)
            self._episode_lengths.append(ep_len)

        # Snapshot episode data for HER before clearing
        self._last_episode = {
            'obs': list(self._current_ep_obs),
            'actions': list(self._current_ep_actions),
            'next_obs': list(self._current_ep_next_obs),
            'dones': list(self._current_ep_dones),
        }
        self._clear_episode_accumulator()

        self._episode_start_idx = self._ptr

        # Remove episodes that have been overwritten by the circular buffer
        self._clean_old_episodes()

    def _clear_episode_accumulator(self):
        """Reset episode-level accumulators."""
        self._current_ep_obs.clear()
        self._current_ep_actions.clear()
        self._current_ep_next_obs.clear()
        self._current_ep_dones.clear()

    def get_last_episode(self) -> Optional[dict]:
        """Return the most recently completed episode's transitions.

        Returns:
            Dict with keys 'obs', 'actions', 'next_obs', 'dones' (lists),
            or None if no episode has been completed.
        """
        return getattr(self, '_last_episode', None)

    def _clean_old_episodes(self):
        """Remove episode records that have been overwritten."""
        while self._episode_starts and self._episode_lengths:
            start = self._episode_starts[0]
            length = self._episode_lengths[0]
            # Check if this episode's data is still valid
            if self.size == self.capacity:
                # Buffer is full — check if the episode's range has been overwritten
                end = (start + length) % self.capacity
                if start < end:
                    # Episode doesn't wrap
                    if self._ptr > start and self._ptr <= end:
                        self._episode_starts.pop(0)
                        self._episode_lengths.pop(0)
                        continue
                else:
                    # Episode wraps around
                    if self._ptr > start or self._ptr <= end:
                        self._episode_starts.pop(0)
                        self._episode_lengths.pop(0)
                        continue
            break

    # ==================== Hindsight Experience Replay ====================

    def add_her_transitions(
        self,
        goal_indices: Tuple[int, int] = (4, 6),
        k: int = 4,
        goal_check_fn: Optional[Callable[[np.ndarray, np.ndarray], bool]] = None,
        reward_fn: Optional[Callable[[np.ndarray], float]] = None,
    ):
        """Generate and store HER-relabeled transitions from the last episode.

        For each transition in the episode, picks up to *k* future achieved
        positions as substitute goals ("future" strategy from Andrychowicz
        et al., 2017).  Observations are relabeled by replacing the goal
        coordinates, rewards are recomputed, and ``done`` is set if the
        achieved position matches the substitute goal.

        Must be called after ``end_episode()`` which snapshots the episode.

        Args:
            goal_indices: Slice ``obs[goal_indices[0]:goal_indices[1]]``
                containing the goal coordinates to be relabeled.
                Default (4, 6) matches PointMaze 6D obs: [x, y, vx, vy, gx, gy].
            k: Number of future goals to sample per transition.
            goal_check_fn: ``(achieved_xy, goal_xy) -> bool`` for terminal check.
                Default: Euclidean distance < 0.45 (PointMaze threshold).
            reward_fn: ``obs_relabeled -> float`` to compute reward for the
                relabeled observation.  If None, uses a simple proximity
                reward: +1 if terminal, 0 otherwise.
        """
        episode = self.get_last_episode()
        if episode is None or len(episode['obs']) == 0:
            return

        ep_obs = episode['obs']        # list of (obs_dim,) arrays
        ep_actions = episode['actions']  # list of int
        ep_next_obs = episode['next_obs']
        ep_len = len(ep_obs)

        gi_start, gi_end = goal_indices

        # Default goal check: Euclidean < 0.45 (PointMaze standard)
        if goal_check_fn is None:
            def goal_check_fn(achieved: np.ndarray, goal: np.ndarray) -> bool:
                return float(np.linalg.norm(achieved - goal)) < 0.45

        for t in range(ep_len):
            # Pick k future indices (or fewer if near end of episode)
            future_indices = range(t + 1, ep_len)
            if len(future_indices) == 0:
                continue
            n_samples = min(k, len(future_indices))
            sampled = np.random.choice(
                list(future_indices), size=n_samples, replace=False
            )

            for future_t in sampled:
                # The substitute goal is the position achieved at future_t
                substitute_goal = ep_next_obs[future_t][:2].copy()

                # Relabel obs and next_obs with the substitute goal
                obs_relabeled = ep_obs[t].copy()
                obs_relabeled[gi_start:gi_end] = substitute_goal

                next_obs_relabeled = ep_next_obs[t].copy()
                next_obs_relabeled[gi_start:gi_end] = substitute_goal

                # Check if this transition reaches the substitute goal
                achieved_pos = ep_next_obs[t][:2]
                done = goal_check_fn(achieved_pos, substitute_goal)

                # Compute reward for relabeled transition
                if reward_fn is not None:
                    reward = reward_fn(next_obs_relabeled)
                    if done:
                        reward += 1.0  # terminal bonus
                else:
                    reward = 1.0 if done else 0.0

                # Store the relabeled transition
                self.add(obs_relabeled, ep_actions[t], reward,
                         next_obs_relabeled, done)

    def truncate(self, keep_fraction: float):
        """Keep only the most recent fraction of stored transitions.

        Used at phase boundaries to remove stale data from previous
        training phases that no longer reflects the current distribution.

        Args:
            keep_fraction: Fraction of current buffer to keep (0.0 to 1.0).
                E.g., 0.3 keeps the most recent 30% of transitions.
        """
        if keep_fraction >= 1.0 or self.size == 0:
            return

        n_keep = max(1, int(self.size * keep_fraction))

        if self._ptr >= n_keep:
            # Most recent data is contiguous: [ptr - n_keep, ptr)
            start = self._ptr - n_keep
            self.obs[:n_keep] = self.obs[start:self._ptr].copy()
            self.actions[:n_keep] = self.actions[start:self._ptr].copy()
            self.rewards[:n_keep] = self.rewards[start:self._ptr].copy()
            self.next_obs[:n_keep] = self.next_obs[start:self._ptr].copy()
            self.dones[:n_keep] = self.dones[start:self._ptr].copy()
        else:
            # Data wraps around: [capacity - (n_keep - ptr), capacity) + [0, ptr)
            tail_len = n_keep - self._ptr
            tail_start = self.capacity - tail_len
            new_obs = np.concatenate([
                self.obs[tail_start:self.capacity],
                self.obs[:self._ptr]
            ], axis=0)
            new_actions = np.concatenate([
                self.actions[tail_start:self.capacity],
                self.actions[:self._ptr]
            ])
            new_rewards = np.concatenate([
                self.rewards[tail_start:self.capacity],
                self.rewards[:self._ptr]
            ])
            new_next_obs = np.concatenate([
                self.next_obs[tail_start:self.capacity],
                self.next_obs[:self._ptr]
            ], axis=0)
            new_dones = np.concatenate([
                self.dones[tail_start:self.capacity],
                self.dones[:self._ptr]
            ])
            self.obs[:n_keep] = new_obs
            self.actions[:n_keep] = new_actions
            self.rewards[:n_keep] = new_rewards
            self.next_obs[:n_keep] = new_next_obs
            self.dones[:n_keep] = new_dones

        self.size = n_keep
        self._ptr = n_keep % self.capacity

        # Reset episode tracking (no longer valid after truncation)
        self._episode_starts.clear()
        self._episode_lengths.clear()
        self._episode_start_idx = self._ptr

        # Rebuild SumTree — assign max priority to all kept transitions
        # (old indices are invalidated by data compaction)
        if self._use_per:
            self._sum_tree.reset()
            priority = self._max_priority ** self._per_alpha
            for i in range(self.size):
                self._sum_tree.update(i, priority)

    # ==================== Sampling ====================

    def sample_uniform(self, batch_size: int,
                       device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.
            device: Torch device for the returned tensors.

        Returns:
            Dict with keys 'obs', 'actions', 'rewards', 'next_obs', 'dones',
            each as a torch.Tensor.
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        return self._indices_to_batch(indices, device)

    def sample_prioritized(
        self, batch_size: int, beta: float,
        device: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """Sample a batch with priorities proportional to TD errors.

        Uses stratified sampling: divides total priority into batch_size
        equal segments and samples one transition from each.

        Args:
            batch_size: Number of transitions to sample.
            beta: IS weight exponent (0 = no correction, 1 = full).
            device: Torch device for returned tensors.

        Returns:
            Tuple of (batch_dict, indices, is_weights):
                batch_dict: Same format as sample_uniform().
                indices: Buffer indices for priority updates, shape (B,).
                is_weights: Importance sampling weights, shape (B,).
        """
        indices = np.zeros(batch_size, dtype=np.int64)
        priorities = np.zeros(batch_size, dtype=np.float64)

        total = self._sum_tree.total
        segment = total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            value = np.random.uniform(lo, hi)
            idx = self._sum_tree.sample(value)
            # Clamp to valid buffer range
            idx = max(0, min(idx, self.size - 1))
            indices[i] = idx
            tree_idx = idx + self._sum_tree.capacity - 1
            priorities[i] = self._sum_tree._tree[tree_idx]

        # Importance sampling weights
        probs = priorities / max(total, 1e-10)
        min_prob = max(self._sum_tree.min_priority, 1e-10) / max(total, 1e-10)
        max_weight = (min_prob * self.size) ** (-beta)
        weights = (probs * self.size) ** (-beta)
        weights /= max(max_weight, 1e-10)  # normalize so max weight = 1

        batch = self._indices_to_batch(indices, device)
        is_weights = weights.astype(np.float32)

        return batch, indices, is_weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities for sampled transitions.

        Args:
            indices: Buffer indices, shape (batch_size,).
            td_errors: Absolute TD errors, shape (batch_size,).
                For SF learning, this is the L2 norm of the TD error vector.
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(float(td_error)) + self._per_eps) ** self._per_alpha
            self._sum_tree.update(int(idx), priority)
            self._max_priority = max(self._max_priority, priority)

    def sample_episodes(self, n_episodes: int,
                        device: Optional[torch.device] = None) -> List[Dict[str, torch.Tensor]]:
        """Sample complete episodes for sequential replay.

        This is the neural analog of the tabular agent's hippocampal replay,
        where episodes are replayed in temporal order.

        Args:
            n_episodes: Number of episodes to sample.
            device: Torch device for the returned tensors.

        Returns:
            List of episode dicts, each with 'obs', 'actions', 'rewards',
            'next_obs', 'dones' tensors.
        """
        if not self._episode_starts:
            return []

        n = min(n_episodes, len(self._episode_starts))
        ep_indices = np.random.choice(len(self._episode_starts), size=n, replace=False)

        episodes = []
        for ei in ep_indices:
            start = self._episode_starts[ei]
            length = self._episode_lengths[ei]
            indices = np.array([(start + i) % self.capacity for i in range(length)])
            episodes.append(self._indices_to_batch(indices, device))

        return episodes

    def _indices_to_batch(self, indices: np.ndarray,
                          device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """Convert buffer indices to a batch of tensors."""
        dev = device or torch.device('cpu')
        return {
            'obs': torch.as_tensor(self.obs[indices], device=dev),
            'actions': torch.as_tensor(self.actions[indices], device=dev),
            'rewards': torch.as_tensor(self.rewards[indices], device=dev),
            'next_obs': torch.as_tensor(self.next_obs[indices], device=dev),
            'dones': torch.as_tensor(self.dones[indices], device=dev),
        }

    @property
    def n_episodes(self) -> int:
        """Number of complete episodes stored."""
        return len(self._episode_starts)

    def __len__(self) -> int:
        return self.size
