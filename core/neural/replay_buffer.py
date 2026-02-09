"""Experience replay buffer for neural SR training.

Supports two sampling modes:
- Uniform: random transitions for standard DRL training.
- Episodic: complete episodes for hippocampal-style sequential replay,
  preserving the biological plausibility of the tabular agent's replay.
"""

from typing import Dict, List, Optional

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity replay buffer with pre-allocated numpy arrays.

    Stores transitions as (obs, action, reward, next_obs, done) and
    tracks episode boundaries for optional episodic sampling.

    Args:
        capacity: Maximum number of transitions to store.
        obs_dim: Dimensionality of observations.
    """

    def __init__(self, capacity: int, obs_dim: int):
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
        self.obs[self._ptr] = obs
        self.actions[self._ptr] = action
        self.rewards[self._ptr] = reward
        self.next_obs[self._ptr] = next_obs
        self.dones[self._ptr] = float(done)

        self._ptr = (self._ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def end_episode(self):
        """Mark the end of the current episode for episodic sampling.

        Call this after the last transition of each episode.
        """
        if self.size == 0:
            return

        # Compute episode length
        if self._ptr > self._episode_start_idx:
            ep_len = self._ptr - self._episode_start_idx
        elif self._ptr == self._episode_start_idx:
            return  # empty episode
        else:
            # Wrapped around the buffer
            ep_len = (self.capacity - self._episode_start_idx) + self._ptr

        if ep_len > 0:
            self._episode_starts.append(self._episode_start_idx)
            self._episode_lengths.append(ep_len)

        self._episode_start_idx = self._ptr

        # Remove episodes that have been overwritten by the circular buffer
        self._clean_old_episodes()

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
