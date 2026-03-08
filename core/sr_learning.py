"""SR learning mixin for HierarchicalSRAgent.

Provides the successor representation learning methods: experience-based
TD learning, transition counting, experience replay, and reward lookup.

Part of the mixin decomposition of the monolithic HierarchicalSRAgent.
All attributes are initialized in HierarchicalSRAgent.__init__; this
mixin only reads/writes them through ``self``.
"""

import random
from typing import List, Optional, Tuple

import numpy as np


class SRLearningMixin:
    """Mixin providing SR learning for HierarchicalSRAgent."""

    def _learn_sr_from_experience(self, num_episodes: int,
                                    goal_states: List[int] = None,
                                    incremental: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Learn transition and successor matrices from random exploration.

        For continuous environments (Mountain Car, Acrobot), uses smooth stepping
        to allow the discretized state to actually change between updates.

        Args:
            num_episodes: Number of episodes to explore
            goal_states: States to make absorbing in B.  Pass None to skip
                         (e.g. for shaped rewards with no absorbing states).
            incremental: If True, build on existing B and M matrices instead
                         of starting fresh.  Matches legacy incremental learning
                         where the same agent accumulates experience across
                         multiple calls.

        Returns:
            Tuple of (B, M) matrices
        """
        if incremental and self.B is not None and self.M is not None:
            # Reuse existing matrices — new transitions accumulate on top
            B = self.B.copy()
            M = self.M.copy()
        else:
            B = self.adapter.create_empty_transition_matrix()
            M = self.adapter.create_empty_successor_matrix()

        episode_length = 40
        experiences = []
        current_episode = []  # transitions for replay buffer

        # Use the effective train smooth steps resolved in learn_environment()
        smooth_steps = self._effective_train_smooth

        # Diverse-start exploration: if the adapter provides sample_random_state(),
        # use it for most episodes to fill in the transition model uniformly.
        # When using diverse starts, we ignore Gym's terminal/truncated flags
        # because we want to learn the full transition structure — a transition
        # from state X -> state Y is valid physics regardless of whether Gym
        # considers the episode "done".  Episodes that start from random high-
        # energy states would otherwise terminate immediately, wasting samples.
        # Continuous adapters provide sample_random_state() and step_with_info()
        # for diverse-start exploration and termination signals.
        has_diverse_starts = self.adapter.is_continuous
        has_step_info = self.adapter.is_continuous

        for ep in range(num_episodes):
            if (ep + 1) % 100 == 0:
                print(f"Learning episode {ep + 1}/{num_episodes}", end='\r')

            if has_diverse_starts:
                self.adapter.sample_random_state()
            else:
                self.adapter.reset()
            s = self.adapter.get_current_state_index()
            done = False

            for step in range(episode_length):
                if done:
                    break

                action = random.randrange(self.adapter.n_actions)
                reseeded = False

                # For continuous envs, take multiple steps to let state actually change
                for _ in range(smooth_steps):
                    if has_step_info:
                        _, _, terminated, truncated, _ = self.adapter.step_with_info(action)
                        if has_diverse_starts and (terminated or truncated):
                            # Record this final transition, then re-seed
                            s_next = self.adapter.get_current_state_index()
                            self._update_transition_count(B, s, s_next, action)
                            # Store terminal transition for replay
                            reward = self._get_reward(s_next) if self.C is not None else 0
                            current_episode.append((s, action, s_next, reward, True))
                            if current_episode:
                                self._store_replay_episode(current_episode)
                                current_episode = []
                            self.adapter.sample_random_state()
                            s = self.adapter.get_current_state_index()
                            reseeded = True
                            break
                        elif not has_diverse_starts:
                            done = terminated or truncated
                    else:
                        self.adapter.step(action)

                    s_next = self.adapter.get_current_state_index()

                    # Break if state changed or episode done
                    if s_next != s or done:
                        break

                # If we reseeded after termination, skip the normal bookkeeping
                # (transition was already recorded above, s is already updated)
                if reseeded:
                    continue

                # Update transition counts
                self._update_transition_count(B, s, s_next, action)

                # Store experience for TD updates
                reward = self._get_reward(s_next) if self.C is not None else 0
                experiences.append([s, action, s_next, reward, False])
                current_episode.append((s, action, s_next, reward, done))

                # TD update for successor matrix
                if len(experiences) > 1:
                    self._update_sr_td(M, experiences[-2], experiences[-1])

                # Final TD update when done
                if done:
                    self._update_sr_td(M, experiences[-1], experiences[-1])

                s = s_next

            # End of episode: flush to replay buffer
            if current_episode:
                self._store_replay_episode(current_episode)
                current_episode = []

        print("\nLearning complete")

        # Normalize transition matrix (and optionally make goal states absorbing)
        B = self.adapter.normalize_transition_matrix(B, goal_states=goal_states)

        if self.use_replay:
            # Bioplausible: learn M via hippocampal-style experience replay.
            # Multiple TD passes over stored trajectories, analogous to
            # sharp-wave ripple replay during rest (Dayan 1993, Stachenfeld 2017).
            replay_lr = self.learning_rate / 2.0
            n_transitions = self._replay_buffer_total
            print(f"Running experience replay ({self.n_replay_epochs} epochs, "
                  f"{n_transitions} transitions, mode={self.replay_mode})...")
            self._replay_sr_updates(M, self.n_replay_epochs, replay_lr)
        else:
            # Analytical fallback: fast but not bioplausible (matrix inverse).
            print("Computing M analytically from B...")
            M = self.adapter.compute_successor_from_transition(B, self.gamma)

        return B, M

    def _get_reward(self, state_idx: int) -> float:
        """Get reward for a state index, handling different C shapes.

        Args:
            state_idx: Flat state index

        Returns:
            Reward value for that state
        """
        if self.C is None:
            return 0.0

        if self.C.ndim == 1:
            # Simple: C is (N,)
            return self.C[state_idx]
        else:
            # Augmented: C is (N, K) - need to convert index
            state_space = self.adapter.state_space
            state = state_space.index_to_state(state_idx)
            return self.C[state[0], state[1]]

    def _update_transition_count(self, B: np.ndarray, s: int, s_next: int, action: int):
        """Update transition count in B matrix.

        Handles different matrix shapes based on environment.
        """
        if B.ndim == 3:
            # Simple: (N, N, A)
            B[s_next, s, action] += 1
        elif B.ndim == 5:
            # Augmented: (N, K, N, K, A)
            # Need to convert flat indices to (base, augment) tuples
            state_space = self.adapter.state_space
            s_state = state_space.index_to_state(s)
            s_next_state = state_space.index_to_state(s_next)
            B[s_next_state[0], s_next_state[1], s_state[0], s_state[1], action] += 1

    def _update_sr_td(self, M: np.ndarray, current_exp, next_exp, lr: float = None):
        """SARSA TD update for successor matrix.

        Args:
            M: Successor matrix to update (modified in place)
            current_exp: [s, a, s', r, done] or tuple
            next_exp: next experience tuple
            lr: Learning rate (default: self.learning_rate)
        """
        if lr is None:
            lr = self.learning_rate

        s1 = current_exp[0]
        s2 = current_exp[2]
        done = current_exp[4]

        if M.ndim == 2:
            # Simple: (N, N)
            I = self.adapter.index_to_onehot(s2)
            if done:
                td_error = I + self.gamma * I - M[s1, :]
            else:
                td_error = I + self.gamma * M[s2, :] - M[s1, :]
            M[s1, :] += lr * td_error

        elif M.ndim == 4:
            # Augmented: (N, K, N, K)
            state_space = self.adapter.state_space
            s1_state = state_space.index_to_state(s1)
            s2_state = state_space.index_to_state(s2)

            I = self.adapter.index_to_onehot(s2)
            if done:
                td_error = I + self.gamma * I - M[s1_state[0], s1_state[1], :, :]
            else:
                td_error = I + self.gamma * M[s2_state[0], s2_state[1], :, :] - M[s1_state[0], s1_state[1], :, :]
            M[s1_state[0], s1_state[1], :, :] += lr * td_error

    # ==================== Experience Replay ====================

    def _store_replay_episode(self, episode: list):
        """Store an episode in the replay buffer with FIFO eviction."""
        self.replay_buffer.append(episode)
        self._replay_buffer_total += len(episode)

        # Evict oldest episodes when buffer exceeds capacity
        while self._replay_buffer_total > self.replay_buffer_size and self.replay_buffer:
            evicted = self.replay_buffer.pop(0)
            self._replay_buffer_total -= len(evicted)

    def _replay_sr_updates(self, M: np.ndarray, n_epochs: int, lr: float):
        """Hippocampal-style experience replay for successor matrix learning.

        Replays stored trajectory episodes multiple times, applying TD updates
        to M. Analogous to sharp-wave ripple replay during rest, where the
        hippocampus replays stored sequences to consolidate the successor
        representation (Dayan 1993, Stachenfeld et al. 2017).

        Args:
            M: Successor matrix to update (modified in place)
            n_epochs: Number of complete passes through the replay buffer
            lr: Learning rate for replay updates
        """
        if not self.replay_buffer:
            print("Warning: replay buffer is empty, skipping replay")
            return

        log_interval = max(1, n_epochs // 5)

        for epoch in range(n_epochs):
            # Determine episode ordering
            episode_indices = list(range(len(self.replay_buffer)))
            if self.replay_mode == 'shuffle':
                random.shuffle(episode_indices)

            n_updates = 0
            for ep_idx in episode_indices:
                episode = self.replay_buffer[ep_idx]
                for t in range(len(episode)):
                    current = episode[t]
                    # Use next transition for bootstrapping; at episode end,
                    # use the last transition itself (terminal)
                    nxt = episode[t + 1] if t + 1 < len(episode) else episode[t]
                    self._update_sr_td(M, current, nxt, lr)
                    n_updates += 1

            if (epoch + 1) % log_interval == 0:
                print(f"  Replay epoch {epoch + 1}/{n_epochs}: "
                      f"{n_updates} updates, M norm={np.linalg.norm(M):.4f}")

        print(f"Replay complete. M Frobenius norm: {np.linalg.norm(M):.4f}")
