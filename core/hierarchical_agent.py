"""Unified Hierarchical Successor Representation Agent.

This module provides a generic hierarchical SR agent that works with any environment
through the adapter interface. The core algorithm (SR learning, clustering,
hierarchical planning) is environment-agnostic.
"""

import random
from collections import Counter, defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding

from .base_environment import BaseEnvironmentAdapter
from .visualization import VisualizationMixin


class HierarchicalSRAgent(VisualizationMixin):
    """Unified Hierarchical Successor Representation Agent.

    This agent implements:
    1. Successor Representation learning (micro-level)
    2. Spectral clustering for macro state discovery
    3. Adjacency learning between macro states
    4. Hierarchical planning (macro-level then micro-level)

    Works with any environment through the BaseEnvironmentAdapter interface.
    """

    def __init__(
        self,
        adapter: BaseEnvironmentAdapter,
        n_clusters: int = 4,
        gamma: float = 0.99,
        learning_rate: float = 0.05,
        learn_from_experience: bool = True,
        use_replay: bool = True,
        n_replay_epochs: int = 10,
        replay_buffer_size: int = 50000,
        replay_mode: str = 'sequential',
        train_smooth_steps: Optional[int] = None,
        test_smooth_steps: int = 1,
    ):
        """
        Args:
            adapter: Environment adapter implementing BaseEnvironmentAdapter
            n_clusters: Number of macro states for clustering
            gamma: Discount factor for SR
            learning_rate: TD learning rate for SR updates
            learn_from_experience: If True, learn B and M from experience.
                                   If False, use analytical computation.
            use_replay: If True, learn M via TD with hippocampal-style
                        experience replay (bioplausible). If False, compute
                        M analytically from B (fast but not bioplausible).
                        Only applies when learn_from_experience=True.
            n_replay_epochs: Number of replay passes over stored experiences.
            replay_buffer_size: Maximum transitions to store for replay.
            replay_mode: 'sequential' (bioplausible, preserves temporal order)
                         or 'shuffle' (randomize episode order each epoch).
            train_smooth_steps: Number of physics steps per action during
                learning.  ``None`` = auto-detect (10 for continuous, 1
                for discrete).
            test_smooth_steps: Number of physics steps per action during
                test-time episode execution.  Defaults to 1 (single step).
        """
        # --- Input validation ---
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")
        if not (0 < gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if not (0 < learning_rate <= 1):
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
        if use_replay and n_replay_epochs < 1:
            raise ValueError(f"n_replay_epochs must be >= 1 when use_replay is True, got {n_replay_epochs}")
        if test_smooth_steps < 1:
            raise ValueError(f"test_smooth_steps must be >= 1, got {test_smooth_steps}")

        self.adapter = adapter
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.learn_from_experience = learn_from_experience
        self.train_smooth_steps = train_smooth_steps
        self.test_smooth_steps = test_smooth_steps

        # Matrices
        self.B = None  # Transition matrix
        self.M = None  # Successor matrix (micro)
        self.B_macro = None  # Macro transition matrix
        self.M_macro = None  # Macro successor matrix

        # Goal/preference
        self.C = None  # Micro-level preference
        self.C_macro = None  # Macro-level preference
        self.goal_states = []
        self._shaped_goal = False  # True when using continuous shaped reward

        # Clustering results
        self.macro_state_list = None  # List of lists: micro states in each macro state
        self.micro_to_macro = None  # Dict: micro state -> macro state
        self.n_macro_actions = None

        # Adjacency
        self.adj_list = None  # Dict: macro state -> list of adjacent macro states
        self.bottleneck_states = None  # Dict: (macro, macro) -> list of entry micro states

        # Episode tracking
        self.current_state = None
        self.state_history = []
        self.action_history = []

        # Experience replay (hippocampal replay)
        self.use_replay = use_replay
        self.n_replay_epochs = n_replay_epochs
        self.replay_buffer_size = replay_buffer_size
        self.replay_mode = replay_mode
        self.replay_buffer = deque()  # deque of episodes; each episode is a list of (s, a, s', r, done)
        self._replay_buffer_total = 0

        # Policy caching
        self._policy_compiled = False
        self._goal_policy = None
        self._bottleneck_policies = None
        self._macro_policy = None

    # ==================== Core Learning ====================

    def learn_environment(self, num_episodes: int = 1000, flat_only: bool = False):
        """Main learning routine: learn SR, cluster, learn adjacency.

        Args:
            num_episodes: Number of episodes for learning
            flat_only: If True, skip adjacency/macro learning and dedicate ALL
                episodes to SR learning.  Matches legacy flat agent behavior
                where no hierarchy overhead exists.
        """
        print("Learning environment dynamics...")

        # Resolve effective train smooth steps once for the whole learning phase
        if self.train_smooth_steps is not None:
            self._effective_train_smooth = self.train_smooth_steps
        else:
            # Auto-detect: 10 for continuous environments, 1 for discrete
            self._effective_train_smooth = 10 if self.adapter.is_continuous else 1

        # Set learning mode on adapter (for POMDP adapters that support it)
        self.adapter.set_learning_mode(True)

        if flat_only:
            # Flat mode: ALL episodes go to SR learning (no adjacency overhead).
            # Matches legacy flat.py where learn_env_likelikood passes all
            # episodes to learn_successor_transition_matrix.
            sr_episodes = num_episodes
            adj_episodes = 0
        else:
            # Hierarchy mode: proportional split matching legacy hierarchy.py.
            # Adjacency gets 1/n_actions of budget, SR gets the rest.
            # No hard minimum floor — at low episode counts, SR learning needs
            # every episode it can get to converge early.
            adj_episodes = max(num_episodes // 5,
                               num_episodes // self.adapter.n_actions)
            sr_episodes = num_episodes - adj_episodes

        # Learn or compute transition and successor matrices
        # Non-absorbing goal: avoids M(goal,goal) → 1/(1−γ) spike that
        # drowns the value gradient.  Policy only needs relative V ordering,
        # which is preserved without the absorbing self-loop.
        if self.learn_from_experience:
            self.B, self.M = self._learn_sr_from_experience(sr_episodes,
                                                             goal_states=None)
        else:
            self.B = self.adapter.get_transition_matrix()
            self.M = self.adapter.compute_successor_from_transition(self.B, self.gamma)

        if flat_only:
            # Skip macro-level learning entirely — flat agent only needs M
            print("Flat-only mode: skipping macro clustering/adjacency")
        else:
            print("Learning macro state clusters...")
            self.macro_state_list, self.micro_to_macro = self._learn_macro_clusters()

            # Create macro-level preference from micro-level
            self._compute_macro_preference()

            print("Learning macro state adjacency...")
            self.adj_list, self.bottleneck_states = self._learn_adjacency(adj_episodes)
            print(f"Adjacency list: {self.adj_list}")

            # Compute macro-level transition and successor matrices
            self.B_macro, self.M_macro = self._compute_macro_matrices()

        # Exit learning mode
        self.adapter.set_learning_mode(False)

    def learn_environment_incremental(self, delta_episodes: int, flat_only: bool = False):
        """Incremental learning: add more episodes of experience to existing B/M.

        Unlike ``learn_environment()`` which starts fresh, this method builds on
        the existing B and M matrices — matching the legacy ``learn_env_likelikood``
        behavior where the same agent is trained with delta episodes at each
        checkpoint.

        The interaction between partial M, re-clustering, and re-adjacency at each
        checkpoint is what produces the hierarchy vs flat divergence: hierarchy can
        exploit a partially-learned M better than flat navigation.

        Args:
            delta_episodes: Number of *additional* episodes to train
            flat_only: If True, skip adjacency/macro learning and dedicate ALL
                episodes to SR learning.  Matches legacy flat agent behavior.
        """
        print(f"Incremental learning: {delta_episodes} more episodes...")

        # Resolve effective train smooth steps
        if self.train_smooth_steps is not None:
            self._effective_train_smooth = self.train_smooth_steps
        else:
            self._effective_train_smooth = 10 if self.adapter.is_continuous else 1

        self.adapter.set_learning_mode(True)

        if flat_only:
            # Flat mode: ALL episodes go to SR (no adjacency overhead)
            sr_episodes = delta_episodes
            adj_episodes = 0
        else:
            # Hierarchy mode: proportional split (no hard floor)
            adj_episodes = max(delta_episodes // 5,
                               delta_episodes // self.adapter.n_actions)
            sr_episodes = delta_episodes - adj_episodes

        # Learn SR incrementally (reuses existing B, M) — non-absorbing
        if self.learn_from_experience:
            self.B, self.M = self._learn_sr_from_experience(
                sr_episodes, goal_states=None, incremental=True)
        else:
            self.B = self.adapter.get_transition_matrix()
            self.M = self.adapter.compute_successor_from_transition(self.B, self.gamma)

        if flat_only:
            print("Flat-only mode: skipping macro re-clustering/adjacency")
        else:
            # Re-cluster on the updated M
            print("Re-clustering macro states...")
            self.macro_state_list, self.micro_to_macro = self._learn_macro_clusters()
            self._compute_macro_preference()

            # Re-learn adjacency
            print("Re-learning adjacency...")
            self.adj_list, self.bottleneck_states = self._learn_adjacency(adj_episodes)
            print(f"Adjacency list: {self.adj_list}")

            # Recompute macro matrices
            self.B_macro, self.M_macro = self._compute_macro_matrices()

        self.adapter.set_learning_mode(False)

    def set_goal(self, goal_spec: Any, reward: float = 100.0, default_cost: float = -0.1):
        """Set a sparse goal for the agent.

        Goal states become absorbing in B and receive high reward in C;
        all other states receive ``default_cost``.

        Args:
            goal_spec: Environment-specific goal specification
            reward: Reward value at goal states
            default_cost: Default cost for non-goal states
        """
        self.goal_states = self.adapter.get_goal_states(goal_spec)
        self.C = self.adapter.create_goal_prior(self.goal_states, reward, default_cost)
        self._shaped_goal = False
        self._policy_compiled = False
        print(f"Goal states: {self.goal_states}")

    def set_shaped_goal(self, C: np.ndarray, goal_threshold: float = 0.0):
        """Set a continuous shaped reward prior.

        Unlike ``set_goal``, no states are made absorbing in B.  The reward
        landscape C is used directly, and ``_is_at_goal`` returns True when
        the current state's C value exceeds ``goal_threshold``.

        Goal states (for macro-preference and stage-diagram detection) are
        inferred as the top-valued states whose C value is ≥ 80% of max(C).

        Args:
            C: Shaped reward vector, one entry per micro state.
            goal_threshold: C value above which the agent is considered
                            "at goal" (used by ``_is_at_goal``).
        """
        self.C = C
        self._shaped_goal = True
        self._goal_threshold = goal_threshold
        self._policy_compiled = False

        # Infer "goal states" as states whose C value meets the threshold.
        # For negative-range C (e.g. -(θ² + 0.1·ω²) ∈ [-10, 0]), the old
        # "0.8 * max(C)" heuristic fails because 0.8 * 0 ≈ 0.  Using the
        # explicit goal_threshold is robust for any C range.
        self.goal_states = [i for i in range(len(C)) if C[i] >= goal_threshold]
        print(f"Shaped goal: {len(self.goal_states)} high-reward states "
              f"(C ≥ {goal_threshold:.1f}), threshold={goal_threshold:.1f}")

        # Recompute macro-level preference if clusters already exist
        if hasattr(self, 'macro_state_list') and self.macro_state_list is not None:
            self._compute_macro_preference()

    def _is_at_goal(self) -> bool:
        """Check if the agent has reached a goal state.

        For sparse goals, uses the discrete goal-bin check.
        For shaped goals, checks whether the current state's C value
        exceeds the goal threshold.
        """
        s_idx = self.adapter.get_current_state_index()

        if self._shaped_goal:
            return self.C[s_idx] >= self._goal_threshold

        in_goal_bin = s_idx in self.goal_states

        # If no discrete match, definitely not at goal
        if not in_goal_bin:
            return False

        # If the adapter supports continuous terminal checks, require both
        continuous_check = self.adapter.is_terminal()
        if continuous_check is not None:
            return continuous_check  # True only if continuous state is also terminal

        # Pure discrete: goal bin membership is sufficient
        return True

    # ==================== Successor Representation Learning ====================

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
        # from state X → state Y is valid physics regardless of whether Gym
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
            # No replay: keep the TD-learned M as-is (raw incremental learning).
            # This matches legacy behavior where M converges slowly via online
            # TD updates — hierarchy can exploit a partially-learned M better
            # than flat, producing a visible convergence gap.
            print("Using raw TD-learned M (no replay, no analytical fallback)")

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
            evicted = self.replay_buffer.popleft()
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

    # ==================== Macro State Clustering ====================

    def _learn_macro_clusters(self) -> Tuple[List[List[int]], Dict[int, int]]:
        """Cluster micro states into macro states using spectral clustering.

        Returns:
            Tuple of (macro_state_list, micro_to_macro):
                - macro_state_list[i] = list of micro state indices in macro state i
                - micro_to_macro[s] = macro state index for micro state s
        """
        # Get valid (non-wall) states
        valid_mask = self.adapter.get_valid_state_mask()
        valid_indices = np.where(valid_mask)[0]

        # Flatten successor matrix for clustering
        M_flat = self.adapter.flatten_successor_for_clustering(self.M)

        # Extract valid states only
        if len(valid_indices) < self.adapter.n_states:
            # Has walls - need to mask
            M_valid = M_flat[np.ix_(valid_indices, valid_indices)]
        else:
            M_valid = M_flat

        # Make symmetric for spectral clustering
        M_symmetric = np.maximum(M_valid, M_valid.T)

        # Let the adapter provide a custom affinity (e.g. cylindrical
        # distance blended with M for periodic state spaces like Pendulum).
        M_symmetric = self.adapter.get_clustering_affinity(M_symmetric)

        # Compute spectral embedding for visualization
        try:
            pos_valid = SpectralEmbedding(
                n_components=2,
                affinity='precomputed'
            ).fit_transform(M_symmetric)

            # Store positions for all states (walls get zeros)
            self.spectral_positions = np.zeros((self.adapter.n_states, 2))
            self.spectral_positions[valid_indices] = pos_valid
        except Exception as e:
            print(f"Spectral embedding failed: {e}")
            self.spectral_positions = None

        # Spectral clustering
        try:
            sc = SpectralClustering(
                self.n_clusters,
                affinity='precomputed',
                n_init=100,
                assign_labels='discretize'
            )
            labels_valid = sc.fit_predict(M_symmetric)
        except Exception as e:
            print(f"Spectral clustering failed: {e}")
            # Fallback: assign randomly
            labels_valid = np.random.randint(0, self.n_clusters, len(valid_indices))

        # Build full label array (walls get label = n_clusters)
        labels = np.ones(self.adapter.n_states, dtype=int) * self.n_clusters
        for i, valid_idx in enumerate(valid_indices):
            labels[valid_idx] = labels_valid[i]

        # Reorder labels for consistency
        unique_labels = np.unique(labels[labels < self.n_clusters])
        label_order = unique_labels[np.argsort([np.min(np.where(labels == l)[0]) for l in unique_labels])]
        mapping = {old: new for new, old in enumerate(label_order)}
        mapping[self.n_clusters] = self.n_clusters  # Keep wall label

        labels = np.array([mapping.get(l, l) for l in labels])
        self.n_clusters = len(unique_labels)

        # Build macro state lists
        macro_state_list = [[] for _ in range(self.n_clusters)]
        micro_to_macro = {}

        for micro_idx, macro_idx in enumerate(labels):
            if macro_idx < self.n_clusters:  # Not a wall
                macro_state_list[macro_idx].append(micro_idx)
                micro_to_macro[micro_idx] = macro_idx

        print(f"Created {self.n_clusters} macro states")
        for i, states in enumerate(macro_state_list):
            print(f"  Macro {i}: {len(states)} states")

        return macro_state_list, micro_to_macro

    def _compute_macro_preference(self):
        """Compute macro-level preference from micro-level."""
        if self.C is None:
            return

        self.C_macro = np.zeros(self.n_clusters)
        for macro_idx in range(self.n_clusters):
            micro_states = self.macro_state_list[macro_idx]
            if not micro_states:
                continue
            if self.C.ndim == 1:
                self.C_macro[macro_idx] = np.mean(self.C[micro_states])
            else:
                # Augmented state space - need to handle differently
                values = []
                for micro_idx in micro_states:
                    state = self.adapter.state_space.index_to_state(micro_idx)
                    values.append(self.C[state])
                self.C_macro[macro_idx] = np.mean(values) if values else 0.0

        print(f"Macro preference C_macro: {self.C_macro}")

    # ==================== Adjacency Learning ====================

    def _learn_adjacency(self, num_episodes: int) -> Tuple[Dict, Dict]:
        """Learn which macro states are adjacent through exploration.

        Args:
            num_episodes: Number of episodes to explore

        Returns:
            Tuple of (adj_list, bottleneck_states)
        """
        adj_list = defaultdict(set)
        bottleneck_states = defaultdict(set)

        episode_length = 50
        has_diverse_starts = self.adapter.is_continuous

        for ep in range(num_episodes):
            if has_diverse_starts:
                self.adapter.sample_random_state()
            else:
                self.adapter.reset()
            s = self.adapter.get_current_state_index()

            for step in range(episode_length):
                action = random.randrange(self.adapter.n_actions)
                # Use same smooth stepping as SR learning for consistency
                for _ in range(self._effective_train_smooth):
                    self.adapter.step(action)
                    s_next = self.adapter.get_current_state_index()
                    if s_next != s:
                        break

                # Check if macro state changed
                if s in self.micro_to_macro and s_next in self.micro_to_macro:
                    s_macro = self.micro_to_macro[s]
                    s_next_macro = self.micro_to_macro[s_next]

                    if s_macro != s_next_macro:
                        adj_list[s_macro].add(s_next_macro)
                        bottleneck_states[(s_macro, s_next_macro)].add(s_next)

                s = s_next

        # Convert to regular dicts with lists
        adj_list = {k: list(v) for k, v in adj_list.items()}
        bottleneck_states = {k: list(v) for k, v in bottleneck_states.items()}

        # Compute max number of macro actions
        self.n_macro_actions = max(len(v) for v in adj_list.values()) if adj_list else 1

        return adj_list, bottleneck_states

    def _compute_macro_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute macro-level transition and successor matrices.

        Returns:
            Tuple of (B_macro, M_macro)
        """
        NC = self.n_clusters
        A = self.n_macro_actions

        B_macro = np.zeros((NC, NC, A))

        for state in range(NC):
            if state not in self.adj_list:
                B_macro[state, state, 0] = 1
                continue

            for action_idx, next_state in enumerate(self.adj_list[state]):
                if action_idx < A:
                    B_macro[next_state, state, action_idx] = 1

        # Handle unvisited transitions
        for col in range(NC):
            for action in range(A):
                if np.sum(B_macro[:, col, action]) == 0:
                    B_macro[col, col, action] = 1

        # Normalize
        B_macro = B_macro / B_macro.sum(axis=0, keepdims=True)

        # Compute successor matrix
        # Average transition over actions (column-stochastic convention)
        B_avg = np.sum(B_macro, axis=2) / B_macro.shape[2]
        I = np.eye(NC)
        M_macro = np.linalg.pinv(I - self.gamma * B_avg)

        return B_macro, M_macro

    # ==================== Episode Execution ====================

    def _get_planning_state(self) -> np.ndarray:
        """Get a one-hot state vector for planning from the adapter's current state.

        In MDP environments, the adapter already returns a one-hot vector.
        In POMDP environments, the adapter returns a belief distribution (spread
        over many states), which causes ``_select_micro_action`` to compute
        blurry expected values and make poor action choices.  This method
        converts the belief to a clean one-hot at the MAP estimate so the
        micro-level planner can differentiate actions properly.

        Uses the adapter's state_space.index_to_onehot() to produce the correct
        shape (e.g., (N,) for simple gridworld, (N,2) for key gridworld).
        """
        s_idx = self.adapter.get_current_state_index()
        return self.adapter.state_space.index_to_onehot(s_idx)

    def _step_with_smooth(self, action: int, smooth_steps: int) -> Tuple[int, float]:
        """Take an action with smooth stepping for continuous environments.

        Repeats ``adapter.step(action)`` up to *smooth_steps* times, breaking
        early if the discrete state changes or the episode terminates.

        Args:
            action: Action to execute.
            smooth_steps: Maximum number of physics steps to take.

        Returns:
            Tuple of (n_physics_steps, env_reward) where env_reward is the
            sum of actual environment rewards across all sub-steps.
        """
        if smooth_steps <= 1:
            # Fast path: no looping needed
            step_result = self.adapter.step_with_info(action)
            if step_result is not None:
                _, reward, _, _, _ = step_result
                return 1, reward
            else:
                self.adapter.step(action)
                return 1, 0.0

        s_before = self.adapter.get_current_state_index()
        env_reward = 0.0
        for i in range(smooth_steps):
            step_result = self.adapter.step_with_info(action)
            if step_result is not None:
                _, reward, terminated, truncated, _ = step_result
                env_reward += reward
                if terminated or truncated:
                    return i + 1, env_reward
            else:
                self.adapter.step(action)
            s_after = self.adapter.get_current_state_index()
            if s_after != s_before:
                return i + 1, env_reward
            # Stop immediately when the continuous state reaches the goal
            # (e.g. within 0.45 units for PointMaze).  Without this check
            # the ball can overshoot the goal during the remaining sub-steps.
            terminal = self.adapter.is_terminal()
            if terminal is True:
                return i + 1, env_reward
        return smooth_steps, env_reward

    def reset_episode(self, init_state: Optional[Any] = None):
        """Reset for a new episode.

        Args:
            init_state: Optional initial state
        """
        self.adapter.reset(init_state)
        self.current_state = self._get_planning_state()
        self.state_history = [self.current_state.copy()]
        self.action_history = []

    def run_episode_hierarchical(self, max_steps: int = 200) -> Dict[str, Any]:
        """Run an episode using hierarchical planning.

        For sparse goals: macro-level navigation to the goal cluster, then
        micro-level fine-tuning.  Terminates on goal arrival.

        For shaped goals: the hierarchy is used only for the initial
        approach — once the agent enters the best macro state it switches
        to flat micro-level control for the remainder of the episode,
        accumulating reward by staying in high-value states.

        If ``compile_policy()`` has been called, uses O(1) cached lookups.

        Args:
            max_steps: Maximum number of steps

        Returns:
            Dict with episode statistics
        """
        if self._policy_compiled:
            return self._run_episode_hierarchical_cached(max_steps)

        total_steps = 0
        total_reward = 0.0
        macro_decisions = 0         # k²-cost macro planning decisions
        micro_phase = False         # N²-cost micro phase computation

        s_idx = self.adapter.get_current_state_index()

        # Determine goal macro state(s)
        goal_macro_states = set()
        for gs in self.goal_states:
            if gs in self.micro_to_macro:
                goal_macro_states.add(self.micro_to_macro[gs])

        # Hierarchical planning phase: navigate to the goal macro state
        while total_steps < max_steps:
            if s_idx not in self.micro_to_macro:
                break

            s_macro = self.micro_to_macro[s_idx]

            if s_macro in goal_macro_states:
                break  # Reached goal macro state — switch to micro

            # Compute macro-level values
            V_macro = self.M_macro @ self.C_macro

            # Find best macro action
            best_macro_action = self._select_macro_action(s_macro, V_macro)

            if best_macro_action is None:
                break  # No valid macro actions

            target_macro = self.adj_list[s_macro][best_macro_action]
            macro_decisions += 1     # one k²-cost macro planning decision

            # Execute macro action (navigate to target macro state)
            # Note: each macro action also involves an N²-cost bottleneck
            # policy computation inside _execute_macro_action.
            steps, reward = self._execute_macro_action(s_macro, target_macro, max_steps - total_steps)
            total_steps += steps
            total_reward += reward

            s_idx = self.adapter.get_current_state_index()

            if not self._shaped_goal and self._is_at_goal():
                break

        # Micro-level phase: fine-grained control
        # For sparse goals: reach the exact goal state.
        # For shaped goals: run the full remaining episode, accumulating reward.
        if total_steps < max_steps:
            V = self.adapter.multiply_M_C(self.M, self.C)
            stop_at_goal = not self._shaped_goal
            micro_phase = True       # one N²-cost computation

            while total_steps < max_steps:
                if stop_at_goal and self._is_at_goal():
                    break

                action = self._select_micro_action(V)
                n_phys, step_reward = self._step_with_smooth(action, self.test_smooth_steps)
                total_reward += step_reward
                self.current_state = self._get_planning_state()
                self.state_history.append(self.current_state.copy())
                self.action_history.append(action)

                s_idx = self.adapter.get_current_state_index()
                total_steps += n_phys

        return {
            'steps': total_steps,
            'reward': total_reward,
            'reached_goal': self._is_at_goal(),
            'final_state': self.adapter.get_current_state(),
            'macro_decisions': macro_decisions,
            'micro_phase': micro_phase,
            'planning_steps': macro_decisions + (1 if micro_phase else 0),
        }

    def run_episode_hierarchical_reentrant(self, max_steps: int = 200) -> Dict[str, Any]:
        """Hierarchical episode with re-entrant macro control.

        Follows the same global goal policy as flat (``V = M @ C``), but
        counts each cluster boundary crossing as a macro-level planning
        decision that the hierarchical agent would need to make.

        This gives an honest macro-decision count for planning-step
        comparisons.  Not the default episode method — used for analysis
        figures only.
        """
        total_steps = 0
        total_reward = 0.0
        macro_decisions = 0
        micro_phase_used = False

        s_idx = self.adapter.get_current_state_index()

        # Determine goal macro state(s)
        goal_macro_states = set()
        for gs in self.goal_states:
            if gs in self.micro_to_macro:
                goal_macro_states.add(self.micro_to_macro[gs])

        # Compute goal-level value function once — same policy as flat.
        V_goal = self.adapter.multiply_M_C(self.M, self.C)

        prev_macro = self.micro_to_macro.get(s_idx)

        while total_steps < max_steps:
            if not self._shaped_goal and self._is_at_goal():
                break

            if s_idx not in self.micro_to_macro:
                break

            s_macro = self.micro_to_macro[s_idx]

            # Count a macro decision whenever we enter a new cluster
            if s_macro != prev_macro:
                if s_macro in goal_macro_states:
                    micro_phase_used = True
                else:
                    macro_decisions += 1
                prev_macro = s_macro

            # Always follow the global goal policy (same trajectory as flat)
            action = self._select_micro_action(V_goal)
            n_phys, step_reward = self._step_with_smooth(action, self.test_smooth_steps)
            total_reward += step_reward
            self.current_state = self._get_planning_state()
            self.state_history.append(self.current_state.copy())
            self.action_history.append(action)
            total_steps += n_phys

            s_idx = self.adapter.get_current_state_index()

        return {
            'steps': total_steps,
            'reward': total_reward,
            'reached_goal': self._is_at_goal(),
            'final_state': self.adapter.get_current_state(),
            'macro_decisions': macro_decisions,
            'micro_phase': micro_phase_used,
            'planning_steps': macro_decisions + (1 if micro_phase_used else 0),
        }

    def _select_macro_action(self, s_macro: int, V_macro: np.ndarray) -> Optional[int]:
        """Select best macro action from current macro state."""
        if s_macro not in self.adj_list:
            return None

        adj_states = self.adj_list[s_macro]
        if not adj_states:
            return None

        # Compute values for each adjacent macro state
        values = []
        for adj_macro in adj_states:
            values.append(V_macro[adj_macro])

        # Select best (that actually moves to different state)
        sorted_indices = np.argsort(values)[::-1]

        for idx in sorted_indices:
            if adj_states[idx] != s_macro:
                return idx

        return None

    def _execute_macro_action(self, init_macro: int, target_macro: int, max_steps: int) -> Tuple[int, float]:
        """Execute a macro action by navigating to target macro state.

        Args:
            init_macro: Starting macro state
            target_macro: Target macro state
            max_steps: Maximum steps allowed

        Returns:
            Tuple of (steps_taken, total_reward)
        """
        bottleneck = self.bottleneck_states.get((init_macro, target_macro), [])
        if not bottleneck:
            # Fallback: use all micro states in the target cluster as goal
            bottleneck = self.macro_state_list[target_macro]
            if not bottleneck:
                return 0, 0.0

        # Create temporary goal at bottleneck states
        C_temp = self.adapter.create_goal_prior(bottleneck, reward=10.0, default_cost=0.0)
        V = self.adapter.multiply_M_C(self.M, C_temp)

        steps = 0
        env_reward = 0.0
        s_idx = self.adapter.get_current_state_index()

        while steps < max_steps:
            if s_idx in bottleneck or self._is_at_goal():
                break

            action = self._select_micro_action(V)
            n_phys, step_reward = self._step_with_smooth(action, self.test_smooth_steps)
            env_reward += step_reward
            self.current_state = self._get_planning_state()
            self.state_history.append(self.current_state.copy())
            self.action_history.append(action)

            s_idx = self.adapter.get_current_state_index()
            steps += n_phys

            # Check if we've changed macro state
            if s_idx in self.micro_to_macro:
                current_macro = self.micro_to_macro[s_idx]
                if current_macro == target_macro:
                    break
                # Entered an unexpected cluster — replan at macro level
                if current_macro != init_macro:
                    break

        return steps, env_reward

    def _run_micro_to_goal(self, max_steps: int) -> Tuple[int, float]:
        """Run micro-level policy to reach goal.

        Args:
            max_steps: Maximum steps allowed

        Returns:
            Tuple of (steps_taken, total_reward)
        """
        V = self.adapter.multiply_M_C(self.M, self.C)

        steps = 0
        reward = 0.0
        s_idx = self.adapter.get_current_state_index()

        while steps < max_steps and not self._is_at_goal():
            action = self._select_micro_action(V)
            n_phys, step_reward = self._step_with_smooth(action, self.test_smooth_steps)
            reward += step_reward
            self.current_state = self._get_planning_state()
            self.state_history.append(self.current_state.copy())
            self.action_history.append(action)

            s_idx = self.adapter.get_current_state_index()
            steps += n_phys

        return steps, reward

    def _select_micro_action(self, V: np.ndarray) -> int:
        """Select best micro action based on expected value.

        For each action, computes the expected next-state value under the
        learned transition distribution: E[V(s')] = B(:,s,a) · V.

        This properly handles stochastic transitions (e.g. Acrobot) where
        the argmax of the transition distribution might be the same state
        for all actions, masking real value differences.

        When the agent is already at a goal state, the best-value action is
        returned directly (staying in place is desirable for maintenance).

        Args:
            V: Value function (flat array)

        Returns:
            Best action index
        """
        s_onehot = self.current_state

        # Compute expected value for each action
        values = []
        for action in range(self.adapter.n_actions):
            s_next_dist = self.adapter.multiply_B_s(self.B, s_onehot, action)
            # Flatten for augmented state spaces (e.g., key gridworld: shape (N,2))
            s_flat = s_next_dist.flatten('F') if s_next_dist.ndim > 1 else s_next_dist
            expected_value = float(s_flat @ V)
            values.append(expected_value)

        sorted_actions = np.argsort(values)[::-1]

        # At the goal, staying in place is the right thing to do —
        # skip the "must change state" filter and return highest-value action.
        if self._is_at_goal():
            return sorted_actions[0]

        # Away from goal: prefer actions that actually change expected state
        for action in sorted_actions:
            s_next = self.adapter.multiply_B_s(self.B, s_onehot, action)
            if not np.allclose(s_next, s_onehot):
                return action

        # If all actions keep us in same state, return best anyway
        return sorted_actions[0]

    def run_episode_flat(self, max_steps: int = 200) -> Dict[str, Any]:
        """Run an episode using only micro-level (flat) planning.

        For sparse goals the episode terminates when the goal is reached.
        For shaped goals the episode runs for the full ``max_steps`` so
        the agent accumulates reward by staying near the optimum.

        If ``compile_policy()`` has been called, uses O(1) cached lookups.

        Args:
            max_steps: Maximum number of steps

        Returns:
            Dict with episode statistics
        """
        if self._policy_compiled:
            return self._run_episode_flat_cached(max_steps)

        V = self.adapter.multiply_M_C(self.M, self.C)

        steps = 0
        reward = 0.0
        s_idx = self.adapter.get_current_state_index()
        stop_at_goal = not self._shaped_goal

        while steps < max_steps:
            if stop_at_goal and self._is_at_goal():
                break

            action = self._select_micro_action(V)
            n_phys, step_reward = self._step_with_smooth(action, self.test_smooth_steps)
            reward += step_reward
            self.current_state = self._get_planning_state()
            self.state_history.append(self.current_state.copy())
            self.action_history.append(action)

            s_idx = self.adapter.get_current_state_index()
            steps += n_phys

        return {
            'steps': steps,
            'reward': reward,
            'reached_goal': self._is_at_goal(),
            'final_state': self.adapter.get_current_state(),
            'macro_decisions': 0,     # flat has no macro decisions
            'micro_phase': True,      # flat is entirely micro
            'planning_steps': steps,  # flat plans every step (each N²)
        }

    # ==================== Policy Compilation ====================

    def compile_policy(self):
        """Precompute state→action lookup tables for O(1) inference.

        After calling this, ``run_episode_hierarchical`` and ``run_episode_flat``
        will use cached dictionary lookups instead of per-step matrix
        multiplications.  Call again (or ``set_goal``) to invalidate.

        Requires that ``learn_environment()`` has been called first.
        """
        if self.B is None or self.M is None:
            raise RuntimeError("Must call learn_environment() before compile_policy()")

        print("Compiling policy tables...")

        # 1. Goal policy: best action toward final goal for every state
        V_goal = self.adapter.multiply_M_C(self.M, self.C)
        self._goal_policy = self._compute_policy_table(V_goal)

        # 2. Bottleneck policies: best action to reach each bottleneck set
        self._bottleneck_policies = {}
        if self.bottleneck_states:
            for (src, tgt), bottleneck in self.bottleneck_states.items():
                C_temp = self.adapter.create_goal_prior(
                    bottleneck, reward=10.0, default_cost=0.0
                )
                V_bn = self.adapter.multiply_M_C(self.M, C_temp)
                self._bottleneck_policies[(src, tgt)] = self._compute_policy_table(V_bn)

        # 2b. Fallback policies for macro transitions without observed bottlenecks
        for s_macro in range(self.n_clusters):
            if s_macro not in self.adj_list:
                continue
            for target_macro in self.adj_list[s_macro]:
                if (s_macro, target_macro) not in self._bottleneck_policies:
                    target_states = self.macro_state_list[target_macro]
                    if target_states:
                        C_temp = self.adapter.create_goal_prior(
                            target_states, reward=10.0, default_cost=0.0
                        )
                        V_bn = self.adapter.multiply_M_C(self.M, C_temp)
                        self._bottleneck_policies[(s_macro, target_macro)] = \
                            self._compute_policy_table(V_bn)

        # 3. Macro policy: best macro action for each macro state
        V_macro = self.M_macro @ self.C_macro
        self._macro_policy = {}
        for s_macro in range(self.n_clusters):
            best = self._select_macro_action(s_macro, V_macro)
            self._macro_policy[s_macro] = best  # int or None

        self._policy_compiled = True
        print(
            f"Policy compiled: {len(self._goal_policy)} micro states, "
            f"{len(self._bottleneck_policies)} bottleneck policies, "
            f"{len(self._macro_policy)} macro states"
        )

    def _compute_policy_table(self, V: np.ndarray) -> dict:
        """Precompute best action for every state given value function *V*.

        Uses the same expected-value + state-change tie-breaking logic as
        ``_select_micro_action``.

        Args:
            V: Value function (flat 1D or multi-dim for augmented spaces)

        Returns:
            Dict mapping ``state_idx`` → ``best_action`` (int)
        """
        policy = {}
        n_states = self.adapter.n_states
        V_flat = V.ravel()

        for s_idx in range(n_states):
            s_onehot = self.adapter.index_to_onehot(s_idx)

            # Expected value under each action's transition distribution
            values = []
            for action in range(self.adapter.n_actions):
                s_next_dist = self.adapter.multiply_B_s(self.B, s_onehot, action)
                expected_value = float(np.dot(s_next_dist.ravel(), V_flat))
                values.append(expected_value)

            # Best action that actually changes expected state
            sorted_actions = np.argsort(values)[::-1]
            best = int(sorted_actions[0])
            for action in sorted_actions:
                s_next = self.adapter.multiply_B_s(self.B, s_onehot, int(action))
                if not np.allclose(s_next, s_onehot):
                    best = int(action)
                    break

            policy[s_idx] = best

        return policy

    # ==================== Cached Episode Execution ====================

    def _run_episode_hierarchical_cached(self, max_steps: int) -> Dict[str, Any]:
        """Run hierarchical episode using precompiled policy tables (O(1) per step)."""
        total_steps = 0
        total_reward = 0.0
        macro_decisions = 0         # macro-level planning decisions
        micro_phase = False         # micro phase used

        goal_macro_states = set()
        for gs in self.goal_states:
            if gs in self.micro_to_macro:
                goal_macro_states.add(self.micro_to_macro[gs])

        # Hierarchical phase: navigate through macro states
        while total_steps < max_steps:
            s_idx = self.adapter.get_current_state_index()
            if s_idx not in self.micro_to_macro:
                break
            s_macro = self.micro_to_macro[s_idx]
            if s_macro in goal_macro_states:
                break

            best_macro = self._macro_policy.get(s_macro)
            if best_macro is None:
                break
            target_macro = self.adj_list[s_macro][best_macro]
            macro_decisions += 1     # one macro planning decision (O(1) cached)

            # Navigate to bottleneck using cached bottleneck policy
            bn_policy = self._bottleneck_policies.get((s_macro, target_macro))
            if bn_policy is None:
                # Fallback: use goal policy to navigate toward the goal
                bn_policy = self._goal_policy
            bottleneck_set = set(
                self.bottleneck_states.get((s_macro, target_macro), [])
            )
            if not bottleneck_set:
                # Fallback: use all target cluster states
                bottleneck_set = set(self.macro_state_list[target_macro])

            while total_steps < max_steps:
                s_idx = self.adapter.get_current_state_index()
                if s_idx in bottleneck_set or self._is_at_goal():
                    break

                action = bn_policy.get(s_idx, 0)  # O(1) lookup
                n_phys, step_reward = self._step_with_smooth(action, self.test_smooth_steps)
                total_reward += step_reward
                self.current_state = self._get_planning_state()
                self.state_history.append(self.current_state.copy())
                self.action_history.append(action)
                total_steps += n_phys

                s_idx = self.adapter.get_current_state_index()

                # Check if we've reached the target macro state
                if s_idx in self.micro_to_macro:
                    if self.micro_to_macro[s_idx] == target_macro:
                        break

            if self._is_at_goal():
                break

        # Micro phase: navigate to exact goal using cached goal policy
        if total_steps < max_steps and not self._is_at_goal():
            micro_phase = True       # micro phase uses cached goal policy (O(1))
        while total_steps < max_steps and not self._is_at_goal():
            s_idx = self.adapter.get_current_state_index()
            action = self._goal_policy.get(s_idx, 0)  # O(1) lookup
            n_phys, step_reward = self._step_with_smooth(action, self.test_smooth_steps)
            total_reward += step_reward
            self.current_state = self._get_planning_state()
            self.state_history.append(self.current_state.copy())
            self.action_history.append(action)
            total_steps += n_phys

            s_idx = self.adapter.get_current_state_index()

        return {
            'steps': total_steps,
            'reward': total_reward,
            'reached_goal': self._is_at_goal(),
            'final_state': self.adapter.get_current_state(),
            'macro_decisions': macro_decisions,
            'micro_phase': micro_phase,
            'planning_steps': macro_decisions + (1 if micro_phase else 0),
        }

    def _run_episode_flat_cached(self, max_steps: int) -> Dict[str, Any]:
        """Run flat episode using precompiled goal policy table (O(1) per step)."""
        steps = 0
        reward = 0.0

        while steps < max_steps and not self._is_at_goal():
            s_idx = self.adapter.get_current_state_index()
            action = self._goal_policy.get(s_idx, 0)  # O(1) lookup
            n_phys, step_reward = self._step_with_smooth(action, self.test_smooth_steps)
            reward += step_reward
            self.current_state = self._get_planning_state()
            self.state_history.append(self.current_state.copy())
            self.action_history.append(action)

            s_idx = self.adapter.get_current_state_index()
            steps += n_phys

        return {
            'steps': steps,
            'reward': reward,
            'reached_goal': self._is_at_goal(),
            'final_state': self.adapter.get_current_state(),
            'macro_decisions': 0,     # flat has no macro decisions
            'micro_phase': True,      # flat is entirely micro
            'planning_steps': steps,  # flat plans every step (each N²)
        }

    # ==================== Policy Save / Load ====================

    def save_compiled_policy(self, path: str):
        """Save compiled policy tables to an ``.npz`` file for deployment.

        The saved file contains only the policy lookup tables and structural
        metadata (adjacency, bottleneck states, cluster assignments).  No B or
        M matrices are needed — the file is typically ~100× smaller.

        Args:
            path: File path (e.g. ``"acrobot_policy.npz"``)
        """
        import json

        if not self._policy_compiled:
            raise RuntimeError("Must call compile_policy() before save")

        data = {
            'goal_policy_keys': np.array(list(self._goal_policy.keys())),
            'goal_policy_vals': np.array(list(self._goal_policy.values())),
            'macro_policy_keys': np.array(list(self._macro_policy.keys())),
            'macro_policy_vals': np.array(
                [v if v is not None else -1 for v in self._macro_policy.values()]
            ),
            'goal_states': np.array(self.goal_states),
            'n_clusters': np.array(self.n_clusters),
            'micro_to_macro_keys': np.array(list(self.micro_to_macro.keys())),
            'micro_to_macro_vals': np.array(list(self.micro_to_macro.values())),
        }

        # Save each bottleneck policy
        for (src, tgt), policy in self._bottleneck_policies.items():
            data[f'bn_{src}_{tgt}_keys'] = np.array(list(policy.keys()))
            data[f'bn_{src}_{tgt}_vals'] = np.array(list(policy.values()))

        # Save adjacency and structural metadata as JSON
        # Convert numpy ints to Python ints for JSON serialization
        def _to_python(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {_to_python(k): _to_python(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_python(x) for x in obj]
            if isinstance(obj, set):
                return [_to_python(x) for x in obj]
            return obj

        metadata = _to_python({
            'adj_list': {str(k): v for k, v in self.adj_list.items()},
            'bottleneck_states': {
                f"{k[0]}_{k[1]}": v for k, v in self.bottleneck_states.items()
            },
            'bottleneck_policy_keys': [
                f"{s}_{t}" for (s, t) in self._bottleneck_policies.keys()
            ],
        })
        data['metadata_json'] = np.array([json.dumps(metadata)])

        np.savez_compressed(path, **data)
        print(f"Saved compiled policy to {path}")

    @classmethod
    def load_compiled_policy(cls, path: str, adapter: 'BaseEnvironmentAdapter') -> 'HierarchicalSRAgent':
        """Load a compiled policy for inference-only use.

        No B or M matrices are loaded — only the precomputed policy lookup
        tables and structural metadata.

        Args:
            path: Path to ``.npz`` file created by ``save_compiled_policy``
            adapter: An environment adapter (needed for ``step`` / ``reset``)

        Returns:
            A ``HierarchicalSRAgent`` with compiled policy ready for execution
        """
        import json

        data = np.load(path, allow_pickle=True)

        agent = cls(adapter, n_clusters=int(data['n_clusters']))

        # Restore goal policy
        agent._goal_policy = dict(zip(
            data['goal_policy_keys'].tolist(),
            data['goal_policy_vals'].tolist(),
        ))

        # Restore macro policy
        macro_keys = data['macro_policy_keys'].tolist()
        macro_vals = data['macro_policy_vals'].tolist()
        agent._macro_policy = {
            k: (v if v != -1 else None) for k, v in zip(macro_keys, macro_vals)
        }

        # Restore goal states and micro_to_macro
        agent.goal_states = data['goal_states'].tolist()
        agent.micro_to_macro = dict(zip(
            data['micro_to_macro_keys'].tolist(),
            data['micro_to_macro_vals'].tolist(),
        ))

        # Restore structural metadata
        metadata = json.loads(str(data['metadata_json'][0]))
        agent.adj_list = {int(k): v for k, v in metadata['adj_list'].items()}

        agent.bottleneck_states = {}
        for key_str, states in metadata['bottleneck_states'].items():
            s, t = key_str.split('_')
            agent.bottleneck_states[(int(s), int(t))] = states

        # Restore bottleneck policies
        agent._bottleneck_policies = {}
        for key_str in metadata['bottleneck_policy_keys']:
            s, t = key_str.split('_')
            keys = data[f'bn_{s}_{t}_keys'].tolist()
            vals = data[f'bn_{s}_{t}_vals'].tolist()
            agent._bottleneck_policies[(int(s), int(t))] = dict(zip(keys, vals))

        agent._policy_compiled = True
        print(f"Loaded compiled policy from {path}: {len(agent._goal_policy)} states")
        return agent
