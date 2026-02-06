"""Unified Hierarchical Successor Representation Agent.

This module provides a generic hierarchical SR agent that works with any environment
through the adapter interface. The core algorithm (SR learning, clustering,
hierarchical planning) is environment-agnostic.
"""

import random
from collections import Counter, defaultdict
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
    ):
        """
        Args:
            adapter: Environment adapter implementing BaseEnvironmentAdapter
            n_clusters: Number of macro states for clustering
            gamma: Discount factor for SR
            learning_rate: TD learning rate for SR updates
            learn_from_experience: If True, learn B and M from experience.
                                   If False, use analytical computation.
        """
        self.adapter = adapter
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.learn_from_experience = learn_from_experience

        # Matrices
        self.B = None  # Transition matrix
        self.M = None  # Successor matrix (micro)
        self.B_macro = None  # Macro transition matrix
        self.M_macro = None  # Macro successor matrix

        # Goal/preference
        self.C = None  # Micro-level preference
        self.C_macro = None  # Macro-level preference
        self.goal_states = []

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

    # ==================== Core Learning ====================

    def learn_environment(self, num_episodes: int = 1000):
        """Main learning routine: learn SR, cluster, learn adjacency.

        Args:
            num_episodes: Number of episodes for learning
        """
        print("Learning environment dynamics...")

        # Set learning mode on adapter (for POMDP adapters that support it)
        if hasattr(self.adapter, 'set_learning_mode'):
            self.adapter.set_learning_mode(True)

        # Learn or compute transition and successor matrices
        if self.learn_from_experience:
            self.B, self.M = self._learn_sr_from_experience(
                num_episodes - num_episodes // self.adapter.n_actions
            )
        else:
            self.B = self.adapter.get_transition_matrix()
            self.M = self.adapter.compute_successor_from_transition(self.B, self.gamma)

        print("Learning macro state clusters...")
        self.macro_state_list, self.micro_to_macro = self._learn_macro_clusters()

        # Create macro-level preference from micro-level
        self._compute_macro_preference()

        print("Learning macro state adjacency...")
        self.adj_list, self.bottleneck_states = self._learn_adjacency(
            num_episodes // self.adapter.n_actions
        )
        print(f"Adjacency list: {self.adj_list}")

        # Compute macro-level transition and successor matrices
        self.B_macro, self.M_macro = self._compute_macro_matrices()

        # Exit learning mode
        if hasattr(self.adapter, 'set_learning_mode'):
            self.adapter.set_learning_mode(False)

    def set_goal(self, goal_spec: Any, reward: float = 100.0, default_cost: float = -0.1):
        """Set the goal for the agent.

        Args:
            goal_spec: Environment-specific goal specification
            reward: Reward value at goal states
            default_cost: Default cost for non-goal states
        """
        self.goal_states = self.adapter.get_goal_states(goal_spec)
        self.C = self.adapter.create_goal_prior(self.goal_states, reward, default_cost)
        print(f"Goal states: {self.goal_states}")

    # ==================== Successor Representation Learning ====================

    def _learn_sr_from_experience(self, num_episodes: int) -> Tuple[np.ndarray, np.ndarray]:
        """Learn transition and successor matrices from random exploration.

        For continuous environments (Mountain Car, Acrobot), uses smooth stepping
        to allow the discretized state to actually change between updates.

        Args:
            num_episodes: Number of episodes to explore

        Returns:
            Tuple of (B, M) matrices
        """
        B = self.adapter.create_empty_transition_matrix()
        M = self.adapter.create_empty_successor_matrix()

        episode_length = 40
        experiences = []

        # Determine if this is a continuous environment that needs smooth stepping
        # Check if adapter has a step_with_info method (continuous envs) or grid_size (discrete)
        is_continuous = not hasattr(self.adapter, 'grid_size')
        smooth_steps = 10 if is_continuous else 1

        for ep in range(num_episodes):
            if (ep + 1) % 100 == 0:
                print(f"Learning episode {ep + 1}/{num_episodes}", end='\r')

            self.adapter.reset()
            s = self.adapter.get_current_state_index()
            done = False

            for step in range(episode_length):
                if done:
                    break

                action = random.randrange(self.adapter.n_actions)

                # For continuous envs, take multiple steps to let state actually change
                for _ in range(smooth_steps):
                    if hasattr(self.adapter, 'step_with_info'):
                        _, _, terminated, truncated, _ = self.adapter.step_with_info(action)
                        done = terminated or truncated
                    else:
                        self.adapter.step(action)

                    s_next = self.adapter.get_current_state_index()

                    # Break if state changed or episode done
                    if s_next != s or done:
                        break

                # Update transition counts
                self._update_transition_count(B, s, s_next, action)

                # Store experience
                # Note: done is always False during exploration to match original behavior
                # The successor matrix learns the full transition structure without early termination
                reward = self._get_reward(s_next) if self.C is not None else 0
                experiences.append([s, action, s_next, reward, False])

                # TD update for successor matrix
                if len(experiences) > 1:
                    self._update_sr_td(M, experiences[-2], experiences[-1])

                # Final TD update when done
                if done:
                    self._update_sr_td(M, experiences[-1], experiences[-1])

                s = s_next

        print("\nLearning complete")

        # Normalize transition matrix and make goal states absorbing
        B = self.adapter.normalize_transition_matrix(B, goal_states=self.goal_states)

        # For continuous environments, compute M analytically from B instead of using TD learning
        # This handles unvisited states better by using the transition matrix structure
        if is_continuous:
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

    def _update_sr_td(self, M: np.ndarray, current_exp: List, next_exp: List):
        """SARSA TD update for successor matrix.

        Args:
            M: Successor matrix to update (modified in place)
            current_exp: [s, a, s', r, done]
            next_exp: next experience tuple
        """
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
            M[s1, :] += self.learning_rate * td_error

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
            M[s1_state[0], s1_state[1], :, :] += self.learning_rate * td_error

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
        if hasattr(self.adapter, 'flatten_successor_for_clustering'):
            M_flat = self.adapter.flatten_successor_for_clustering(self.M)
        else:
            M_flat = self.M if self.M.ndim == 2 else self.M.reshape(
                self.adapter.n_states, self.adapter.n_states
            )

        # Extract valid states only
        if len(valid_indices) < self.adapter.n_states:
            # Has walls - need to mask
            M_valid = M_flat[np.ix_(valid_indices, valid_indices)]
        else:
            M_valid = M_flat

        # Make symmetric for spectral clustering
        M_symmetric = np.maximum(M_valid, M_valid.T)

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
            if self.C.ndim == 1:
                self.C_macro[macro_idx] = np.sum(self.C[micro_states])
            else:
                # Augmented state space - need to handle differently
                for micro_idx in micro_states:
                    state = self.adapter.state_space.index_to_state(micro_idx)
                    self.C_macro[macro_idx] += self.C[state]

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

        for ep in range(num_episodes):
            self.adapter.reset()
            s = self.adapter.get_current_state_index()

            for step in range(episode_length):
                action = random.randrange(self.adapter.n_actions)
                self.adapter.step(action)
                s_next = self.adapter.get_current_state_index()

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
        B_avg = np.sum(B_macro, axis=2)
        B_avg = B_avg / np.sum(B_avg, axis=1, keepdims=True)
        I = np.eye(NC)
        M_macro = np.linalg.pinv(I - 0.9 * B_avg)

        return B_macro, M_macro

    # ==================== Episode Execution ====================

    def reset_episode(self, init_state: Optional[Any] = None):
        """Reset for a new episode.

        Args:
            init_state: Optional initial state
        """
        self.current_state = self.adapter.reset(init_state)
        self.state_history = [self.current_state.copy()]
        self.action_history = []

    def run_episode_hierarchical(self, max_steps: int = 200) -> Dict[str, Any]:
        """Run an episode using hierarchical planning.

        First uses macro-level planning, then switches to micro-level.

        Args:
            max_steps: Maximum number of steps

        Returns:
            Dict with episode statistics
        """
        total_steps = 0
        total_reward = 0.0

        s_idx = self.adapter.get_current_state_index()

        # Determine goal macro state(s)
        goal_macro_states = set()
        for gs in self.goal_states:
            if gs in self.micro_to_macro:
                goal_macro_states.add(self.micro_to_macro[gs])

        # Hierarchical planning phase
        while total_steps < max_steps:
            if s_idx not in self.micro_to_macro:
                break

            s_macro = self.micro_to_macro[s_idx]

            if s_macro in goal_macro_states:
                break  # Reached goal macro state

            # Compute macro-level values
            V_macro = self.M_macro @ self.C_macro

            # Find best macro action
            best_macro_action = self._select_macro_action(s_macro, V_macro)

            if best_macro_action is None:
                break  # No valid macro actions

            target_macro = self.adj_list[s_macro][best_macro_action]

            # Execute macro action (navigate to target macro state)
            steps, reward = self._execute_macro_action(s_macro, target_macro, max_steps - total_steps)
            total_steps += steps
            total_reward += reward

            s_idx = self.adapter.get_current_state_index()

            if s_idx in self.goal_states:
                break

        # Micro-level planning to reach exact goal
        if s_idx not in self.goal_states and total_steps < max_steps:
            steps, reward = self._run_micro_to_goal(max_steps - total_steps)
            total_steps += steps
            total_reward += reward

        return {
            'steps': total_steps,
            'reward': total_reward,
            'reached_goal': self.adapter.get_current_state_index() in self.goal_states,
            'final_state': self.adapter.get_current_state(),
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
            return 0, 0.0

        # Create temporary goal at bottleneck states
        C_temp = self.adapter.create_goal_prior(bottleneck, reward=10.0, default_cost=0.0)
        V = self.adapter.multiply_M_C(self.M, C_temp)

        steps = 0
        reward = 0.0
        s_idx = self.adapter.get_current_state_index()

        while steps < max_steps:
            if s_idx in bottleneck or s_idx in self.goal_states:
                break

            action = self._select_micro_action(V)
            self.current_state = self.adapter.step(action)
            self.state_history.append(self.current_state.copy())
            self.action_history.append(action)

            s_idx = self.adapter.get_current_state_index()
            if self.C is not None:
                if self.C.ndim == 1:
                    reward += self.C[s_idx]
                else:
                    state = self.adapter.state_space.index_to_state(s_idx)
                    reward += self.C[state]

            steps += 1

            # Check if we've changed macro state
            if s_idx in self.micro_to_macro:
                if self.micro_to_macro[s_idx] == target_macro:
                    break

        return steps, reward

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

        while steps < max_steps and s_idx not in self.goal_states:
            action = self._select_micro_action(V)
            self.current_state = self.adapter.step(action)
            self.state_history.append(self.current_state.copy())
            self.action_history.append(action)

            s_idx = self.adapter.get_current_state_index()
            if self.C.ndim == 1:
                reward += self.C[s_idx]
            else:
                state = self.adapter.state_space.index_to_state(s_idx)
                reward += self.C[state]

            steps += 1

        return steps, reward

    def _select_micro_action(self, V: np.ndarray) -> int:
        """Select best micro action based on value function.

        Args:
            V: Value function (flat array)

        Returns:
            Best action index
        """
        s_onehot = self.current_state

        # Compute values for each action
        values = []
        for action in range(self.adapter.n_actions):
            s_next = self.adapter.multiply_B_s(self.B, s_onehot, action)
            next_idx = self.adapter.onehot_to_index(s_next)
            values.append(V[next_idx])

        # Select best action that actually changes state
        sorted_actions = np.argsort(values)[::-1]

        for action in sorted_actions:
            s_next = self.adapter.multiply_B_s(self.B, s_onehot, action)
            if not np.allclose(s_next, s_onehot):
                return action

        # If all actions keep us in same state, return best anyway
        return sorted_actions[0]

    def run_episode_flat(self, max_steps: int = 200) -> Dict[str, Any]:
        """Run an episode using only micro-level (flat) planning.

        Args:
            max_steps: Maximum number of steps

        Returns:
            Dict with episode statistics
        """
        V = self.adapter.multiply_M_C(self.M, self.C)

        steps = 0
        reward = 0.0
        s_idx = self.adapter.get_current_state_index()

        while steps < max_steps and s_idx not in self.goal_states:
            action = self._select_micro_action(V)
            self.current_state = self.adapter.step(action)
            self.state_history.append(self.current_state.copy())
            self.action_history.append(action)

            s_idx = self.adapter.get_current_state_index()
            if self.C.ndim == 1:
                reward += self.C[s_idx]
            else:
                state = self.adapter.state_space.index_to_state(s_idx)
                reward += self.C[state]

            steps += 1

        return {
            'steps': steps,
            'reward': reward,
            'reached_goal': s_idx in self.goal_states,
            'final_state': self.adapter.get_current_state(),
        }
