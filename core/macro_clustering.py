"""Macro-state clustering mixin for HierarchicalSRAgent.

Provides spectral clustering of micro states into macro states, adjacency
learning between macro states, and macro-level matrix computation.

Part of the mixin decomposition of the monolithic HierarchicalSRAgent.
All attributes are initialized in HierarchicalSRAgent.__init__; this
mixin only reads/writes them through ``self``.
"""

import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding


class MacroClusteringMixin:
    """Mixin providing macro-state clustering for HierarchicalSRAgent."""

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

        # Let the adapter provide a custom affinity (e.g. cylindrical
        # distance blended with M for periodic state spaces like Pendulum).
        if hasattr(self.adapter, 'get_clustering_affinity'):
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
        has_diverse_starts = hasattr(self.adapter, 'sample_random_state')

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
