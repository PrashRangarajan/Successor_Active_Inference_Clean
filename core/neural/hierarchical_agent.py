"""Hierarchical Neural Successor Feature agent.

Extends NeuralSRAgent with macro-state hierarchy:

1. Micro-level: Flat SF agent — Q(s,a) = φ(s,a)ᵀ · w
2. Macro-level: Cluster SF embeddings into macro states, learn macro-state
   adjacency via exploration, build a small tabular macro SR matrix for
   high-level planning.

Hierarchical episode execution:
  Macro planning: select target macro state via V_macro = M_macro @ C_macro
  Micro execution: use SF Q-values to navigate toward bottleneck observations
  at the boundary of the target macro state.

This mirrors the tabular HierarchicalSRAgent's structure but operates on
continuous state spaces via neural function approximation.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .agent import NeuralSRAgent
from .clustering import SFClustering
from .continuous_adapter import ContinuousAdapter


class HierarchicalNeuralSRAgent(NeuralSRAgent):
    """Hierarchical agent combining neural SFs with macro-state planning.

    Inherits all flat SF capabilities from NeuralSRAgent. Adds:
    - Macro-state discovery via SF embedding clustering
    - Macro-level adjacency learning
    - Macro SR matrix for high-level planning
    - Two-phase episode execution (macro navigate → micro goal-seek)

    Args:
        adapter: ContinuousAdapter wrapping a BinnedContinuousAdapter.
        n_clusters: Number of macro states for clustering.
        cluster_method: Clustering algorithm — 'spectral' or 'kmeans'.
        n_cluster_samples: Number of observations to collect for clustering.
        adjacency_episodes: Number of random exploration episodes for
            learning macro-state adjacency.
        adjacency_episode_length: Steps per adjacency exploration episode.
        **kwargs: Additional arguments passed to NeuralSRAgent.
    """

    def __init__(
        self,
        adapter: ContinuousAdapter,
        n_clusters: int = 4,
        cluster_method: str = 'spectral',
        n_cluster_samples: int = 5000,
        adjacency_episodes: int = 500,
        adjacency_episode_length: int = 50,
        **kwargs,
    ):
        super().__init__(adapter=adapter, **kwargs)

        self.n_clusters = n_clusters
        self.cluster_method = cluster_method
        self._n_cluster_samples = n_cluster_samples
        self._adjacency_episodes = adjacency_episodes
        self._adjacency_episode_length = adjacency_episode_length

        # Macro-state structures (populated by learn_hierarchy())
        self.clustering: Optional[SFClustering] = None
        self.adj_list: Dict[int, List[int]] = {}
        self.bottleneck_obs: Dict[Tuple[int, int], List[np.ndarray]] = {}
        self.n_macro_actions: int = 1
        self.B_macro: Optional[np.ndarray] = None
        self.M_macro: Optional[np.ndarray] = None
        self.C_macro: Optional[np.ndarray] = None
        self._hierarchy_learned = False

    # ==================== Hierarchy Learning ====================

    def learn_hierarchy(self, n_cluster_samples: Optional[int] = None,
                        adjacency_episodes: Optional[int] = None):
        """Learn macro-state structure from the trained SF network.

        Call this after learn_environment() has trained the SF network.
        Steps:
        1. Cluster SF embeddings → macro states
        2. Random exploration → macro-state adjacency + bottleneck obs
        3. Build tabular macro SR matrix

        Args:
            n_cluster_samples: Override default number of clustering samples.
            adjacency_episodes: Override default number of adjacency episodes.
        """
        n_samples = n_cluster_samples or self._n_cluster_samples
        n_adj_episodes = adjacency_episodes or self._adjacency_episodes

        # Step 1: Cluster SF embeddings
        print(f"Step 1: Clustering SF embeddings ({n_samples} samples, "
              f"{self.n_clusters} clusters, method={self.cluster_method})...")
        self.clustering = SFClustering(
            n_clusters=self.n_clusters,
            method=self.cluster_method,
        )
        observations, embeddings = self.clustering.collect_embeddings(
            self, n_samples=n_samples
        )
        labels = self.clustering.fit(observations, embeddings)

        stats = self.clustering.get_cluster_stats()
        for c, size in stats['cluster_sizes'].items():
            print(f"  Cluster {c}: {size} samples")

        # Step 2: Learn adjacency
        print(f"\nStep 2: Learning adjacency ({n_adj_episodes} episodes)...")
        self.adj_list, self.bottleneck_obs = self._learn_adjacency(
            n_adj_episodes
        )
        print(f"  Adjacency graph:")
        for macro, neighbors in sorted(self.adj_list.items()):
            n_bottleneck = sum(
                len(self.bottleneck_obs.get((macro, nb), []))
                for nb in neighbors
            )
            print(f"    Cluster {macro} → {neighbors} "
                  f"({n_bottleneck} bottleneck obs)")

        # Step 3: Build macro SR
        print(f"\nStep 3: Building macro SR matrix...")
        self.B_macro, self.M_macro = self._compute_macro_matrices()
        self._compute_macro_preference()
        print(f"  C_macro: {self.C_macro}")
        print(f"  M_macro diagonal: {np.diag(self.M_macro).round(2)}")

        self._hierarchy_learned = True
        print("\nHierarchy learning complete.")

    def _learn_adjacency(self, num_episodes: int
                         ) -> Tuple[Dict[int, List[int]],
                                    Dict[Tuple[int, int], List[np.ndarray]]]:
        """Learn macro-state adjacency through random exploration.

        At each step, classify the current and next observation. If the
        macro state changes, record the adjacency and the next observation
        as a bottleneck observation (boundary entry point).

        Args:
            num_episodes: Number of exploration episodes.

        Returns:
            Tuple of (adj_list, bottleneck_obs):
                adj_list[macro] = list of reachable neighbor macro states
                bottleneck_obs[(src, tgt)] = list of boundary observations
        """
        adj_sets: Dict[int, set] = defaultdict(set)
        bottleneck_lists: Dict[Tuple[int, int], List[np.ndarray]] = \
            defaultdict(list)

        max_bottleneck_per_pair = 50  # cap memory usage

        for ep in range(num_episodes):
            obs = self.adapter.sample_random_state()
            s_macro = self.clustering.predict(obs)

            for step in range(self._adjacency_episode_length):
                action = np.random.randint(self.n_actions)
                next_obs, _, terminated, truncated, _ = self.adapter.step(
                    action
                )
                s_next_macro = self.clustering.predict(next_obs)

                if s_macro != s_next_macro:
                    adj_sets[s_macro].add(s_next_macro)
                    key = (s_macro, s_next_macro)
                    if len(bottleneck_lists[key]) < max_bottleneck_per_pair:
                        bottleneck_lists[key].append(next_obs.copy())

                s_macro = s_next_macro
                obs = next_obs
                if terminated or truncated:
                    break

        # Convert to regular dicts
        adj_list = {k: sorted(v) for k, v in adj_sets.items()}
        bottleneck_obs = dict(bottleneck_lists)

        self.n_macro_actions = max(
            (len(v) for v in adj_list.values()), default=1
        )

        return adj_list, bottleneck_obs

    def _compute_macro_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Build macro-level transition and successor matrices.

        The macro-level is small (n_clusters × n_clusters), so tabular
        computation is efficient. Mirrors the tabular agent's approach:
        B_macro is deterministic (one adjacency = one macro action),
        M_macro = (I - γ B_avg)^{-1}.

        Returns:
            Tuple of (B_macro, M_macro).
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

        # Fill unvisited transitions with self-loops
        for col in range(NC):
            for action in range(A):
                if np.sum(B_macro[:, col, action]) == 0:
                    B_macro[col, col, action] = 1

        # Normalize columns
        col_sums = B_macro.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums == 0, 1, col_sums)
        B_macro = B_macro / col_sums

        # Average over actions → successor matrix
        B_avg = np.sum(B_macro, axis=2) / A
        I = np.eye(NC)
        M_macro = np.linalg.pinv(I - self.gamma * B_avg)

        return B_macro, M_macro

    def _compute_macro_preference(self):
        """Compute macro-level preference C_macro from micro-level rewards.

        Samples observations from each cluster and averages the reward
        prediction ψ(s)ᵀ · w to get the per-cluster expected reward.
        """
        self.C_macro = np.zeros(self.n_clusters)

        if self.clustering is None or self.clustering.labels is None:
            return

        obs = self.clustering.observations
        labels = self.clustering.labels

        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                    device=self.device)
            psi = self.reward_net(obs_t)  # (n, sf_dim)
            rewards = (psi * self.w).sum(dim=-1).cpu().numpy()  # (n,)

        for c in range(self.n_clusters):
            mask = labels == c
            if mask.any():
                self.C_macro[c] = rewards[mask].mean()

    # ==================== Hierarchical Episode Execution ====================

    def run_episode_hierarchical(self, init_state: Optional[Any] = None,
                                 max_steps: int = 200
                                 ) -> Dict[str, Any]:
        """Run an episode using hierarchical macro → micro planning.

        Phase 1 (macro): Select target macro state via V_macro = M_macro @ C_macro.
            Navigate toward bottleneck observations of that cluster using
            temporary reward weights.
        Phase 2 (micro): Once in the goal cluster, use standard SF Q-values
            to reach the exact goal.

        Args:
            init_state: Initial state for reset.
            max_steps: Maximum steps.

        Returns:
            Dict with 'steps', 'reward', 'reached_goal', 'final_state'.
        """
        if not self._hierarchy_learned:
            raise RuntimeError("Must call learn_hierarchy() before "
                               "run_episode_hierarchical()")

        obs = self.adapter.reset(init_state)
        total_steps = 0
        total_reward = 0.0

        # Determine goal macro state(s)
        goal_macros = self._get_goal_macro_states()

        # ---- Phase 1: Macro-level navigation ----
        while total_steps < max_steps:
            s_macro = self.clustering.predict(obs)

            if s_macro in goal_macros:
                break  # Reached goal cluster

            # Select best adjacent macro state
            V_macro = self.M_macro @ self.C_macro
            target_macro = self._select_macro_target(s_macro, V_macro)

            if target_macro is None:
                break  # No valid macro actions

            # Navigate toward target cluster via bottleneck observations
            steps, reward, obs = self._navigate_to_macro(
                s_macro, target_macro, max_steps - total_steps
            )
            total_steps += steps
            total_reward += reward

            # Check if we hit the goal during navigation
            terminal = self.adapter.is_terminal(obs)
            if terminal is True:
                return {
                    'steps': total_steps,
                    'reward': total_reward,
                    'reached_goal': True,
                    'final_state': obs,
                }
            if terminal is None and self.goal_states:
                if self.adapter.is_in_goal_bin(self.goal_states, obs):
                    return {
                        'steps': total_steps,
                        'reward': total_reward,
                        'reached_goal': True,
                        'final_state': obs,
                    }

        # ---- Phase 2: Micro-level goal seeking ----
        while total_steps < max_steps:
            action = self.select_action(obs, greedy=True)
            next_obs, env_reward, terminated, truncated, info = \
                self.adapter.step(action)

            total_reward += env_reward
            total_steps += 1

            terminal = self.adapter.is_terminal(next_obs)
            if terminal is True:
                return {
                    'steps': total_steps,
                    'reward': total_reward,
                    'reached_goal': True,
                    'final_state': next_obs,
                }
            if terminal is None and self.goal_states:
                if self.adapter.is_in_goal_bin(self.goal_states, next_obs):
                    return {
                        'steps': total_steps,
                        'reward': total_reward,
                        'reached_goal': True,
                        'final_state': next_obs,
                    }

            if terminated or truncated:
                break

            obs = next_obs

        return {
            'steps': total_steps,
            'reward': total_reward,
            'reached_goal': False,
            'final_state': self.adapter.get_current_obs(),
        }

    def _get_goal_macro_states(self) -> set:
        """Determine which macro states contain goal states.

        Uses the clustering samples to find which clusters contain
        observations that are in goal bins.
        """
        goal_macros = set()
        if self.clustering is None or self.clustering.observations is None:
            return goal_macros

        for obs, label in zip(self.clustering.observations,
                              self.clustering.labels):
            if self.goal_states and self.adapter.is_in_goal_bin(
                self.goal_states, obs
            ):
                goal_macros.add(int(label))

        return goal_macros

    def _select_macro_target(self, s_macro: int,
                             V_macro: np.ndarray) -> Optional[int]:
        """Select best adjacent macro state to navigate toward.

        Args:
            s_macro: Current macro state.
            V_macro: Macro-level value function.

        Returns:
            Target macro state index, or None if no valid targets.
        """
        if s_macro not in self.adj_list:
            return None

        neighbors = self.adj_list[s_macro]
        if not neighbors:
            return None

        # Pick neighbor with highest macro value (that isn't current)
        values = [V_macro[n] for n in neighbors]
        sorted_indices = np.argsort(values)[::-1]

        for idx in sorted_indices:
            if neighbors[idx] != s_macro:
                return neighbors[idx]

        return None

    def _navigate_to_macro(self, src_macro: int, tgt_macro: int,
                           max_steps: int
                           ) -> Tuple[int, float, np.ndarray]:
        """Navigate from current position toward the target macro state.

        Uses a temporary reward weight vector w_temp that rewards reaching
        the bottleneck observations of the target cluster. The SF network
        φ stays fixed; only the reward signal changes.

        Args:
            src_macro: Source macro state.
            tgt_macro: Target macro state.
            max_steps: Maximum steps for this navigation.

        Returns:
            Tuple of (steps_taken, total_reward, final_obs).
        """
        # Get bottleneck observations for this transition
        bottleneck = self.bottleneck_obs.get((src_macro, tgt_macro), [])

        if not bottleneck:
            # Fallback: use cluster center as a single target
            if self.clustering.cluster_centers is not None:
                bottleneck = [self.clustering.cluster_centers[tgt_macro]]
            else:
                return 0, 0.0, self.adapter.get_current_obs()

        # Compute temporary w that rewards reaching bottleneck obs.
        # We find the mean reward feature ψ across bottleneck observations
        # and set w_temp so that Q → high near bottleneck.
        w_temp = self._compute_bottleneck_w(bottleneck)

        obs = self.adapter.get_current_obs()
        steps = 0
        reward = 0.0

        while steps < max_steps:
            # Action selection using temporary w
            action = self._select_action_with_w(obs, w_temp)
            next_obs, env_reward, terminated, truncated, info = \
                self.adapter.step(action)

            reward += env_reward
            steps += 1

            # Check if we've entered the target macro state
            next_macro = self.clustering.predict(next_obs)
            if next_macro == tgt_macro:
                obs = next_obs
                break

            # Check terminal
            if terminated or truncated:
                obs = next_obs
                break

            obs = next_obs

        return steps, reward, obs

    def _compute_bottleneck_w(self, bottleneck_obs: List[np.ndarray]
                              ) -> torch.Tensor:
        """Compute temporary reward weights that reward reaching bottleneck obs.

        Strategy: compute the mean ψ(s) over bottleneck observations and
        use it as the reward weight. This makes Q(s,a) = φ(s,a)ᵀ · ψ_mean
        high for states whose successor distribution visits the bottleneck
        region — exactly what we want for navigation.

        Args:
            bottleneck_obs: List of boundary observation arrays.

        Returns:
            Temporary w vector.
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(
                np.array(bottleneck_obs, dtype=np.float32),
                device=self.device,
            )
            psi = self.reward_net(obs_t)  # (n_bottleneck, sf_dim)
            w_temp = psi.mean(dim=0)  # (sf_dim,)
            # Normalize to prevent scale issues
            norm = w_temp.norm()
            if norm > 0:
                w_temp = w_temp / norm
            return w_temp

    def _select_action_with_w(self, obs: np.ndarray,
                              w: torch.Tensor) -> int:
        """Select greedy action using a custom reward weight vector.

        Q(s,a) = φ(s,a)ᵀ · w  — same as select_action but with custom w.

        Args:
            obs: Raw observation.
            w: Reward weight vector (possibly temporary).

        Returns:
            Action index.
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            all_sf = self.sf_net(obs_t)  # (1, n_actions, sf_dim)
            q_values = (all_sf * w).sum(dim=-1)  # (1, n_actions)
            return q_values.argmax(dim=1).item()

    # ==================== Diagnostics ====================

    def get_macro_state(self, obs: np.ndarray) -> int:
        """Get the macro state for an observation.

        Args:
            obs: Raw observation.

        Returns:
            Macro state index.
        """
        if self.clustering is None:
            raise RuntimeError("Hierarchy not learned yet.")
        return self.clustering.predict(obs)

    def get_macro_values(self) -> np.ndarray:
        """Compute macro-level value function V_macro = M_macro @ C_macro."""
        if self.M_macro is None or self.C_macro is None:
            raise RuntimeError("Hierarchy not learned yet.")
        return self.M_macro @ self.C_macro

    # ==================== Save / Load ====================

    def save(self, path: str):
        """Save all networks, weights, training state, and hierarchy.

        Args:
            path: File path for the checkpoint.
        """
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path)
                     else '.', exist_ok=True)

        hierarchy_data = None
        if self._hierarchy_learned:
            hierarchy_data = {
                'n_clusters': self.n_clusters,
                'adj_list': self.adj_list,
                'bottleneck_obs': {
                    str(k): [o.tolist() for o in v]
                    for k, v in self.bottleneck_obs.items()
                },
                'n_macro_actions': self.n_macro_actions,
                'B_macro': self.B_macro,
                'M_macro': self.M_macro,
                'C_macro': self.C_macro,
                'clustering_observations': self.clustering.observations,
                'clustering_labels': self.clustering.labels,
                'clustering_method': self.cluster_method,
            }

        torch.save({
            'sf_net': self.sf_net.state_dict(),
            'sf_target': self.sf_target.state_dict(),
            'reward_net': self.reward_net.state_dict(),
            'w': self.w.data,
            'sf_optimizer': self.sf_optimizer.state_dict(),
            'reward_optimizer': self.reward_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'config': {
                'obs_dim': self.obs_dim,
                'n_actions': self.n_actions,
                'sf_dim': self.sf_dim,
                'gamma': self.gamma,
            },
            'hierarchy': hierarchy_data,
        }, path)

    def load(self, path: str):
        """Load networks, weights, training state, and hierarchy.

        Args:
            path: File path of the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.sf_net.load_state_dict(checkpoint['sf_net'])
        self.sf_target.load_state_dict(checkpoint['sf_target'])
        self.reward_net.load_state_dict(checkpoint['reward_net'])
        self.w.data.copy_(checkpoint['w'])
        self.sf_optimizer.load_state_dict(checkpoint['sf_optimizer'])
        self.reward_optimizer.load_state_dict(checkpoint['reward_optimizer'])
        self.total_steps = checkpoint['total_steps']
        self.epsilon = checkpoint['epsilon']

        # Restore hierarchy if present
        hierarchy_data = checkpoint.get('hierarchy')
        if hierarchy_data is not None:
            self.n_clusters = hierarchy_data['n_clusters']
            self.adj_list = hierarchy_data['adj_list']
            self.n_macro_actions = hierarchy_data['n_macro_actions']
            self.B_macro = hierarchy_data['B_macro']
            self.M_macro = hierarchy_data['M_macro']
            self.C_macro = hierarchy_data['C_macro']

            # Restore bottleneck obs (keys were stringified for serialization)
            self.bottleneck_obs = {}
            for k_str, v_list in hierarchy_data['bottleneck_obs'].items():
                key = eval(k_str)  # "(src, tgt)" → tuple
                self.bottleneck_obs[key] = [
                    np.array(o, dtype=np.float32) for o in v_list
                ]

            # Rebuild clustering classifier
            obs = hierarchy_data['clustering_observations']
            labels = hierarchy_data['clustering_labels']
            self.clustering = SFClustering(
                n_clusters=self.n_clusters,
                method=hierarchy_data.get('clustering_method', 'spectral'),
            )
            self.clustering.observations = obs
            self.clustering.labels = labels
            from sklearn.neighbors import KNeighborsClassifier
            self.clustering.classifier = KNeighborsClassifier(
                n_neighbors=min(5, len(obs))
            )
            self.clustering.classifier.fit(obs, labels)

            self._hierarchy_learned = True
