"""Macro state discovery via clustering of successor feature embeddings.

Tabular analog: Spectral clustering on the M matrix (N x N), where M[i,j]
encodes the expected discounted future occupancy of state j starting from
state i.  States with similar rows in M (similar successor distributions)
cluster together.

Neural analog: Cluster the SF embeddings φ_mean(s) = mean_a φ(s,a), each
in R^sf_dim.  These embeddings carry the same information as M rows but
live in a learned low-dimensional space, making clustering tractable even
for continuous state spaces.

Pipeline:
1. Sample representative observations via random exploration.
2. Compute φ_mean(s) for each observation via the trained SF network.
3. Cluster embeddings (spectral on RBF affinity or k-means).
4. Build a KNN classifier: obs → macro state, for assigning new
   observations at runtime.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


class SFClustering:
    """Cluster successor feature embeddings to discover macro states.

    Args:
        n_clusters: Number of macro states.
        method: Clustering method — 'spectral' or 'kmeans'.
        n_neighbors_knn: Number of neighbors for the runtime classifier.
    """

    def __init__(self, n_clusters: int = 4, method: str = 'spectral',
                 n_neighbors_knn: int = 5):
        self.n_clusters = n_clusters
        self.method = method
        self.n_neighbors_knn = n_neighbors_knn

        # Populated after fit()
        self.labels: Optional[np.ndarray] = None
        self.observations: Optional[np.ndarray] = None
        self.embeddings: Optional[np.ndarray] = None
        self.classifier: Optional[KNeighborsClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.cluster_centers: Optional[np.ndarray] = None

    def collect_embeddings(self, agent, n_samples: int = 5000,
                           steps_per_episode: int = 200
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """Collect observation-embedding pairs for clustering.

        Prefers sampling from the agent's replay buffer (which contains
        diverse training experience) for better state coverage. Falls
        back to random exploration if the buffer is too small.

        Args:
            agent: A NeuralSRAgent with a trained SF network.
            n_samples: Target number of observations to collect.
            steps_per_episode: Steps per exploration episode (fallback).

        Returns:
            Tuple of (observations, embeddings) arrays.
        """
        obs_list = []

        # Try to sample from replay buffer first — it has diverse training
        # experience that better covers the state space than random walks
        if hasattr(agent, 'buffer') and agent.buffer.size >= n_samples:
            indices = np.random.choice(
                agent.buffer.size, size=n_samples, replace=False
            )
            obs_list = [agent.buffer.obs[i].copy() for i in indices]
        else:
            # Fallback: collect via random exploration
            adapter = agent.adapter
            while len(obs_list) < n_samples:
                obs = adapter.sample_random_state()
                obs_list.append(obs.copy())
                for _ in range(steps_per_episode):
                    action = np.random.randint(adapter.n_actions)
                    next_obs, _, terminated, truncated, _ = adapter.step(action)
                    obs_list.append(next_obs.copy())
                    if terminated or truncated:
                        break
                    if len(obs_list) >= n_samples:
                        break

        observations = np.array(obs_list[:n_samples], dtype=np.float32)
        embeddings = agent.get_sf_embedding(observations)
        return observations, embeddings

    def fit(self, observations: np.ndarray, embeddings: np.ndarray,
            cluster_on: str = 'embeddings') -> np.ndarray:
        """Cluster embeddings (or observations) and build the runtime classifier.

        Args:
            observations: Raw observations, shape (n, obs_dim).
            embeddings: SF embeddings, shape (n, sf_dim).
            cluster_on: What to cluster on — 'embeddings' for SF embeddings,
                'observations' for raw observation features. Observation-space
                clustering is more robust when SF embeddings are poorly
                differentiated (e.g., early training, sparse rewards).

        Returns:
            Cluster labels, shape (n,).
        """
        self.observations = observations
        self.embeddings = embeddings

        # Select features for clustering
        if cluster_on == 'observations':
            features = observations
        else:
            features = embeddings

        # Normalize features for better clustering
        self.scaler = StandardScaler()
        feat_scaled = self.scaler.fit_transform(features)

        if self.method == 'spectral':
            self.labels = self._spectral_cluster(feat_scaled)
        elif self.method == 'kmeans':
            self.labels = self._kmeans_cluster(feat_scaled)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        # Reorder labels for deterministic ordering (by first occurrence)
        self.labels = self._reorder_labels(self.labels)

        # Build KNN classifier for assigning new observations at runtime
        self.classifier = KNeighborsClassifier(
            n_neighbors=min(self.n_neighbors_knn, len(observations))
        )
        self.classifier.fit(observations, self.labels)

        # Compute cluster centers in observation space
        self.cluster_centers = np.zeros(
            (self.n_clusters, observations.shape[1]), dtype=np.float32
        )
        for c in range(self.n_clusters):
            mask = self.labels == c
            if mask.any():
                self.cluster_centers[c] = observations[mask].mean(axis=0)

        return self.labels

    def predict(self, obs: np.ndarray) -> int:
        """Assign a single observation to a macro state.

        Args:
            obs: Raw observation, shape (obs_dim,).

        Returns:
            Macro state index.
        """
        if self.classifier is None:
            raise RuntimeError("Must call fit() before predict()")
        return int(self.classifier.predict(obs.reshape(1, -1))[0])

    def predict_batch(self, obs_batch: np.ndarray) -> np.ndarray:
        """Assign a batch of observations to macro states.

        Args:
            obs_batch: Raw observations, shape (n, obs_dim).

        Returns:
            Macro state labels, shape (n,).
        """
        if self.classifier is None:
            raise RuntimeError("Must call fit() before predict_batch()")
        return self.classifier.predict(obs_batch)

    def _spectral_cluster(self, emb_scaled: np.ndarray) -> np.ndarray:
        """Spectral clustering on RBF affinity of SF embeddings."""
        # Compute RBF affinity matrix from embeddings
        from sklearn.metrics import pairwise_distances
        dists = pairwise_distances(emb_scaled, metric='euclidean')
        # Use median distance as bandwidth (robust heuristic)
        sigma = np.median(dists[dists > 0])
        if sigma == 0:
            sigma = 1.0
        affinity = np.exp(-dists ** 2 / (2 * sigma ** 2))

        sc = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            n_init=50,
            assign_labels='discretize',
            random_state=42,
        )
        return sc.fit_predict(affinity)

    def _kmeans_cluster(self, emb_scaled: np.ndarray) -> np.ndarray:
        """K-means clustering on SF embeddings."""
        km = KMeans(
            n_clusters=self.n_clusters,
            n_init=50,
            random_state=42,
        )
        return km.fit_predict(emb_scaled)

    @staticmethod
    def _reorder_labels(labels: np.ndarray) -> np.ndarray:
        """Reorder cluster labels by first occurrence for determinism."""
        seen = {}
        new_id = 0
        mapping = {}
        for label in labels:
            if label not in seen:
                seen[label] = new_id
                mapping[label] = new_id
                new_id += 1
        return np.array([mapping[l] for l in labels])

    def get_cluster_stats(self) -> Dict[str, any]:
        """Return summary statistics about the clustering."""
        if self.labels is None:
            return {}
        unique, counts = np.unique(self.labels, return_counts=True)
        return {
            'n_clusters': len(unique),
            'cluster_sizes': dict(zip(unique.tolist(), counts.tolist())),
            'n_samples': len(self.labels),
        }
