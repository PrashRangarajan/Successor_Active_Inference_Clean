"""Tests for SFClustering and HierarchicalNeuralSRAgent (Phase 3).

Covers: SF embedding clustering, macro-state assignment, adjacency
learning, macro SR construction, hierarchical episode execution,
and save/load with hierarchy state.
"""

import os
import tempfile

import gymnasium as gym
import numpy as np
import pytest
import torch

from environments.acrobot import AcrobotAdapter
from environments.mountain_car import MountainCarAdapter
from core.neural.continuous_adapter import ContinuousAdapter
from core.neural.clustering import SFClustering
from core.neural.hierarchical_agent import HierarchicalNeuralSRAgent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def acrobot_adapter():
    env = gym.make('Acrobot-v1')
    base = AcrobotAdapter(env, n_theta_bins=6, n_dtheta_bins=5,
                          goal_velocity_filter=True)
    return ContinuousAdapter(base)


@pytest.fixture
def mountain_car_adapter():
    env = gym.make('MountainCar-v0')
    base = MountainCarAdapter(env, n_pos_bins=10, n_vel_bins=10)
    return ContinuousAdapter(base)


@pytest.fixture
def trained_agent(acrobot_adapter):
    """Agent with a small amount of training for testing."""
    agent = HierarchicalNeuralSRAgent(
        adapter=acrobot_adapter,
        n_clusters=3,
        cluster_method='kmeans',
        n_cluster_samples=200,
        adjacency_episodes=20,
        adjacency_episode_length=20,
        sf_dim=16,
        hidden_sizes=(16, 16),
        gamma=0.99,
        lr=1e-3,
        lr_w=1e-3,
        batch_size=16,
        buffer_size=1000,
        target_update_freq=50,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=200,
    )
    agent.set_goal(None, reward=1.0, default_cost=0.0)
    agent.learn_environment(
        num_episodes=5, steps_per_episode=30, log_interval=100
    )
    return agent


# ---------------------------------------------------------------------------
# SFClustering tests
# ---------------------------------------------------------------------------

class TestSFClustering:

    def test_collect_embeddings(self, trained_agent):
        clustering = SFClustering(n_clusters=3, method='kmeans')
        obs, emb = clustering.collect_embeddings(
            trained_agent, n_samples=100, steps_per_episode=20
        )
        assert obs.shape[0] == 100
        assert emb.shape[0] == 100
        assert obs.shape[1] == trained_agent.obs_dim
        assert emb.shape[1] == trained_agent.sf_dim

    def test_fit_kmeans(self, trained_agent):
        clustering = SFClustering(n_clusters=3, method='kmeans')
        obs, emb = clustering.collect_embeddings(
            trained_agent, n_samples=100
        )
        labels = clustering.fit(obs, emb)
        assert labels.shape == (100,)
        assert set(labels) == {0, 1, 2}

    def test_fit_spectral(self, trained_agent):
        clustering = SFClustering(n_clusters=3, method='spectral')
        obs, emb = clustering.collect_embeddings(
            trained_agent, n_samples=100
        )
        labels = clustering.fit(obs, emb)
        assert labels.shape == (100,)
        assert len(set(labels)) == 3

    def test_predict_single(self, trained_agent):
        clustering = SFClustering(n_clusters=3, method='kmeans')
        obs, emb = clustering.collect_embeddings(
            trained_agent, n_samples=100
        )
        clustering.fit(obs, emb)

        test_obs = trained_agent.adapter.reset()
        label = clustering.predict(test_obs)
        assert 0 <= label < 3

    def test_predict_batch(self, trained_agent):
        clustering = SFClustering(n_clusters=3, method='kmeans')
        obs, emb = clustering.collect_embeddings(
            trained_agent, n_samples=100
        )
        clustering.fit(obs, emb)

        test_obs = np.random.randn(10, trained_agent.obs_dim).astype(
            np.float32
        )
        labels = clustering.predict_batch(test_obs)
        assert labels.shape == (10,)
        assert all(0 <= l < 3 for l in labels)

    def test_predict_before_fit_raises(self):
        clustering = SFClustering(n_clusters=3)
        with pytest.raises(RuntimeError, match="fit"):
            clustering.predict(np.zeros(6))

    def test_cluster_stats(self, trained_agent):
        clustering = SFClustering(n_clusters=3, method='kmeans')
        obs, emb = clustering.collect_embeddings(
            trained_agent, n_samples=100
        )
        clustering.fit(obs, emb)
        stats = clustering.get_cluster_stats()
        assert stats['n_clusters'] == 3
        assert stats['n_samples'] == 100
        assert sum(stats['cluster_sizes'].values()) == 100

    def test_cluster_centers(self, trained_agent):
        clustering = SFClustering(n_clusters=3, method='kmeans')
        obs, emb = clustering.collect_embeddings(
            trained_agent, n_samples=100
        )
        clustering.fit(obs, emb)
        assert clustering.cluster_centers.shape == (3, trained_agent.obs_dim)

    def test_invalid_method(self, trained_agent):
        clustering = SFClustering(n_clusters=3, method='invalid')
        obs, emb = clustering.collect_embeddings(
            trained_agent, n_samples=50
        )
        with pytest.raises(ValueError, match="Unknown"):
            clustering.fit(obs, emb)

    def test_reorder_labels_deterministic(self):
        labels = np.array([2, 2, 0, 0, 1, 1])
        reordered = SFClustering._reorder_labels(labels)
        assert list(reordered) == [0, 0, 1, 1, 2, 2]


# ---------------------------------------------------------------------------
# HierarchicalNeuralSRAgent construction tests
# ---------------------------------------------------------------------------

class TestHierarchicalConstruction:

    def test_inherits_neural_sr(self, acrobot_adapter):
        agent = HierarchicalNeuralSRAgent(
            adapter=acrobot_adapter,
            n_clusters=3,
            sf_dim=16,
            hidden_sizes=(16, 16),
        )
        assert agent.obs_dim == 6
        assert agent.n_actions == 3
        assert agent.sf_dim == 16
        assert agent.n_clusters == 3
        assert agent._hierarchy_learned is False

    def test_flat_episode_still_works(self, trained_agent):
        """Flat episode execution should work without hierarchy."""
        result = trained_agent.run_episode(
            init_state=[0, 0, 0, 0], max_steps=30
        )
        assert 'steps' in result
        assert 'reward' in result
        assert result['steps'] > 0

    def test_hierarchical_before_learn_raises(self, trained_agent):
        with pytest.raises(RuntimeError, match="learn_hierarchy"):
            trained_agent.run_episode_hierarchical(max_steps=10)


# ---------------------------------------------------------------------------
# Hierarchy learning tests
# ---------------------------------------------------------------------------

class TestHierarchyLearning:

    def test_learn_hierarchy(self, trained_agent):
        trained_agent.learn_hierarchy(
            n_cluster_samples=100,
            adjacency_episodes=20,
        )
        assert trained_agent._hierarchy_learned
        assert trained_agent.clustering is not None
        assert trained_agent.M_macro is not None
        assert trained_agent.C_macro is not None
        assert trained_agent.M_macro.shape == (3, 3)
        assert trained_agent.C_macro.shape == (3,)

    def test_adjacency_discovered(self, trained_agent):
        trained_agent.learn_hierarchy(
            n_cluster_samples=100,
            adjacency_episodes=50,
        )
        # At least some adjacency should be found
        total_edges = sum(len(v) for v in trained_agent.adj_list.values())
        assert total_edges > 0

    def test_bottleneck_obs_collected(self, trained_agent):
        trained_agent.learn_hierarchy(
            n_cluster_samples=100,
            adjacency_episodes=50,
        )
        # Should have bottleneck obs for at least some transitions
        total_bottleneck = sum(
            len(v) for v in trained_agent.bottleneck_obs.values()
        )
        assert total_bottleneck > 0

    def test_macro_preference(self, trained_agent):
        trained_agent.learn_hierarchy(n_cluster_samples=100)
        assert trained_agent.C_macro is not None
        assert trained_agent.C_macro.shape == (3,)
        # C_macro should not be all zeros after training
        # (reward weights learned during learn_environment)
        # Note: with minimal training it might be near zero, so just check shape

    def test_macro_values(self, trained_agent):
        trained_agent.learn_hierarchy(n_cluster_samples=100)
        V_macro = trained_agent.get_macro_values()
        assert V_macro.shape == (3,)

    def test_get_macro_state(self, trained_agent):
        trained_agent.learn_hierarchy(n_cluster_samples=100)
        obs = trained_agent.adapter.reset()
        macro = trained_agent.get_macro_state(obs)
        assert 0 <= macro < 3


# ---------------------------------------------------------------------------
# Hierarchical episode execution tests
# ---------------------------------------------------------------------------

class TestHierarchicalExecution:

    @pytest.fixture
    def hierarchical_agent(self, trained_agent):
        trained_agent.learn_hierarchy(
            n_cluster_samples=100,
            adjacency_episodes=50,
        )
        return trained_agent

    def test_hierarchical_episode_runs(self, hierarchical_agent):
        result = hierarchical_agent.run_episode_hierarchical(
            init_state=[0, 0, 0, 0], max_steps=50
        )
        assert 'steps' in result
        assert 'reward' in result
        assert 'reached_goal' in result
        assert 'final_state' in result
        assert result['steps'] > 0

    def test_hierarchical_returns_obs(self, hierarchical_agent):
        result = hierarchical_agent.run_episode_hierarchical(
            init_state=[0, 0, 0, 0], max_steps=50
        )
        assert result['final_state'].shape == (6,)

    def test_flat_vs_hierarchical_both_work(self, hierarchical_agent):
        result_flat = hierarchical_agent.run_episode(
            init_state=[0, 0, 0, 0], max_steps=50
        )
        result_hier = hierarchical_agent.run_episode_hierarchical(
            init_state=[0, 0, 0, 0], max_steps=50
        )
        # Both should return valid results (don't compare performance
        # since this is with minimal training)
        assert result_flat['steps'] > 0
        assert result_hier['steps'] > 0


# ---------------------------------------------------------------------------
# Save / Load tests
# ---------------------------------------------------------------------------

class TestHierarchicalSaveLoad:

    def test_save_load_with_hierarchy(self, trained_agent):
        trained_agent.learn_hierarchy(
            n_cluster_samples=100,
            adjacency_episodes=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "hier_checkpoint.pt")
            trained_agent.save(path)
            assert os.path.exists(path)

            # Create a fresh agent and load
            env = gym.make('Acrobot-v1')
            base = AcrobotAdapter(env, n_theta_bins=6, n_dtheta_bins=5,
                                  goal_velocity_filter=True)
            adapter = ContinuousAdapter(base)
            new_agent = HierarchicalNeuralSRAgent(
                adapter=adapter,
                n_clusters=3,
                sf_dim=16,
                hidden_sizes=(16, 16),
                batch_size=16,
                buffer_size=1000,
            )
            new_agent.set_goal(None, reward=1.0, default_cost=0.0)
            new_agent.load(path)

            assert new_agent._hierarchy_learned
            assert new_agent.M_macro is not None
            assert new_agent.n_clusters == 3
            assert new_agent.clustering is not None

    def test_save_load_without_hierarchy(self, trained_agent):
        """Save/load should work even before hierarchy is learned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "flat_checkpoint.pt")
            trained_agent.save(path)
            assert os.path.exists(path)

    def test_loaded_agent_can_predict_macro(self, trained_agent):
        trained_agent.learn_hierarchy(
            n_cluster_samples=100,
            adjacency_episodes=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "hier_checkpoint.pt")
            trained_agent.save(path)

            env = gym.make('Acrobot-v1')
            base = AcrobotAdapter(env, n_theta_bins=6, n_dtheta_bins=5,
                                  goal_velocity_filter=True)
            adapter = ContinuousAdapter(base)
            new_agent = HierarchicalNeuralSRAgent(
                adapter=adapter,
                n_clusters=3,
                sf_dim=16,
                hidden_sizes=(16, 16),
                batch_size=16,
                buffer_size=1000,
            )
            new_agent.set_goal(None, reward=1.0, default_cost=0.0)
            new_agent.load(path)

            obs = adapter.reset()
            macro = new_agent.get_macro_state(obs)
            assert 0 <= macro < 3


# ---------------------------------------------------------------------------
# Mountain Car cross-environment test
# ---------------------------------------------------------------------------

class TestMountainCarHierarchy:

    def test_mountain_car_clustering(self, mountain_car_adapter):
        agent = HierarchicalNeuralSRAgent(
            adapter=mountain_car_adapter,
            n_clusters=3,
            cluster_method='kmeans',
            sf_dim=16,
            hidden_sizes=(16, 16),
            batch_size=16,
            buffer_size=500,
            target_update_freq=50,
        )
        agent.set_goal(None, reward=1.0, default_cost=0.0)
        agent.learn_environment(
            num_episodes=3, steps_per_episode=20, log_interval=100
        )
        agent.learn_hierarchy(n_cluster_samples=50, adjacency_episodes=10)
        assert agent._hierarchy_learned
        assert agent.M_macro.shape == (3, 3)
