"""Tests for HalfCheetah adapter and ActionConditionedSFNetwork.

Skips automatically if mujoco is not installed. Covers:
- HalfCheetahAdapter: construction, actions, step, random state
- ActionConditionedSFNetwork: forward, get_sf, shapes, compatibility
- End-to-end integration with NeuralSRAgent
"""

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco", reason="MuJoCo not installed")

from environments.mujoco.half_cheetah import HalfCheetahAdapter
from core.neural.networks import ActionConditionedSFNetwork, SFNetwork


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cheetah_adapter():
    """HalfCheetah adapter with 3 bins per joint (729 actions)."""
    return HalfCheetahAdapter(n_bins_per_joint=3)


@pytest.fixture
def cheetah_adapter_small():
    """HalfCheetah adapter with 2 bins per joint (64 actions)."""
    return HalfCheetahAdapter(n_bins_per_joint=2)


# ---------------------------------------------------------------------------
# HalfCheetahAdapter Tests
# ---------------------------------------------------------------------------

class TestHalfCheetahConstruction:

    def test_obs_dim(self, cheetah_adapter):
        assert cheetah_adapter.obs_dim == 17

    def test_n_actions_3bins(self, cheetah_adapter):
        assert cheetah_adapter.n_actions == 3 ** 6  # 729

    def test_n_actions_2bins(self, cheetah_adapter_small):
        assert cheetah_adapter_small.n_actions == 2 ** 6  # 64

    def test_discrete_actions_shape(self, cheetah_adapter):
        actions = cheetah_adapter._discrete_actions
        assert actions.shape == (729, 6)

    def test_discrete_actions_range(self, cheetah_adapter):
        actions = cheetah_adapter._discrete_actions
        assert np.all(actions >= -1.0)
        assert np.all(actions <= 1.0)

    def test_discrete_actions_include_extremes(self, cheetah_adapter):
        actions = cheetah_adapter._discrete_actions
        # Should include all-negative and all-positive corners
        all_neg = np.array([-1.0] * 6, dtype=np.float32)
        all_pos = np.array([1.0] * 6, dtype=np.float32)
        assert any(np.allclose(a, all_neg) for a in actions)
        assert any(np.allclose(a, all_pos) for a in actions)


class TestHalfCheetahResetStep:

    def test_reset(self, cheetah_adapter):
        obs = cheetah_adapter.reset()
        assert obs.shape == (17,)
        assert obs.dtype == np.float32

    def test_step(self, cheetah_adapter):
        cheetah_adapter.reset()
        obs, reward, terminated, truncated, info = cheetah_adapter.step(0)
        assert obs.shape == (17,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert 'x_velocity' in info or 'reward_run' in info

    def test_never_terminates(self, cheetah_adapter):
        """HalfCheetah never terminates early."""
        cheetah_adapter.reset()
        for _ in range(50):
            _, _, terminated, _, _ = cheetah_adapter.step(
                np.random.randint(cheetah_adapter.n_actions))
            assert not terminated

    def test_is_terminal_returns_none(self, cheetah_adapter):
        """Locomotion task: is_terminal always returns None."""
        obs = cheetah_adapter.reset()
        assert cheetah_adapter.is_terminal(obs) is None

    def test_multiple_episodes(self, cheetah_adapter):
        for _ in range(3):
            cheetah_adapter.reset()
            for _ in range(20):
                cheetah_adapter.step(
                    np.random.randint(cheetah_adapter.n_actions))


class TestHalfCheetahGoal:

    def test_goal_state_fast(self, cheetah_adapter):
        # obs[8] = x_velocity > 2.0
        obs = np.zeros(17, dtype=np.float32)
        obs[8] = 3.0  # fast forward
        assert cheetah_adapter.is_goal_state(obs) is True

    def test_goal_state_slow(self, cheetah_adapter):
        obs = np.zeros(17, dtype=np.float32)
        obs[8] = 0.5  # slow
        assert cheetah_adapter.is_goal_state(obs) is False

    def test_get_goal_states_empty(self, cheetah_adapter):
        assert cheetah_adapter.get_goal_states() == []


class TestHalfCheetahRandomState:

    def test_sample_random_state(self, cheetah_adapter):
        obs = cheetah_adapter.sample_random_state()
        assert obs.shape == (17,)
        assert obs.dtype == np.float32

    def test_random_states_diverse(self, cheetah_adapter):
        obs_list = [cheetah_adapter.sample_random_state() for _ in range(10)]
        obs_array = np.array(obs_list)
        assert np.std(obs_array, axis=0).sum() > 0.01

    def test_get_state_for_reset(self, cheetah_adapter):
        cheetah_adapter.reset()
        state = cheetah_adapter.get_state_for_reset()
        # nq=9 + nv=9 = 18
        assert len(state) == 18


# ---------------------------------------------------------------------------
# ActionConditionedSFNetwork Tests
# ---------------------------------------------------------------------------

class TestActionConditionedSFNetwork:

    def test_construction(self):
        net = ActionConditionedSFNetwork(
            obs_dim=17, n_actions=729, sf_dim=64,
            hidden_sizes=(128, 128))
        assert net.obs_dim == 17
        assert net.n_actions == 729
        assert net.sf_dim == 64

    def test_forward_shape(self):
        net = ActionConditionedSFNetwork(
            obs_dim=17, n_actions=64, sf_dim=32,
            hidden_sizes=(64, 64))
        obs = torch.randn(4, 17)
        out = net(obs)
        assert out.shape == (4, 64, 32)  # (batch, n_actions, sf_dim)

    def test_get_sf_shape(self):
        net = ActionConditionedSFNetwork(
            obs_dim=17, n_actions=64, sf_dim=32,
            hidden_sizes=(64, 64))
        obs = torch.randn(8, 17)
        actions = torch.randint(0, 64, (8,))
        out = net.get_sf(obs, actions)
        assert out.shape == (8, 32)  # (batch, sf_dim)

    def test_forward_get_sf_consistency(self):
        """forward() and get_sf() should give same results for same actions."""
        net = ActionConditionedSFNetwork(
            obs_dim=4, n_actions=8, sf_dim=16,
            hidden_sizes=(32, 32))
        obs = torch.randn(3, 4)
        actions = torch.tensor([0, 3, 7])

        all_sf = net(obs)  # (3, 8, 16)
        specific_sf = net.get_sf(obs, actions)  # (3, 16)

        # Extract same actions from forward output
        batch_idx = torch.arange(3)
        from_forward = all_sf[batch_idx, actions]  # (3, 16)

        torch.testing.assert_close(specific_sf, from_forward)

    def test_same_interface_as_sfnetwork(self):
        """ActionConditionedSFNetwork has same external interface as SFNetwork."""
        for NetCls in [SFNetwork, ActionConditionedSFNetwork]:
            net = NetCls(obs_dim=6, n_actions=3, sf_dim=8,
                         hidden_sizes=(16, 16))
            obs = torch.randn(2, 6)
            actions = torch.tensor([0, 2])

            all_sf = net(obs)
            assert all_sf.shape == (2, 3, 8)

            specific = net.get_sf(obs, actions)
            assert specific.shape == (2, 8)

    def test_forward_actions_shape(self):
        """forward_actions() returns correct shape for a subset of actions."""
        net = ActionConditionedSFNetwork(
            obs_dim=17, n_actions=729, sf_dim=64,
            hidden_sizes=(64, 64))
        obs = torch.randn(4, 17)
        subset = torch.tensor([0, 10, 50, 100, 200, 500, 728])
        out = net.forward_actions(obs, subset)
        assert out.shape == (4, 7, 64)  # (batch, n_subset, sf_dim)

    def test_forward_actions_consistency(self):
        """forward_actions() gives same results as forward() for same actions."""
        net = ActionConditionedSFNetwork(
            obs_dim=4, n_actions=16, sf_dim=8,
            hidden_sizes=(32, 32))
        obs = torch.randn(3, 4)
        subset = torch.tensor([0, 5, 10, 15])

        all_sf = net(obs)  # (3, 16, 8)
        subset_sf = net.forward_actions(obs, subset)  # (3, 4, 8)

        # They should match for the selected actions
        for i, a in enumerate(subset):
            torch.testing.assert_close(subset_sf[:, i], all_sf[:, a])

    def test_gradients_flow(self):
        """Verify gradients flow through the network."""
        net = ActionConditionedSFNetwork(
            obs_dim=4, n_actions=8, sf_dim=16,
            hidden_sizes=(32, 32))
        obs = torch.randn(4, 4)
        actions = torch.randint(0, 8, (4,))

        sf = net.get_sf(obs, actions)
        loss = sf.sum()
        loss.backward()

        # All parameters should have gradients
        for p in net.parameters():
            assert p.grad is not None
            assert p.grad.abs().sum() > 0

    def test_gradients_flow_forward_actions(self):
        """Verify gradients flow through forward_actions()."""
        net = ActionConditionedSFNetwork(
            obs_dim=4, n_actions=64, sf_dim=16,
            hidden_sizes=(32, 32))
        obs = torch.randn(4, 4)
        subset = torch.tensor([0, 10, 30, 63])

        sf_subset = net.forward_actions(obs, subset)  # (4, 4, 16)
        loss = sf_subset.sum()
        loss.backward()

        for p in net.parameters():
            assert p.grad is not None
            assert p.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# End-to-End Integration
# ---------------------------------------------------------------------------

class TestHalfCheetahIntegration:

    def test_agent_with_action_conditioned_net(self, cheetah_adapter_small):
        """Create agent with ActionConditionedSFNetwork and run briefly."""
        from core.neural.agent import NeuralSRAgent

        agent = NeuralSRAgent(
            adapter=cheetah_adapter_small,
            sf_dim=16,
            hidden_sizes=(32, 32),
            gamma=0.99,
            lr=1e-3,
            lr_w=1e-3,
            batch_size=16,
            buffer_size=500,
            target_update_freq=50,
            tau=0.01,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay_steps=100,
            sf_network_cls='action_conditioned',
        )

        agent.set_goal(reward=1.0, default_cost=0.0, use_env_reward=True)

        # Brief training
        agent.learn_environment(
            num_episodes=3,
            steps_per_episode=20,
            diverse_start=True,
            log_interval=3,
        )

        # Run evaluation episode
        result = agent.run_episode(max_steps=20)
        assert 'steps' in result
        assert result['steps'] > 0

    def test_hierarchical_agent_with_action_conditioned(self, cheetah_adapter_small):
        """HierarchicalNeuralSRAgent with ActionConditionedSFNetwork."""
        from core.neural.hierarchical_agent import HierarchicalNeuralSRAgent

        agent = HierarchicalNeuralSRAgent(
            adapter=cheetah_adapter_small,
            sf_dim=16,
            hidden_sizes=(32, 32),
            gamma=0.99,
            lr=1e-3,
            lr_w=1e-3,
            batch_size=16,
            buffer_size=500,
            target_update_freq=50,
            tau=0.01,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay_steps=100,
            sf_network_cls='action_conditioned',
            n_clusters=2,
            cluster_method='kmeans',
            cluster_on='observations',
            n_cluster_samples=50,
            adjacency_episodes=10,
            adjacency_episode_length=10,
        )

        agent.set_goal(reward=1.0, default_cost=0.0, use_env_reward=True)

        # Brief training
        agent.learn_environment(
            num_episodes=3,
            steps_per_episode=20,
            diverse_start=True,
            log_interval=3,
        )

        result = agent.run_episode(max_steps=20)
        assert result['steps'] > 0
