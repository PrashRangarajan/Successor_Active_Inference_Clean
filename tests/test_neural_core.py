"""Tests for neural SR core infrastructure (Phase 1).

Covers: network forward passes, shapes, gradient flow, replay buffer,
loss functions, and utility functions.
"""

import numpy as np
import pytest
import torch

from core.neural.networks import SFNetwork, RewardFeatureNetwork, StateEncoder
from core.neural.replay_buffer import ReplayBuffer
from core.neural.losses import sf_td_loss, reward_prediction_loss
from core.neural.utils import soft_update, hard_update, RunningMeanStd


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def obs_dim():
    return 6  # Acrobot observation dimensionality


@pytest.fixture
def n_actions():
    return 3


@pytest.fixture
def sf_dim():
    return 64


@pytest.fixture
def batch_size():
    return 32


@pytest.fixture
def sf_net(obs_dim, n_actions, sf_dim):
    return SFNetwork(obs_dim, n_actions, sf_dim, hidden_sizes=(64, 64))


@pytest.fixture
def reward_net(obs_dim, sf_dim):
    return RewardFeatureNetwork(obs_dim, sf_dim, hidden_sizes=(32, 32))


# ---------------------------------------------------------------------------
# SFNetwork tests
# ---------------------------------------------------------------------------

class TestSFNetwork:

    def test_forward_shape(self, sf_net, batch_size, obs_dim, n_actions, sf_dim):
        obs = torch.randn(batch_size, obs_dim)
        out = sf_net(obs)
        assert out.shape == (batch_size, n_actions, sf_dim)

    def test_get_sf_shape(self, sf_net, batch_size, obs_dim, sf_dim):
        obs = torch.randn(batch_size, obs_dim)
        actions = torch.randint(0, sf_net.n_actions, (batch_size,))
        out = sf_net.get_sf(obs, actions)
        assert out.shape == (batch_size, sf_dim)

    def test_single_observation(self, sf_net, obs_dim, n_actions, sf_dim):
        obs = torch.randn(1, obs_dim)
        out = sf_net(obs)
        assert out.shape == (1, n_actions, sf_dim)

    def test_gradient_flow(self, sf_net, batch_size, obs_dim):
        obs = torch.randn(batch_size, obs_dim)
        out = sf_net(obs)
        loss = out.sum()
        loss.backward()
        # All parameters should have gradients
        for name, param in sf_net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_different_actions_different_sf(self, sf_net, obs_dim, sf_dim):
        obs = torch.randn(1, obs_dim)
        all_sf = sf_net(obs)  # (1, n_actions, sf_dim)
        # Different action heads should produce different outputs
        # (with random init, this is almost certainly true)
        sf_0 = all_sf[0, 0]
        sf_1 = all_sf[0, 1]
        assert not torch.allclose(sf_0, sf_1, atol=1e-6)


# ---------------------------------------------------------------------------
# RewardFeatureNetwork tests
# ---------------------------------------------------------------------------

class TestRewardFeatureNetwork:

    def test_forward_shape(self, reward_net, batch_size, obs_dim, sf_dim):
        obs = torch.randn(batch_size, obs_dim)
        out = reward_net(obs)
        assert out.shape == (batch_size, sf_dim)

    def test_gradient_flow(self, reward_net, batch_size, obs_dim):
        obs = torch.randn(batch_size, obs_dim)
        out = reward_net(obs)
        loss = out.sum()
        loss.backward()
        for name, param in reward_net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"


# ---------------------------------------------------------------------------
# StateEncoder tests
# ---------------------------------------------------------------------------

class TestStateEncoder:

    def test_forward_shape(self, obs_dim):
        latent_dim = 32
        enc = StateEncoder(obs_dim, latent_dim, hidden_sizes=(64, 64))
        obs = torch.randn(16, obs_dim)
        out = enc(obs)
        assert out.shape == (16, latent_dim)


# ---------------------------------------------------------------------------
# ReplayBuffer tests
# ---------------------------------------------------------------------------

class TestReplayBuffer:

    def test_add_and_size(self, obs_dim):
        buf = ReplayBuffer(capacity=100, obs_dim=obs_dim)
        assert len(buf) == 0

        for i in range(50):
            buf.add(np.random.randn(obs_dim), 0, 1.0,
                    np.random.randn(obs_dim), False)
        assert len(buf) == 50

    def test_capacity_overflow(self, obs_dim):
        buf = ReplayBuffer(capacity=10, obs_dim=obs_dim)
        for i in range(25):
            buf.add(np.random.randn(obs_dim), 0, 1.0,
                    np.random.randn(obs_dim), False)
        assert len(buf) == 10  # Capped at capacity

    def test_sample_uniform_shape(self, obs_dim):
        buf = ReplayBuffer(capacity=100, obs_dim=obs_dim)
        for i in range(50):
            buf.add(np.random.randn(obs_dim), i % 3, float(i),
                    np.random.randn(obs_dim), i == 49)

        batch = buf.sample_uniform(16)
        assert batch['obs'].shape == (16, obs_dim)
        assert batch['actions'].shape == (16,)
        assert batch['rewards'].shape == (16,)
        assert batch['next_obs'].shape == (16, obs_dim)
        assert batch['dones'].shape == (16,)

    def test_sample_uniform_dtypes(self, obs_dim):
        buf = ReplayBuffer(capacity=100, obs_dim=obs_dim)
        for i in range(20):
            buf.add(np.random.randn(obs_dim), 1, 0.5,
                    np.random.randn(obs_dim), False)

        batch = buf.sample_uniform(8)
        assert batch['obs'].dtype == torch.float32
        assert batch['actions'].dtype == torch.int64
        assert batch['rewards'].dtype == torch.float32
        assert batch['dones'].dtype == torch.float32

    def test_episode_tracking(self, obs_dim):
        buf = ReplayBuffer(capacity=1000, obs_dim=obs_dim)

        # Add 3 episodes
        for ep in range(3):
            for step in range(10):
                buf.add(np.random.randn(obs_dim), 0, 1.0,
                        np.random.randn(obs_dim), step == 9)
            buf.end_episode()

        assert buf.n_episodes == 3

    def test_sample_episodes(self, obs_dim):
        buf = ReplayBuffer(capacity=1000, obs_dim=obs_dim)
        ep_len = 15

        for ep in range(5):
            for step in range(ep_len):
                buf.add(np.random.randn(obs_dim), 0, 1.0,
                        np.random.randn(obs_dim), step == ep_len - 1)
            buf.end_episode()

        episodes = buf.sample_episodes(2)
        assert len(episodes) == 2
        for ep in episodes:
            assert ep['obs'].shape == (ep_len, obs_dim)

    def test_empty_buffer_sample(self, obs_dim):
        buf = ReplayBuffer(capacity=100, obs_dim=obs_dim)
        episodes = buf.sample_episodes(1)
        assert episodes == []


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------

class TestLosses:

    def test_sf_td_loss_shape(self, sf_dim, batch_size):
        sf_current = torch.randn(batch_size, sf_dim)
        reward_features = torch.randn(batch_size, sf_dim)
        sf_next_target = torch.randn(batch_size, sf_dim)
        dones = torch.zeros(batch_size)

        loss = sf_td_loss(sf_current, reward_features, sf_next_target, dones, 0.99)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_sf_td_loss_terminal(self, sf_dim):
        """At terminal states, target = ψ(s') only (no bootstrap)."""
        sf_current = torch.randn(1, sf_dim)
        reward_features = torch.ones(1, sf_dim)
        sf_next_target = torch.ones(1, sf_dim) * 999  # should be zeroed out
        dones = torch.ones(1)  # terminal

        loss = sf_td_loss(sf_current, reward_features, sf_next_target, dones, 0.99)
        # Target should be just reward_features (dones=1 kills bootstrap)
        expected = torch.nn.functional.mse_loss(sf_current, reward_features.detach())
        assert torch.allclose(loss, expected, atol=1e-6)

    def test_sf_td_loss_gradient(self, sf_dim, batch_size):
        sf_current = torch.randn(batch_size, sf_dim, requires_grad=True)
        reward_features = torch.randn(batch_size, sf_dim)
        sf_next_target = torch.randn(batch_size, sf_dim)
        dones = torch.zeros(batch_size)

        loss = sf_td_loss(sf_current, reward_features, sf_next_target, dones, 0.99)
        loss.backward()
        assert sf_current.grad is not None

    def test_reward_prediction_loss(self, batch_size):
        predicted = torch.randn(batch_size)
        actual = torch.randn(batch_size)
        loss = reward_prediction_loss(predicted, actual)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_perfect_prediction_zero_loss(self, batch_size):
        vals = torch.randn(batch_size)
        loss = reward_prediction_loss(vals, vals)
        assert loss.item() < 1e-10


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestUtils:

    def test_hard_update(self, obs_dim, sf_dim):
        net1 = SFNetwork(obs_dim, 3, sf_dim, hidden_sizes=(32,))
        net2 = SFNetwork(obs_dim, 3, sf_dim, hidden_sizes=(32,))

        # They should differ initially
        p1 = list(net1.parameters())[0].data.clone()
        p2 = list(net2.parameters())[0].data.clone()
        assert not torch.allclose(p1, p2)

        hard_update(net2, net1)

        # Now they should match
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            assert torch.allclose(p1.data, p2.data)

    def test_soft_update(self, obs_dim, sf_dim):
        net1 = SFNetwork(obs_dim, 3, sf_dim, hidden_sizes=(32,))
        net2 = SFNetwork(obs_dim, 3, sf_dim, hidden_sizes=(32,))
        hard_update(net2, net1)

        # Modify net1
        with torch.no_grad():
            for p in net1.parameters():
                p.add_(torch.ones_like(p))

        tau = 0.1
        # Save net2 params before update
        old_p2 = [p.data.clone() for p in net2.parameters()]

        soft_update(net2, net1, tau)

        for old, p1, p2 in zip(old_p2, net1.parameters(), net2.parameters()):
            expected = tau * p1.data + (1 - tau) * old
            assert torch.allclose(p2.data, expected, atol=1e-6)

    def test_running_mean_std(self):
        rms = RunningMeanStd(shape=(4,))

        # Update with known data
        data = np.random.randn(100, 4).astype(np.float64)
        rms.update(data)

        assert np.allclose(rms.mean, data.mean(axis=0), atol=0.01)
        assert np.allclose(rms.var, data.var(axis=0), atol=0.01)

    def test_normalize(self):
        rms = RunningMeanStd(shape=(2,))
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        rms.update(data)

        normalized = rms.normalize(np.array([3.0, 4.0]))
        # Mean is [3,4], so center should be near [0,0]
        assert abs(normalized[0]) < 0.1
        assert abs(normalized[1]) < 0.1
