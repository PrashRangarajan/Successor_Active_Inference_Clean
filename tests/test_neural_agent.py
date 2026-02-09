"""Tests for NeuralSRAgent and ContinuousAdapter (Phase 2).

Covers: adapter wrapping, agent construction, learning loop, action
selection, episode execution, save/load, and goal transfer.
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
from core.neural.agent import NeuralSRAgent


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
def acrobot_agent(acrobot_adapter):
    agent = NeuralSRAgent(
        adapter=acrobot_adapter,
        sf_dim=32,
        hidden_sizes=(32, 32),
        gamma=0.99,
        lr=1e-3,
        lr_w=1e-3,
        batch_size=16,
        buffer_size=1000,
        target_update_freq=50,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay_steps=500,
    )
    agent.set_goal(None, reward=1.0, default_cost=0.0)
    return agent


# ---------------------------------------------------------------------------
# ContinuousAdapter tests
# ---------------------------------------------------------------------------

class TestContinuousAdapter:

    def test_obs_dim(self, acrobot_adapter):
        assert acrobot_adapter.obs_dim == 6  # Acrobot has 6D observations

    def test_n_actions(self, acrobot_adapter):
        assert acrobot_adapter.n_actions == 3

    def test_reset_returns_raw_obs(self, acrobot_adapter):
        obs = acrobot_adapter.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (6,)
        # Should NOT be one-hot (all entries should be non-zero or varied)
        assert not (np.sum(obs != 0) == 1)  # not one-hot

    def test_step_returns_tuple(self, acrobot_adapter):
        acrobot_adapter.reset()
        result = acrobot_adapter.step(0)
        assert len(result) == 5  # (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (6,)
        assert isinstance(reward, (float, int, np.floating))
        assert isinstance(terminated, bool)

    def test_sample_random_state(self, acrobot_adapter):
        obs = acrobot_adapter.sample_random_state()
        assert obs.shape == (6,)

    def test_goal_states(self, acrobot_adapter):
        goals = acrobot_adapter.get_goal_states()
        assert len(goals) > 0
        assert all(isinstance(g, (int, np.integer)) for g in goals)

    def test_goal_reward_fn(self, acrobot_adapter):
        reward_fn = acrobot_adapter.create_goal_reward_fn(
            None, reward=1.0, default_cost=0.0
        )
        obs = acrobot_adapter.reset()
        r = reward_fn(obs)
        assert isinstance(r, float)
        assert r in (0.0, 1.0)

    def test_mountain_car_adapter(self, mountain_car_adapter):
        assert mountain_car_adapter.obs_dim == 2
        assert mountain_car_adapter.n_actions == 3
        obs = mountain_car_adapter.reset()
        assert obs.shape == (2,)


# ---------------------------------------------------------------------------
# NeuralSRAgent tests
# ---------------------------------------------------------------------------

class TestNeuralSRAgent:

    def test_construction(self, acrobot_agent):
        assert acrobot_agent.obs_dim == 6
        assert acrobot_agent.n_actions == 3
        assert acrobot_agent.sf_dim == 32

    def test_invalid_gamma(self, acrobot_adapter):
        with pytest.raises(ValueError, match="gamma"):
            NeuralSRAgent(acrobot_adapter, gamma=1.5)

    def test_invalid_sf_dim(self, acrobot_adapter):
        with pytest.raises(ValueError, match="sf_dim"):
            NeuralSRAgent(acrobot_adapter, sf_dim=0)

    def test_select_action_random(self, acrobot_agent):
        obs = acrobot_agent.adapter.reset()
        acrobot_agent.epsilon = 1.0  # fully random
        actions = set()
        for _ in range(100):
            a = acrobot_agent.select_action(obs, greedy=False)
            assert 0 <= a < 3
            actions.add(a)
        # Should have explored multiple actions
        assert len(actions) > 1

    def test_select_action_greedy(self, acrobot_agent):
        obs = acrobot_agent.adapter.reset()
        a = acrobot_agent.select_action(obs, greedy=True)
        assert 0 <= a < 3

    def test_learn_short(self, acrobot_agent):
        """Verify learn_environment runs without errors."""
        acrobot_agent.learn_environment(
            num_episodes=5,
            steps_per_episode=20,
            log_interval=100,  # suppress output
        )
        assert acrobot_agent.total_steps > 0
        assert len(acrobot_agent.training_log['sf_loss']) > 0
        assert len(acrobot_agent.training_log['episode_reward']) == 5

    def test_run_episode(self, acrobot_agent):
        # Quick train so the agent has some data
        acrobot_agent.learn_environment(
            num_episodes=3, steps_per_episode=20, log_interval=100
        )
        result = acrobot_agent.run_episode(
            init_state=[0, 0, 0, 0], max_steps=50
        )
        assert 'steps' in result
        assert 'reward' in result
        assert 'reached_goal' in result
        assert 'final_state' in result
        assert result['steps'] > 0
        assert result['final_state'].shape == (6,)

    def test_get_q_values(self, acrobot_agent):
        obs = acrobot_agent.adapter.reset()
        q = acrobot_agent.get_q_values(obs)
        assert q.shape == (3,)

    def test_get_sf_embedding(self, acrobot_agent):
        obs_batch = np.random.randn(10, 6).astype(np.float32)
        emb = acrobot_agent.get_sf_embedding(obs_batch)
        assert emb.shape == (10, 32)

    def test_epsilon_decay(self, acrobot_agent):
        acrobot_agent.epsilon = 1.0
        acrobot_agent.total_steps = 0
        acrobot_agent._epsilon_decay_steps = 100

        for _ in range(100):
            acrobot_agent.total_steps += 1
            acrobot_agent._decay_epsilon()

        assert abs(acrobot_agent.epsilon - acrobot_agent._epsilon_end) < 0.01

    def test_save_load(self, acrobot_agent):
        acrobot_agent.learn_environment(
            num_episodes=3, steps_per_episode=10, log_interval=100
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_checkpoint.pt")
            acrobot_agent.save(path)
            assert os.path.exists(path)

            # Modify w to verify load restores it
            old_w = acrobot_agent.w.data.clone()
            acrobot_agent.w.data.fill_(999.0)

            acrobot_agent.load(path)
            assert torch.allclose(acrobot_agent.w.data, old_w)

    def test_run_episode_flat_alias(self, acrobot_agent):
        """run_episode_flat should call the same underlying function."""
        assert NeuralSRAgent.run_episode_flat is NeuralSRAgent.run_episode

    def test_relearn_reward_weights(self, acrobot_agent):
        """Verify reward weight relearning runs without errors."""
        acrobot_agent.learn_environment(
            num_episodes=5, steps_per_episode=30, log_interval=100
        )
        old_w = acrobot_agent.w.data.clone()
        acrobot_agent.relearn_reward_weights(n_updates=10)
        # w should have changed
        assert not torch.allclose(acrobot_agent.w.data, old_w)
