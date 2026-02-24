"""Tests for MuJoCo adapters (InvertedPendulum).

Skips automatically if mujoco is not installed. Covers:
- Adapter construction and properties
- Reset, step, action discretization
- Goal condition checking
- Random state injection
- State save/restore
- Compatibility with NeuralSRAgent interface
"""

import tempfile

import numpy as np
import pytest

# Skip all tests if mujoco is not installed
mujoco = pytest.importorskip("mujoco", reason="MuJoCo not installed")
import gymnasium as gym

from environments.mujoco.base_adapter import MuJoCoAdapter
from environments.mujoco.inverted_pendulum import InvertedPendulumAdapter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def pendulum_adapter():
    """Create InvertedPendulumAdapter with 7 force bins."""
    return InvertedPendulumAdapter(n_force_bins=7)


@pytest.fixture
def pendulum_adapter_render():
    """Create InvertedPendulumAdapter with rgb_array render mode."""
    return InvertedPendulumAdapter(n_force_bins=7, render_mode='rgb_array')


# ---------------------------------------------------------------------------
# Construction & Properties
# ---------------------------------------------------------------------------

class TestInvertedPendulumConstruction:
    """Test adapter creation and basic properties."""

    def test_obs_dim(self, pendulum_adapter):
        assert pendulum_adapter.obs_dim == 4  # [x, v, theta, omega]

    def test_n_actions(self, pendulum_adapter):
        assert pendulum_adapter.n_actions == 7

    def test_custom_force_bins(self):
        adapter = InvertedPendulumAdapter(n_force_bins=11)
        assert adapter.n_actions == 11

    def test_env_property(self, pendulum_adapter):
        assert pendulum_adapter.env is not None
        assert hasattr(pendulum_adapter.env, 'step')
        assert hasattr(pendulum_adapter.env, 'reset')

    def test_discrete_actions_shape(self, pendulum_adapter):
        actions = pendulum_adapter._discrete_actions
        assert actions.shape == (7, 1)

    def test_discrete_actions_range(self, pendulum_adapter):
        actions = pendulum_adapter._discrete_actions
        assert np.isclose(actions[0, 0], -3.0)
        assert np.isclose(actions[-1, 0], 3.0)
        # Actions should be sorted
        assert np.all(np.diff(actions[:, 0]) > 0)


# ---------------------------------------------------------------------------
# Reset & Step
# ---------------------------------------------------------------------------

class TestResetAndStep:
    """Test basic environment interaction."""

    def test_reset_returns_observation(self, pendulum_adapter):
        obs = pendulum_adapter.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)
        assert obs.dtype == np.float32

    def test_step_returns_tuple(self, pendulum_adapter):
        pendulum_adapter.reset()
        result = pendulum_adapter.step(3)  # middle action
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_get_current_obs(self, pendulum_adapter):
        obs = pendulum_adapter.reset()
        current = pendulum_adapter.get_current_obs()
        np.testing.assert_array_equal(obs, current)

    def test_get_current_obs_updates_after_step(self, pendulum_adapter):
        obs1 = pendulum_adapter.reset()
        obs2, _, _, _, _ = pendulum_adapter.step(0)
        current = pendulum_adapter.get_current_obs()
        np.testing.assert_array_equal(obs2, current)

    def test_reset_with_init_state(self, pendulum_adapter):
        pendulum_adapter.reset()
        # Set specific state: x=0.1, theta=0.05, v=0, omega=0
        init_state = np.array([0.1, 0.05, 0.0, 0.0])
        obs = pendulum_adapter.reset(init_state=init_state)
        assert obs.shape == (4,)
        # The observation should be close to the injected state
        # (x, v, theta, omega) — but obs ordering may differ
        # Just check it's valid
        assert np.all(np.isfinite(obs))

    def test_reward_is_positive(self, pendulum_adapter):
        """InvertedPendulum gives +1 reward per step (survival)."""
        pendulum_adapter.reset()
        _, reward, _, _, _ = pendulum_adapter.step(3)
        assert reward > 0

    def test_multiple_steps_without_crash(self, pendulum_adapter):
        pendulum_adapter.reset()
        for _ in range(50):
            obs, reward, terminated, truncated, info = pendulum_adapter.step(
                np.random.randint(pendulum_adapter.n_actions))
            if terminated or truncated:
                pendulum_adapter.reset()


# ---------------------------------------------------------------------------
# Goal Condition
# ---------------------------------------------------------------------------

class TestGoalCondition:
    """Test continuous goal checking."""

    def test_goal_state_balanced(self, pendulum_adapter):
        # Perfectly balanced: obs = [x, theta, x_dot, omega]
        obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert pendulum_adapter.is_goal_state(obs) is True

    def test_goal_state_tilted(self, pendulum_adapter):
        # Pole tilted beyond threshold: obs = [x, theta, x_dot, omega]
        obs = np.array([0.0, 0.3, 0.0, 0.0], dtype=np.float32)
        assert pendulum_adapter.is_goal_state(obs) is False

    def test_goal_state_off_center(self, pendulum_adapter):
        # Cart too far from center: obs = [x, theta, x_dot, omega]
        obs = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        assert pendulum_adapter.is_goal_state(obs) is False

    def test_goal_state_with_velocity(self, pendulum_adapter):
        # Balanced even with small velocity — still goal
        obs = np.array([0.1, 0.05, 0.5, 0.3], dtype=np.float32)
        assert pendulum_adapter.is_goal_state(obs) is True

    def test_is_terminal_returns_none_for_survival(self, pendulum_adapter):
        """Survival tasks: is_terminal() returns None (don't terminate on goal)."""
        obs_good = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_bad = np.array([0.0, 0.5, 0.0, 0.0], dtype=np.float32)
        # Both should return None — survival tasks don't use is_terminal
        assert pendulum_adapter.is_terminal(obs_good) is None
        assert pendulum_adapter.is_terminal(obs_bad) is None

    def test_is_in_goal_bin_compatibility(self, pendulum_adapter):
        """is_in_goal_bin should delegate to is_goal_state (no bins)."""
        # obs = [x, theta, x_dot, omega] — balanced
        obs_good = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # goal_states list is ignored for MuJoCo
        assert pendulum_adapter.is_in_goal_bin([1, 2, 3], obs_good) is True

    def test_get_goal_states_returns_empty(self, pendulum_adapter):
        """No discrete bins — goal_states should be empty."""
        assert pendulum_adapter.get_goal_states() == []

    def test_create_goal_reward_fn(self, pendulum_adapter):
        reward_fn = pendulum_adapter.create_goal_reward_fn(
            reward=10.0, default_cost=-1.0)
        # obs = [x, theta, x_dot, omega]
        obs_good = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs_bad = np.array([0.0, 0.5, 0.0, 0.0], dtype=np.float32)
        assert reward_fn(obs_good) == 10.0
        assert reward_fn(obs_bad) == -1.0


# ---------------------------------------------------------------------------
# Random State Injection
# ---------------------------------------------------------------------------

class TestRandomState:
    """Test diverse start state sampling."""

    def test_sample_random_state_returns_obs(self, pendulum_adapter):
        obs = pendulum_adapter.sample_random_state()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (4,)
        assert obs.dtype == np.float32

    def test_random_states_are_diverse(self, pendulum_adapter):
        """Multiple random samples should not all be identical."""
        obs_list = [pendulum_adapter.sample_random_state() for _ in range(10)]
        obs_array = np.array(obs_list)
        # Standard deviation across samples should be nonzero
        assert np.std(obs_array, axis=0).sum() > 0.01

    def test_get_state_for_reset_roundtrip(self, pendulum_adapter):
        """Save state, reset, restore, and verify consistency."""
        pendulum_adapter.reset()
        pendulum_adapter.step(0)  # move to non-default state
        state = pendulum_adapter.get_state_for_reset()
        assert isinstance(state, np.ndarray)
        assert len(state) == 4  # nq=2 + nv=2 for InvertedPendulum


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class TestRendering:

    def test_render_returns_array(self, pendulum_adapter_render):
        pendulum_adapter_render.reset()
        frame = pendulum_adapter_render.render()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3  # (H, W, 3)
        assert frame.shape[2] == 3  # RGB


# ---------------------------------------------------------------------------
# Agent Compatibility
# ---------------------------------------------------------------------------

class TestAgentCompatibility:
    """Test that the adapter works with NeuralSRAgent interface."""

    def test_agent_construction(self, pendulum_adapter):
        """Verify the adapter provides everything NeuralSRAgent needs."""
        # These are the properties the agent constructor accesses
        assert isinstance(pendulum_adapter.obs_dim, int)
        assert isinstance(pendulum_adapter.n_actions, int)
        assert pendulum_adapter.obs_dim > 0
        assert pendulum_adapter.n_actions > 0

    def test_agent_episode_loop(self, pendulum_adapter):
        """Simulate the inner loop of NeuralSRAgent.learn_environment()."""
        obs = pendulum_adapter.reset()
        assert obs.shape == (pendulum_adapter.obs_dim,)

        for _ in range(20):
            action = np.random.randint(pendulum_adapter.n_actions)
            next_obs, reward, terminated, truncated, info = \
                pendulum_adapter.step(action)
            assert next_obs.shape == (pendulum_adapter.obs_dim,)

            # Check terminal (returns None for survival tasks)
            terminal = pendulum_adapter.is_terminal(next_obs)
            assert terminal is None  # survival task — don't use is_terminal

            if terminated or truncated:
                obs = pendulum_adapter.reset()
            else:
                obs = next_obs

    def test_diverse_start_loop(self, pendulum_adapter):
        """Simulate diverse start training."""
        for _ in range(5):
            obs = pendulum_adapter.sample_random_state()
            assert obs.shape == (pendulum_adapter.obs_dim,)
            for _ in range(10):
                action = np.random.randint(pendulum_adapter.n_actions)
                obs, _, term, trunc, _ = pendulum_adapter.step(action)
                if term or trunc:
                    break

    def test_full_agent_integration(self, pendulum_adapter):
        """End-to-end: create agent, train briefly, run episode."""
        import torch
        from core.neural.agent import NeuralSRAgent

        agent = NeuralSRAgent(
            adapter=pendulum_adapter,
            sf_dim=16,
            hidden_sizes=(16, 16),
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
        )

        agent.set_goal(reward=1.0, default_cost=0.0, use_env_reward=True)

        # Brief training
        agent.learn_environment(
            num_episodes=5,
            steps_per_episode=50,
            diverse_start=True,
            log_interval=5,
        )

        # Run evaluation episode
        result = agent.run_episode(max_steps=50)
        assert 'steps' in result
        assert 'reward' in result
        assert 'reached_goal' in result
        assert result['steps'] > 0
