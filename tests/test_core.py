"""Core test suite for state spaces, agent validation, eval utilities, and helpers.

Covers:
1. State space roundtrip tests (state_to_index / index_to_state)
2. One-hot roundtrip tests (index_to_onehot / onehot_to_index)
3. Input validation for HierarchicalSRAgent and QLearningAgent
4. eval_utils metric and plotting functions
5. BinnedContinuousAdapter clamp helper
"""

import os
import math
import tempfile

import numpy as np
import pytest

from core.state_space import (
    SimpleStateSpace,
    GridStateSpace,
    BinnedContinuousStateSpace,
    AugmentedStateSpace,
)
from core.eval_utils import (
    relative_stability,
    compute_stability_array,
    plot_reward_curves,
    plot_stability_bars,
)
from environments.binned_continuous_adapter import clamp
from core.hierarchical_agent import HierarchicalSRAgent
from core.q_learning import QLearningAgent


# ---------------------------------------------------------------------------
# Fixtures: state space instances
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_ss():
    return SimpleStateSpace(n_states=25)


@pytest.fixture
def grid_ss():
    return GridStateSpace(grid_size=5)


@pytest.fixture
def binned_2d_ss():
    return BinnedContinuousStateSpace(n_bins_per_dim=(10, 10))


@pytest.fixture
def binned_4d_ss():
    """4-dimensional binned space (like Acrobot)."""
    return BinnedContinuousStateSpace(n_bins_per_dim=(6, 6, 5, 5))


@pytest.fixture
def augmented_ss():
    return AugmentedStateSpace(base_n_states=25, n_augment=2)


# ===================================================================
# 1. State space roundtrip tests  (state_to_index(index_to_state(i)) == i)
# ===================================================================

class TestStateSpaceRoundtrip:
    """Verify state_to_index(index_to_state(i)) == i for every valid index."""

    def test_simple(self, simple_ss):
        for i in range(simple_ss.n_states):
            state = simple_ss.index_to_state(i)
            assert simple_ss.state_to_index(state) == i

    def test_grid(self, grid_ss):
        for i in range(grid_ss.n_states):
            state = grid_ss.index_to_state(i)
            assert grid_ss.state_to_index(state) == i

    def test_binned_2d(self, binned_2d_ss):
        for i in range(binned_2d_ss.n_states):
            state = binned_2d_ss.index_to_state(i)
            assert binned_2d_ss.state_to_index(state) == i

    def test_binned_4d(self, binned_4d_ss):
        for i in range(binned_4d_ss.n_states):
            state = binned_4d_ss.index_to_state(i)
            assert binned_4d_ss.state_to_index(state) == i

    def test_augmented(self, augmented_ss):
        for i in range(augmented_ss.n_states):
            state = augmented_ss.index_to_state(i)
            assert augmented_ss.state_to_index(state) == i


# ===================================================================
# 2. One-hot roundtrip tests  (onehot_to_index(index_to_onehot(i)) == i)
# ===================================================================

class TestOnehotRoundtrip:
    """Verify onehot_to_index(index_to_onehot(i)) == i for every valid index."""

    def test_simple(self, simple_ss):
        for i in range(simple_ss.n_states):
            onehot = simple_ss.index_to_onehot(i)
            assert simple_ss.onehot_to_index(onehot) == i

    def test_grid(self, grid_ss):
        for i in range(grid_ss.n_states):
            onehot = grid_ss.index_to_onehot(i)
            assert grid_ss.onehot_to_index(onehot) == i

    def test_binned_2d(self, binned_2d_ss):
        for i in range(binned_2d_ss.n_states):
            onehot = binned_2d_ss.index_to_onehot(i)
            assert binned_2d_ss.onehot_to_index(onehot) == i

    def test_binned_4d(self, binned_4d_ss):
        for i in range(binned_4d_ss.n_states):
            onehot = binned_4d_ss.index_to_onehot(i)
            assert binned_4d_ss.onehot_to_index(onehot) == i

    def test_augmented(self, augmented_ss):
        for i in range(augmented_ss.n_states):
            onehot = augmented_ss.index_to_onehot(i)
            assert augmented_ss.onehot_to_index(onehot) == i


# ===================================================================
# 3. Input validation tests
# ===================================================================

class MockAdapter:
    """Minimal adapter that satisfies the constructor just enough for
    validation to fire before any real adapter usage occurs.
    """
    pass


class TestHierarchicalSRAgentValidation:
    """HierarchicalSRAgent must raise ValueError for invalid hyperparameters."""

    def test_n_clusters_too_small(self):
        with pytest.raises(ValueError, match="n_clusters"):
            HierarchicalSRAgent(adapter=MockAdapter(), n_clusters=1)

    def test_gamma_zero(self):
        with pytest.raises(ValueError, match="gamma"):
            HierarchicalSRAgent(adapter=MockAdapter(), gamma=0)

    def test_gamma_above_one(self):
        with pytest.raises(ValueError, match="gamma"):
            HierarchicalSRAgent(adapter=MockAdapter(), gamma=1.5)

    def test_learning_rate_negative(self):
        with pytest.raises(ValueError, match="learning_rate"):
            HierarchicalSRAgent(adapter=MockAdapter(), learning_rate=-0.1)

    def test_learning_rate_too_large(self):
        with pytest.raises(ValueError, match="learning_rate"):
            HierarchicalSRAgent(adapter=MockAdapter(), learning_rate=2.0)

    def test_test_smooth_steps_zero(self):
        with pytest.raises(ValueError, match="test_smooth_steps"):
            HierarchicalSRAgent(adapter=MockAdapter(), test_smooth_steps=0)


class TestQLearningAgentValidation:
    """QLearningAgent must raise ValueError for invalid hyperparameters."""

    def _make(self, **overrides):
        """Helper to build a QLearningAgent with overridden kwargs."""
        defaults = dict(
            adapter=MockAdapter(),
            goal_states=[],
            C=np.zeros(1),
        )
        defaults.update(overrides)
        return QLearningAgent(**defaults)

    def test_gamma_zero(self):
        with pytest.raises(ValueError, match="gamma"):
            self._make(gamma=0)

    def test_gamma_above_one(self):
        with pytest.raises(ValueError, match="gamma"):
            self._make(gamma=2.0)

    def test_alpha_negative(self):
        with pytest.raises(ValueError, match="alpha"):
            self._make(alpha=-0.1)

    def test_epsilon_start_above_one(self):
        with pytest.raises(ValueError, match="epsilon_start"):
            self._make(epsilon_start=2.0)


# ===================================================================
# 4. eval_utils tests
# ===================================================================

class TestRelativeStability:
    """Tests for relative_stability."""

    def test_returns_float(self):
        result = relative_stability([1, 2, 3, 4, 5])
        assert isinstance(result, float)

    def test_empty_returns_nan(self):
        result = relative_stability([])
        assert math.isnan(result)


class TestComputeStabilityArray:
    """Tests for compute_stability_array."""

    def test_constant_returns_zero_stability(self):
        data = np.ones((3, 5))
        stab = compute_stability_array(data)
        assert stab.shape == (3,)
        np.testing.assert_allclose(stab, 0.0, atol=1e-7)


class TestPlotRewardCurves:
    """plot_reward_curves should run without error on valid inputs."""

    def test_no_crash(self):
        from collections import OrderedDict

        eps_range = np.arange(5)
        data_dict = OrderedDict({
            "Hierarchy": np.random.randn(3, 5),
            "Flat": np.random.randn(3, 5),
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "reward_curves.png")
            plot_reward_curves(eps_range, data_dict, save_path)
            assert os.path.isfile(save_path)


class TestPlotStabilityBars:
    """plot_stability_bars should run without error on valid inputs."""

    def test_no_crash(self):
        data_dict = {
            "Hierarchy": np.array([0.1, 0.2, 0.15]),
            "Flat": np.array([0.3, 0.25, 0.35]),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "stability_bars.png")
            plot_stability_bars(data_dict, save_path)
            assert os.path.isfile(save_path)


# ===================================================================
# 5. BinnedContinuousAdapter clamp test
# ===================================================================

class TestClamp:
    """Tests for the clamp helper function."""

    def test_within_range(self):
        assert clamp(5, 0, 10) == 5

    def test_below_minimum(self):
        assert clamp(-1, 0, 10) == 0

    def test_above_maximum(self):
        assert clamp(15, 0, 10) == 10
