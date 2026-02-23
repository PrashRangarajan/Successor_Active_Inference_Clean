"""End-to-end visualization tests for the Key Gridworld environment.

Verifies that all visualization paths work correctly with the augmented
(location, has_key) state space, including:
- Split value function panels (Without key / With key)
- Augmented cluster maps (1×2 layout)
- Spectral embedding with covariance ellipses
- Composite 2×2 figure
- Matrix visualizations (B, M, M-from-origin with 2×2 augmented panels)
"""

import os
import tempfile

import numpy as np
import pytest

from core.hierarchical_agent import HierarchicalSRAgent
from environments.key_gridworld import KeyGridworldAdapter
from unified_env import KeyGridworld


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_agent():
    """Build a small key gridworld, learn, cluster, and return (agent, locs).

    Uses a 5×5 grid with a short training run — enough for all viz paths
    to execute without errors, not meant for performance evaluation.
    """
    grid_size = 5
    key_loc = (0, 3)
    init_loc = (0, 0)
    goal_loc = (grid_size - 1, grid_size - 1)
    walls = [(1, x) for x in range(grid_size // 2 + 1)]

    env = KeyGridworld(grid_size, key_loc=key_loc, pickup=False)
    env.set_walls(walls)
    adapter = KeyGridworldAdapter(env, grid_size, has_pickup_action=False)

    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=4,
        gamma=0.95,
        learning_rate=0.1,
        learn_from_experience=True,
    )

    goal_spec = (goal_loc[0], goal_loc[1], 1)
    agent.set_goal(goal_spec, reward=100.0)
    agent.learn_environment(num_episodes=200)

    # Run one episode so trajectory data is available
    agent.reset_episode(init_state=init_loc + (0,))
    agent.run_episode_flat(max_steps=50)

    return agent, dict(init_loc=init_loc, goal_loc=goal_loc, key_loc=key_loc)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestValueFunctionViz:
    """plot_value_function with augmented state space."""

    def test_produces_file(self, trained_agent):
        agent, _ = trained_agent
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "value_function.png")
            agent.plot_value_function(save_path=path)
            assert os.path.isfile(path), "Value function figure not created"
            assert os.path.getsize(path) > 0, "Value function figure is empty"


class TestClusterViz:
    """visualize_clusters with augmented state space."""

    def test_cluster_map_produced(self, trained_agent):
        agent, _ = trained_agent
        with tempfile.TemporaryDirectory() as tmpdir:
            agent.visualize_clusters(save_dir=tmpdir)
            macro_path = os.path.join(tmpdir, "Macro_s.png")
            assert os.path.isfile(macro_path), "Cluster map not created"
            assert os.path.getsize(macro_path) > 0

    def test_spectral_embedding_produced(self, trained_agent):
        agent, _ = trained_agent
        with tempfile.TemporaryDirectory() as tmpdir:
            agent.visualize_clusters(save_dir=tmpdir)
            scatter_path = os.path.join(tmpdir, "macro_state_viz.png")
            assert os.path.isfile(scatter_path), "Spectral embedding not created"

    def test_spectral_ellipses_produced(self, trained_agent):
        agent, _ = trained_agent
        with tempfile.TemporaryDirectory() as tmpdir:
            agent.visualize_clusters(save_dir=tmpdir)
            ellipse_path = os.path.join(tmpdir, "macro_state_viz_ellipses.png")
            assert os.path.isfile(ellipse_path), "Ellipse embedding not created"


class TestMatrixViz:
    """view_matrices with augmented state space."""

    def test_matrix_figures_produced(self, trained_agent):
        agent, locs = trained_agent
        origin_idx = locs["init_loc"][0] * 5 + locs["init_loc"][1]
        with tempfile.TemporaryDirectory() as tmpdir:
            agent.view_matrices(save_dir=tmpdir, learned=True,
                                origin_state=origin_idx)
            # B matrix
            assert os.path.isfile(os.path.join(tmpdir, "B_matrix_micro.png"))
            # M matrix
            assert os.path.isfile(os.path.join(tmpdir, "m_estimated_micro.png"))
            # M from origin (augmented: 2×2 panel)
            assert os.path.isfile(os.path.join(tmpdir, "m_origin_estimated.png"))
            # Macro matrices
            assert os.path.isfile(os.path.join(tmpdir, "B_matrix_macro.png"))
            assert os.path.isfile(os.path.join(tmpdir, "m_estimated_macro.png"))


class TestCompositeFigure:
    """visualize_key_gridworld_composite produces a 2×2 composite."""

    def test_composite_produced(self, trained_agent):
        agent, locs = trained_agent
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "composite.png")
            agent.visualize_key_gridworld_composite(
                save_path=path,
                init_loc=locs["init_loc"],
                goal_loc=locs["goal_loc"],
                key_loc=locs["key_loc"],
            )
            assert os.path.isfile(path), "Composite figure not created"
            assert os.path.getsize(path) > 0
