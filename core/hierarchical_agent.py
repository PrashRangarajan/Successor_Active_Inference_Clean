"""Unified Hierarchical Successor Representation Agent.

This module provides a generic hierarchical SR agent that works with any environment
through the adapter interface. The core algorithm (SR learning, clustering,
hierarchical planning) is environment-agnostic.

The agent's functionality is split across mixin modules for maintainability:
- goal_setting:          set_goal, set_shaped_goal, _is_at_goal
- sr_learning:           _learn_sr_from_experience, TD updates, replay
- macro_clustering:      spectral clustering, adjacency, macro matrices
- hierarchical_planning: episode execution (hierarchical & flat), action selection
- policy_compilation:    compile_policy, cached execution, save/load
- visualization:         plotting (VisualizationMixin)
"""

from collections import deque
from typing import Optional

import numpy as np

from .base_environment import BaseEnvironmentAdapter
from .goal_setting import GoalSettingMixin
from .sr_learning import SRLearningMixin
from .macro_clustering import MacroClusteringMixin
from .hierarchical_planning import HierarchicalPlanningMixin
from .policy_compilation import PolicyCompilationMixin
from .visualization import VisualizationMixin


class HierarchicalSRAgent(
    GoalSettingMixin,
    SRLearningMixin,
    MacroClusteringMixin,
    HierarchicalPlanningMixin,
    PolicyCompilationMixin,
    VisualizationMixin,
):
    """Unified Hierarchical Successor Representation Agent.

    This agent implements:
    1. Successor Representation learning (micro-level)
    2. Spectral clustering for macro state discovery
    3. Adjacency learning between macro states
    4. Hierarchical planning (macro-level then micro-level)

    Works with any environment through the BaseEnvironmentAdapter interface.
    """

    def __init__(
        self,
        adapter: BaseEnvironmentAdapter,
        n_clusters: int = 4,
        gamma: float = 0.99,
        learning_rate: float = 0.05,
        learn_from_experience: bool = True,
        use_replay: bool = True,
        n_replay_epochs: int = 10,
        replay_buffer_size: int = 50000,
        replay_mode: str = 'sequential',
        train_smooth_steps: Optional[int] = None,
        test_smooth_steps: int = 1,
    ):
        """
        Args:
            adapter: Environment adapter implementing BaseEnvironmentAdapter
            n_clusters: Number of macro states for clustering
            gamma: Discount factor for SR
            learning_rate: TD learning rate for SR updates
            learn_from_experience: If True, learn B and M from experience.
                                   If False, use analytical computation.
            use_replay: If True, learn M via TD with hippocampal-style
                        experience replay (bioplausible). If False, compute
                        M analytically from B (fast but not bioplausible).
                        Only applies when learn_from_experience=True.
            n_replay_epochs: Number of replay passes over stored experiences.
            replay_buffer_size: Maximum transitions to store for replay.
            replay_mode: 'sequential' (bioplausible, preserves temporal order)
                         or 'shuffle' (randomize episode order each epoch).
            train_smooth_steps: Number of physics steps per action during
                learning.  ``None`` = auto-detect (10 for continuous, 1
                for discrete).
            test_smooth_steps: Number of physics steps per action during
                test-time episode execution.  Defaults to 1 (single step).
        """
        # --- Input validation ---
        if n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")
        if not (0 < gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if not (0 < learning_rate <= 1):
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
        if use_replay and n_replay_epochs < 1:
            raise ValueError(f"n_replay_epochs must be >= 1 when use_replay is True, got {n_replay_epochs}")
        if test_smooth_steps < 1:
            raise ValueError(f"test_smooth_steps must be >= 1, got {test_smooth_steps}")

        self.adapter = adapter
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.learn_from_experience = learn_from_experience
        self.train_smooth_steps = train_smooth_steps
        self.test_smooth_steps = test_smooth_steps

        # Matrices
        self.B = None  # Transition matrix
        self.M = None  # Successor matrix (micro)
        self.B_macro = None  # Macro transition matrix
        self.M_macro = None  # Macro successor matrix

        # Goal/preference
        self.C = None  # Micro-level preference
        self.C_macro = None  # Macro-level preference
        self.goal_states = []
        self._shaped_goal = False  # True when using continuous shaped reward

        # Clustering results
        self.macro_state_list = None  # List of lists: micro states in each macro state
        self.micro_to_macro = None  # Dict: micro state -> macro state
        self.n_macro_actions = None

        # Adjacency
        self.adj_list = None  # Dict: macro state -> list of adjacent macro states
        self.bottleneck_states = None  # Dict: (macro, macro) -> list of entry micro states

        # Episode tracking
        self.current_state = None
        self.state_history = []
        self.action_history = []

        # Experience replay (hippocampal replay)
        self.use_replay = use_replay
        self.n_replay_epochs = n_replay_epochs
        self.replay_buffer_size = replay_buffer_size
        self.replay_mode = replay_mode
        self.replay_buffer = deque()  # deque of episodes; each episode is a list of (s, a, s', r, done)
        self._replay_buffer_total = 0

        # Policy caching
        self._policy_compiled = False
        self._goal_policy = None
        self._bottleneck_policies = None
        self._macro_policy = None

    # ==================== Core Learning ====================

    def learn_environment(self, num_episodes: int = 1000, flat_only: bool = False):
        """Main learning routine: learn SR, cluster, learn adjacency.

        Args:
            num_episodes: Number of episodes for learning
            flat_only: If True, skip adjacency/macro learning and dedicate ALL
                episodes to SR learning.  Matches legacy flat agent behavior
                where no hierarchy overhead exists.
        """
        print("Learning environment dynamics...")

        # Resolve effective train smooth steps once for the whole learning phase
        if self.train_smooth_steps is not None:
            self._effective_train_smooth = self.train_smooth_steps
        else:
            # Auto-detect: 10 for continuous environments, 1 for discrete
            self._effective_train_smooth = 10 if self.adapter.is_continuous else 1

        # Set learning mode on adapter (for POMDP adapters that support it)
        self.adapter.set_learning_mode(True)

        if flat_only:
            # Flat mode: ALL episodes go to SR learning (no adjacency overhead).
            sr_episodes = num_episodes
            adj_episodes = 0
        else:
            # Hierarchy mode: proportional split matching legacy hierarchy.py.
            adj_episodes = max(num_episodes // 5,
                               num_episodes // self.adapter.n_actions)
            sr_episodes = num_episodes - adj_episodes

        # Learn or compute transition and successor matrices
        # Non-absorbing goal: avoids M(goal,goal) -> 1/(1-gamma) spike that
        # drowns the value gradient.  Policy only needs relative V ordering,
        # which is preserved without the absorbing self-loop.
        if self.learn_from_experience:
            self.B, self.M = self._learn_sr_from_experience(sr_episodes,
                                                             goal_states=None)
        else:
            self.B = self.adapter.get_transition_matrix()
            self.M = self.adapter.compute_successor_from_transition(self.B, self.gamma)

        if flat_only:
            # Skip macro-level learning entirely — flat agent only needs M
            print("Flat-only mode: skipping macro clustering/adjacency")
        else:
            print("Learning macro state clusters...")
            self.macro_state_list, self.micro_to_macro = self._learn_macro_clusters()

            # Create macro-level preference from micro-level
            self._compute_macro_preference()

            print("Learning macro state adjacency...")
            self.adj_list, self.bottleneck_states = self._learn_adjacency(adj_episodes)
            print(f"Adjacency list: {self.adj_list}")

            # Compute macro-level transition and successor matrices
            self.B_macro, self.M_macro = self._compute_macro_matrices()

        # Exit learning mode
        self.adapter.set_learning_mode(False)

    def learn_environment_incremental(self, delta_episodes: int, flat_only: bool = False):
        """Incremental learning: add more episodes of experience to existing B/M.

        Unlike ``learn_environment()`` which starts fresh, this method builds on
        the existing B and M matrices — matching the legacy ``learn_env_likelikood``
        behavior where the same agent is trained with delta episodes at each
        checkpoint.

        The interaction between partial M, re-clustering, and re-adjacency at each
        checkpoint is what produces the hierarchy vs flat divergence: hierarchy can
        exploit a partially-learned M better than flat navigation.

        Args:
            delta_episodes: Number of *additional* episodes to train
            flat_only: If True, skip adjacency/macro learning and dedicate ALL
                episodes to SR learning.  Matches legacy flat agent behavior.
        """
        print(f"Incremental learning: {delta_episodes} more episodes...")

        # Resolve effective train smooth steps
        if self.train_smooth_steps is not None:
            self._effective_train_smooth = self.train_smooth_steps
        else:
            self._effective_train_smooth = 10 if self.adapter.is_continuous else 1

        self.adapter.set_learning_mode(True)

        if flat_only:
            sr_episodes = delta_episodes
            adj_episodes = 0
        else:
            adj_episodes = max(delta_episodes // 5,
                               delta_episodes // self.adapter.n_actions)
            sr_episodes = delta_episodes - adj_episodes

        # Learn SR incrementally (reuses existing B, M) — non-absorbing
        if self.learn_from_experience:
            self.B, self.M = self._learn_sr_from_experience(
                sr_episodes, goal_states=None, incremental=True)
        else:
            self.B = self.adapter.get_transition_matrix()
            self.M = self.adapter.compute_successor_from_transition(self.B, self.gamma)

        if flat_only:
            print("Flat-only mode: skipping macro re-clustering/adjacency")
        else:
            # Re-cluster on the updated M
            print("Re-clustering macro states...")
            self.macro_state_list, self.micro_to_macro = self._learn_macro_clusters()
            self._compute_macro_preference()

            # Re-learn adjacency
            print("Re-learning adjacency...")
            self.adj_list, self.bottleneck_states = self._learn_adjacency(adj_episodes)
            print(f"Adjacency list: {self.adj_list}")

            # Recompute macro matrices
            self.B_macro, self.M_macro = self._compute_macro_matrices()

        self.adapter.set_learning_mode(False)
