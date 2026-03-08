"""Goal-setting mixin for HierarchicalSRAgent.

Provides set_goal(), set_shaped_goal(), and _is_at_goal() — the three
methods that manage the agent's goal state and preference vector C.

Part of the mixin decomposition of the monolithic HierarchicalSRAgent.
All attributes are initialized in HierarchicalSRAgent.__init__; this
mixin only reads/writes them through ``self``.
"""

import numpy as np
from typing import Any


class GoalSettingMixin:
    """Mixin providing goal management for HierarchicalSRAgent."""

    def set_goal(self, goal_spec: Any, reward: float = 100.0, default_cost: float = -0.1):
        """Set a sparse goal for the agent.

        Goal states become absorbing in B and receive high reward in C;
        all other states receive ``default_cost``.

        Args:
            goal_spec: Environment-specific goal specification
            reward: Reward value at goal states
            default_cost: Default cost for non-goal states
        """
        self.goal_states = self.adapter.get_goal_states(goal_spec)
        self.C = self.adapter.create_goal_prior(self.goal_states, reward, default_cost)
        self._shaped_goal = False
        self._policy_compiled = False
        print(f"Goal states: {self.goal_states}")

    def set_shaped_goal(self, C: np.ndarray, goal_threshold: float = 0.0):
        """Set a continuous shaped reward prior.

        Unlike ``set_goal``, no states are made absorbing in B.  The reward
        landscape C is used directly, and ``_is_at_goal`` returns True when
        the current state's C value exceeds ``goal_threshold``.

        Goal states (for macro-preference and stage-diagram detection) are
        inferred as the top-valued states whose C value is >= 80% of max(C).

        Args:
            C: Shaped reward vector, one entry per micro state.
            goal_threshold: C value above which the agent is considered
                            "at goal" (used by ``_is_at_goal``).
        """
        self.C = C
        self._shaped_goal = True
        self._goal_threshold = goal_threshold
        self._policy_compiled = False

        # Infer "goal states" as states whose C value meets the threshold.
        # For negative-range C (e.g. -(theta^2 + 0.1*omega^2) in [-10, 0]), the old
        # "0.8 * max(C)" heuristic fails because 0.8 * 0 ~ 0.  Using the
        # explicit goal_threshold is robust for any C range.
        self.goal_states = [i for i in range(len(C)) if C[i] >= goal_threshold]
        print(f"Shaped goal: {len(self.goal_states)} high-reward states "
              f"(C >= {goal_threshold:.1f}), threshold={goal_threshold:.1f}")

        # Recompute macro-level preference if clusters already exist
        if hasattr(self, 'macro_state_list') and self.macro_state_list is not None:
            self._compute_macro_preference()

    def _is_at_goal(self) -> bool:
        """Check if the agent has reached a goal state.

        For sparse goals, uses the discrete goal-bin check.
        For shaped goals, checks whether the current state's C value
        exceeds the goal threshold.
        """
        s_idx = self.adapter.get_current_state_index()

        if self._shaped_goal:
            return self.C[s_idx] >= self._goal_threshold

        in_goal_bin = s_idx in self.goal_states

        # If no discrete match, definitely not at goal
        if not in_goal_bin:
            return False

        # If the adapter supports continuous terminal checks, require both
        continuous_check = self.adapter.is_terminal()
        if continuous_check is not None:
            return continuous_check  # True only if continuous state is also terminal

        # Pure discrete: goal bin membership is sufficient
        return True
