"""Hierarchical planning mixin for HierarchicalSRAgent.

Provides episode execution (hierarchical and flat), action selection at
both macro and micro levels, and environment stepping utilities.

Part of the mixin decomposition of the monolithic HierarchicalSRAgent.
All attributes are initialized in HierarchicalSRAgent.__init__; this
mixin only reads/writes them through ``self``.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np


class HierarchicalPlanningMixin:
    """Mixin providing hierarchical planning for HierarchicalSRAgent."""

    def _get_planning_state(self) -> np.ndarray:
        """Get a one-hot state vector for planning from the adapter's current state.

        In MDP environments, the adapter already returns a one-hot vector.
        In POMDP environments, the adapter returns a belief distribution (spread
        over many states), which causes ``_select_micro_action`` to compute
        blurry expected values and make poor action choices.  This method
        converts the belief to a clean one-hot at the MAP estimate so the
        micro-level planner can differentiate actions properly.

        Uses the adapter's state_space.index_to_onehot() to produce the correct
        shape (e.g., (N,) for simple gridworld, (N,2) for key gridworld).
        """
        s_idx = self.adapter.get_current_state_index()
        return self.adapter.state_space.index_to_onehot(s_idx)

    def _step_with_smooth(self, action: int, smooth_steps: int) -> Tuple[int, float]:
        """Take an action with smooth stepping for continuous environments.

        Repeats ``adapter.step(action)`` up to *smooth_steps* times, breaking
        early if the discrete state changes or the episode terminates.

        Args:
            action: Action to execute.
            smooth_steps: Maximum number of physics steps to take.

        Returns:
            Tuple of (n_physics_steps, env_reward) where env_reward is the
            sum of actual environment rewards across all sub-steps.
        """
        if smooth_steps <= 1:
            # Fast path: no looping needed
            step_result = self.adapter.step_with_info(action)
            if step_result is not None:
                _, reward, _, _, _ = step_result
                return 1, reward
            else:
                self.adapter.step(action)
                return 1, 0.0

        s_before = self.adapter.get_current_state_index()
        env_reward = 0.0
        for i in range(smooth_steps):
            step_result = self.adapter.step_with_info(action)
            if step_result is not None:
                _, reward, terminated, truncated, _ = step_result
                env_reward += reward
                if terminated or truncated:
                    return i + 1, env_reward
            else:
                self.adapter.step(action)
            s_after = self.adapter.get_current_state_index()
            if s_after != s_before:
                return i + 1, env_reward
            # Stop immediately when the continuous state reaches the goal
            # (e.g. within 0.45 units for PointMaze).  Without this check
            # the ball can overshoot the goal during the remaining sub-steps.
            terminal = self.adapter.is_terminal()
            if terminal is True:
                return i + 1, env_reward
        return smooth_steps, env_reward

    def reset_episode(self, init_state: Optional[Any] = None):
        """Reset for a new episode.

        Args:
            init_state: Optional initial state
        """
        self.adapter.reset(init_state)
        self.current_state = self._get_planning_state()
        self.state_history = [self.current_state.copy()]
        self.action_history = []

    def run_episode_hierarchical(self, max_steps: int = 200) -> Dict[str, Any]:
        """Run an episode using hierarchical planning.

        For sparse goals: macro-level navigation to the goal cluster, then
        micro-level fine-tuning.  Terminates on goal arrival.

        For shaped goals: the hierarchy is used only for the initial
        approach — once the agent enters the best macro state it switches
        to flat micro-level control for the remainder of the episode,
        accumulating reward by staying in high-value states.

        If ``compile_policy()`` has been called, uses O(1) cached lookups.

        Args:
            max_steps: Maximum number of steps

        Returns:
            Dict with episode statistics
        """
        if self._policy_compiled:
            return self._run_episode_hierarchical_cached(max_steps)

        total_steps = 0
        total_reward = 0.0
        macro_decisions = 0         # k²-cost macro planning decisions
        micro_phase = False         # N²-cost micro phase computation

        s_idx = self.adapter.get_current_state_index()

        # Determine goal macro state(s)
        goal_macro_states = set()
        for gs in self.goal_states:
            if gs in self.micro_to_macro:
                goal_macro_states.add(self.micro_to_macro[gs])

        # Hierarchical planning phase: navigate to the goal macro state
        while total_steps < max_steps:
            if s_idx not in self.micro_to_macro:
                break

            s_macro = self.micro_to_macro[s_idx]

            if s_macro in goal_macro_states:
                break  # Reached goal macro state — switch to micro

            # Compute macro-level values
            V_macro = self.M_macro @ self.C_macro

            # Find best macro action
            best_macro_action = self._select_macro_action(s_macro, V_macro)

            if best_macro_action is None:
                break  # No valid macro actions

            target_macro = self.adj_list[s_macro][best_macro_action]
            macro_decisions += 1     # one k²-cost macro planning decision

            # Execute macro action (navigate to target macro state)
            steps, reward = self._execute_macro_action(s_macro, target_macro, max_steps - total_steps)
            total_steps += steps
            total_reward += reward

            s_idx = self.adapter.get_current_state_index()

            if not self._shaped_goal and self._is_at_goal():
                break

        # Micro-level phase: fine-grained control
        # For sparse goals: reach the exact goal state.
        # For shaped goals: run the full remaining episode, accumulating reward.
        if total_steps < max_steps:
            V = self.adapter.multiply_M_C(self.M, self.C)
            stop_at_goal = not self._shaped_goal
            micro_phase = True       # one N²-cost computation

            while total_steps < max_steps:
                if stop_at_goal and self._is_at_goal():
                    break

                action = self._select_micro_action(V)
                n_phys, step_reward = self._step_with_smooth(action, self.test_smooth_steps)
                total_reward += step_reward
                self.current_state = self._get_planning_state()
                self.state_history.append(self.current_state.copy())
                self.action_history.append(action)

                s_idx = self.adapter.get_current_state_index()
                total_steps += n_phys

        return {
            'steps': total_steps,
            'reward': total_reward,
            'reached_goal': self._is_at_goal(),
            'final_state': self.adapter.get_current_state(),
            'macro_decisions': macro_decisions,
            'micro_phase': micro_phase,
            'planning_steps': macro_decisions + (1 if micro_phase else 0),
        }

    def run_episode_hierarchical_reentrant(self, max_steps: int = 200) -> Dict[str, Any]:
        """Hierarchical episode with re-entrant macro control.

        Follows the same global goal policy as flat (``V = M @ C``), but
        counts each cluster boundary crossing as a macro-level planning
        decision that the hierarchical agent would need to make.

        This gives an honest macro-decision count for planning-step
        comparisons.  Not the default episode method — used for analysis
        figures only.
        """
        total_steps = 0
        total_reward = 0.0
        macro_decisions = 0
        micro_phase_used = False

        s_idx = self.adapter.get_current_state_index()

        # Determine goal macro state(s)
        goal_macro_states = set()
        for gs in self.goal_states:
            if gs in self.micro_to_macro:
                goal_macro_states.add(self.micro_to_macro[gs])

        # Compute goal-level value function once — same policy as flat.
        V_goal = self.adapter.multiply_M_C(self.M, self.C)

        prev_macro = self.micro_to_macro.get(s_idx)

        while total_steps < max_steps:
            if not self._shaped_goal and self._is_at_goal():
                break

            if s_idx not in self.micro_to_macro:
                break

            s_macro = self.micro_to_macro[s_idx]

            # Count a macro decision whenever we enter a new cluster
            if s_macro != prev_macro:
                if s_macro in goal_macro_states:
                    micro_phase_used = True
                else:
                    macro_decisions += 1
                prev_macro = s_macro

            # Always follow the global goal policy (same trajectory as flat)
            action = self._select_micro_action(V_goal)
            n_phys, step_reward = self._step_with_smooth(action, self.test_smooth_steps)
            total_reward += step_reward
            self.current_state = self._get_planning_state()
            self.state_history.append(self.current_state.copy())
            self.action_history.append(action)
            total_steps += n_phys

            s_idx = self.adapter.get_current_state_index()

        return {
            'steps': total_steps,
            'reward': total_reward,
            'reached_goal': self._is_at_goal(),
            'final_state': self.adapter.get_current_state(),
            'macro_decisions': macro_decisions,
            'micro_phase': micro_phase_used,
            'planning_steps': macro_decisions + (1 if micro_phase_used else 0),
        }

    def _select_macro_action(self, s_macro: int, V_macro: np.ndarray) -> Optional[int]:
        """Select best macro action from current macro state."""
        if s_macro not in self.adj_list:
            return None

        adj_states = self.adj_list[s_macro]
        if not adj_states:
            return None

        # Compute values for each adjacent macro state
        values = []
        for adj_macro in adj_states:
            values.append(V_macro[adj_macro])

        # Select best (that actually moves to different state)
        sorted_indices = np.argsort(values)[::-1]

        for idx in sorted_indices:
            if adj_states[idx] != s_macro:
                return idx

        return None

    def _execute_macro_action(self, init_macro: int, target_macro: int, max_steps: int) -> Tuple[int, float]:
        """Execute a macro action by navigating to target macro state.

        Args:
            init_macro: Starting macro state
            target_macro: Target macro state
            max_steps: Maximum steps allowed

        Returns:
            Tuple of (steps_taken, total_reward)
        """
        bottleneck = self.bottleneck_states.get((init_macro, target_macro), [])
        if not bottleneck:
            # Fallback: use all micro states in the target cluster as goal
            bottleneck = self.macro_state_list[target_macro]
            if not bottleneck:
                return 0, 0.0

        # Create temporary goal at bottleneck states
        C_temp = self.adapter.create_goal_prior(bottleneck, reward=10.0, default_cost=0.0)
        V = self.adapter.multiply_M_C(self.M, C_temp)

        steps = 0
        env_reward = 0.0
        s_idx = self.adapter.get_current_state_index()

        while steps < max_steps:
            if s_idx in bottleneck or self._is_at_goal():
                break

            action = self._select_micro_action(V)
            n_phys, step_reward = self._step_with_smooth(action, self.test_smooth_steps)
            env_reward += step_reward
            self.current_state = self._get_planning_state()
            self.state_history.append(self.current_state.copy())
            self.action_history.append(action)

            s_idx = self.adapter.get_current_state_index()
            steps += n_phys

            # Check if we've changed macro state
            if s_idx in self.micro_to_macro:
                current_macro = self.micro_to_macro[s_idx]
                if current_macro == target_macro:
                    break
                # Entered an unexpected cluster — replan at macro level
                if current_macro != init_macro:
                    break

        return steps, env_reward

    def _run_micro_to_goal(self, max_steps: int) -> Tuple[int, float]:
        """Run micro-level policy to reach goal.

        Args:
            max_steps: Maximum steps allowed

        Returns:
            Tuple of (steps_taken, total_reward)
        """
        V = self.adapter.multiply_M_C(self.M, self.C)

        steps = 0
        reward = 0.0
        s_idx = self.adapter.get_current_state_index()

        while steps < max_steps and not self._is_at_goal():
            action = self._select_micro_action(V)
            n_phys, step_reward = self._step_with_smooth(action, self.test_smooth_steps)
            reward += step_reward
            self.current_state = self._get_planning_state()
            self.state_history.append(self.current_state.copy())
            self.action_history.append(action)

            s_idx = self.adapter.get_current_state_index()
            steps += n_phys

        return steps, reward

    def _select_micro_action(self, V: np.ndarray) -> int:
        """Select best micro action based on expected value.

        For each action, computes the expected next-state value under the
        learned transition distribution: E[V(s')] = B(:,s,a) . V.

        This properly handles stochastic transitions (e.g. Acrobot) where
        the argmax of the transition distribution might be the same state
        for all actions, masking real value differences.

        When the agent is already at a goal state, the best-value action is
        returned directly (staying in place is desirable for maintenance).

        Args:
            V: Value function (flat array)

        Returns:
            Best action index
        """
        s_onehot = self.current_state

        # Compute expected value for each action
        values = []
        for action in range(self.adapter.n_actions):
            s_next_dist = self.adapter.multiply_B_s(self.B, s_onehot, action)
            # Flatten for augmented state spaces (e.g., key gridworld: shape (N,2))
            s_flat = s_next_dist.flatten('F') if s_next_dist.ndim > 1 else s_next_dist
            expected_value = float(s_flat @ V)
            values.append(expected_value)

        sorted_actions = np.argsort(values)[::-1]

        # At the goal, staying in place is the right thing to do —
        # skip the "must change state" filter and return highest-value action.
        if self._is_at_goal():
            return sorted_actions[0]

        # Away from goal: prefer actions that actually change expected state
        for action in sorted_actions:
            s_next = self.adapter.multiply_B_s(self.B, s_onehot, action)
            if not np.allclose(s_next, s_onehot):
                return action

        # If all actions keep us in same state, return best anyway
        return sorted_actions[0]

    def run_episode_flat(self, max_steps: int = 200) -> Dict[str, Any]:
        """Run an episode using only micro-level (flat) planning.

        For sparse goals the episode terminates when the goal is reached.
        For shaped goals the episode runs for the full ``max_steps`` so
        the agent accumulates reward by staying near the optimum.

        If ``compile_policy()`` has been called, uses O(1) cached lookups.

        Args:
            max_steps: Maximum number of steps

        Returns:
            Dict with episode statistics
        """
        if self._policy_compiled:
            return self._run_episode_flat_cached(max_steps)

        V = self.adapter.multiply_M_C(self.M, self.C)

        steps = 0
        reward = 0.0
        s_idx = self.adapter.get_current_state_index()
        stop_at_goal = not self._shaped_goal

        while steps < max_steps:
            if stop_at_goal and self._is_at_goal():
                break

            action = self._select_micro_action(V)
            n_phys, step_reward = self._step_with_smooth(action, self.test_smooth_steps)
            reward += step_reward
            self.current_state = self._get_planning_state()
            self.state_history.append(self.current_state.copy())
            self.action_history.append(action)

            s_idx = self.adapter.get_current_state_index()
            steps += n_phys

        return {
            'steps': steps,
            'reward': reward,
            'reached_goal': self._is_at_goal(),
            'final_state': self.adapter.get_current_state(),
            'macro_decisions': 0,     # flat has no macro decisions
            'micro_phase': True,      # flat is entirely micro
            'planning_steps': steps,  # flat plans every step (each N²)
        }
