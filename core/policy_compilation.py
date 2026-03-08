"""Policy compilation mixin for HierarchicalSRAgent.

Provides compile_policy() for precomputing state->action lookup tables,
cached episode execution, and policy save/load for deployment.

Part of the mixin decomposition of the monolithic HierarchicalSRAgent.
All attributes are initialized in HierarchicalSRAgent.__init__; this
mixin only reads/writes them through ``self``.
"""

import json
from typing import Any, Dict

import numpy as np


class PolicyCompilationMixin:
    """Mixin providing policy compilation for HierarchicalSRAgent."""

    def compile_policy(self):
        """Precompute state->action lookup tables for O(1) inference.

        After calling this, ``run_episode_hierarchical`` and ``run_episode_flat``
        will use cached dictionary lookups instead of per-step matrix
        multiplications.  Call again (or ``set_goal``) to invalidate.

        Requires that ``learn_environment()`` has been called first.
        """
        if self.B is None or self.M is None:
            raise RuntimeError("Must call learn_environment() before compile_policy()")

        print("Compiling policy tables...")

        # 1. Goal policy: best action toward final goal for every state
        V_goal = self.adapter.multiply_M_C(self.M, self.C)
        self._goal_policy = self._compute_policy_table(V_goal)

        # 2. Bottleneck policies: best action to reach each bottleneck set
        self._bottleneck_policies = {}
        if self.bottleneck_states:
            for (src, tgt), bottleneck in self.bottleneck_states.items():
                C_temp = self.adapter.create_goal_prior(
                    bottleneck, reward=10.0, default_cost=0.0
                )
                V_bn = self.adapter.multiply_M_C(self.M, C_temp)
                self._bottleneck_policies[(src, tgt)] = self._compute_policy_table(V_bn)

        # 2b. Fallback policies for macro transitions without observed bottlenecks
        for s_macro in range(self.n_clusters):
            if s_macro not in self.adj_list:
                continue
            for target_macro in self.adj_list[s_macro]:
                if (s_macro, target_macro) not in self._bottleneck_policies:
                    target_states = self.macro_state_list[target_macro]
                    if target_states:
                        C_temp = self.adapter.create_goal_prior(
                            target_states, reward=10.0, default_cost=0.0
                        )
                        V_bn = self.adapter.multiply_M_C(self.M, C_temp)
                        self._bottleneck_policies[(s_macro, target_macro)] = \
                            self._compute_policy_table(V_bn)

        # 3. Macro policy: best macro action for each macro state
        V_macro = self.M_macro @ self.C_macro
        self._macro_policy = {}
        for s_macro in range(self.n_clusters):
            best = self._select_macro_action(s_macro, V_macro)
            self._macro_policy[s_macro] = best  # int or None

        self._policy_compiled = True
        print(
            f"Policy compiled: {len(self._goal_policy)} micro states, "
            f"{len(self._bottleneck_policies)} bottleneck policies, "
            f"{len(self._macro_policy)} macro states"
        )

    def _compute_policy_table(self, V: np.ndarray) -> dict:
        """Precompute best action for every state given value function *V*.

        Uses the same expected-value + state-change tie-breaking logic as
        ``_select_micro_action``.

        Args:
            V: Value function (flat 1D or multi-dim for augmented spaces)

        Returns:
            Dict mapping ``state_idx`` -> ``best_action`` (int)
        """
        policy = {}
        n_states = self.adapter.n_states
        V_flat = V.ravel()

        for s_idx in range(n_states):
            s_onehot = self.adapter.index_to_onehot(s_idx)

            # Expected value under each action's transition distribution
            values = []
            for action in range(self.adapter.n_actions):
                s_next_dist = self.adapter.multiply_B_s(self.B, s_onehot, action)
                expected_value = float(np.dot(s_next_dist.ravel(), V_flat))
                values.append(expected_value)

            # Best action that actually changes expected state
            sorted_actions = np.argsort(values)[::-1]
            best = int(sorted_actions[0])
            for action in sorted_actions:
                s_next = self.adapter.multiply_B_s(self.B, s_onehot, int(action))
                if not np.allclose(s_next, s_onehot):
                    best = int(action)
                    break

            policy[s_idx] = best

        return policy

    # ==================== Cached Episode Execution ====================

    def _run_episode_hierarchical_cached(self, max_steps: int) -> Dict[str, Any]:
        """Run hierarchical episode using precompiled policy tables (O(1) per step)."""
        total_steps = 0
        total_reward = 0.0
        macro_decisions = 0
        micro_phase = False

        goal_macro_states = set()
        for gs in self.goal_states:
            if gs in self.micro_to_macro:
                goal_macro_states.add(self.micro_to_macro[gs])

        # Hierarchical phase: navigate through macro states
        while total_steps < max_steps:
            s_idx = self.adapter.get_current_state_index()
            if s_idx not in self.micro_to_macro:
                break
            s_macro = self.micro_to_macro[s_idx]
            if s_macro in goal_macro_states:
                break

            best_macro = self._macro_policy.get(s_macro)
            if best_macro is None:
                break
            target_macro = self.adj_list[s_macro][best_macro]
            macro_decisions += 1     # one macro planning decision (O(1) cached)

            # Navigate to bottleneck using cached bottleneck policy
            bn_policy = self._bottleneck_policies.get((s_macro, target_macro))
            if bn_policy is None:
                # Fallback: use goal policy to navigate toward the goal
                bn_policy = self._goal_policy
            bottleneck_set = set(
                self.bottleneck_states.get((s_macro, target_macro), [])
            )
            if not bottleneck_set:
                # Fallback: use all target cluster states
                bottleneck_set = set(self.macro_state_list[target_macro])

            while total_steps < max_steps:
                s_idx = self.adapter.get_current_state_index()
                if s_idx in bottleneck_set or self._is_at_goal():
                    break

                action = bn_policy.get(s_idx, 0)  # O(1) lookup
                n_phys, step_reward = self._step_with_smooth(action, self.test_smooth_steps)
                total_reward += step_reward
                self.current_state = self._get_planning_state()
                self.state_history.append(self.current_state.copy())
                self.action_history.append(action)
                total_steps += n_phys

                s_idx = self.adapter.get_current_state_index()

                # Check if we've reached the target macro state
                if s_idx in self.micro_to_macro:
                    if self.micro_to_macro[s_idx] == target_macro:
                        break

            if self._is_at_goal():
                break

        # Micro phase: navigate to exact goal using cached goal policy
        if total_steps < max_steps and not self._is_at_goal():
            micro_phase = True       # micro phase uses cached goal policy (O(1))
        while total_steps < max_steps and not self._is_at_goal():
            s_idx = self.adapter.get_current_state_index()
            action = self._goal_policy.get(s_idx, 0)  # O(1) lookup
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
            'micro_phase': micro_phase,
            'planning_steps': macro_decisions + (1 if micro_phase else 0),
        }

    def _run_episode_flat_cached(self, max_steps: int) -> Dict[str, Any]:
        """Run flat episode using precompiled goal policy table (O(1) per step)."""
        steps = 0
        reward = 0.0

        while steps < max_steps and not self._is_at_goal():
            s_idx = self.adapter.get_current_state_index()
            action = self._goal_policy.get(s_idx, 0)  # O(1) lookup
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

    # ==================== Policy Save / Load ====================

    def save_compiled_policy(self, path: str):
        """Save compiled policy tables to an ``.npz`` file for deployment.

        The saved file contains only the policy lookup tables and structural
        metadata (adjacency, bottleneck states, cluster assignments).  No B or
        M matrices are needed — the file is typically ~100x smaller.

        Args:
            path: File path (e.g. ``"acrobot_policy.npz"``)
        """
        if not self._policy_compiled:
            raise RuntimeError("Must call compile_policy() before save")

        data = {
            'goal_policy_keys': np.array(list(self._goal_policy.keys())),
            'goal_policy_vals': np.array(list(self._goal_policy.values())),
            'macro_policy_keys': np.array(list(self._macro_policy.keys())),
            'macro_policy_vals': np.array(
                [v if v is not None else -1 for v in self._macro_policy.values()]
            ),
            'goal_states': np.array(self.goal_states),
            'n_clusters': np.array(self.n_clusters),
            'micro_to_macro_keys': np.array(list(self.micro_to_macro.keys())),
            'micro_to_macro_vals': np.array(list(self.micro_to_macro.values())),
        }

        # Save each bottleneck policy
        for (src, tgt), policy in self._bottleneck_policies.items():
            data[f'bn_{src}_{tgt}_keys'] = np.array(list(policy.keys()))
            data[f'bn_{src}_{tgt}_vals'] = np.array(list(policy.values()))

        # Save adjacency and structural metadata as JSON
        # Convert numpy ints to Python ints for JSON serialization
        def _to_python(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {_to_python(k): _to_python(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_python(x) for x in obj]
            if isinstance(obj, set):
                return [_to_python(x) for x in obj]
            return obj

        metadata = _to_python({
            'adj_list': {str(k): v for k, v in self.adj_list.items()},
            'bottleneck_states': {
                f"{k[0]}_{k[1]}": v for k, v in self.bottleneck_states.items()
            },
            'bottleneck_policy_keys': [
                f"{s}_{t}" for (s, t) in self._bottleneck_policies.keys()
            ],
        })
        data['metadata_json'] = np.array([json.dumps(metadata)])

        np.savez_compressed(path, **data)
        print(f"Saved compiled policy to {path}")

    @classmethod
    def load_compiled_policy(cls, path: str, adapter: 'BaseEnvironmentAdapter') -> 'HierarchicalSRAgent':
        """Load a compiled policy for inference-only use.

        No B or M matrices are loaded — only the precomputed policy lookup
        tables and structural metadata.

        Args:
            path: Path to ``.npz`` file created by ``save_compiled_policy``
            adapter: An environment adapter (needed for ``step`` / ``reset``)

        Returns:
            A ``HierarchicalSRAgent`` with compiled policy ready for execution
        """
        data = np.load(path, allow_pickle=True)

        agent = cls(adapter, n_clusters=int(data['n_clusters']))

        # Restore goal policy
        agent._goal_policy = dict(zip(
            data['goal_policy_keys'].tolist(),
            data['goal_policy_vals'].tolist(),
        ))

        # Restore macro policy
        macro_keys = data['macro_policy_keys'].tolist()
        macro_vals = data['macro_policy_vals'].tolist()
        agent._macro_policy = {
            k: (v if v != -1 else None) for k, v in zip(macro_keys, macro_vals)
        }

        # Restore goal states and micro_to_macro
        agent.goal_states = data['goal_states'].tolist()
        agent.micro_to_macro = dict(zip(
            data['micro_to_macro_keys'].tolist(),
            data['micro_to_macro_vals'].tolist(),
        ))

        # Restore structural metadata
        metadata = json.loads(str(data['metadata_json'][0]))
        agent.adj_list = {int(k): v for k, v in metadata['adj_list'].items()}

        agent.bottleneck_states = {}
        for key_str, states in metadata['bottleneck_states'].items():
            s, t = key_str.split('_')
            agent.bottleneck_states[(int(s), int(t))] = states

        # Restore bottleneck policies
        agent._bottleneck_policies = {}
        for key_str in metadata['bottleneck_policy_keys']:
            s, t = key_str.split('_')
            keys = data[f'bn_{s}_{t}_keys'].tolist()
            vals = data[f'bn_{s}_{t}_vals'].tolist()
            agent._bottleneck_policies[(int(s), int(t))] = dict(zip(keys, vals))

        agent._policy_compiled = True
        print(f"Loaded compiled policy from {path}: {len(agent._goal_policy)} states")
        return agent
