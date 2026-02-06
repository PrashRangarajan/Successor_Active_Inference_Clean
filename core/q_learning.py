"""Tabular Q-Learning baseline agent.

This serves as a baseline comparison for Successor Representation agents.
Ported from the legacy q_learning.py to use the new adapter API.
"""

import random
from typing import Any, Dict, List, Optional

import numpy as np

from .base_environment import BaseEnvironmentAdapter


class QLearningAgent:
    """Tabular Q-Learning agent compatible with the adapter interface.

    Uses epsilon-greedy exploration with decay and standard Q-learning updates.
    Unlike HierarchicalSRAgent.learn_environment(), the learn() method is
    incremental — Q-values accumulate across calls.
    """

    def __init__(
        self,
        adapter: BaseEnvironmentAdapter,
        goal_states: List[int],
        C: np.ndarray,
        gamma: float = 0.99,
        alpha: float = 0.1,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        """
        Args:
            adapter: Environment adapter implementing BaseEnvironmentAdapter
            goal_states: List of goal state indices
            C: Preference/reward vector (same as SR agents use)
            gamma: Discount factor (default: 0.99)
            alpha: Learning rate for Q-updates (default: 0.1)
            epsilon_start: Initial exploration rate (default: 1.0)
            epsilon_end: Minimum exploration rate (default: 0.05)
            epsilon_decay: Decay rate for epsilon per episode (default: 0.995)
        """
        self.adapter = adapter
        self.goal_states = set(goal_states)
        self.C = C
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.n_states = adapter.n_states
        self.n_actions = adapter.n_actions

        # Initialize Q-table
        self.Q = np.zeros((self.n_states, self.n_actions))

        # Training episode counter
        self.total_episodes = 0

    def select_action(self, state_idx: int, greedy: bool = False) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state_idx: Current state index
            greedy: If True, always select best action (for evaluation)

        Returns:
            Selected action index
        """
        if not greedy and np.random.rand() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            return int(np.argmax(self.Q[state_idx]))

    def _update_q(self, state_idx: int, action: int, reward: float,
                  next_state_idx: int, done: bool) -> float:
        """Q-learning update: Q(s,a) += alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

        Returns:
            TD error
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state_idx])

        td_error = target - self.Q[state_idx, action]
        self.Q[state_idx, action] += self.alpha * td_error
        return td_error

    def learn(self, num_episodes: int = 500):
        """Train the Q-learning agent through exploration.

        This method is INCREMENTAL — Q-values accumulate across calls.
        This matches the legacy learn_env_likelikood() behavior.

        Args:
            num_episodes: Number of training episodes to run
        """
        # Use grid_size-based step limit if available, otherwise default
        if hasattr(self.adapter, 'grid_size'):
            max_steps = 5 * self.adapter.grid_size
        else:
            max_steps = 200

        for ep in range(num_episodes):
            if (ep + 1) % 100 == 0:
                print(f"Q-learning episode {self.total_episodes + ep + 1}", end='\r')

            self.adapter.reset()
            state_idx = self.adapter.get_current_state_index()

            for t in range(max_steps):
                # Select action with exploration
                action = self.select_action(state_idx)

                # Take action
                self.adapter.step(action)
                next_state_idx = self.adapter.get_current_state_index()

                # Get reward from preference vector
                reward = self.C[next_state_idx]
                done = next_state_idx in self.goal_states

                # Update Q-value
                self._update_q(state_idx, action, reward, next_state_idx, done)

                state_idx = next_state_idx

                if done:
                    break

            # Decay epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self.total_episodes += num_episodes

    def run_episode(self, init_state: Optional[Any] = None,
                    max_steps: int = 200) -> Dict[str, Any]:
        """Run a single evaluation episode using greedy policy.

        Args:
            init_state: Initial state (index, tuple, or None for random)
            max_steps: Maximum steps per episode

        Returns:
            Dict matching HierarchicalSRAgent result format:
                'steps': number of steps taken
                'reward': cumulative reward
                'reached_goal': whether goal was reached
        """
        self.adapter.reset(init_state)
        state_idx = self.adapter.get_current_state_index()

        total_reward = 0.0
        steps = 0
        reached_goal = False

        while steps < max_steps:
            # Greedy action selection
            action = self.select_action(state_idx, greedy=True)

            # Take action
            self.adapter.step(action)
            next_state_idx = self.adapter.get_current_state_index()

            # Accumulate reward
            total_reward += self.C[next_state_idx]
            steps += 1

            # Check goal
            if next_state_idx in self.goal_states:
                reached_goal = True
                break

            # Handle being stuck (wall) — try other actions
            if next_state_idx == state_idx:
                moved = False
                for alt_action in range(self.n_actions):
                    if alt_action != action:
                        self.adapter.reset(int(state_idx))
                        self.adapter.step(alt_action)
                        alt_next = self.adapter.get_current_state_index()
                        if alt_next != state_idx:
                            next_state_idx = alt_next
                            moved = True
                            break
                if not moved:
                    # Truly stuck, reset back to current position
                    self.adapter.reset(int(state_idx))

            state_idx = next_state_idx

        return {
            'steps': steps,
            'reward': total_reward,
            'reached_goal': reached_goal,
        }

    def get_value_function(self) -> np.ndarray:
        """Return value function V(s) = max_a Q(s,a)."""
        return np.max(self.Q, axis=1)
