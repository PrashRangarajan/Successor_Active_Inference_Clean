"""Neural Successor Feature agent for continuous state spaces.

Replaces the tabular SR (M matrix) with neural successor features:
    Q(s, a) = φ(s, a)ᵀ · w

where φ(s, a) are learned successor features and w encodes reward preferences.

The SF Bellman equation (neural analog of tabular TD update):
    φ(s, a) = ψ(s') + γ · φ(s', a')

Training uses the same TD structure as the tabular agent but with gradient
descent on neural networks instead of direct matrix updates.
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import SFNetwork, ActionConditionedSFNetwork, RewardFeatureNetwork
from .replay_buffer import ReplayBuffer
from .losses import sf_td_loss, sf_td_loss_per_sample, reward_prediction_loss
from .utils import soft_update, hard_update
from .continuous_adapter import ContinuousAdapter


class NeuralSRAgent:
    """Neural Successor Feature agent.

    Core factorization: Q(s, a) = φ(s, a)ᵀ · w
    - φ is learned via TD on successor features
    - w is learned via reward regression: r(s) ≈ ψ(s)ᵀ · w
    - When the goal changes, only w needs relearning (SF transfer)

    Args:
        adapter: ContinuousAdapter wrapping a BinnedContinuousAdapter.
        sf_dim: Dimensionality of successor features.
        hidden_sizes: Hidden layer sizes for networks.
        gamma: Discount factor.
        lr: Learning rate for SF network.
        lr_w: Learning rate for reward weight and reward feature network.
        batch_size: Batch size for training.
        buffer_size: Replay buffer capacity.
        target_update_freq: Steps between target network updates.
        tau: Soft update coefficient for target network.
        epsilon_start: Initial exploration rate.
        epsilon_end: Final exploration rate.
        epsilon_decay_steps: Steps over which to decay epsilon linearly.
        grad_clip: Maximum gradient norm for clipping.
        device: Torch device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        adapter: ContinuousAdapter,
        sf_dim: int = 128,
        hidden_sizes: Tuple[int, ...] = (256, 256),
        gamma: float = 0.99,
        lr: float = 3e-4,
        lr_w: float = 1e-3,
        batch_size: int = 256,
        buffer_size: int = 100_000,
        target_update_freq: int = 1000,
        tau: float = 0.005,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 50_000,
        grad_clip: float = 1.0,
        device: str = 'cpu',
        sf_network_cls: str = 'per_action',
        use_per: bool = False,
        per_alpha: float = 0.6,
        per_beta_start: float = 0.4,
        per_beta_end: float = 1.0,
        per_beta_annealing_steps: Optional[int] = None,
        use_episodic_replay: bool = False,
        episodic_replay_episodes: int = 2,
        use_her: bool = False,
        her_k: int = 4,
        her_goal_indices: Tuple[int, int] = (4, 6),
        train_every: int = 1,
    ):
        # Validation
        if not (0 < gamma <= 1):
            raise ValueError(f"gamma must be in (0, 1], got {gamma}")
        if sf_dim < 1:
            raise ValueError(f"sf_dim must be >= 1, got {sf_dim}")

        self.adapter = adapter
        self.obs_dim = adapter.obs_dim
        self.n_actions = adapter.n_actions
        self.sf_dim = sf_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.tau = tau
        self.grad_clip = grad_clip
        self.device = torch.device(device)
        self._sf_network_cls = sf_network_cls

        # Exploration schedule
        self.epsilon = epsilon_start
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay_steps = epsilon_decay_steps
        self._epsilon_phase_step_offset = 0

        # Action subset sampling for large action spaces (used in _update_sf).
        # When n_actions exceeds the threshold, the batch argmax in the Double
        # DQN target uses a random subset of candidate actions instead of
        # evaluating all actions — a ~10x speedup for 729-action spaces.
        self._action_candidate_threshold = 64
        self._n_action_candidates = min(64, self.n_actions)

        # Networks — select architecture based on action space size
        NetworkCls = self._resolve_sf_network_cls(sf_network_cls)
        self.sf_net = NetworkCls(
            self.obs_dim, self.n_actions, sf_dim, hidden_sizes
        ).to(self.device)
        self.sf_target = NetworkCls(
            self.obs_dim, self.n_actions, sf_dim, hidden_sizes
        ).to(self.device)
        hard_update(self.sf_target, self.sf_net)

        self.reward_net = RewardFeatureNetwork(
            self.obs_dim, sf_dim, hidden_sizes=(hidden_sizes[0],)
        ).to(self.device)

        # Reward weight vector w: Q(s,a) = φ(s,a)ᵀ · w
        self.w = nn.Parameter(torch.zeros(sf_dim, device=self.device))

        # Optimizers
        self.sf_optimizer = optim.Adam(self.sf_net.parameters(), lr=lr)
        self.reward_optimizer = optim.Adam(
            list(self.reward_net.parameters()) + [self.w], lr=lr_w
        )

        # LR schedulers (created on-demand via reset_lr())
        self._sf_scheduler = None
        self._rw_scheduler = None

        # Replay buffer
        self._use_per = use_per
        self._per_beta_start = per_beta_start
        self._per_beta_end = per_beta_end
        self._per_beta_annealing_steps = per_beta_annealing_steps or 1_000_000
        self._use_episodic_replay = use_episodic_replay
        self._episodic_replay_episodes = episodic_replay_episodes
        self._use_her = use_her
        self._her_k = her_k
        self._her_goal_indices = her_goal_indices
        self._train_every = max(1, train_every)
        self.buffer = ReplayBuffer(
            buffer_size, self.obs_dim,
            use_per=use_per, per_alpha=per_alpha,
        )

        # State
        self.total_steps = 0
        self._reward_fn = None
        self._goal_reward_fn = None
        self._reward_shaping_fn = None
        self._use_env_reward = True
        self._terminal_bonus = 0.0
        self.goal_states = []

        # Training logs
        self.training_log: Dict[str, List[float]] = {
            'sf_loss': [],
            'reward_loss': [],
            'episode_reward': [],
            'episode_steps': [],
            # Diagnostics for phase-transition analysis
            'q_mean': [],
            'q_std': [],
            'q_max': [],
            'sf_grad_norm': [],
            'rw_grad_norm': [],
            'w_norm': [],
            'epsilon': [],
            'sf_lr': [],
            'rw_lr': [],
        }

    @staticmethod
    def _resolve_sf_network_cls(name: str):
        """Resolve SF network class from string name.

        Args:
            name: 'per_action' for SFNetwork (small action spaces),
                  'action_conditioned' for ActionConditionedSFNetwork (large).

        Returns:
            Network class.
        """
        if name == 'per_action':
            return SFNetwork
        elif name == 'action_conditioned':
            return ActionConditionedSFNetwork
        else:
            raise ValueError(f"Unknown sf_network_cls: {name}. "
                             f"Use 'per_action' or 'action_conditioned'.")

    # ==================== Goal / Preference ====================

    def set_goal(self, goal_spec: Any = None, reward: float = 1.0,
                 default_cost: float = 0.0,
                 use_env_reward: bool = True,
                 terminal_bonus: float = 0.0,
                 reward_shaping_fn=None):
        """Set the goal for the agent.

        Creates a reward function from the goal specification. In the SF
        framework, changing the goal only requires re-learning w (the reward
        weights), not φ (the successor features).

        Args:
            goal_spec: Goal specification (environment-specific).
            reward: Reward for reaching goal states.
            default_cost: Cost for non-goal states.
            use_env_reward: If True, use the environment's native reward
                signal for training (denser, better for learning). The
                goal-based reward is still used for terminal checking
                during evaluation. If False, use the sparse goal reward.
            terminal_bonus: Extra reward added when the environment signals
                termination (goal reached). Helps w learn to distinguish
                goal states when the base env reward is flat (e.g., Acrobot
                gives -1 every step). Set to 0 to disable.
            reward_shaping_fn: Optional function obs → float providing dense
                shaped reward for training. When set, overrides both
                use_env_reward and goal reward. Essential for long-horizon
                tasks where the env reward is too sparse for SF learning.
        """
        self._goal_reward_fn = self.adapter.create_goal_reward_fn(
            goal_spec, reward, default_cost
        )
        self.goal_states = self.adapter.get_goal_states(goal_spec)
        self._use_env_reward = use_env_reward
        self._terminal_bonus = terminal_bonus
        self._reward_shaping_fn = reward_shaping_fn
        if not use_env_reward:
            self._reward_fn = self._goal_reward_fn

    # ==================== Staged Learning ====================

    def freeze_sf(self):
        """Freeze successor feature network (stop gradient updates).

        Use during staged learning Phase 3: only w adapts to the goal.
        """
        for param in self.sf_net.parameters():
            param.requires_grad = False
        for param in self.sf_target.parameters():
            param.requires_grad = False

    def unfreeze_sf(self):
        """Unfreeze successor feature network."""
        for param in self.sf_net.parameters():
            param.requires_grad = True
        # Target network doesn't need grad (updated via soft copy)

    def freeze_reward_weights(self):
        """Freeze reward weight w and reward feature network.

        Use during staged learning Phase 2: consolidate SF representation
        without the moving target of a changing reward signal.
        """
        self.w.requires_grad = False
        for param in self.reward_net.parameters():
            param.requires_grad = False

    def unfreeze_reward_weights(self):
        """Unfreeze reward weight w and reward feature network."""
        self.w.requires_grad = True
        for param in self.reward_net.parameters():
            param.requires_grad = True

    # ==================== Learning ====================

    def learn_environment(self, num_episodes: int = 1000,
                          steps_per_episode: int = 200,
                          diverse_start: bool = True,
                          diverse_fraction: float = 1.0,
                          log_interval: int = 100):
        """Learn successor features by exploring the environment.

        For each step:
        1. Select action via ε-greedy on Q(s,a) = φ(s,a)ᵀ·w
        2. Store transition in replay buffer
        3. Sample batch and update SF network (TD loss) and w (reward loss)
        4. Periodically soft-update target network

        Args:
            num_episodes: Number of training episodes.
            steps_per_episode: Maximum steps per episode.
            diverse_start: If True, start episodes from random states.
            diverse_fraction: Fraction of episodes using diverse starts
                (rest use default start). Only applies when diverse_start=True.
                Set to 1.0 for all-diverse, 0.5 for half-and-half, etc.
            log_interval: Episodes between log prints.
        """
        print(f"Learning with Neural SF ({num_episodes} episodes, "
              f"sf_dim={self.sf_dim}, ε: {self.epsilon:.3f}→{self._epsilon_end})...")

        for episode in range(num_episodes):
            use_diverse = diverse_start and (np.random.rand() < diverse_fraction)
            if use_diverse:
                obs = self.adapter.sample_random_state()
            else:
                obs = self.adapter.reset()

            ep_reward = 0.0
            ep_steps = 0

            for step in range(steps_per_episode):
                action = self.select_action(obs, greedy=False)
                next_obs, env_reward, terminated, truncated, info = \
                    self.adapter.step(action)

                # Compute reward for training
                if self._reward_shaping_fn is not None:
                    # Dense shaped reward — provides gradient signal at every
                    # step, essential for long-horizon tasks like Acrobot
                    reward = self._reward_shaping_fn(next_obs)
                elif self._use_env_reward:
                    reward = env_reward
                elif self._reward_fn:
                    reward = self._reward_fn(next_obs)
                else:
                    reward = env_reward
                # Terminal bonus on top of any reward source
                if terminated and self._terminal_bonus != 0:
                    reward += self._terminal_bonus

                self.buffer.add(obs, action, reward, next_obs,
                                terminated or truncated)

                ep_reward += reward
                ep_steps += 1
                self.total_steps += 1

                # Train on a batch (skip steps to reduce CPU/GPU overhead)
                if (self.buffer.size >= self.batch_size
                        and self.total_steps % self._train_every == 0):
                    if self._use_per:
                        beta = self._get_per_beta()
                        batch, per_indices, is_weights = \
                            self.buffer.sample_prioritized(
                                self.batch_size, beta, self.device)
                        is_weights_t = torch.as_tensor(
                            is_weights, dtype=torch.float32,
                            device=self.device)
                        sf_loss, td_norms = self._update_sf_per(
                            batch, is_weights_t)
                        self.buffer.update_priorities(
                            per_indices, td_norms.cpu().numpy())
                        rw_loss = self._update_reward_weights(batch)
                    else:
                        batch = self.buffer.sample_uniform(
                            self.batch_size, self.device
                        )
                        sf_loss = self._update_sf(batch)
                        rw_loss = self._update_reward_weights(batch)

                    self.training_log['sf_loss'].append(sf_loss)
                    self.training_log['reward_loss'].append(rw_loss)

                    # Step LR schedulers if active
                    if self._sf_scheduler is not None:
                        self._sf_scheduler.step()
                    if self._rw_scheduler is not None:
                        self._rw_scheduler.step()

                # Target network update
                if self.total_steps % self.target_update_freq == 0:
                    soft_update(self.sf_target, self.sf_net, self.tau)

                # Epsilon decay
                self._decay_epsilon()

                obs = next_obs
                if terminated or truncated:
                    break

            self.buffer.end_episode()

            # HER: relabel failed trajectory with achieved goals
            # Multiplies effective data ~k× without extra env steps.
            if self._use_her:
                self.buffer.add_her_transitions(
                    goal_indices=self._her_goal_indices,
                    k=self._her_k,
                    reward_fn=self._reward_shaping_fn,
                )

            # Hippocampal replay: replay complete past episodes
            # at the end of each training episode for temporal coherence
            if (self._use_episodic_replay
                    and self.buffer._episode_starts):
                episodes = self.buffer.sample_episodes(
                    self._episodic_replay_episodes, self.device)
                for ep_batch in episodes:
                    self._update_sf(ep_batch)
                    self._update_reward_weights(ep_batch)

            self.training_log['episode_reward'].append(ep_reward)
            self.training_log['episode_steps'].append(ep_steps)
            self.training_log['epsilon'].append(self.epsilon)
            self.training_log['sf_lr'].append(
                self.sf_optimizer.param_groups[0]['lr']
            )
            self.training_log['rw_lr'].append(
                self.reward_optimizer.param_groups[0]['lr']
            )

            # Q-value diagnostics from last observation of the episode
            if obs is not None:
                q_vals = self.get_q_values(obs)
                self.training_log['q_mean'].append(float(np.mean(q_vals)))
                self.training_log['q_std'].append(float(np.std(q_vals)))
                self.training_log['q_max'].append(float(np.max(q_vals)))

            if (episode + 1) % log_interval == 0:
                avg_reward = np.mean(
                    self.training_log['episode_reward'][-log_interval:]
                )
                avg_sf_loss = np.mean(
                    self.training_log['sf_loss'][-100:]
                ) if self.training_log['sf_loss'] else 0.0
                print(f"  Episode {episode + 1}/{num_episodes} | "
                      f"Avg reward: {avg_reward:.2f} | "
                      f"SF loss: {avg_sf_loss:.4f} | "
                      f"ε: {self.epsilon:.3f} | "
                      f"Buffer: {self.buffer.size}")

        print(f"Learning complete. Total steps: {self.total_steps}")

    def _update_sf(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single successor feature update step.

        Implements the SF Bellman equation via TD learning:
            target = ψ(s') + γ · (1-done) · φ_target(s', argmax_a' Q(s',a'))
            loss = MSE(φ(s,a), target)

        For action-conditioned networks with large action spaces (729+),
        uses random action subset sampling to approximate the argmax
        over next-state actions. This reduces the batch forward from
        O(B * n_actions) to O(B * n_action_candidates), a ~10x speedup
        for HalfCheetah.

        Skips when SF network is frozen (staged learning Phase 3).

        Args:
            batch: Dict with 'obs', 'actions', 'rewards', 'next_obs', 'dones'.

        Returns:
            Scalar loss value (0.0 if frozen).
        """
        # Skip if SF net is frozen (staged learning goal-focused phase)
        if not next(self.sf_net.parameters()).requires_grad:
            self.training_log['sf_grad_norm'].append(0.0)
            return 0.0

        obs = batch['obs']
        actions = batch['actions']
        next_obs = batch['next_obs']
        dones = batch['dones']

        # Current SFs: φ(s, a)
        sf_current = self.sf_net.get_sf(obs, actions)  # (B, sf_dim)

        # Reward features for next state: ψ(s')
        psi_next = self.reward_net(next_obs)  # (B, sf_dim)

        with torch.no_grad():
            # Action selection via online network (Double DQN-style)
            # For action-conditioned nets with many actions, use subset sampling
            if (isinstance(self.sf_net, ActionConditionedSFNetwork)
                    and self.n_actions > self._action_candidate_threshold):
                next_actions = self._select_next_actions_subset(next_obs)
            else:
                all_sf_next = self.sf_net(next_obs)  # (B, n_actions, sf_dim)
                q_next = (all_sf_next * self.w).sum(dim=-1)  # (B, n_actions)
                next_actions = q_next.argmax(dim=1)  # (B,)

            # Target SF values
            sf_target_next = self.sf_target.get_sf(
                next_obs, next_actions
            )  # (B, sf_dim)

        loss = sf_td_loss(sf_current, psi_next, sf_target_next, dones, self.gamma)

        self.sf_optimizer.zero_grad()
        loss.backward()
        sf_grad_norm = nn.utils.clip_grad_norm_(
            self.sf_net.parameters(), self.grad_clip
        )
        self.sf_optimizer.step()

        self.training_log['sf_grad_norm'].append(sf_grad_norm.item())
        return loss.item()

    def _get_per_beta(self) -> float:
        """Compute current PER importance-sampling beta via linear annealing."""
        fraction = min(
            1.0,
            self.total_steps / max(1, self._per_beta_annealing_steps),
        )
        return self._per_beta_start + fraction * (
            self._per_beta_end - self._per_beta_start
        )

    def _update_sf_per(
        self, batch: Dict[str, torch.Tensor], is_weights: torch.Tensor,
    ) -> Tuple[float, torch.Tensor]:
        """SF update with PER importance sampling weights.

        Same forward pass as _update_sf(), but uses the per-sample loss
        function and returns TD error norms for priority updates.

        Skips when SF network is frozen (staged learning Phase 3).

        Args:
            batch: Transition batch dict.
            is_weights: IS weights from prioritized sampling, shape (B,).

        Returns:
            Tuple of (scalar_loss, td_error_norms).
        """
        # Skip if SF net is frozen (staged learning goal-focused phase)
        if not next(self.sf_net.parameters()).requires_grad:
            self.training_log['sf_grad_norm'].append(0.0)
            dummy_norms = torch.zeros(batch['obs'].shape[0], device=self.device)
            return 0.0, dummy_norms

        obs = batch['obs']
        actions = batch['actions']
        next_obs = batch['next_obs']
        dones = batch['dones']

        sf_current = self.sf_net.get_sf(obs, actions)
        psi_next = self.reward_net(next_obs)

        with torch.no_grad():
            if (isinstance(self.sf_net, ActionConditionedSFNetwork)
                    and self.n_actions > self._action_candidate_threshold):
                next_actions = self._select_next_actions_subset(next_obs)
            else:
                all_sf_next = self.sf_net(next_obs)
                q_next = (all_sf_next * self.w).sum(dim=-1)
                next_actions = q_next.argmax(dim=1)

            sf_target_next = self.sf_target.get_sf(next_obs, next_actions)

        loss, td_error_norms = sf_td_loss_per_sample(
            sf_current, psi_next, sf_target_next, dones, self.gamma,
            weights=is_weights,
        )

        self.sf_optimizer.zero_grad()
        loss.backward()
        sf_grad_norm = nn.utils.clip_grad_norm_(
            self.sf_net.parameters(), self.grad_clip
        )
        self.sf_optimizer.step()

        self.training_log['sf_grad_norm'].append(sf_grad_norm.item())
        return loss.item(), td_error_norms

    def _select_next_actions_subset(self, next_obs: torch.Tensor) -> torch.Tensor:
        """Approximate argmax_a Q(s', a) using random action subset sampling.

        Instead of evaluating all n_actions (e.g. 729) for each sample in the
        batch, evaluates a random subset of candidates. The argmax over the
        subset is a good approximation of the true argmax, especially early in
        training when Q-values are noisy anyway.

        Uses forward_actions() which is O(B * K) instead of O(B * A) where
        K << A (e.g., K=64 vs A=729 → ~11x speedup).

        Args:
            next_obs: Next observations, shape (B, obs_dim).

        Returns:
            Best action indices (from full action space), shape (B,).
        """
        K = self._n_action_candidates
        # Sample a random subset of action indices
        candidate_indices = torch.randint(
            0, self.n_actions, (K,), device=self.device
        )
        # Evaluate the subset: (B, K, sf_dim)
        sf_subset = self.sf_net.forward_actions(next_obs, candidate_indices)
        # Compute Q-values for subset
        q_subset = (sf_subset * self.w).sum(dim=-1)  # (B, K)
        # Best within the subset
        best_in_subset = q_subset.argmax(dim=1)  # (B,)
        # Map back to original action indices
        return candidate_indices[best_in_subset]  # (B,)

    def _update_reward_weights(self, batch: Dict[str, torch.Tensor]) -> float:
        """Learn reward weights w such that r(s) ≈ ψ(s)ᵀ · w.

        Skips the update when reward weights are frozen (staged learning
        Phase 2: consolidating SF representation without moving reward target).

        Args:
            batch: Dict with 'obs', 'rewards'.

        Returns:
            Scalar loss value (0.0 if frozen).
        """
        # Skip if w is frozen (staged learning consolidation phase)
        if not self.w.requires_grad:
            self.training_log['rw_grad_norm'].append(0.0)
            self.training_log['w_norm'].append(self.w.data.norm().item())
            return 0.0

        obs = batch['obs']
        rewards = batch['rewards']

        psi = self.reward_net(obs)  # (B, sf_dim)
        predicted = (psi * self.w).sum(dim=-1)  # (B,)

        loss = reward_prediction_loss(predicted, rewards)

        self.reward_optimizer.zero_grad()
        loss.backward()

        rw_grad_norm = nn.utils.clip_grad_norm_(
            list(self.reward_net.parameters()) + [self.w], self.grad_clip
        )
        self.training_log['rw_grad_norm'].append(rw_grad_norm.item())
        self.training_log['w_norm'].append(self.w.data.norm().item())

        self.reward_optimizer.step()

        return loss.item()

    def _decay_epsilon(self):
        """Linearly decay epsilon from start to end over decay_steps.

        Uses phase-relative steps so that epsilon restarts correctly
        after reset_epsilon() is called at phase boundaries.
        """
        phase_steps = self.total_steps - self._epsilon_phase_step_offset
        fraction = min(1.0, phase_steps / max(1, self._epsilon_decay_steps))
        self.epsilon = self._epsilon_start + fraction * (
            self._epsilon_end - self._epsilon_start
        )

    def reset_epsilon(self, new_start: float, new_decay_steps: int):
        """Reset epsilon for a new training phase.

        At phase transitions, the exploration schedule needs to restart
        to give the agent budget to discover the new start-state distribution.

        Args:
            new_start: New epsilon starting value (e.g., 0.3).
            new_decay_steps: Steps over which to decay to epsilon_end.
        """
        self._epsilon_start = new_start
        self._epsilon_decay_steps = new_decay_steps
        self.epsilon = new_start
        self._epsilon_phase_step_offset = self.total_steps

    def reset_lr(self, sf_lr: float, rw_lr: float, decay_steps: int):
        """Reset learning rates for a new training phase with cosine annealing.

        Creates cosine annealing schedulers that decay LR from the given
        peak to 10% of peak over the specified number of training steps.

        Args:
            sf_lr: New peak LR for the SF optimizer.
            rw_lr: New peak LR for the reward optimizer.
            decay_steps: Number of training steps for the cosine decay period.
        """
        for pg in self.sf_optimizer.param_groups:
            pg['lr'] = sf_lr
            pg['initial_lr'] = sf_lr
        for pg in self.reward_optimizer.param_groups:
            pg['lr'] = rw_lr
            pg['initial_lr'] = rw_lr

        self._sf_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.sf_optimizer, T_max=max(1, decay_steps), eta_min=sf_lr * 0.1,
        )
        self._rw_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.reward_optimizer, T_max=max(1, decay_steps), eta_min=rw_lr * 0.1,
        )
        # Advance past epoch 0 so the first scheduler.step() in the training
        # loop doesn't trigger PyTorch's "step() before optimizer.step()" warning.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self._sf_scheduler.step()
            self._rw_scheduler.step()
        print(f"  LR reset: SF={sf_lr:.1e}, RW={rw_lr:.1e}, "
              f"cosine decay over {decay_steps} steps")

    # ==================== Action Selection ====================

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Select action using ε-greedy over SF-derived Q-values.

        Q(s, a) = φ(s, a)ᵀ · w

        Args:
            obs: Raw observation.
            greedy: If True, always pick the best action.

        Returns:
            Action index.
        """
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            all_sf = self.sf_net(obs_t)  # (1, n_actions, sf_dim)
            q_values = (all_sf * self.w).sum(dim=-1)  # (1, n_actions)
            return q_values.argmax(dim=1).item()

    # ==================== Episode Execution ====================

    def run_episode(self, init_state: Optional[Any] = None,
                    max_steps: int = 200) -> Dict[str, Any]:
        """Run a single evaluation episode with greedy policy.

        Returns a dict matching HierarchicalSRAgent's format for
        compatibility with the shared evaluation infrastructure.

        Args:
            init_state: Initial state for reset (environment-specific format).
            max_steps: Maximum steps before truncation.

        Returns:
            Dict with 'steps', 'reward', 'reached_goal', 'final_state'.
        """
        obs = self.adapter.reset(init_state)
        total_reward = 0.0
        steps = 0
        reached_goal = False

        for _ in range(max_steps):
            action = self.select_action(obs, greedy=True)
            next_obs, env_reward, terminated, truncated, info = \
                self.adapter.step(action)

            total_reward += env_reward
            steps += 1

            # Check terminal condition
            terminal = self.adapter.is_terminal(next_obs)
            if terminal is True:
                reached_goal = True
                break

            # Also check discrete goal bins for consistency with tabular agent
            if terminal is None and self.goal_states:
                if self.adapter.is_in_goal_bin(self.goal_states, next_obs):
                    reached_goal = True
                    break

            # Environment terminated (e.g., Acrobot reached height threshold)
            # counts as goal success; truncation (step limit) does not.
            if terminated:
                reached_goal = True
                break
            if truncated:
                break

            obs = next_obs

        return {
            'steps': steps,
            'reward': total_reward,
            'reached_goal': reached_goal,
            'final_state': self.adapter.get_current_obs(),
        }

    # Alias for compatibility with evaluation scripts expecting both modes
    run_episode_flat = run_episode

    # ==================== Goal Transfer ====================

    def truncate_buffer(self, keep_fraction: float):
        """Truncate the replay buffer, keeping only the most recent data.

        Call at phase boundaries to remove stale data from previous phases
        that no longer reflects the current training distribution.

        Args:
            keep_fraction: Fraction of buffer to keep (0.0 to 1.0).
        """
        old_size = self.buffer.size
        self.buffer.truncate(keep_fraction)
        print(f"  Buffer truncated: {old_size} -> {self.buffer.size} "
              f"(kept {keep_fraction:.0%})")

    def relearn_reward_weights(self, n_updates: int = 1000):
        """Re-learn w for a new goal without re-training φ.

        This is the key SF transfer advantage: when the goal changes,
        only the reward weights need updating. The successor features φ
        capture task-independent environment dynamics.

        Uses buffered experience with the new reward function.

        Args:
            n_updates: Number of gradient steps for reward weight learning.
        """
        if self.buffer.size < self.batch_size:
            print("Warning: buffer too small for reward weight relearning.")
            return

        print(f"Re-learning reward weights ({n_updates} updates)...")
        for i in range(n_updates):
            batch = self.buffer.sample_uniform(self.batch_size, self.device)
            # Recompute rewards with the goal-based reward function
            reward_fn = self._goal_reward_fn or self._reward_fn
            if reward_fn:
                obs_np = batch['obs'].cpu().numpy()
                new_rewards = torch.tensor(
                    [reward_fn(obs_np[j]) for j in range(len(obs_np))],
                    dtype=torch.float32, device=self.device,
                )
                batch['rewards'] = new_rewards
            self._update_reward_weights(batch)

    # ==================== Diagnostics ====================

    def get_sf_embedding(self, obs_batch: np.ndarray) -> np.ndarray:
        """Compute mean SF embeddings for a batch of observations.

        Returns the mean SF across actions: mean_a φ(s, a). Useful for
        visualization, t-SNE/UMAP plots, and clustering (Phase 3).

        For large action spaces (action-conditioned networks), samples a
        subset of actions and computes the mean over that subset, avoiding
        the expensive all-actions forward pass.

        Args:
            obs_batch: Observations, shape (n, obs_dim).

        Returns:
            SF embeddings, shape (n, sf_dim).
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(
                obs_batch, dtype=torch.float32, device=self.device
            )
            if (isinstance(self.sf_net, ActionConditionedSFNetwork)
                    and self.n_actions > self._action_candidate_threshold):
                # Sample a subset of actions for efficiency
                K = self._n_action_candidates
                action_indices = torch.randint(
                    0, self.n_actions, (K,), device=self.device
                )
                sf_subset = self.sf_net.forward_actions(obs_t, action_indices)
                return sf_subset.mean(dim=1).cpu().numpy()
            else:
                all_sf = self.sf_net(obs_t)  # (n, n_actions, sf_dim)
                return all_sf.mean(dim=1).cpu().numpy()

    def get_q_values(self, obs: np.ndarray) -> np.ndarray:
        """Compute Q-values for all actions at a single observation.

        Args:
            obs: Single observation, shape (obs_dim,).

        Returns:
            Q-values, shape (n_actions,).
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            all_sf = self.sf_net(obs_t)  # (1, n_actions, sf_dim)
            q_values = (all_sf * self.w).sum(dim=-1)  # (1, n_actions)
            return q_values.squeeze(0).cpu().numpy()

    # ==================== Save / Load ====================

    def save(self, path: str):
        """Save all networks, weights, and training state.

        Args:
            path: File path for the checkpoint.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'sf_net': self.sf_net.state_dict(),
            'sf_target': self.sf_target.state_dict(),
            'reward_net': self.reward_net.state_dict(),
            'w': self.w.data,
            'sf_optimizer': self.sf_optimizer.state_dict(),
            'reward_optimizer': self.reward_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'epsilon': self.epsilon,
            'epsilon_phase_step_offset': self._epsilon_phase_step_offset,
            'sf_scheduler': (self._sf_scheduler.state_dict()
                             if self._sf_scheduler else None),
            'rw_scheduler': (self._rw_scheduler.state_dict()
                             if self._rw_scheduler else None),
            'config': {
                'obs_dim': self.obs_dim,
                'n_actions': self.n_actions,
                'sf_dim': self.sf_dim,
                'gamma': self.gamma,
                'sf_network_cls': self._sf_network_cls,
            },
        }, path)

    def load(self, path: str):
        """Load networks, weights, and training state from checkpoint.

        Args:
            path: File path of the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.sf_net.load_state_dict(checkpoint['sf_net'])
        self.sf_target.load_state_dict(checkpoint['sf_target'])
        self.reward_net.load_state_dict(checkpoint['reward_net'])
        self.w.data.copy_(checkpoint['w'])
        self.sf_optimizer.load_state_dict(checkpoint['sf_optimizer'])
        self.reward_optimizer.load_state_dict(checkpoint['reward_optimizer'])
        self.total_steps = checkpoint['total_steps']
        self.epsilon = checkpoint['epsilon']
        self._epsilon_phase_step_offset = checkpoint.get(
            'epsilon_phase_step_offset', 0
        )
        if checkpoint.get('sf_scheduler') and self._sf_scheduler:
            self._sf_scheduler.load_state_dict(checkpoint['sf_scheduler'])
        if checkpoint.get('rw_scheduler') and self._rw_scheduler:
            self._rw_scheduler.load_state_dict(checkpoint['rw_scheduler'])
