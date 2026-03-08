"""Demo: Neural Successor Feature agent on Acrobot.

Trains a neural SF agent on Acrobot-v1 and evaluates it. Unlike the
tabular agent which discretizes the 4D state space into bins (leading to
exponential blowup), the neural agent operates directly on raw 6D
observations.

Training uses three phases with gradual distribution shift:
1. Diverse exploration (100%): random starts to build the SF representation
2. Transition (60% diverse): gradual shift toward task-relevant states
3. Task-focused (30% diverse): mostly fixed-start to learn swing-up

Usage:
    python examples/run_neural_acrobot.py
    python examples/run_neural_acrobot.py --quick
"""

import argparse
import os
import sys
import time

# Ensure imports resolve from the project root (not from examples/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import math

import gymnasium as gym
import numpy as np

from environments.acrobot import AcrobotAdapter
from core.neural.continuous_adapter import ContinuousAdapter
from core.neural.agent import NeuralSRAgent
from examples.configs import NEURAL_ACROBOT
from examples.neural_experiment import setup_device


def acrobot_height_reward(obs):
    """Dense shaped reward based on Acrobot end-effector height + velocity.

    Height = -cos(θ1) - cos(θ1 + θ2), ranges from -2 (hanging) to +2 (upright).
    Goal: height > 1.0. Base reward normalized to [-1, 1] range.

    Near the goal region (height > 0.5), adds a velocity bonus to encourage
    swinging THROUGH the goal with sufficient angular velocity — aligning the
    shaped reward with the velocity-filtered goal criterion.
    """
    c1, s1, c2, s2 = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
    dtheta1 = float(obs[4])
    theta1 = math.atan2(s1, c1)
    theta2 = math.atan2(s2, c2)
    height = -np.cos(theta1) - np.cos(theta1 + theta2)

    # Base height reward
    reward = height / 2.0

    # Velocity bonus near goal: reward upward angular velocity
    # so w learns to value high-height + correct-velocity states
    if height > 0.5:
        velocity_bonus = 0.3 * np.clip(dtheta1 / (4 * np.pi), -1.0, 1.0)
        reward += velocity_bonus

    # Goal threshold bonus: sharp signal when at goal height
    if height > 1.0:
        reward += 1.0

    return reward


def main():
    parser = argparse.ArgumentParser(description="Neural SF agent on Acrobot")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with fewer episodes")
    parser.add_argument("--n-eval", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--save-dir", type=str, default="data/neural_acrobot",
                        help="Directory to save checkpoints")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device ('cpu' or 'cuda')")
    args = parser.parse_args()

    cfg = NEURAL_ACROBOT

    # ==================== Setup ====================
    print("=" * 60)
    print("Neural Successor Feature Agent — Acrobot")
    print("=" * 60)

    env_train = gym.make('Acrobot-v1')
    base_adapter = AcrobotAdapter(
        env_train,
        n_theta_bins=cfg["n_theta_bins"],
        n_dtheta_bins=cfg["n_dtheta_bins"],
        goal_velocity_filter=cfg["goal_velocity_filter"],
    )
    adapter = ContinuousAdapter(base_adapter)

    print(f"Observation dim: {adapter.obs_dim}")
    print(f"Actions: {adapter.n_actions}")
    print(f"SF dim: {cfg['sf_dim']}")
    print(f"Goal states (discrete): {len(base_adapter.get_goal_states())} bins")
    print()

    # ==================== Create Agent ====================
    device = setup_device(args.device)
    print(f"Device: {device}")

    agent = NeuralSRAgent(
        adapter=adapter,
        sf_dim=cfg["sf_dim"],
        hidden_sizes=cfg["hidden_sizes"],
        gamma=cfg["gamma"],
        lr=cfg["lr"],
        lr_w=cfg["lr_w"],
        batch_size=cfg["batch_size"],
        buffer_size=cfg["buffer_size"],
        target_update_freq=cfg["target_update_freq"],
        tau=cfg["tau"],
        epsilon_start=cfg["epsilon_start"],
        epsilon_end=cfg["epsilon_end"],
        epsilon_decay_steps=cfg["epsilon_decay_steps"],
        device=device,
        use_per=cfg.get("use_per", False),
        per_alpha=cfg.get("per_alpha", 0.6),
        per_beta_start=cfg.get("per_beta_start", 0.4),
        per_beta_end=cfg.get("per_beta_end", 1.0),
        use_episodic_replay=cfg.get("use_episodic_replay", False),
        episodic_replay_episodes=cfg.get("episodic_replay_episodes", 2),
    )

    agent.set_goal(
        goal_spec=None,
        reward=cfg["reward"],
        default_cost=cfg["default_cost"],
        use_env_reward=True,
        terminal_bonus=cfg.get("terminal_bonus", 0.0),
        reward_shaping_fn=acrobot_height_reward,
    )

    # ==================== Three-Phase Training ====================
    # Gradual transition from diverse to task-focused to avoid the
    # hard distribution shift that destabilized SF learning.
    ep1 = cfg["train_episodes_phase1"]
    ep2 = cfg["train_episodes_phase2"]
    ep3 = cfg["train_episodes_phase3"]
    frac2 = cfg["diverse_fraction_phase2"]
    frac3 = cfg["diverse_fraction_phase3"]
    if args.quick:
        ep1 = 150
        ep2 = 150
        ep3 = 200
        args.n_eval = 5

    t0 = time.time()

    # Phase 1: Diverse exploration — build SF representation
    print(f"Phase 1: Diverse exploration ({ep1} episodes, 100% diverse)")
    agent.learn_environment(
        num_episodes=ep1,
        steps_per_episode=cfg["steps_per_episode"],
        diverse_start=True,
        log_interval=max(1, ep1 // 5),
    )
    agent.save(os.path.join(args.save_dir, "checkpoint_phase1.pt"))

    # Phase 2: Gradual transition — intermediate diversity
    agent.truncate_buffer(keep_fraction=cfg["buffer_keep_phase2"])
    agent.reset_epsilon(
        new_start=cfg["epsilon_phase2_start"],
        new_decay_steps=cfg["epsilon_phase2_decay_steps"],
    )
    agent.reset_lr(
        sf_lr=cfg["lr"] * cfg["lr_phase2_fraction"],
        rw_lr=cfg["lr_w"] * cfg["lr_phase2_fraction"],
        decay_steps=ep2 * cfg["steps_per_episode"],
    )
    print(f"\nPhase 2: Transition ({ep2} episodes, {frac2:.0%} diverse)")
    agent.learn_environment(
        num_episodes=ep2,
        steps_per_episode=cfg["steps_per_episode"],
        diverse_start=True,
        diverse_fraction=frac2,
        log_interval=max(1, ep2 // 5),
    )
    agent.save(os.path.join(args.save_dir, "checkpoint_phase2.pt"))

    # Phase 3: Task-focused — mostly fixed start
    agent.truncate_buffer(keep_fraction=cfg["buffer_keep_phase3"])
    agent.reset_epsilon(
        new_start=cfg["epsilon_phase3_start"],
        new_decay_steps=cfg["epsilon_phase3_decay_steps"],
    )
    agent.reset_lr(
        sf_lr=cfg["lr"] * cfg["lr_phase3_fraction"],
        rw_lr=cfg["lr_w"] * cfg["lr_phase3_fraction"],
        decay_steps=ep3 * cfg["steps_per_episode"],
    )
    print(f"\nPhase 3: Task-focused ({ep3} episodes, {frac3:.0%} diverse)")
    agent.learn_environment(
        num_episodes=ep3,
        steps_per_episode=cfg["steps_per_episode"],
        diverse_start=True,
        diverse_fraction=frac3,
        log_interval=max(1, ep3 // 5),
    )

    train_time = time.time() - t0
    print(f"\nTotal training time: {train_time:.1f}s")

    # ==================== Save ====================
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "checkpoint.pt")
    agent.save(save_path)
    print(f"Saved checkpoint to {save_path}")

    # ==================== Evaluate ====================
    print(f"\nEvaluating ({args.n_eval} episodes)...")

    env_test = gym.make('Acrobot-v1', max_episode_steps=cfg["test_max_steps"])
    test_base = AcrobotAdapter(
        env_test,
        n_theta_bins=cfg["n_theta_bins"],
        n_dtheta_bins=cfg["n_dtheta_bins"],
        goal_velocity_filter=cfg["goal_velocity_filter"],
    )
    agent.adapter = ContinuousAdapter(test_base)
    agent.goal_states = test_base.get_goal_states()

    results = []
    for i in range(args.n_eval):
        result = agent.run_episode(
            init_state=[0, 0, 0, 0],
            max_steps=cfg["test_max_steps"],
        )
        results.append(result)
        status = "GOAL" if result['reached_goal'] else "timeout"
        print(f"  Episode {i + 1}: {result['steps']:4d} steps, "
              f"reward={result['reward']:.1f}, {status}")

    # ==================== Summary ====================
    steps = [r['steps'] for r in results]
    successes = [r['reached_goal'] for r in results]
    print(f"\n{'=' * 40}")
    print(f"Success rate: {sum(successes)}/{len(successes)} "
          f"({100 * sum(successes) / len(successes):.0f}%)")
    print(f"Avg steps (all): {np.mean(steps):.1f} ± {np.std(steps):.1f}")
    if any(successes):
        success_steps = [s for s, ok in zip(steps, successes) if ok]
        print(f"Avg steps (success): {np.mean(success_steps):.1f} "
              f"± {np.std(success_steps):.1f}")
    print(f"{'=' * 40}")


if __name__ == "__main__":
    main()
