"""Demo: Neural Successor Feature agent on Acrobot.

Trains a neural SF agent on Acrobot-v1 and evaluates it. Unlike the
tabular agent which discretizes the 4D state space into bins (leading to
exponential blowup), the neural agent operates directly on raw 6D
observations.

Training uses two phases:
1. Diverse exploration: random starts to build the SF representation
2. Fixed-start training: learn to solve the task from the default state

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


def acrobot_height_reward(obs):
    """Dense shaped reward based on Acrobot end-effector height.

    Height = -cos(θ1) - cos(θ1 + θ2), ranges from -2 (hanging) to +2 (upright).
    Goal: height > 1.0. We normalize to [-1, 1] range.
    """
    c1, s1, c2, s2 = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
    theta1 = math.atan2(s1, c1)
    theta2 = math.atan2(s2, c2)
    height = -np.cos(theta1) - np.cos(theta1 + theta2)
    return height / 2.0


def main():
    parser = argparse.ArgumentParser(description="Neural SF agent on Acrobot")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with fewer episodes")
    parser.add_argument("--n-eval", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--save-dir", type=str, default="data/neural_acrobot",
                        help="Directory to save checkpoints")
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
    )

    agent.set_goal(
        goal_spec=None,
        reward=cfg["reward"],
        default_cost=cfg["default_cost"],
        use_env_reward=True,
        terminal_bonus=cfg.get("terminal_bonus", 0.0),
        reward_shaping_fn=acrobot_height_reward,
    )

    # ==================== Two-Phase Training ====================
    ep_diverse = cfg["train_episodes_diverse"]
    ep_fixed = cfg["train_episodes_fixed"]
    if args.quick:
        ep_diverse = 200
        ep_fixed = 300
        args.n_eval = 5

    t0 = time.time()

    # Phase 1: Diverse exploration — build SF representation
    print(f"Phase 1: Diverse exploration ({ep_diverse} episodes)")
    agent.learn_environment(
        num_episodes=ep_diverse,
        steps_per_episode=cfg["steps_per_episode"],
        diverse_start=True,
        log_interval=max(1, ep_diverse // 5),
    )

    # Phase 2: Mixed training — mostly fixed start with some diverse
    diverse_frac = cfg.get("diverse_fraction", 0.3)
    print(f"\nPhase 2: Mixed training ({ep_fixed} episodes, "
          f"{diverse_frac:.0%} diverse)")
    agent.learn_environment(
        num_episodes=ep_fixed,
        steps_per_episode=cfg["steps_per_episode"],
        diverse_start=True,
        diverse_fraction=diverse_frac,
        log_interval=max(1, ep_fixed // 5),
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
