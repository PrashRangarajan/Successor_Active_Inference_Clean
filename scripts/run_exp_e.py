"""Experiment E: Extended Phase 1 only.

Load Exp C's Phase 1 checkpoint (4000 eps) and continue Phase 1
for another 4000 episodes. Tests whether more Phase 1 exploration
alone is sufficient, without Phase 2/3.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from examples.configs import NEURAL_POINTMAZE

# Same episode config as Exp C
NEURAL_POINTMAZE["train_episodes_phase1"] = 4000  # additional episodes
NEURAL_POINTMAZE["steps_per_episode"] = 500
NEURAL_POINTMAZE["epsilon_decay_steps"] = 1_300_000

# No Phase 2 or 3
NEURAL_POINTMAZE["consolidation_episodes"] = 0
NEURAL_POINTMAZE["train_episodes_phase2"] = 0

import examples.configs
examples.configs.NEURAL_POINTMAZE = NEURAL_POINTMAZE

save_dir = sys.argv[1] if len(sys.argv) > 1 else "data/neural_point_maze_exp_e"
device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
source_ckpt = sys.argv[3] if len(sys.argv) > 3 else "data/neural_point_maze_exp_c/checkpoint_phase1.pt"

sys.argv = [
    "run_neural_point_maze.py",
    "--save-dir", save_dir,
    "--device", device,
    "--no-staging",
]

# Import and set up everything
from examples.run_neural_point_maze import *
import numpy as np
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

cfg = NEURAL_POINTMAZE

env_train = gym.make(cfg["maze_id"], max_episode_steps=cfg["steps_per_episode"])
base_adapter = PointMazeAdapter(env_train, n_x_bins=cfg["n_x_bins"], n_y_bins=cfg["n_y_bins"])
adapter = ContinuousAdapter(base_adapter)
adapter.reset()

dev = setup_device(device)

her_goal_indices = tuple(cfg.get("her_goal_indices", [4, 6]))
agent = NeuralSRAgent(
    adapter=adapter, sf_dim=cfg["sf_dim"], hidden_sizes=cfg["hidden_sizes"],
    gamma=cfg["gamma"], lr=cfg["lr"], lr_w=cfg["lr_w"],
    batch_size=cfg["batch_size"], buffer_size=cfg["buffer_size"],
    target_update_freq=cfg["target_update_freq"], tau=cfg["tau"],
    epsilon_start=cfg["epsilon_end"],  # start at low epsilon (already explored)
    epsilon_end=cfg["epsilon_end"],
    epsilon_decay_steps=1,
    device=dev,
    use_per=cfg.get("use_per", False), per_alpha=cfg.get("per_alpha", 0.6),
    per_beta_start=cfg.get("per_beta_start", 0.4), per_beta_end=cfg.get("per_beta_end", 1.0),
    use_episodic_replay=cfg.get("use_episodic_replay", False),
    episodic_replay_episodes=cfg.get("episodic_replay_episodes", 2),
    use_her=cfg.get("use_her", False), her_k=cfg.get("her_k", 4),
    her_goal_indices=her_goal_indices,
    train_every=cfg.get("train_every", 1),
)

reward_fn = make_maze_distance_reward(base_adapter)
agent.set_goal(
    goal_spec=None, reward=cfg["reward"], default_cost=cfg["default_cost"],
    use_env_reward=True, terminal_bonus=cfg.get("terminal_bonus", 0.0),
    reward_shaping_fn=reward_fn,
)

# Load Exp C Phase 1 checkpoint
agent.load(source_ckpt)
print(f"Loaded Phase 1 checkpoint from {source_ckpt}")
print(f"Continuing Phase 1 for {cfg['train_episodes_phase1']} more episodes (ε={cfg['epsilon_end']})")

os.makedirs(save_dir, exist_ok=True)

# Continue Phase 1: diverse exploration
agent.learn_environment(
    num_episodes=cfg["train_episodes_phase1"],
    steps_per_episode=cfg["steps_per_episode"],
    diverse_start=True,
    log_interval=max(1, cfg["train_episodes_phase1"] // 5),
)

agent.save(os.path.join(save_dir, "checkpoint.pt"))
print(f"Saved checkpoint to {save_dir}/checkpoint.pt")

# Evaluate
print(f"\nEvaluating (10 episodes)...")
env_test = gym.make(cfg["maze_id"], max_episode_steps=cfg["test_max_steps"])
test_base = PointMazeAdapter(env_test, n_x_bins=cfg["n_x_bins"], n_y_bins=cfg["n_y_bins"])
agent.adapter = ContinuousAdapter(test_base)
test_base.reset()
agent.goal_states = test_base.get_goal_states()

results = []
for i in range(10):
    result = agent.run_episode(init_state=None, max_steps=cfg["test_max_steps"])
    results.append(result)
    status = "GOAL" if result['reached_goal'] else "timeout"
    print(f"  Episode {i + 1}: {result['steps']:4d} steps, reward={result['reward']:.1f}, {status}")

steps = [r['steps'] for r in results]
successes = [r['reached_goal'] for r in results]
print(f"\n{'=' * 40}")
print(f"Success rate: {sum(successes)}/{len(successes)} ({100 * sum(successes) / len(successes):.0f}%)")
print(f"Avg steps (all): {np.mean(steps):.1f} +/- {np.std(steps):.1f}")
if any(successes):
    success_steps = [s for s, ok in zip(steps, successes) if ok]
    print(f"Avg steps (success): {np.mean(success_steps):.1f} +/- {np.std(success_steps):.1f}")
print(f"{'=' * 40}")

env_train.close()
env_test.close()
