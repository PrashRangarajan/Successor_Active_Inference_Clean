"""Experiment I continuation: Train best 3 seeds for 4000 more episodes.

Resumes from checkpoint.pt for seeds 0, 1, 2 with epsilon held at 0.20
(already decayed) for 4000 additional Phase 1 episodes.

Also re-evaluates the BEFORE checkpoint with the same eval seed for comparison.

Usage:
    # Run seeds 0,1 on GPU 0:
    CUDA_VISIBLE_DEVICES=0 conda run -n sai python scripts/run_exp_i_continue.py cuda 0 1

    # Run seed 2 on GPU 1:
    CUDA_VISIBLE_DEVICES=1 conda run -n sai python scripts/run_exp_i_continue.py cuda 2
"""
import sys
import os

# Use EGL for headless MuJoCo rendering
os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import numpy as np
import torch
import gymnasium as gym

from core.neural.agent import NeuralSRAgent
from core.neural.continuous_adapter import ContinuousAdapter
from environments.point_maze import PointMazeAdapter
from examples.configs import NEURAL_POINTMAZE as BASE_CONFIG
from examples.run_neural_point_maze import make_maze_distance_reward, setup_device

N_EVAL = 50
EVAL_SEED = 42  # Fixed eval seed for fair comparison


def build_agent_from_config(cfg, device):
    """Build agent + env from config dict."""
    env = gym.make(cfg["maze_id"], max_episode_steps=cfg["steps_per_episode"])
    base_adapter = PointMazeAdapter(env, n_x_bins=cfg["n_x_bins"], n_y_bins=cfg["n_y_bins"])
    adapter = ContinuousAdapter(base_adapter)
    adapter.reset()

    her_goal_indices = tuple(cfg.get("her_goal_indices", [4, 6]))

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
        epsilon_start=cfg["epsilon_end"],  # Start at final epsilon
        epsilon_end=cfg["epsilon_end"],
        epsilon_decay_steps=cfg["epsilon_decay_steps"],
        device=device,
        use_per=cfg.get("use_per", False),
        per_alpha=cfg.get("per_alpha", 0.6),
        per_beta_start=cfg.get("per_beta_start", 0.4),
        per_beta_end=cfg.get("per_beta_end", 1.0),
        use_episodic_replay=cfg.get("use_episodic_replay", False),
        episodic_replay_episodes=cfg.get("episodic_replay_episodes", 2),
        use_her=cfg.get("use_her", False),
        her_k=cfg.get("her_k", 4),
        her_goal_indices=her_goal_indices,
        train_every=cfg.get("train_every", 1),
    )

    reward_fn = make_maze_distance_reward(base_adapter)
    agent.set_goal(
        goal_spec=None,
        reward=cfg["reward"],
        default_cost=cfg["default_cost"],
        use_env_reward=True,
        terminal_bonus=cfg.get("terminal_bonus", 0.0),
        reward_shaping_fn=reward_fn,
    )

    return agent, env, base_adapter


def evaluate(agent, cfg, n_episodes=N_EVAL, seed=EVAL_SEED):
    """Evaluate agent with fixed seed for reproducibility."""
    env_test = gym.make(cfg["maze_id"], max_episode_steps=cfg["test_max_steps"])
    test_base = PointMazeAdapter(env_test, n_x_bins=cfg["n_x_bins"], n_y_bins=cfg["n_y_bins"])
    agent.adapter = ContinuousAdapter(test_base)
    test_base.reset()
    agent.goal_states = test_base.get_goal_states()

    # Fixed eval seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    results = []
    for i in range(n_episodes):
        result = agent.run_episode(init_state=None, max_steps=cfg["test_max_steps"])
        results.append(result)
        status = "GOAL" if result['reached_goal'] else "timeout"
        print(f"  Episode {i + 1}: {result['steps']:4d} steps, "
              f"reward={result['reward']:.1f}, {status}")

    steps = [r['steps'] for r in results]
    successes = [r['reached_goal'] for r in results]
    n_succ = sum(successes)
    print(f"\n  Success rate: {n_succ}/{len(successes)} ({100 * n_succ / len(successes):.0f}%)")
    print(f"  Avg steps (all): {np.mean(steps):.1f} +/- {np.std(steps):.1f}")
    if any(successes):
        success_steps = [s for s, ok in zip(steps, successes) if ok]
        print(f"  Avg steps (success): {np.mean(success_steps):.1f} +/- {np.std(success_steps):.1f}")

    env_test.close()
    return n_succ, len(successes)


def run_continuation(seed, device="cuda"):
    """Resume training for one seed and compare before/after."""
    cfg = dict(BASE_CONFIG)
    cfg["train_episodes_phase1"] = 4000
    cfg["steps_per_episode"] = 500
    cfg["epsilon_decay_steps"] = 1_300_000
    cfg["epsilon_end"] = 0.20

    checkpoint_path = f"data/neural_point_maze_exp_i/seed_{seed}/checkpoint.pt"
    continued_dir = f"data/neural_point_maze_exp_i_continued/seed_{seed}"
    os.makedirs(continued_dir, exist_ok=True)

    dev = setup_device(device)

    # --- BEFORE: evaluate existing checkpoint ---
    print(f"--- BEFORE (4000 episodes) ---")
    agent, env_train, base_adapter = build_agent_from_config(cfg, dev)
    agent.load(checkpoint_path)
    print(f"Loaded {checkpoint_path} (steps={agent.total_steps}, ε={agent.epsilon:.3f})")
    before_succ, before_total = evaluate(agent, cfg)
    env_train.close()

    # --- TRAIN: 4000 more episodes ---
    print(f"\n--- TRAINING 4000 more episodes ---")
    # Rebuild agent fresh for training (clean env)
    torch.manual_seed(seed)
    np.random.seed(seed + 1000)  # Offset so continuation sees new transitions
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 1000)

    agent, env_train, base_adapter = build_agent_from_config(cfg, dev)
    agent.load(checkpoint_path)
    print(f"Continuing from steps={agent.total_steps}, ε={agent.epsilon:.3f}")

    t0 = time.time()
    agent.learn_environment(
        num_episodes=4000,
        steps_per_episode=cfg["steps_per_episode"],
        diverse_start=True,
        log_interval=800,
        checkpoint_dir=continued_dir,
        checkpoint_interval=1000,
    )
    elapsed = time.time() - t0
    print(f"Continuation training time: {elapsed:.1f}s")

    agent.save(os.path.join(continued_dir, "checkpoint.pt"))
    print(f"Saved to {continued_dir}/checkpoint.pt")

    # --- AFTER: evaluate continued checkpoint ---
    print(f"\n--- AFTER (8000 total episodes) ---")
    after_succ, after_total = evaluate(agent, cfg)
    env_train.close()

    print(f"\n--- SEED {seed} COMPARISON ---")
    print(f"  Before: {before_succ}/{before_total} ({100*before_succ/before_total:.0f}%)")
    print(f"  After:  {after_succ}/{after_total} ({100*after_succ/after_total:.0f}%)")

    return before_succ, after_succ


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    seeds = [int(s) for s in sys.argv[2:]] if len(sys.argv) > 2 else [0, 1, 2]

    print(f"=== Experiment I Continuation: +4000 episodes ===")
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print(f"Eval: {N_EVAL} episodes, seed={EVAL_SEED}")

    all_results = {}
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}\n")
        before, after = run_continuation(seed, device)
        all_results[seed] = (before, after)

    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for seed, (before, after) in all_results.items():
        delta = after - before
        sign = "+" if delta >= 0 else ""
        print(f"  Seed {seed}: {before}/{N_EVAL} -> {after}/{N_EVAL} ({sign}{delta})")
    print(f"{'='*60}")
