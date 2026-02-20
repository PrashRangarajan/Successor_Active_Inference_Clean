"""Evaluation: Neural Successor Feature agent on Acrobot.

Trains the neural SF agent with configurable training budgets, produces:
  - Training curves: episode reward, SF loss, reward loss
  - Evaluation metrics: success rate, steps-to-goal across checkpoints
  - Video recording of a successful swing-up episode

Saves .npy data to data/eval/neural_acrobot/ and figures to figures/eval/neural_acrobot/.

Usage:
    # Full evaluation (train + plot + video):
    python examples/run_eval_neural_acrobot.py --train

    # Quick test (fewer episodes, fewer seeds):
    python examples/run_eval_neural_acrobot.py --train --quick

    # Plot from saved data (no training):
    python examples/run_eval_neural_acrobot.py

    # Just record a video from a saved checkpoint:
    python examples/run_eval_neural_acrobot.py --video-only
"""

import argparse
import json
import math
import os
import sys
import time

# Ensure imports resolve from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-poster")

from environments.acrobot import AcrobotAdapter
from core.neural.continuous_adapter import ContinuousAdapter
from core.neural.agent import NeuralSRAgent
from core.neural.hierarchical_agent import HierarchicalNeuralSRAgent
from examples.configs import NEURAL_ACROBOT


# ==================== Reward Shaping ====================

def acrobot_height_reward(obs):
    """Dense shaped reward based on Acrobot end-effector height + velocity.

    Height = -cos(theta1) - cos(theta1 + theta2), ranges [-2, +2].
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


# ==================== Agent Factory ====================

def create_neural_agent(cfg):
    """Create a fresh hierarchical neural SR agent with standard Acrobot config.

    Uses HierarchicalNeuralSRAgent which extends NeuralSRAgent with macro-state
    hierarchy support. Even for flat evaluation, this agent class is needed
    because it properly handles the training dynamics that lead to convergence.

    Returns:
        (agent, env_train, base_adapter) tuple.
    """
    env_train = gym.make('Acrobot-v1')
    base_adapter = AcrobotAdapter(
        env_train,
        n_theta_bins=cfg["n_theta_bins"],
        n_dtheta_bins=cfg["n_dtheta_bins"],
        goal_velocity_filter=cfg["goal_velocity_filter"],
    )
    adapter = ContinuousAdapter(base_adapter)

    agent = HierarchicalNeuralSRAgent(
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
        n_clusters=4,
        cluster_method='kmeans',
        cluster_on='observations',
        n_cluster_samples=5000,
        adjacency_episodes=500,
        adjacency_episode_length=50,
    )

    agent.set_goal(
        goal_spec=None,
        reward=cfg["reward"],
        default_cost=cfg["default_cost"],
        use_env_reward=True,
        terminal_bonus=cfg.get("terminal_bonus", 0.0),
        reward_shaping_fn=acrobot_height_reward,
    )

    return agent, env_train, base_adapter


def train_agent(agent, cfg, ep1, ep2, ep3):
    """Three-phase training: gradual transition from diverse to task-focused.

    Avoids the hard distribution shift that destabilized SF learning in the
    two-phase schedule.
    """
    frac2 = cfg.get("diverse_fraction_phase2", 0.6)
    frac3 = cfg.get("diverse_fraction_phase3", 0.3)

    # Phase 1: Diverse exploration — build SF representation
    print(f"Phase 1: Diverse exploration ({ep1} episodes, 100% diverse)")
    agent.learn_environment(
        num_episodes=ep1,
        steps_per_episode=cfg["steps_per_episode"],
        diverse_start=True,
        log_interval=max(1, ep1 // 5),
    )

    # Phase 2: Gradual transition — intermediate diversity
    print(f"\nPhase 2: Transition ({ep2} episodes, {frac2:.0%} diverse)")
    agent.learn_environment(
        num_episodes=ep2,
        steps_per_episode=cfg["steps_per_episode"],
        diverse_start=True,
        diverse_fraction=frac2,
        log_interval=max(1, ep2 // 5),
    )

    # Phase 3: Task-focused — mostly fixed start
    print(f"\nPhase 3: Task-focused ({ep3} episodes, {frac3:.0%} diverse)")
    agent.learn_environment(
        num_episodes=ep3,
        steps_per_episode=cfg["steps_per_episode"],
        diverse_start=True,
        diverse_fraction=frac3,
        log_interval=max(1, ep3 // 5),
    )


# ==================== Evaluation ====================

def evaluate_agent(agent, cfg, n_eval=10):
    """Evaluate trained agent from default start.

    Returns:
        List of result dicts with 'steps', 'reward', 'reached_goal'.
    """
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
    for i in range(n_eval):
        result = agent.run_episode(
            init_state=[0, 0, 0, 0],
            max_steps=cfg["test_max_steps"],
        )
        results.append(result)
    return results


# ==================== Multi-Checkpoint Experiment ====================

def checkpoint_experiment(cfg, args):
    """Train across multiple training budgets and evaluate each.

    Evaluates the agent at several training checkpoints to show learning
    progress. Each checkpoint trains from scratch (not incremental).

    Returns:
        (rewards, steps, successes, episodes_list)
    """
    episodes_list = args.episodes
    n_trials = len(episodes_list)
    n_runs = args.n_runs
    n_eval = args.n_eval

    rewards = np.zeros((n_runs, n_trials))
    steps = np.zeros((n_runs, n_trials))
    successes = np.zeros((n_runs, n_trials))

    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Run {run+1}/{n_runs}")
        print(f"{'='*60}")

        for trial, total_eps in enumerate(episodes_list):
            # Split total budget into three phases (30/30/40)
            ep1 = int(total_eps * 0.30)
            ep2 = int(total_eps * 0.30)
            ep3 = total_eps - ep1 - ep2

            print(f"\n--- Checkpoint: {total_eps} episodes "
                  f"(phase1={ep1}, phase2={ep2}, phase3={ep3}) ---")

            agent, _, _ = create_neural_agent(cfg)
            train_agent(agent, cfg, ep1, ep2, ep3)

            results = evaluate_agent(agent, cfg, n_eval)

            avg_reward = np.mean([r['reward'] for r in results])
            avg_steps = np.mean([r['steps'] for r in results])
            success_rate = np.mean([r['reached_goal'] for r in results])

            rewards[run, trial] = avg_reward
            steps[run, trial] = avg_steps
            successes[run, trial] = success_rate

            print(f"  => Reward: {avg_reward:.1f}, Steps: {avg_steps:.1f}, "
                  f"Success: {success_rate:.0%}")

    return rewards, steps, successes, episodes_list


# ==================== Video Recording ====================

def record_episode_video(agent, cfg, save_path, max_steps=500):
    """Record a single episode as an MP4 video.

    Uses gymnasium's rgb_array render mode to capture frames.

    Args:
        agent: Trained NeuralSRAgent.
        cfg: Config dict.
        save_path: Path to save the .mp4 file.
        max_steps: Max steps for the recorded episode.

    Returns:
        Result dict from the episode.
    """
    env_render = gym.make('Acrobot-v1', render_mode='rgb_array',
                          max_episode_steps=max_steps)
    render_base = AcrobotAdapter(
        env_render,
        n_theta_bins=cfg["n_theta_bins"],
        n_dtheta_bins=cfg["n_dtheta_bins"],
        goal_velocity_filter=cfg["goal_velocity_filter"],
    )
    agent.adapter = ContinuousAdapter(render_base)
    agent.goal_states = render_base.get_goal_states()

    # Reset and collect frames
    obs = agent.adapter.reset([0, 0, 0, 0])
    frames = [env_render.render()]
    heights = []
    total_reward = 0.0
    reached_goal = False

    for step in range(max_steps):
        # Track height for trajectory plot
        c1, s1, c2, s2 = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
        theta1 = math.atan2(s1, c1)
        theta2 = math.atan2(s2, c2)
        height = -np.cos(theta1) - np.cos(theta1 + theta2)
        heights.append(height)

        action = agent.select_action(obs, greedy=True)
        next_obs, env_reward, terminated, truncated, info = agent.adapter.step(action)
        total_reward += env_reward
        frames.append(env_render.render())

        if terminated or truncated:
            # Record the final observation's height so the plot shows
            # the actual goal-reaching state (not just the last pre-goal obs)
            c1f, s1f, c2f, s2f = (float(next_obs[0]), float(next_obs[1]),
                                   float(next_obs[2]), float(next_obs[3]))
            theta1f = math.atan2(s1f, c1f)
            theta2f = math.atan2(s2f, c2f)
            heights.append(-np.cos(theta1f) - np.cos(theta1f + theta2f))
            if terminated:
                reached_goal = True
            break
        obs = next_obs

    env_render.close()

    # Save height trajectory plot
    if heights:
        save_dir_traj = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
        fig_h, ax_h = plt.subplots(figsize=(10, 5))
        ax_h.plot(heights, color='C0', linewidth=2)
        ax_h.axhline(y=1.0, color='red', linestyle='--', alpha=0.7,
                      label='Goal threshold (height > 1)')
        ax_h.fill_between(range(len(heights)), 1.0, 2.0, alpha=0.1, color='green')
        ax_h.set_xlabel('Step', fontsize=14)
        ax_h.set_ylabel('End-Effector Height', fontsize=14)
        status_str = f"{len(heights)} steps to goal" if reached_goal else "timeout"
        ax_h.set_title(f'Acrobot Swing-Up: Neural SF Agent ({status_str})',
                        fontsize=16)
        ax_h.legend(fontsize=12)
        ax_h.grid(True, alpha=0.3)
        ax_h.set_xlim(0, len(heights))
        ax_h.set_ylim(-2.2, 2.2)
        plt.tight_layout()
        traj_path = os.path.join(save_dir_traj, 'height_trajectory.png')
        plt.savefig(traj_path, dpi=150)
        plt.close(fig_h)
        print(f"  Saved {traj_path}")

    # Save as MP4 using matplotlib animation
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    try:
        import matplotlib.animation as animation

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis('off')
        img = ax.imshow(frames[0])

        def update(frame_idx):
            img.set_data(frames[frame_idx])
            ax.set_title(f"Step {frame_idx}/{len(frames)-1}"
                         f"{'  GOAL!' if frame_idx == len(frames)-1 and reached_goal else ''}",
                         fontsize=14)
            return [img]

        anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                       interval=50, blit=True)
        anim.save(save_path, writer='ffmpeg', fps=30, dpi=100)
        plt.close(fig)
        print(f"  Saved video to {save_path}")
    except Exception as e:
        print(f"  Warning: Could not save video ({e})")
        print(f"  Saving frames as GIF instead...")
        # Fallback: save as GIF using PIL
        try:
            from PIL import Image
            gif_path = save_path.replace('.mp4', '.gif')
            pil_frames = [Image.fromarray(f) for f in frames]
            pil_frames[0].save(gif_path, save_all=True,
                               append_images=pil_frames[1:],
                               duration=33, loop=0)
            print(f"  Saved GIF to {gif_path}")
        except ImportError:
            print("  Could not save video (no ffmpeg) or GIF (no PIL).")
            # Last resort: save frames as numpy
            np_path = save_path.replace('.mp4', '_frames.npy')
            np.save(np_path, np.array(frames))
            print(f"  Saved raw frames to {np_path}")

    result = {
        'steps': len(frames) - 1,
        'reward': total_reward,
        'reached_goal': reached_goal,
        'n_frames': len(frames),
    }
    return result


# ==================== Plotting ====================

def plot_training_curves(training_log, save_dir):
    """Plot training curves from the agent's training log.

    Generates:
        - Episode reward over training
        - SF loss over training steps
        - Reward prediction loss over training steps
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Episode Reward ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ep_rewards = training_log['episode_reward']
    ax.plot(ep_rewards, alpha=0.3, color='C0', linewidth=0.5)
    # Smoothed curve
    window = min(100, len(ep_rewards) // 10 + 1)
    if window > 1 and len(ep_rewards) >= window:
        smoothed = np.convolve(ep_rewards, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, window-1+len(smoothed)), smoothed,
                color='C0', linewidth=2, label=f'Smoothed ({window}-ep)')
    ax.set_xlabel('Episode', fontsize=16)
    ax.set_ylabel('Episode Reward', fontsize=16)
    ax.set_title('Neural SF Agent — Training Reward', fontsize=18)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'training_reward.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

    # --- SF Loss ---
    fig, ax = plt.subplots(figsize=(12, 6))
    sf_loss = training_log['sf_loss']
    if sf_loss:
        ax.plot(sf_loss, alpha=0.2, color='C1', linewidth=0.5)
        window = min(500, len(sf_loss) // 10 + 1)
        if window > 1 and len(sf_loss) >= window:
            smoothed = np.convolve(sf_loss, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, window-1+len(smoothed)), smoothed,
                    color='C1', linewidth=2, label=f'Smoothed ({window}-step)')
        ax.set_xlabel('Training Step', fontsize=16)
        ax.set_ylabel('SF TD Loss', fontsize=16)
        ax.set_title('Successor Feature Loss', fontsize=18)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    plt.tight_layout()
    path = os.path.join(save_dir, 'sf_loss.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

    # --- Reward Prediction Loss ---
    fig, ax = plt.subplots(figsize=(12, 6))
    rw_loss = training_log['reward_loss']
    if rw_loss:
        ax.plot(rw_loss, alpha=0.2, color='C2', linewidth=0.5)
        window = min(500, len(rw_loss) // 10 + 1)
        if window > 1 and len(rw_loss) >= window:
            smoothed = np.convolve(rw_loss, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, window-1+len(smoothed)), smoothed,
                    color='C2', linewidth=2, label=f'Smoothed ({window}-step)')
        ax.set_xlabel('Training Step', fontsize=16)
        ax.set_ylabel('Reward Prediction Loss', fontsize=16)
        ax.set_title('Reward Weight (w) Learning', fontsize=18)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    plt.tight_layout()
    path = os.path.join(save_dir, 'reward_loss.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")

    # --- Episode Steps ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ep_steps = training_log['episode_steps']
    ax.plot(ep_steps, alpha=0.3, color='C3', linewidth=0.5)
    window = min(100, len(ep_steps) // 10 + 1)
    if window > 1 and len(ep_steps) >= window:
        smoothed = np.convolve(ep_steps, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, window-1+len(smoothed)), smoothed,
                color='C3', linewidth=2, label=f'Smoothed ({window}-ep)')
    ax.set_xlabel('Episode', fontsize=16)
    ax.set_ylabel('Steps per Episode', fontsize=16)
    ax.set_title('Episode Length During Training', fontsize=18)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, 'episode_steps.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


def plot_checkpoint_curves(episodes_list, rewards, steps, successes, save_dir):
    """Plot evaluation metrics across training checkpoints.

    Generates:
        - Success rate vs training episodes
        - Steps to goal vs training episodes
        - Total reward vs training episodes
    """
    os.makedirs(save_dir, exist_ok=True)
    n_runs = rewards.shape[0]

    def _plot(data, ylabel, title, filename, ylim=None):
        fig, ax = plt.subplots(figsize=(12, 7))
        mean = np.mean(data, axis=0)
        sem = np.std(data, axis=0) / np.sqrt(n_runs)
        ax.plot(episodes_list, mean, 'o-', color='C0', linewidth=2, markersize=8)
        ax.fill_between(episodes_list, mean - sem, mean + sem, alpha=0.3, color='C0')
        ax.set_xlabel('Training Episodes', fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_title(title, fontsize=20)
        ax.tick_params(labelsize=14)
        ax.grid(True, alpha=0.3)
        if ylim is not None:
            ax.set_ylim(ylim)
        plt.tight_layout()
        path = os.path.join(save_dir, filename)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")

    _plot(successes, 'Success Rate', 'Neural SF — Acrobot Success Rate',
          'checkpoint_success_rate.png', ylim=(-0.05, 1.05))
    _plot(steps, 'Steps to Goal', 'Neural SF — Steps to Goal',
          'checkpoint_steps.png')
    _plot(rewards, 'Total Reward', 'Neural SF — Total Reward',
          'checkpoint_reward.png')


def plot_q_value_heatmap(agent, cfg, save_dir):
    """Plot Q-values as a function of Acrobot height.

    Samples observations at various heights and plots Q-value spread.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Sample observations at different theta1 values
    theta1_range = np.linspace(-np.pi, np.pi, 50)
    heights = []
    q_spreads = []
    q_means = []
    q_maxes = []

    for theta1 in theta1_range:
        # Construct observation: [cos(t1), sin(t1), cos(t2), sin(t2), dt1, dt2]
        obs = np.array([
            np.cos(theta1), np.sin(theta1),
            1.0, 0.0,  # theta2=0
            0.0, 0.0,  # zero velocities
        ], dtype=np.float32)

        height = -np.cos(theta1) - np.cos(theta1 + 0)
        heights.append(height)

        q = agent.get_q_values(obs)
        q_spreads.append(np.max(q) - np.min(q))
        q_means.append(np.mean(q))
        q_maxes.append(np.max(q))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Q-value spread vs height
    axes[0].plot(heights, q_spreads, 'o-', color='C0', markersize=4)
    axes[0].set_xlabel('End-Effector Height', fontsize=14)
    axes[0].set_ylabel('Q-Value Spread (max - min)', fontsize=14)
    axes[0].set_title('Action Differentiation vs Height', fontsize=16)
    axes[0].axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Goal threshold')
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Max Q-value vs height
    axes[1].plot(heights, q_maxes, 'o-', color='C1', markersize=4)
    axes[1].set_xlabel('End-Effector Height', fontsize=14)
    axes[1].set_ylabel('Max Q-Value', fontsize=14)
    axes[1].set_title('Expected Value vs Height', fontsize=16)
    axes[1].axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='Goal threshold')
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, 'q_value_analysis.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Neural SF Acrobot — Evaluation with Plots & Video")
    parser.add_argument("--train", action="store_true",
                        help="Run training experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with fewer episodes")
    parser.add_argument("--video-only", action="store_true",
                        help="Only record video from checkpoint")
    parser.add_argument("--n-runs", type=int, default=None,
                        help="Number of evaluation runs")
    parser.add_argument("--n-eval", type=int, default=10,
                        help="Eval episodes per checkpoint")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Override figure save directory")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to saved checkpoint for --video-only")
    args_cli = parser.parse_args()

    cfg = NEURAL_ACROBOT

    data_dir = "data/eval/neural_acrobot"
    fig_dir = args_cli.save_dir or "figures/eval/neural_acrobot"
    checkpoint_dir = "data/neural_acrobot"

    if args_cli.quick:
        episodes = list(cfg["eval_quick_episodes"])
        n_runs = args_cli.n_runs or cfg["eval_quick_n_runs"]
        n_eval = 5
    else:
        episodes = list(cfg["eval_episodes"])
        n_runs = args_cli.n_runs or cfg["eval_n_runs"]
        n_eval = args_cli.n_eval

    # ==================== Video-Only Mode ====================
    if args_cli.video_only:
        print("=" * 60)
        print("VIDEO RECORDING — Loading checkpoint")
        print("=" * 60)

        agent, _, _ = create_neural_agent(cfg)
        ckpt = args_cli.checkpoint or os.path.join(checkpoint_dir, "checkpoint_working.pt")
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found at {ckpt}")
            print("Run with --train first, or provide --checkpoint path.")
            return
        agent.load(ckpt)
        print(f"Loaded checkpoint from {ckpt}")

        os.makedirs(fig_dir, exist_ok=True)
        video_result = record_episode_video(
            agent, cfg,
            os.path.join(fig_dir, 'acrobot_neural_episode.mp4'),
            max_steps=cfg["test_max_steps"],
        )
        status = "GOAL" if video_result['reached_goal'] else "timeout"
        print(f"  Episode: {video_result['steps']} steps, "
              f"reward={video_result['reward']:.1f}, {status}")
        return

    # ==================== Full Training Run ====================
    if args_cli.train:
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)

        print("=" * 60)
        print("NEURAL SF ACROBOT EVALUATION")
        print("=" * 60)
        print(f"Checkpoints: {episodes}")
        print(f"Runs: {n_runs}, Eval episodes per checkpoint: {n_eval}")
        print()

        # Save config
        with open(os.path.join(data_dir, "args.json"), "w") as f:
            json.dump({
                "episodes": episodes,
                "n_runs": n_runs,
                "n_eval": n_eval,
                "config": {k: v if not isinstance(v, tuple) else list(v)
                           for k, v in cfg.items()},
            }, f, indent=2)

        # --- Single full training run for training curves + video ---
        print("\n" + "=" * 60)
        print("PHASE A: Full training run (for training curves & video)")
        print("=" * 60)

        ep1 = cfg["train_episodes_phase1"]
        ep2 = cfg["train_episodes_phase2"]
        ep3 = cfg["train_episodes_phase3"]
        if args_cli.quick:
            ep1 = 150
            ep2 = 150
            ep3 = 200

        agent, _, _ = create_neural_agent(cfg)
        t0 = time.time()
        train_agent(agent, cfg, ep1, ep2, ep3)
        train_time = time.time() - t0
        print(f"\nTraining time: {train_time:.1f}s")

        # Save checkpoint
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        agent.save(save_path)
        print(f"Saved checkpoint to {save_path}")

        # Save training log
        for key, vals in agent.training_log.items():
            np.save(os.path.join(data_dir, f"training_{key}.npy"),
                    np.array(vals))

        # Plot training curves
        print("\n--- Training Curves ---")
        plot_training_curves(agent.training_log, fig_dir)

        # Q-value analysis
        print("\n--- Q-Value Analysis ---")
        plot_q_value_heatmap(agent, cfg, fig_dir)

        # Evaluate the fully trained agent
        print("\n--- Final Evaluation ---")
        results = evaluate_agent(agent, cfg, n_eval)
        for i, r in enumerate(results):
            status = "GOAL" if r['reached_goal'] else "timeout"
            print(f"  Episode {i+1}: {r['steps']:4d} steps, "
                  f"reward={r['reward']:.1f}, {status}")

        success_rate = np.mean([r['reached_goal'] for r in results])
        avg_steps = np.mean([r['steps'] for r in results])
        print(f"\n  Success rate: {success_rate:.0%}")
        print(f"  Avg steps: {avg_steps:.1f}")

        # Record video
        print("\n--- Video Recording ---")
        video_result = record_episode_video(
            agent, cfg,
            os.path.join(fig_dir, 'acrobot_neural_episode.mp4'),
            max_steps=cfg["test_max_steps"],
        )
        status = "GOAL" if video_result['reached_goal'] else "timeout"
        print(f"  Video episode: {video_result['steps']} steps, "
              f"reward={video_result['reward']:.1f}, {status}")

        # --- Multi-checkpoint experiment ---
        print("\n" + "=" * 60)
        print("PHASE B: Checkpoint evaluation (learning curves)")
        print("=" * 60)

        exp_args = argparse.Namespace(
            episodes=episodes,
            n_runs=n_runs,
            n_eval=n_eval,
        )

        t0 = time.time()
        rewards, steps_arr, successes, eps_list = checkpoint_experiment(cfg, exp_args)
        exp_time = time.time() - t0
        print(f"\nCheckpoint experiment completed in {exp_time:.0f}s")

        # Save checkpoint experiment data
        np.save(os.path.join(data_dir, "checkpoint_rewards.npy"), rewards)
        np.save(os.path.join(data_dir, "checkpoint_steps.npy"), steps_arr)
        np.save(os.path.join(data_dir, "checkpoint_successes.npy"), successes)
        np.save(os.path.join(data_dir, "checkpoint_episodes.npy"),
                np.array(eps_list))
        print(f"Saved checkpoint data to {data_dir}/")

        # Plot checkpoint curves
        print("\n--- Checkpoint Curves ---")
        plot_checkpoint_curves(eps_list, rewards, steps_arr, successes, fig_dir)

    else:
        # ==================== Plot from Saved Data ====================
        print("=" * 60)
        print("PLOTTING FROM SAVED DATA")
        print("=" * 60)
        os.makedirs(fig_dir, exist_ok=True)

    # --- Always try to plot from saved data ---
    # Training curves
    training_keys = ['sf_loss', 'reward_loss', 'episode_reward', 'episode_steps']
    training_log = {}
    all_found = True
    for key in training_keys:
        path = os.path.join(data_dir, f"training_{key}.npy")
        if os.path.exists(path):
            training_log[key] = np.load(path).tolist()
        else:
            all_found = False

    if all_found and training_log:
        print("\n--- Training Curves (from saved data) ---")
        plot_training_curves(training_log, fig_dir)

    # Checkpoint curves
    ckpt_rewards_path = os.path.join(data_dir, "checkpoint_rewards.npy")
    if os.path.exists(ckpt_rewards_path):
        print("\n--- Checkpoint Curves (from saved data) ---")
        rewards = np.load(ckpt_rewards_path)
        steps_arr = np.load(os.path.join(data_dir, "checkpoint_steps.npy"))
        successes = np.load(os.path.join(data_dir, "checkpoint_successes.npy"))
        eps_list = np.load(os.path.join(data_dir, "checkpoint_episodes.npy")).tolist()
        plot_checkpoint_curves(eps_list, rewards, steps_arr, successes, fig_dir)

    print(f"\nDone! Figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
