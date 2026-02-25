"""Neural Successor Feature agent on InvertedPendulum-v4 (MuJoCo).

First MuJoCo environment for the neural SF framework. The InvertedPendulum
is a balancing task: the agent applies horizontal forces to keep a pole
upright on a cart. The env gives +1 reward per timestep (survival reward),
terminating when the pole falls.

This script:
  - Trains the hierarchical neural SF agent with two-phase training
  - Evaluates from default and diverse starts
  - Records a video of a successful balancing episode
  - Plots training curves and evaluation metrics

Usage:
    # Full training + evaluation + video:
    python examples/run_neural_inverted_pendulum.py --train

    # Quick smoke test:
    python examples/run_neural_inverted_pendulum.py --train --quick

    # Record video from saved checkpoint:
    python examples/run_neural_inverted_pendulum.py --video-only

    # Plot from saved data (no training):
    python examples/run_neural_inverted_pendulum.py
"""

import argparse
import json
import os
import sys
import time

# Ensure imports resolve from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np

try:
    plt.style.use("seaborn-v0_8-poster")
except OSError:
    plt.style.use("seaborn-poster")

from environments.mujoco.inverted_pendulum import InvertedPendulumAdapter
from core.neural.hierarchical_agent import HierarchicalNeuralSRAgent
from examples.configs import NEURAL_INVERTED_PENDULUM


# ==================== Reward Shaping ====================

def pendulum_balance_reward(obs):
    """Dense shaped reward for InvertedPendulum balance task.

    The raw env reward (+1/step) is flat — every state looks equally good
    until the pole falls. This shaping provides gradient: states where the
    pole is upright and centered score high, tilted states low.

    Observation layout: [x, theta, x_dot, omega]

    Returns:
        Scalar in roughly [-1, 1].
    """
    x = float(obs[0])          # cart position
    theta = float(obs[1])      # pole angle (0 = upright)
    return float(np.clip(1.0 - 5.0 * abs(theta) - 0.1 * abs(x), -1.0, 1.0))


# ==================== Agent Factory ====================

def create_agent(cfg, render_mode=None):
    """Create a fresh hierarchical neural SF agent for InvertedPendulum.

    Args:
        cfg: Config dict (NEURAL_INVERTED_PENDULUM).
        render_mode: 'rgb_array' for video, None for speed.

    Returns:
        (agent, adapter) tuple.
    """
    adapter = InvertedPendulumAdapter(
        n_force_bins=cfg["n_force_bins"],
        render_mode=render_mode,
        max_episode_steps=cfg["steps_per_episode"],
    )

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
        n_cluster_samples=3000,
        adjacency_episodes=300,
        adjacency_episode_length=100,
    )

    # InvertedPendulum env reward (+1/step) is too flat for SF learning.
    # Use shaped reward: high for upright+centered, low for tilted.
    # Terminal bonus is negative (penalty) because termination = failure.
    agent.set_goal(
        goal_spec=None,
        reward=cfg["reward"],
        default_cost=cfg["default_cost"],
        use_env_reward=False,
        terminal_bonus=cfg.get("terminal_bonus", 0.0),
        reward_shaping_fn=pendulum_balance_reward,
    )

    return agent, adapter


def train_agent(agent, cfg, ep_diverse, ep_fixed):
    """Two-phase training: diverse exploration then mixed training."""
    diverse_frac = cfg.get("diverse_fraction", 0.3)

    # Phase 1: Diverse exploration — SF representation learning
    print(f"Phase 1: Diverse exploration ({ep_diverse} episodes)")
    agent.learn_environment(
        num_episodes=ep_diverse,
        steps_per_episode=cfg["steps_per_episode"],
        diverse_start=True,
        log_interval=max(1, ep_diverse // 5),
    )

    # Phase 2: Mixed training — refine policy
    print(f"\nPhase 2: Mixed training ({ep_fixed} episodes, "
          f"{diverse_frac:.0%} diverse)")
    agent.learn_environment(
        num_episodes=ep_fixed,
        steps_per_episode=cfg["steps_per_episode"],
        diverse_start=True,
        diverse_fraction=diverse_frac,
        log_interval=max(1, ep_fixed // 5),
    )


# ==================== Evaluation ====================

def evaluate_agent(agent, cfg, n_eval=10, diverse=False):
    """Evaluate trained agent.

    Args:
        agent: Trained agent.
        cfg: Config dict.
        n_eval: Number of evaluation episodes.
        diverse: If True, evaluate from diverse starts.

    Returns:
        List of result dicts.
    """
    # Create fresh test adapter (no render, standard max steps)
    test_adapter = InvertedPendulumAdapter(
        n_force_bins=cfg["n_force_bins"],
        max_episode_steps=cfg["test_max_steps"],
    )
    agent.adapter = test_adapter

    results = []
    for i in range(n_eval):
        if diverse:
            # Start from a random state
            init_obs = test_adapter.sample_random_state()
            init_state = test_adapter.get_state_for_reset()
        else:
            init_state = None

        result = agent.run_episode(
            init_state=init_state,
            max_steps=cfg["test_max_steps"],
        )
        results.append(result)

    return results


def print_eval_results(results, label=""):
    """Pretty-print evaluation results."""
    n_survived = sum(1 for r in results if not r.get('reached_goal', True)
                     and r['steps'] >= 999)
    # For InvertedPendulum, "success" = surviving the full episode
    # reached_goal=True means env terminated (pole fell) which is bad
    # reached_goal=False with max steps means the agent balanced successfully
    survived = [r for r in results if r['steps'] >= 950]

    print(f"\n  {label} Results ({len(results)} episodes):")
    for i, r in enumerate(results):
        status = "BALANCED" if r['steps'] >= 950 else f"FELL at step {r['steps']}"
        print(f"    Episode {i+1}: {r['steps']:4d} steps, "
              f"reward={r['reward']:.1f}, {status}")

    avg_steps = np.mean([r['steps'] for r in results])
    avg_reward = np.mean([r['reward'] for r in results])
    survival_rate = len(survived) / len(results)
    print(f"\n  Survival rate (>950 steps): {survival_rate:.0%}")
    print(f"  Avg steps: {avg_steps:.1f}")
    print(f"  Avg reward: {avg_reward:.1f}")

    return survival_rate, avg_steps, avg_reward


# ==================== Video Recording ====================

def record_episode_video(agent, cfg, save_path, max_steps=1000):
    """Record a single episode as an MP4 video.

    Args:
        agent: Trained agent.
        cfg: Config dict.
        save_path: Path to save the .mp4 file.
        max_steps: Maximum steps for the recorded episode.

    Returns:
        Result dict from the episode.
    """
    render_adapter = InvertedPendulumAdapter(
        n_force_bins=cfg["n_force_bins"],
        render_mode='rgb_array',
        max_episode_steps=max_steps,
    )
    agent.adapter = render_adapter

    obs = render_adapter.reset()
    frames = [render_adapter.render()]
    thetas = []
    positions = []
    total_reward = 0.0

    for step in range(max_steps):
        thetas.append(float(obs[1]))      # pole angle (obs layout: x, theta, x_dot, omega)
        positions.append(float(obs[0]))   # cart position

        action = agent.select_action(obs, greedy=True)
        next_obs, env_reward, terminated, truncated, info = render_adapter.step(action)
        total_reward += env_reward
        frames.append(render_adapter.render())

        if terminated or truncated:
            break
        obs = next_obs

    render_adapter.env.close()
    survived = len(thetas) >= 950

    # Save trajectory plots
    save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
    os.makedirs(save_dir, exist_ok=True)

    if thetas:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Pole angle
        axes[0].plot(thetas, color='C0', linewidth=1.5)
        axes[0].axhline(y=0.2, color='red', linestyle='--', alpha=0.7,
                        label='Termination threshold')
        axes[0].axhline(y=-0.2, color='red', linestyle='--', alpha=0.7)
        axes[0].fill_between(range(len(thetas)), -0.1, 0.1,
                             alpha=0.1, color='green', label='Goal region')
        axes[0].set_ylabel('Pole Angle (rad)', fontsize=14)
        axes[0].set_title(
            f'InvertedPendulum: Neural SF Agent '
            f'({"balanced {0} steps".format(len(thetas)) if survived else "fell at step {0}".format(len(thetas))})',
            fontsize=16,
        )
        axes[0].legend(fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Cart position
        axes[1].plot(positions, color='C1', linewidth=1.5)
        axes[1].axhline(y=0.0, color='gray', linestyle='-', alpha=0.3)
        axes[1].set_xlabel('Step', fontsize=14)
        axes[1].set_ylabel('Cart Position', fontsize=14)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        traj_path = os.path.join(save_dir, 'pendulum_trajectory.png')
        plt.savefig(traj_path, dpi=150)
        plt.close()
        print(f"  Saved {traj_path}")

    # Save as MP4
    try:
        import matplotlib.animation as animation

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis('off')
        img = ax.imshow(frames[0])

        def update(frame_idx):
            img.set_data(frames[frame_idx])
            ax.set_title(
                f"Step {frame_idx}/{len(frames)-1}"
                f"{'  BALANCED!' if frame_idx == len(frames)-1 and survived else ''}",
                fontsize=14,
            )
            return [img]

        anim = animation.FuncAnimation(
            fig, update, frames=len(frames), interval=50, blit=True,
        )
        anim.save(save_path, writer='ffmpeg', fps=30, dpi=100)
        plt.close(fig)
        print(f"  Saved video to {save_path}")
    except Exception as e:
        print(f"  Warning: Could not save video ({e})")
        try:
            from PIL import Image
            gif_path = save_path.replace('.mp4', '.gif')
            pil_frames = [Image.fromarray(f) for f in frames[::3]]  # subsample
            pil_frames[0].save(
                gif_path, save_all=True, append_images=pil_frames[1:],
                duration=100, loop=0,
            )
            print(f"  Saved GIF to {gif_path}")
        except ImportError:
            np_path = save_path.replace('.mp4', '_frames.npy')
            np.save(np_path, np.array(frames))
            print(f"  Saved raw frames to {np_path}")

    return {
        'steps': len(thetas),
        'reward': total_reward,
        'survived': survived,
        'n_frames': len(frames),
    }


# ==================== Plotting ====================

def plot_training_curves(training_log, save_dir):
    """Plot training curves from the agent's training log."""
    os.makedirs(save_dir, exist_ok=True)

    # --- Episode Reward ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ep_rewards = training_log['episode_reward']
    ax.plot(ep_rewards, alpha=0.3, color='C0', linewidth=0.5)
    window = min(100, len(ep_rewards) // 10 + 1)
    if window > 1 and len(ep_rewards) >= window:
        smoothed = np.convolve(ep_rewards, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, window-1+len(smoothed)), smoothed,
                color='C0', linewidth=2, label=f'Smoothed ({window}-ep)')
    ax.set_xlabel('Episode', fontsize=16)
    ax.set_ylabel('Episode Reward (survival)', fontsize=16)
    ax.set_title('Neural SF — InvertedPendulum Training Reward', fontsize=18)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_reward.png'), dpi=150)
    plt.close()
    print(f"  Saved {os.path.join(save_dir, 'training_reward.png')}")

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
    plt.savefig(os.path.join(save_dir, 'sf_loss.png'), dpi=150)
    plt.close()
    print(f"  Saved {os.path.join(save_dir, 'sf_loss.png')}")

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
    plt.savefig(os.path.join(save_dir, 'reward_loss.png'), dpi=150)
    plt.close()
    print(f"  Saved {os.path.join(save_dir, 'reward_loss.png')}")

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
    ax.set_ylabel('Steps per Episode (higher = better)', fontsize=16)
    ax.set_title('Episode Length During Training', fontsize=18)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'episode_steps.png'), dpi=150)
    plt.close()
    print(f"  Saved {os.path.join(save_dir, 'episode_steps.png')}")


def plot_checkpoint_curves(episodes_list, survival_rates, avg_steps,
                           avg_rewards, save_dir):
    """Plot evaluation metrics across training checkpoints."""
    os.makedirs(save_dir, exist_ok=True)
    n_runs = survival_rates.shape[0]

    def _plot(data, ylabel, title, filename, ylim=None):
        fig, ax = plt.subplots(figsize=(12, 7))
        mean = np.mean(data, axis=0)
        sem = np.std(data, axis=0) / np.sqrt(n_runs)
        ax.plot(episodes_list, mean, 'o-', color='C0', linewidth=2, markersize=8)
        ax.fill_between(episodes_list, mean - sem, mean + sem,
                        alpha=0.3, color='C0')
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

    _plot(survival_rates, 'Survival Rate',
          'Neural SF — InvertedPendulum Survival Rate',
          'checkpoint_survival_rate.png', ylim=(-0.05, 1.05))
    _plot(avg_steps, 'Avg Steps (higher = better)',
          'Neural SF — Average Episode Length',
          'checkpoint_steps.png')
    _plot(avg_rewards, 'Avg Reward',
          'Neural SF — Average Episode Reward',
          'checkpoint_reward.png')


# ==================== Multi-Checkpoint Experiment ====================

def checkpoint_experiment(cfg, episodes_list, n_runs, n_eval):
    """Train across multiple training budgets and evaluate each.

    Returns:
        (survival_rates, avg_steps, avg_rewards, episodes_list)
    """
    n_trials = len(episodes_list)

    survival_rates = np.zeros((n_runs, n_trials))
    avg_steps_arr = np.zeros((n_runs, n_trials))
    avg_rewards_arr = np.zeros((n_runs, n_trials))

    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Run {run+1}/{n_runs}")
        print(f"{'='*60}")

        for trial, total_eps in enumerate(episodes_list):
            ep_diverse = int(total_eps * 0.4)
            ep_fixed = total_eps - ep_diverse

            print(f"\n--- Checkpoint: {total_eps} episodes "
                  f"(diverse={ep_diverse}, fixed={ep_fixed}) ---")

            agent, adapter = create_agent(cfg)
            train_agent(agent, cfg, ep_diverse, ep_fixed)

            results = evaluate_agent(agent, cfg, n_eval)

            # For InvertedPendulum: survival = steps >= 950
            survived = [r for r in results if r['steps'] >= 950]
            survival_rate = len(survived) / len(results)
            avg_steps = np.mean([r['steps'] for r in results])
            avg_reward = np.mean([r['reward'] for r in results])

            survival_rates[run, trial] = survival_rate
            avg_steps_arr[run, trial] = avg_steps
            avg_rewards_arr[run, trial] = avg_reward

            print(f"  => Survival: {survival_rate:.0%}, "
                  f"Avg steps: {avg_steps:.1f}, Avg reward: {avg_reward:.1f}")

    return survival_rates, avg_steps_arr, avg_rewards_arr


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Neural SF InvertedPendulum — Train, Evaluate, Record")
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

    cfg = NEURAL_INVERTED_PENDULUM

    data_dir = "data/eval/neural_inverted_pendulum"
    fig_dir = args_cli.save_dir or "figures/eval/neural_inverted_pendulum"
    checkpoint_dir = "data/neural_inverted_pendulum"

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

        agent, _ = create_agent(cfg)
        ckpt = args_cli.checkpoint or os.path.join(
            checkpoint_dir, "checkpoint.pt")
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found at {ckpt}")
            print("Run with --train first, or provide --checkpoint path.")
            return
        agent.load(ckpt)
        print(f"Loaded checkpoint from {ckpt}")

        os.makedirs(fig_dir, exist_ok=True)
        video_result = record_episode_video(
            agent, cfg,
            os.path.join(fig_dir, 'inverted_pendulum_episode.mp4'),
            max_steps=cfg["test_max_steps"],
        )
        status = "BALANCED" if video_result['survived'] else "FELL"
        print(f"  Episode: {video_result['steps']} steps, "
              f"reward={video_result['reward']:.1f}, {status}")
        return

    # ==================== Full Training Run ====================
    if args_cli.train:
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)

        print("=" * 60)
        print("NEURAL SF INVERTED PENDULUM")
        print("=" * 60)
        print(f"Config: sf_dim={cfg['sf_dim']}, hidden={cfg['hidden_sizes']}, "
              f"n_force_bins={cfg['n_force_bins']}")
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

        # --- Single full training run ---
        print("\n" + "=" * 60)
        print("PHASE A: Full training run")
        print("=" * 60)

        ep_diverse = cfg["train_episodes_diverse"]
        ep_fixed = cfg["train_episodes_fixed"]
        if args_cli.quick:
            ep_diverse = 200
            ep_fixed = 300

        agent, adapter = create_agent(cfg)
        t0 = time.time()
        train_agent(agent, cfg, ep_diverse, ep_fixed)
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

        # Evaluate the fully trained agent
        print("\n--- Default Start Evaluation ---")
        results_default = evaluate_agent(agent, cfg, n_eval, diverse=False)
        print_eval_results(results_default, "Default Start")

        print("\n--- Diverse Start Evaluation ---")
        results_diverse = evaluate_agent(agent, cfg, n_eval, diverse=True)
        print_eval_results(results_diverse, "Diverse Start")

        # Record video
        print("\n--- Video Recording ---")
        video_result = record_episode_video(
            agent, cfg,
            os.path.join(fig_dir, 'inverted_pendulum_episode.mp4'),
            max_steps=cfg["test_max_steps"],
        )
        status = "BALANCED" if video_result['survived'] else "FELL"
        print(f"  Video: {video_result['steps']} steps, "
              f"reward={video_result['reward']:.1f}, {status}")

        # --- Multi-checkpoint experiment ---
        print("\n" + "=" * 60)
        print("PHASE B: Checkpoint evaluation (learning curves)")
        print("=" * 60)

        t0 = time.time()
        survival_rates, avg_steps_arr, avg_rewards_arr = checkpoint_experiment(
            cfg, episodes, n_runs, n_eval)
        exp_time = time.time() - t0
        print(f"\nCheckpoint experiment completed in {exp_time:.0f}s")

        # Save data
        np.save(os.path.join(data_dir, "checkpoint_survival.npy"), survival_rates)
        np.save(os.path.join(data_dir, "checkpoint_steps.npy"), avg_steps_arr)
        np.save(os.path.join(data_dir, "checkpoint_rewards.npy"), avg_rewards_arr)
        np.save(os.path.join(data_dir, "checkpoint_episodes.npy"),
                np.array(episodes))
        print(f"Saved checkpoint data to {data_dir}/")

        # Plot checkpoint curves
        print("\n--- Checkpoint Curves ---")
        plot_checkpoint_curves(
            episodes, survival_rates, avg_steps_arr, avg_rewards_arr, fig_dir)

    else:
        # ==================== Plot from Saved Data ====================
        print("=" * 60)
        print("PLOTTING FROM SAVED DATA")
        print("=" * 60)
        os.makedirs(fig_dir, exist_ok=True)

    # --- Always try to plot from saved data ---
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

    ckpt_survival_path = os.path.join(data_dir, "checkpoint_survival.npy")
    if os.path.exists(ckpt_survival_path):
        print("\n--- Checkpoint Curves (from saved data) ---")
        survival_rates = np.load(ckpt_survival_path)
        avg_steps_arr = np.load(os.path.join(data_dir, "checkpoint_steps.npy"))
        avg_rewards_arr = np.load(os.path.join(data_dir, "checkpoint_rewards.npy"))
        eps_list = np.load(
            os.path.join(data_dir, "checkpoint_episodes.npy")).tolist()
        plot_checkpoint_curves(
            eps_list, survival_rates, avg_steps_arr, avg_rewards_arr, fig_dir)

    print(f"\nDone! Figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
