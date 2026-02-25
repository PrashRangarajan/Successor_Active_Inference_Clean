"""Neural Successor Feature agent on HalfCheetah-v4 (MuJoCo).

Locomotion task: a 2D cheetah runs forward as fast as possible. Uses the
ActionConditionedSFNetwork to handle 729 discrete actions (3^6 joint grid).

This script:
  - Trains the hierarchical neural SF agent with two-phase training
  - Evaluates from default and diverse starts
  - Records a video of the learned locomotion
  - Plots training curves and evaluation metrics

Usage:
    # Full training + evaluation + video:
    python examples/run_neural_half_cheetah.py --train

    # Quick smoke test:
    python examples/run_neural_half_cheetah.py --train --quick

    # Record video from saved checkpoint:
    python examples/run_neural_half_cheetah.py --video-only

    # Plot from saved data (no training):
    python examples/run_neural_half_cheetah.py
"""

import argparse
import json
import os
import sys
import time

# Use EGL for headless MuJoCo rendering (no X11 display needed)
os.environ.setdefault("MUJOCO_GL", "egl")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import numpy as np

try:
    plt.style.use("seaborn-v0_8-poster")
except OSError:
    plt.style.use("seaborn-poster")

from environments.mujoco.half_cheetah import HalfCheetahAdapter
from core.neural.hierarchical_agent import HierarchicalNeuralSRAgent
from examples.configs import NEURAL_HALF_CHEETAH


# ==================== Reward Shaping ====================

def cheetah_velocity_reward(obs):
    """Dense shaped reward based on forward velocity.

    The native HalfCheetah reward combines velocity and control cost.
    For SF learning, we use a shaping that:
      1. Rewards forward velocity (proportional)
      2. Penalizes near-zero velocity to prevent dead static poses
      3. Penalizes backward motion more heavily

    Observation layout: obs[8] = root x velocity (first element of qvel).

    Returns:
        Scalar reward.
    """
    x_velocity = float(obs[8])
    # Base reward: forward velocity
    reward = x_velocity
    # Penalty for near-zero velocity to discourage static equilibria
    if abs(x_velocity) < 0.3:
        reward -= 0.5
    # Clip to reasonable range
    return float(np.clip(reward, -3.0, 5.0))


# ==================== Agent Factory ====================

def create_agent(cfg, render_mode=None, device='cpu'):
    """Create a fresh hierarchical neural SF agent for HalfCheetah.

    Uses ActionConditionedSFNetwork to handle 729 discrete actions.

    Args:
        cfg: Config dict (NEURAL_HALF_CHEETAH).
        render_mode: 'rgb_array' for video, None for speed.
        device: Torch device ('cpu' or 'cuda').

    Returns:
        (agent, adapter) tuple.
    """
    adapter = HalfCheetahAdapter(
        n_bins_per_joint=cfg["n_bins_per_joint"],
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
        sf_network_cls=cfg["sf_network_cls"],
        n_clusters=4,
        cluster_method='kmeans',
        cluster_on='observations',
        n_cluster_samples=5000,
        adjacency_episodes=500,
        adjacency_episode_length=100,
        device=device,
    )

    # Use velocity-based shaping for richer gradient signal
    agent.set_goal(
        goal_spec=None,
        reward=cfg["reward"],
        default_cost=cfg["default_cost"],
        use_env_reward=False,
        terminal_bonus=cfg.get("terminal_bonus", 0.0),
        reward_shaping_fn=cheetah_velocity_reward,
    )

    return agent, adapter


def train_agent(agent, cfg, ep_diverse, ep_fixed):
    """Two-phase training: diverse exploration then mixed training.

    Phase boundary management:
    - Buffer truncation to remove stale diverse-exploration data
    - Epsilon reset to re-explore under the new start distribution
    - LR warm restart with cosine annealing
    """
    diverse_frac = cfg.get("diverse_fraction", 0.3)

    print(f"Phase 1: Diverse exploration ({ep_diverse} episodes)")
    agent.learn_environment(
        num_episodes=ep_diverse,
        steps_per_episode=cfg["steps_per_episode"],
        diverse_start=True,
        log_interval=max(1, ep_diverse // 5),
    )

    # Phase boundary management
    agent.truncate_buffer(keep_fraction=cfg.get("buffer_keep_phase2", 0.3))
    agent.reset_epsilon(
        new_start=cfg.get("epsilon_phase2_start", 0.3),
        new_decay_steps=cfg.get("epsilon_phase2_decay_steps", 80_000),
    )
    agent.reset_lr(
        sf_lr=cfg["lr"] * cfg.get("lr_phase2_fraction", 0.5),
        rw_lr=cfg["lr_w"] * cfg.get("lr_phase2_fraction", 0.5),
        decay_steps=ep_fixed * cfg["steps_per_episode"],
    )

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
    """Evaluate trained agent on HalfCheetah.

    For locomotion, success is measured by forward distance and velocity,
    not by reaching a goal state.

    Returns:
        List of result dicts.
    """
    test_adapter = HalfCheetahAdapter(
        n_bins_per_joint=cfg["n_bins_per_joint"],
        max_episode_steps=cfg["test_max_steps"],
    )
    agent.adapter = test_adapter

    results = []
    for i in range(n_eval):
        if diverse:
            test_adapter.sample_random_state()
            init_state = test_adapter.get_state_for_reset()
        else:
            init_state = None

        result = agent.run_episode(
            init_state=init_state,
            max_steps=cfg["test_max_steps"],
        )

        # Also capture x_position from info
        final_obs = result.get('final_state', test_adapter.get_current_obs())
        x_vel = float(final_obs[8]) if final_obs is not None else 0.0
        result['final_x_velocity'] = x_vel

        results.append(result)

    return results


def print_eval_results(results, label=""):
    """Pretty-print evaluation results for locomotion."""
    print(f"\n  {label} Results ({len(results)} episodes):")
    for i, r in enumerate(results):
        print(f"    Episode {i+1}: {r['steps']:4d} steps, "
              f"reward={r['reward']:.1f}, "
              f"x_vel={r.get('final_x_velocity', 0):.2f}")

    avg_steps = np.mean([r['steps'] for r in results])
    avg_reward = np.mean([r['reward'] for r in results])
    avg_vel = np.mean([r.get('final_x_velocity', 0) for r in results])
    print(f"\n  Avg steps: {avg_steps:.1f}")
    print(f"  Avg reward: {avg_reward:.1f}")
    print(f"  Avg final x-velocity: {avg_vel:.2f}")

    return avg_reward, avg_steps


# ==================== Video Recording ====================

def record_episode_video(agent, cfg, save_path, max_steps=1000):
    """Record a single locomotion episode as MP4."""
    render_adapter = HalfCheetahAdapter(
        n_bins_per_joint=cfg["n_bins_per_joint"],
        render_mode='rgb_array',
        max_episode_steps=max_steps,
    )
    agent.adapter = render_adapter

    obs = render_adapter.reset()
    frames = [render_adapter.render()]
    x_velocities = []
    total_reward = 0.0

    for step in range(max_steps):
        x_velocities.append(float(obs[8]))  # root x velocity

        action = agent.select_action(obs, greedy=True)
        next_obs, env_reward, terminated, truncated, info = render_adapter.step(action)
        total_reward += env_reward
        frames.append(render_adapter.render())

        if terminated or truncated:
            break
        obs = next_obs

    render_adapter.env.close()

    save_dir = os.path.dirname(save_path) if os.path.dirname(save_path) else '.'
    os.makedirs(save_dir, exist_ok=True)

    # Velocity trajectory plot
    if x_velocities:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x_velocities, color='C0', linewidth=1.5)
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.5,
                    label='Good velocity threshold')
        ax.fill_between(range(len(x_velocities)), 2.0, max(5, max(x_velocities) + 0.5),
                        alpha=0.1, color='green')
        ax.set_xlabel('Step', fontsize=14)
        ax.set_ylabel('Forward Velocity', fontsize=14)
        avg_vel = np.mean(x_velocities)
        ax.set_title(f'HalfCheetah: Neural SF Agent (avg vel={avg_vel:.2f})',
                     fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        traj_path = os.path.join(save_dir, 'velocity_trajectory.png')
        plt.savefig(traj_path, dpi=150)
        plt.close()
        print(f"  Saved {traj_path}")

    # Save video
    try:
        import matplotlib.animation as animation

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')
        img = ax.imshow(frames[0])

        def update(frame_idx):
            img.set_data(frames[frame_idx])
            ax.set_title(f"Step {frame_idx}/{len(frames)-1}", fontsize=14)
            return [img]

        anim = animation.FuncAnimation(
            fig, update, frames=len(frames), interval=33, blit=True)
        anim.save(save_path, writer='ffmpeg', fps=30, dpi=100)
        plt.close(fig)
        print(f"  Saved video to {save_path}")
    except Exception as e:
        print(f"  Warning: Could not save video ({e})")
        try:
            from PIL import Image
            gif_path = save_path.replace('.mp4', '.gif')
            pil_frames = [Image.fromarray(f) for f in frames[::3]]
            pil_frames[0].save(
                gif_path, save_all=True, append_images=pil_frames[1:],
                duration=100, loop=0)
            print(f"  Saved GIF to {gif_path}")
        except ImportError:
            np_path = save_path.replace('.mp4', '_frames.npy')
            np.save(np_path, np.array(frames))
            print(f"  Saved raw frames to {np_path}")

    return {
        'steps': len(x_velocities),
        'reward': total_reward,
        'avg_velocity': float(np.mean(x_velocities)) if x_velocities else 0.0,
        'n_frames': len(frames),
    }


# ==================== Plotting ====================

def plot_training_curves(training_log, save_dir):
    """Plot training curves."""
    os.makedirs(save_dir, exist_ok=True)

    for key, ylabel, title, color, use_log in [
        ('episode_reward', 'Episode Reward', 'Neural SF — HalfCheetah Training Reward', 'C0', False),
        ('sf_loss', 'SF TD Loss', 'Successor Feature Loss', 'C1', True),
        ('reward_loss', 'Reward Prediction Loss', 'Reward Weight (w) Learning', 'C2', True),
        ('episode_steps', 'Steps per Episode', 'Episode Length During Training', 'C3', False),
    ]:
        fig, ax = plt.subplots(figsize=(12, 6))
        data = training_log.get(key, [])
        if not data:
            plt.close()
            continue
        ax.plot(data, alpha=0.2 if len(data) > 500 else 0.5,
                color=color, linewidth=0.5)
        window = min(100 if 'episode' in key else 500,
                     len(data) // 10 + 1)
        if window > 1 and len(data) >= window:
            smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
            ax.plot(np.arange(window-1, window-1+len(smoothed)), smoothed,
                    color=color, linewidth=2, label=f'Smoothed ({window})')
        ax.set_xlabel('Episode' if 'episode' in key else 'Training Step', fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(title, fontsize=18)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_yscale('log')
        plt.tight_layout()
        fname = f"{key.replace('episode_', 'training_') if 'episode' in key else key}.png"
        path = os.path.join(save_dir, fname)
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")


def plot_checkpoint_curves(episodes_list, avg_rewards, avg_steps, save_dir):
    """Plot evaluation metrics across training checkpoints."""
    os.makedirs(save_dir, exist_ok=True)
    n_runs = avg_rewards.shape[0]

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

    _plot(avg_rewards, 'Avg Episode Reward',
          'Neural SF — HalfCheetah Reward',
          'checkpoint_reward.png')
    _plot(avg_steps, 'Avg Episode Steps',
          'Neural SF — HalfCheetah Episode Length',
          'checkpoint_steps.png')


# ==================== Multi-Checkpoint Experiment ====================

def checkpoint_experiment(cfg, episodes_list, n_runs, n_eval, device='cpu'):
    """Train across multiple training budgets and evaluate each."""
    n_trials = len(episodes_list)

    avg_rewards_arr = np.zeros((n_runs, n_trials))
    avg_steps_arr = np.zeros((n_runs, n_trials))

    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Run {run+1}/{n_runs}")
        print(f"{'='*60}")

        for trial, total_eps in enumerate(episodes_list):
            ep_diverse = int(total_eps * 0.4)
            ep_fixed = total_eps - ep_diverse

            print(f"\n--- Checkpoint: {total_eps} episodes "
                  f"(diverse={ep_diverse}, fixed={ep_fixed}) ---")

            agent, adapter = create_agent(cfg, device=device)
            train_agent(agent, cfg, ep_diverse, ep_fixed)

            results = evaluate_agent(agent, cfg, n_eval)
            avg_reward = np.mean([r['reward'] for r in results])
            avg_steps = np.mean([r['steps'] for r in results])

            avg_rewards_arr[run, trial] = avg_reward
            avg_steps_arr[run, trial] = avg_steps

            print(f"  => Avg reward: {avg_reward:.1f}, Avg steps: {avg_steps:.1f}")

    return avg_rewards_arr, avg_steps_arr


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(
        description="Neural SF HalfCheetah — Train, Evaluate, Record")
    parser.add_argument("--train", action="store_true",
                        help="Run training experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run with fewer episodes")
    parser.add_argument("--video-only", action="store_true",
                        help="Only record video from checkpoint")
    parser.add_argument("--n-runs", type=int, default=None)
    parser.add_argument("--n-eval", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu",
                        help="Torch device ('cpu' or 'cuda')")
    args_cli = parser.parse_args()

    import torch
    device = args_cli.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = 'cpu'

    cfg = NEURAL_HALF_CHEETAH

    data_dir = "data/eval/neural_half_cheetah"
    fig_dir = args_cli.save_dir or "figures/eval/neural_half_cheetah"
    checkpoint_dir = "data/neural_half_cheetah"

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

        agent, _ = create_agent(cfg, device=device)
        ckpt = args_cli.checkpoint or os.path.join(
            checkpoint_dir, "checkpoint.pt")
        if not os.path.exists(ckpt):
            print(f"Checkpoint not found at {ckpt}")
            return
        agent.load(ckpt)
        print(f"Loaded checkpoint from {ckpt}")

        os.makedirs(fig_dir, exist_ok=True)
        video_result = record_episode_video(
            agent, cfg,
            os.path.join(fig_dir, 'half_cheetah_episode.mp4'),
            max_steps=cfg["test_max_steps"])
        print(f"  Episode: {video_result['steps']} steps, "
              f"reward={video_result['reward']:.1f}, "
              f"avg_vel={video_result['avg_velocity']:.2f}")
        return

    # ==================== Full Training Run ====================
    if args_cli.train:
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)

        print("=" * 60)
        print("NEURAL SF HALFCHEETAH")
        print("=" * 60)
        n_actions = cfg["n_bins_per_joint"] ** 6
        print(f"Config: sf_dim={cfg['sf_dim']}, hidden={cfg['hidden_sizes']}, "
              f"n_bins={cfg['n_bins_per_joint']} ({n_actions} actions)")
        print(f"Network: {cfg['sf_network_cls']}")
        print(f"Checkpoints: {episodes}")
        print(f"Runs: {n_runs}, Eval episodes per checkpoint: {n_eval}")

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
            ep_diverse = 100
            ep_fixed = 200

        agent, adapter = create_agent(cfg, device=device)
        print(f"Device: {device}")
        t0 = time.time()
        train_agent(agent, cfg, ep_diverse, ep_fixed)
        train_time = time.time() - t0
        print(f"\nTraining time: {train_time:.1f}s")

        os.makedirs(checkpoint_dir, exist_ok=True)
        agent.save(os.path.join(checkpoint_dir, "checkpoint.pt"))
        print(f"Saved checkpoint")

        for key, vals in agent.training_log.items():
            np.save(os.path.join(data_dir, f"training_{key}.npy"),
                    np.array(vals))

        print("\n--- Training Curves ---")
        plot_training_curves(agent.training_log, fig_dir)

        print("\n--- Default Start Evaluation ---")
        results_default = evaluate_agent(agent, cfg, n_eval, diverse=False)
        print_eval_results(results_default, "Default Start")

        print("\n--- Diverse Start Evaluation ---")
        results_diverse = evaluate_agent(agent, cfg, n_eval, diverse=True)
        print_eval_results(results_diverse, "Diverse Start")

        print("\n--- Video Recording ---")
        video_result = record_episode_video(
            agent, cfg,
            os.path.join(fig_dir, 'half_cheetah_episode.mp4'),
            max_steps=cfg["test_max_steps"])
        print(f"  Video: {video_result['steps']} steps, "
              f"reward={video_result['reward']:.1f}, "
              f"avg_vel={video_result['avg_velocity']:.2f}")

        # --- Multi-checkpoint experiment ---
        print("\n" + "=" * 60)
        print("PHASE B: Checkpoint evaluation (learning curves)")
        print("=" * 60)

        t0 = time.time()
        avg_rewards, avg_steps = checkpoint_experiment(
            cfg, episodes, n_runs, n_eval, device=device)
        exp_time = time.time() - t0
        print(f"\nCheckpoint experiment: {exp_time:.0f}s")

        np.save(os.path.join(data_dir, "checkpoint_rewards.npy"), avg_rewards)
        np.save(os.path.join(data_dir, "checkpoint_steps.npy"), avg_steps)
        np.save(os.path.join(data_dir, "checkpoint_episodes.npy"),
                np.array(episodes))

        print("\n--- Checkpoint Curves ---")
        plot_checkpoint_curves(episodes, avg_rewards, avg_steps, fig_dir)

    else:
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

    ckpt_rewards_path = os.path.join(data_dir, "checkpoint_rewards.npy")
    if os.path.exists(ckpt_rewards_path):
        print("\n--- Checkpoint Curves (from saved data) ---")
        avg_rewards = np.load(ckpt_rewards_path)
        avg_steps = np.load(os.path.join(data_dir, "checkpoint_steps.npy"))
        eps_list = np.load(
            os.path.join(data_dir, "checkpoint_episodes.npy")).tolist()
        plot_checkpoint_curves(eps_list, avg_rewards, avg_steps, fig_dir)

    print(f"\nDone! Figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
