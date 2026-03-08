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

from environments.mujoco.half_cheetah import HalfCheetahAdapter
from core.neural.hierarchical_agent import HierarchicalNeuralSRAgent
from examples.configs import NEURAL_HALF_CHEETAH
from examples.neural_experiment import (
    setup_device, train_two_phase, train_to_checkpoint,
    plot_training_curves, save_training_log, load_training_log,
    plot_checkpoint_curves,
)


# ==================== Reward Shaping ====================

def cheetah_velocity_reward(obs):
    """Dense shaped reward for forward locomotion with uprightness penalty.

    Rewards forward velocity while penalizing non-upright body orientation
    to prevent the belly-slide exploit (flipping and sliding on back).

    Observation layout:
      obs[0] = rootz (torso height, slide joint)
      obs[1] = rooty (body pitch angle in rad, 0 = upright)
      obs[8] = root x-velocity (forward speed)

    Returns:
        Scalar reward.
    """
    x_velocity = float(obs[8])
    body_angle = float(obs[1])  # rooty pitch: 0 = upright, ±π = flipped

    # Base reward: forward velocity
    reward = x_velocity

    # Orientation penalty: penalize tilting away from upright.
    # |angle| ~ 0.3 rad (17°) → penalty ~0.09 (negligible)
    # |angle| ~ 1.0 rad (57°) → penalty ~1.0 (comparable to velocity)
    # |angle| ~ π   rad (180°)→ penalty ~9.9 (dominates velocity reward)
    reward -= 1.0 * body_angle ** 2

    # Penalty for near-zero velocity to discourage static equilibria
    if abs(x_velocity) < 0.3:
        reward -= 0.5

    # Clip to reasonable range
    return float(np.clip(reward, -5.0, 5.0))


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
        use_per=cfg.get("use_per", False),
        per_alpha=cfg.get("per_alpha", 0.6),
        per_beta_start=cfg.get("per_beta_start", 0.4),
        per_beta_end=cfg.get("per_beta_end", 1.0),
        use_episodic_replay=cfg.get("use_episodic_replay", False),
        episodic_replay_episodes=cfg.get("episodic_replay_episodes", 2),
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
    """Two-phase training: diverse exploration then mixed training."""
    train_two_phase(agent, cfg, ep_diverse, ep_fixed)


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

def record_episode_video(agent, cfg, save_path, max_steps=200):
    """Record a short locomotion demo as MP4.

    Uses 200 steps (~6.6s at 30fps) — standard demo length for
    HalfCheetah.  Camera tracks the torso and ground plane is
    extended so the checkered floor stays visible throughout.
    """
    render_adapter = HalfCheetahAdapter(
        n_bins_per_joint=cfg["n_bins_per_joint"],
        render_mode='rgb_array',
        max_episode_steps=max_steps,
    )
    agent.adapter = render_adapter

    # Configure camera to track the cheetah torso so it stays in frame
    try:
        import mujoco
        viewer = render_adapter.env.unwrapped.mujoco_renderer._get_viewer('rgb_array')
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = 1  # torso body
        viewer.cam.distance = 4.5
        viewer.cam.elevation = -15.0
    except Exception:
        pass  # Fall back to default camera if MuJoCo API differs

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

def _plot_checkpoint_curves(episodes_list, avg_rewards, avg_steps, save_dir):
    """Plot evaluation metrics across training checkpoints."""
    plot_checkpoint_curves(
        episodes_list,
        {'Avg Episode Reward': avg_rewards,
         'Avg Episode Steps': avg_steps},
        save_dir, env_name='HalfCheetah')


# ==================== Multi-Checkpoint Experiment ====================

def checkpoint_experiment(cfg, episodes_list, n_runs, n_eval, device='cpu'):
    """Train incrementally, evaluating at each checkpoint.

    Instead of retraining from scratch at each episode budget, trains
    one agent through the full schedule and evaluates at intermediate
    points.  This reduces total training from O(sum(episodes_list))
    to O(max(episodes_list)) per run.
    """
    sorted_eps = sorted(episodes_list)
    max_eps = sorted_eps[-1]
    ep_diverse = int(max_eps * 0.4)
    ep_fixed = max_eps - ep_diverse

    avg_rewards_arr = np.zeros((n_runs, len(episodes_list)))
    avg_steps_arr = np.zeros((n_runs, len(episodes_list)))

    for run in range(n_runs):
        print(f"\n{'='*60}")
        print(f"Run {run+1}/{n_runs}")
        print(f"{'='*60}")

        agent, adapter = create_agent(cfg, device=device)
        trained, boundary_done = 0, False

        for target in sorted_eps:
            print(f"\n--- Checkpoint: {target} episodes ---")
            trained, boundary_done = train_to_checkpoint(
                agent, cfg, target, trained,
                ep_diverse, ep_fixed, boundary_done)

            results = evaluate_agent(agent, cfg, n_eval)
            idx = episodes_list.index(target)
            avg_rewards_arr[run, idx] = np.mean([r['reward'] for r in results])
            avg_steps_arr[run, idx] = np.mean([r['steps'] for r in results])

            print(f"  => Avg reward: {avg_rewards_arr[run, idx]:.1f}, "
                  f"Avg steps: {avg_steps_arr[run, idx]:.1f}")

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

    device = setup_device(args_cli.device)

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

        save_training_log(agent.training_log, data_dir)

        print("\n--- Training Curves ---")
        plot_training_curves(agent.training_log, fig_dir, env_name='HalfCheetah')

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
        _plot_checkpoint_curves(episodes, avg_rewards, avg_steps, fig_dir)

    else:
        print("=" * 60)
        print("PLOTTING FROM SAVED DATA")
        print("=" * 60)
        os.makedirs(fig_dir, exist_ok=True)

    # --- Always try to plot from saved data ---
    training_log = load_training_log(data_dir)
    if training_log:
        print("\n--- Training Curves (from saved data) ---")
        plot_training_curves(training_log, fig_dir, env_name='HalfCheetah')

    ckpt_rewards_path = os.path.join(data_dir, "checkpoint_rewards.npy")
    if os.path.exists(ckpt_rewards_path):
        print("\n--- Checkpoint Curves (from saved data) ---")
        avg_rewards = np.load(ckpt_rewards_path)
        avg_steps = np.load(os.path.join(data_dir, "checkpoint_steps.npy"))
        eps_list = np.load(
            os.path.join(data_dir, "checkpoint_episodes.npy")).tolist()
        _plot_checkpoint_curves(eps_list, avg_rewards, avg_steps, fig_dir)

    print(f"\nDone! Figures saved to {fig_dir}/")


if __name__ == "__main__":
    main()
