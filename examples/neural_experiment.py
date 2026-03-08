"""Shared utilities for neural SF experiment scripts.

Provides common functionality used across run_neural_*.py scripts:
device selection, training loop helpers, plotting, and data I/O.

This module extracts duplicated code that was previously copy-pasted
across the five neural example scripts.
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    plt.style.use("seaborn-v0_8-poster")
except OSError:
    plt.style.use("seaborn-poster")


def setup_device(requested: str = 'cpu') -> str:
    """Resolve torch device with CUDA fallback.

    Args:
        requested: Requested device ('cpu' or 'cuda').

    Returns:
        Resolved device string.
    """
    import torch
    if requested == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        return 'cpu'
    return requested


def _apply_phase_boundary(agent, cfg: dict, ep_fixed: int):
    """Apply Phase 1 to Phase 2 boundary: truncate buffer, reset epsilon/LR."""
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


def train_two_phase(agent, cfg: dict, ep_diverse: int, ep_fixed: int,
                    phase_boundary: bool = True):
    """Two-phase training with optional phase boundary management.

    Phase 1: Diverse exploration (100% diverse starts).
    Phase 2: Mixed training (diverse_fraction from config).

    Between phases (if phase_boundary=True): truncate buffer,
    reset epsilon, reset learning rate.

    Args:
        agent: NeuralSRAgent or HierarchicalNeuralSRAgent.
        cfg: Config dict with standard training keys.
        ep_diverse: Episodes for phase 1.
        ep_fixed: Episodes for phase 2.
        phase_boundary: If True, apply buffer/epsilon/LR reset between phases.
    """
    diverse_frac = cfg.get("diverse_fraction", 0.3)

    print(f"Phase 1: Diverse exploration ({ep_diverse} episodes)")
    agent.learn_environment(
        num_episodes=ep_diverse,
        steps_per_episode=cfg["steps_per_episode"],
        diverse_start=True,
        log_interval=max(1, ep_diverse // 5),
    )

    if phase_boundary:
        _apply_phase_boundary(agent, cfg, ep_fixed)

    print(f"\nPhase 2: Mixed training ({ep_fixed} episodes, "
          f"{diverse_frac:.0%} diverse)")
    agent.learn_environment(
        num_episodes=ep_fixed,
        steps_per_episode=cfg["steps_per_episode"],
        diverse_start=True,
        diverse_fraction=diverse_frac,
        log_interval=max(1, ep_fixed // 5),
    )


def train_to_checkpoint(agent, cfg: dict, target: int, trained: int,
                        ep_diverse: int, ep_fixed: int,
                        phase_boundary_done: bool):
    """Train agent incrementally from ``trained`` to ``target`` episodes.

    Handles the Phase 1 (diverse) to Phase 2 (mixed) boundary
    automatically. Call this repeatedly with increasing ``target``
    values to build a learning curve without retraining from scratch.

    Args:
        agent: Neural SR agent.
        cfg: Config dict with standard training keys.
        target: Target total episodes to reach.
        trained: Episodes already completed.
        ep_diverse: Total Phase 1 (diverse) episodes for the full run.
        ep_fixed: Total Phase 2 (mixed) episodes for the full run.
        phase_boundary_done: Whether boundary reset has been applied.

    Returns:
        (trained, phase_boundary_done) updated state tuple.
    """
    remaining = target - trained
    if remaining <= 0:
        return trained, phase_boundary_done

    diverse_frac = cfg.get("diverse_fraction", 0.3)

    # Phase 1 portion
    if trained < ep_diverse:
        p1 = min(remaining, ep_diverse - trained)
        print(f"  Phase 1: +{p1} diverse (total -> {trained + p1})")
        agent.learn_environment(
            num_episodes=p1,
            steps_per_episode=cfg["steps_per_episode"],
            diverse_start=True,
            log_interval=max(1, p1 // 5),
        )
        trained += p1
        remaining -= p1

        if trained >= ep_diverse and not phase_boundary_done:
            phase_boundary_done = True
            _apply_phase_boundary(agent, cfg, ep_fixed)

    # Phase 2 portion
    if remaining > 0:
        if not phase_boundary_done:
            phase_boundary_done = True
            _apply_phase_boundary(agent, cfg, ep_fixed)

        print(f"  Phase 2: +{remaining} mixed (total -> {trained + remaining})")
        agent.learn_environment(
            num_episodes=remaining,
            steps_per_episode=cfg["steps_per_episode"],
            diverse_start=True,
            diverse_fraction=diverse_frac,
            log_interval=max(1, remaining // 5),
        )
        trained += remaining

    return trained, phase_boundary_done


def plot_training_curves(training_log: dict, save_dir: str,
                         env_name: str = ""):
    """Plot the 4 standard training curves.

    Produces: episode_reward, sf_loss, reward_loss, episode_steps.

    Args:
        training_log: Agent's training_log dict.
        save_dir: Directory to save PNG plots.
        env_name: Environment name for plot titles.
    """
    os.makedirs(save_dir, exist_ok=True)

    title_prefix = f"Neural SF — {env_name} " if env_name else "Neural SF — "
    for key, ylabel, title, color, use_log in [
        ('episode_reward', 'Episode Reward',
         f'{title_prefix}Training Reward', 'C0', False),
        ('sf_loss', 'SF TD Loss', 'Successor Feature Loss', 'C1', True),
        ('reward_loss', 'Reward Prediction Loss',
         'Reward Weight (w) Learning', 'C2', True),
        ('episode_steps', 'Steps per Episode',
         'Episode Length During Training', 'C3', False),
    ]:
        data = training_log.get(key, [])
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data, alpha=0.2 if len(data) > 500 else 0.5,
                color=color, linewidth=0.5)
        window = min(100 if 'episode' in key else 500,
                     len(data) // 10 + 1)
        if window > 1 and len(data) >= window:
            smoothed = np.convolve(
                data, np.ones(window) / window, mode='valid')
            ax.plot(np.arange(window - 1, window - 1 + len(smoothed)),
                    smoothed, color=color, linewidth=2,
                    label=f'Smoothed ({window})')
        ax.set_xlabel(
            'Episode' if 'episode' in key else 'Training Step', fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(title, fontsize=18)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        if use_log:
            ax.set_yscale('log')
        plt.tight_layout()
        base = key.replace('episode_', 'training_') if 'episode' in key else key
        path = os.path.join(save_dir, f"{base}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")


def save_training_log(training_log: dict, save_dir: str):
    """Save training log arrays as .npy files.

    Args:
        training_log: Agent's training_log dict.
        save_dir: Directory to save to.
    """
    os.makedirs(save_dir, exist_ok=True)
    for key, vals in training_log.items():
        if vals:
            np.save(os.path.join(save_dir, f"training_{key}.npy"),
                    np.array(vals))


def load_training_log(data_dir: str) -> Optional[dict]:
    """Load training log from saved .npy files.

    Returns:
        Training log dict, or None if not all files found.
    """
    keys = ['sf_loss', 'reward_loss', 'episode_reward', 'episode_steps']
    log = {}
    for key in keys:
        path = os.path.join(data_dir, f"training_{key}.npy")
        if os.path.exists(path):
            log[key] = np.load(path).tolist()
        else:
            return None
    return log


def plot_checkpoint_curves(episodes_list: List[int],
                           metrics: Dict[str, np.ndarray],
                           save_dir: str,
                           env_name: str = ""):
    """Plot evaluation metrics across training checkpoints.

    Args:
        episodes_list: List of episode counts (x-axis).
        metrics: Dict mapping metric_name -> (n_runs, n_checkpoints) array.
            Each entry produces one plot.
        save_dir: Directory to save plots.
        env_name: Environment name for plot titles.
    """
    os.makedirs(save_dir, exist_ok=True)

    for name, data in metrics.items():
        n_runs = data.shape[0]
        fig, ax = plt.subplots(figsize=(12, 7))
        mean = np.mean(data, axis=0)
        sem = np.std(data, axis=0) / np.sqrt(n_runs)
        ax.plot(episodes_list, mean, 'o-', color='C0',
                linewidth=2, markersize=8)
        ax.fill_between(episodes_list, mean - sem, mean + sem,
                        alpha=0.3, color='C0')
        ax.set_xlabel('Training Episodes', fontsize=18)
        ax.set_ylabel(name, fontsize=18)
        title_prefix = f"Neural SF — {env_name} " if env_name else ""
        ax.set_title(f'{title_prefix}{name}', fontsize=20)
        ax.tick_params(labelsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        path = os.path.join(save_dir, f"checkpoint_{filename}.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved {path}")
