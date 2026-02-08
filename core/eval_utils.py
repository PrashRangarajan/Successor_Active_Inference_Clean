"""Shared evaluation utilities for benchmark scripts.

Provides common functions used across all run_eval_*.py scripts:
- relative_stability: Relative stability metric (lower is better)
- plot_reward_curves: Plot reward vs training episodes with confidence bands
- plot_step_curves: Plot steps vs training episodes with confidence bands
- plot_stability_bars: Plot relative stability bar chart
- reconcile_episodes: Fix args.json / data shape mismatches
- save_eval_data: Save experiment arrays to disk
- load_eval_args: Load and reconcile saved args.json
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt


# ==================== Metrics ====================


def relative_stability(returns, Ke=100, smooth_window=1, eps=1e-8):
    """Relative Stability metric (lower is better).

    Args:
        returns: Array of return values across checkpoints.
        Ke: Number of tail entries to evaluate.
        smooth_window: Smoothing kernel width (1 = no smoothing).
        eps: Small constant to avoid division by zero.

    Returns:
        Scalar relative stability value.
    """
    r = np.asarray(returns, dtype=float).reshape(-1)
    if r.size == 0:
        return np.nan
    Ke = int(min(max(1, Ke), r.size))
    w = r[-Ke:]

    if smooth_window is not None and int(smooth_window) > 1 and w.size >= int(smooth_window):
        k = int(smooth_window)
        kernel = np.ones(k, dtype=float) / k
        w_smooth = np.convolve(w, kernel, mode="same")
    else:
        w_smooth = w

    best = np.max(w)
    denom = np.abs(best) + eps
    return float(np.mean(np.abs((w_smooth - best) / denom)))


def compute_stability_array(data_2d):
    """Compute per-run relative stability from a (n_runs, n_trials) array.

    Args:
        data_2d: Array of shape (n_runs, n_checkpoints).

    Returns:
        1D array of length n_runs with stability values.
    """
    return np.array([
        relative_stability(data_2d[i, :])
        for i in range(data_2d.shape[0])
    ])


# ==================== Plotting ====================

# Consistent color scheme across all eval scripts.
AGENT_COLORS = {"Hierarchy": "C0", "Flat": "C1", "Q-Learning": "C2"}


def plot_reward_curves(eps_range, data_dict, save_path, ylabel="Total Reward"):
    """Plot reward (or any metric) curves with confidence bands.

    Args:
        eps_range: List/array of training episode counts (x-axis).
        data_dict: OrderedDict of {label: (n_runs, n_trials) array}.
            Common labels: "Hierarchy", "Flat", "Q-Learning".
        save_path: Full path to save the figure.
        ylabel: Y-axis label (default "Total Reward").
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig = plt.figure(figsize=(14, 10))
    for label, data in data_dict.items():
        arr = data[:, :len(eps_range)]
        mean = np.mean(arr, axis=0)
        sem = np.std(arr, axis=0) / np.sqrt(len(arr))
        color = AGENT_COLORS.get(label, None)
        plt.plot(eps_range, mean, label=label, color=color)
        plt.fill_between(eps_range, mean - sem, mean + sem, alpha=0.5, color=color)

    plt.xlabel("Number of Training Episodes", fontsize=28)
    plt.ylabel(ylabel, fontsize=28)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig(save_path, format="png")
    plt.close()
    print(f"  Saved {save_path}")


def plot_step_curves(eps_range, data_dict, save_path, ylabel="Steps to Goal"):
    """Plot steps-to-goal curves with confidence bands.

    Convenience wrapper around plot_reward_curves with different default ylabel.
    """
    plot_reward_curves(eps_range, data_dict, save_path, ylabel=ylabel)


def plot_stability_bars(data_dict, save_path):
    """Plot relative stability bar chart.

    Args:
        data_dict: Dict of {label: 1D stability array (one value per run)}.
        save_path: Full path to save the figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels, means, sems, bar_colors = [], [], [], []
    for label, st in data_dict.items():
        if st is None or len(st) == 0:
            continue
        labels.append(label)
        means.append(float(np.mean(st)))
        sems.append(float(np.std(st) / np.sqrt(len(st))))
        bar_colors.append(AGENT_COLORS.get(label, "C0"))

    if not labels:
        print("  No stability data found — skipping")
        return

    fig = plt.figure(figsize=(10, 8))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=sems, capsize=8, color=bar_colors)
    plt.xticks(x, labels, fontsize=26)
    plt.yticks(fontsize=26)
    plt.ylabel("Relative Stability", fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path, format="png")
    plt.close()
    print(f"  Saved {save_path}")


# ==================== Data I/O ====================


def save_eval_data(data_dir, arrays_dict):
    """Save experiment arrays and stability metrics to disk.

    Args:
        data_dir: Directory to save .npy files.
        arrays_dict: Dict of {filename_without_ext: ndarray}.
            Example: {"SR_rewards_hierarchy": arr, "Q_rewards": arr, ...}
    """
    os.makedirs(data_dir, exist_ok=True)
    for name, arr in arrays_dict.items():
        path = os.path.join(data_dir, f"{name}.npy")
        np.save(path, arr)
    print(f"  Saved {len(arrays_dict)} arrays to {data_dir}/")


def load_eval_args(data_dir, tuple_keys=None):
    """Load args.json and reconcile with actual data shape.

    Args:
        data_dir: Directory containing args.json and .npy files.
        tuple_keys: List of keys to convert from lists back to tuples
            (e.g. ["init_loc", "goal_loc", "walls"]).

    Returns:
        argparse.Namespace or None if not found.
    """
    import argparse

    args_path = os.path.join(data_dir, "args.json")
    if not os.path.exists(args_path):
        print("No saved args found. Run with --train first.")
        return None

    with open(args_path, "r") as f:
        saved = json.load(f)

    # Convert list fields back to tuples
    if tuple_keys:
        for key in tuple_keys:
            if key in saved:
                val = saved[key]
                if isinstance(val, list):
                    # Handle list of lists (e.g. walls) vs simple list (e.g. init_loc)
                    if val and isinstance(val[0], list):
                        saved[key] = [tuple(v) for v in val]
                    else:
                        saved[key] = tuple(val)

    args = argparse.Namespace(**saved)

    # Reconcile episodes list with actual data shape
    reconcile_episodes(args, data_dir)

    print(f"  Loaded args from {args_path}")
    return args


def reconcile_episodes(args, data_dir,
                       ref_file="SR_rewards_hierarchy.npy"):
    """Fix mismatch between args.episodes and actual data columns.

    When args.json is stale (e.g. a --quick run overwrote .npy files),
    the episodes list won't match the data shape.  This replaces
    args.episodes with interpolated labels so plotting still works.

    Args:
        args: Namespace with an ``episodes`` attribute.
        data_dir: Directory containing .npy data files.
        ref_file: Reference file to check column count against.
    """
    if not hasattr(args, 'episodes'):
        return

    ref_path = os.path.join(data_dir, ref_file)
    if not os.path.exists(ref_path):
        return

    n_data_cols = np.load(ref_path).shape[1]
    if n_data_cols != len(args.episodes):
        print(f"  Warning: args.json lists {len(args.episodes)} episodes "
              f"but data has {n_data_cols} checkpoints.")
        print(f"  Re-run with --train to regenerate consistent data.")
        args.episodes = list(np.linspace(
            args.episodes[0], args.episodes[-1], n_data_cols, dtype=int
        ))
        print(f"  Using interpolated episode labels: {args.episodes}")
