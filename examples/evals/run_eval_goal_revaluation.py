"""Evaluation: Goal Revaluation — SR vs Q-Learning.

Demonstrates the key advantage of Successor Representation (SR) agents:
instant replanning when goals change. SR learns a goal-agnostic successor
matrix M, so switching goals only requires recomputing V = M @ C_new.
Q-learning must retrain from scratch because Q(s,a) values are entangled
with the specific goal.

Protocol:
  Phase 1: Train all agents on Goal A (bottom-right). Evaluate baseline.
  Phase 2: Switch to Goal B (top-right). SR agents replan instantly.
           Q-learning is tested at various retraining budgets (0..5000 eps).

Saves .npy data to data/eval/goal_revaluation/ and figures to
figures/eval/goal_revaluation/.

Usage:
    # Run experiments:
    python examples/run_eval_goal_revaluation.py --train

    # Plot from saved data:
    python examples/run_eval_goal_revaluation.py

    # Quick test (3 seeds, 3 retraining budgets):
    python examples/run_eval_goal_revaluation.py --train --quick
"""

import os

import argparse
import copy
import json
import time

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-poster")

from core import HierarchicalSRAgent
from core.q_learning import QLearningAgent
from environments.gridworld import GridworldAdapter, get_layout, AVAILABLE_LAYOUTS
from unified_env import StandardGridworld as SR_Gridworld

# ==================== Utilities ====================

def switch_q_goal(q_agent, adapter, new_goal_loc, goal_val=100.0, reset_q=False):
    """Switch a Q-learning agent to a new goal.

    Args:
        q_agent: Existing QLearningAgent
        adapter: GridworldAdapter
        new_goal_loc: (x, y) tuple for the new goal
        goal_val: Reward value at goal
        reset_q: If True, zero out Q-table (from-scratch).
                 If False, keep old Q-values (transfer).
    """
    new_goal_states = adapter.get_goal_states(new_goal_loc)
    new_C = adapter.create_goal_prior(new_goal_states, reward=goal_val, default_cost=-0.1)

    q_agent.goal_states = set(new_goal_states)
    q_agent.C = new_C

    if reset_q:
        q_agent.Q = np.zeros_like(q_agent.Q)

    # Reset exploration so agent can discover new goal
    q_agent.epsilon = q_agent.epsilon_start

def evaluate_agent(agent, init_state, max_steps, n_eval, mode="flat"):
    """Run multiple eval episodes and return mean metrics.

    Args:
        agent: HierarchicalSRAgent or QLearningAgent
        init_state: Starting state
        max_steps: Max steps per episode
        n_eval: Number of evaluation episodes
        mode: 'flat', 'hier', or 'q'

    Returns:
        Dict with mean reward, steps, reached_goal fraction
    """
    rewards, steps, reached = [], [], []

    for _ in range(n_eval):
        if mode == "q":
            result = agent.run_episode(init_state=init_state, max_steps=max_steps)
        elif mode == "hier":
            agent.reset_episode(init_state=init_state)
            result = agent.run_episode_hierarchical(max_steps=max_steps)
        else:  # flat
            agent.reset_episode(init_state=init_state)
            result = agent.run_episode_flat(max_steps=max_steps)

        rewards.append(result["reward"])
        steps.append(result["steps"])
        reached.append(float(result["reached_goal"]))

    return {
        "reward": np.mean(rewards),
        "steps": np.mean(steps),
        "reached": np.mean(reached),
    }

def generate_q_video(q_agent, adapter, init_state, goal_loc, max_steps,
                     save_path, init_loc=(0, 0)):
    """Generate a gridworld episode video for a Q-learning agent.

    Since QLearningAgent doesn't have the VisualizationMixin, we build the
    animation directly using matplotlib.animation.

    Args:
        q_agent: QLearningAgent
        adapter: GridworldAdapter
        init_state: Initial state index or tuple
        goal_loc: Goal (x, y) tuple
        max_steps: Max episode steps
        save_path: Path to save MP4
        init_loc: Start (x, y) tuple
    """
    import matplotlib.animation as animation

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    grid_size = adapter.grid_size

    # Run a greedy episode and record state locations
    adapter.reset(init_state)
    state_idx = adapter.get_current_state_index()
    state_locs = [adapter.state_space.index_to_state(state_idx)]

    for _ in range(max_steps):
        action = q_agent.select_action(state_idx, greedy=True)
        adapter.step(action)
        state_idx = adapter.get_current_state_index()
        loc = adapter.state_space.index_to_state(state_idx)
        state_locs.append(loc)
        if state_idx in q_agent.goal_states:
            break

    # Get wall locations
    walls = adapter.get_wall_indices() if hasattr(adapter, 'get_wall_indices') else []
    wall_locs = []
    for w in walls:
        loc = adapter.state_space.index_to_state(w)
        if len(loc) == 2:
            wall_locs.append(loc)

    # Build animation
    fig = plt.figure(figsize=(8, 8))
    grid = np.zeros((grid_size, grid_size))
    if wall_locs:
        for w in wall_locs:
            grid[w] = 0.25
    grid[init_loc] = 1
    grid[goal_loc] = 0.5

    im = plt.imshow(grid.T, aspect='equal', cmap='magma')
    plt.title('Q-Learning Episode')
    ax = plt.gca()
    ax.set_xticks(np.arange(0, grid_size, 1))
    ax.set_yticks(np.arange(0, grid_size, 1))
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)
    ax.tick_params(which='minor', bottom=False, left=False)

    past_trail = []

    def init_func():
        g = np.zeros((grid_size, grid_size))
        if wall_locs:
            for w in wall_locs:
                g[w] = 0.25
        g[init_loc] = 1
        g[goal_loc] = 0.5
        im.set_data(g.T)
        return im,

    def animate_func(i):
        for txt in ax.texts[::-1]:
            txt.remove()

        g = np.zeros((grid_size, grid_size))
        if wall_locs:
            for w in wall_locs:
                g[w] = 0.25

        s_loc = state_locs[i]
        g[s_loc[0], s_loc[1]] = 1
        g[goal_loc] = 0.5
        im.set_data(g.T)

        past_trail.append(s_loc)
        ax.text(s_loc[0], s_loc[1], 'Agent', fontsize=10,
                ha="center", va="center", color="b")
        ax.text(goal_loc[0], goal_loc[1], 'Goal', fontsize=10,
                ha="center", va="center", color="w")
        for w in wall_locs:
            ax.text(w[0], w[1], 'Wall', fontsize=8,
                    ha="center", va="center", color="w")
        if i > 0:
            ax.scatter(past_trail[i-1][0], past_trail[i-1][1], color='y')

        return im,

    # Scale frame interval to keep video duration reasonable
    # Short episodes (≤50 steps): 500ms per frame; longer: speed up proportionally
    n_frames = len(state_locs)
    interval = max(100, min(500, 25000 // n_frames))

    ani = animation.FuncAnimation(fig, animate_func, np.arange(n_frames),
                                  init_func=init_func, interval=interval, blit=True)
    ani.save(save_path)
    plt.close()
    print(f"  Video saved to {save_path}")


# ==================== Agent Factories ====================

def create_sr_agent(grid_size, n_clusters, walls, goal_loc, num_episodes,
                    gamma=0.99, learn_from_experience=False, use_replay=True,
                    learning_rate=0.05, flat_only=False):
    """Create a fresh SR agent trained on a specific goal.

    Args:
        learn_from_experience: If True, learn B from exploration (shows a
            real learning curve). If False, compute B analytically (instant).
        use_replay: If True (default), use experience replay after TD learning.
            If False, raw TD only (slower convergence, matches legacy behavior).
        learning_rate: TD learning rate for SR. Legacy hierarchy uses 0.05,
            legacy flat uses 0.1.
        flat_only: If True, all episodes go to SR (no adjacency overhead).
            Matches legacy flat agent behavior.
    """
    env = SR_Gridworld(grid_size)
    env.set_walls(walls)
    adapter = GridworldAdapter(env, grid_size)

    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=n_clusters,
        gamma=gamma,
        learning_rate=learning_rate,
        learn_from_experience=learn_from_experience,
        use_replay=use_replay,
    )
    agent.set_goal(goal_loc, reward=100.0)
    agent.learn_environment(num_episodes, flat_only=flat_only)

    return agent, adapter

def create_sr_agent_untrained(grid_size, n_clusters, walls, goal_loc,
                              gamma=0.99, use_replay=True, learning_rate=0.05):
    """Create an SR agent with goal set but NOT trained.

    Used for incremental learning where the caller will call
    ``learn_environment_incremental(delta)`` at each checkpoint.

    Args:
        learning_rate: TD learning rate. Legacy hierarchy=0.05, flat=0.1.

    Returns:
        Tuple of (agent, adapter)
    """
    env = SR_Gridworld(grid_size)
    env.set_walls(walls)
    adapter = GridworldAdapter(env, grid_size)

    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=n_clusters,
        gamma=gamma,
        learning_rate=learning_rate,
        learn_from_experience=True,
        use_replay=use_replay,
    )
    agent.set_goal(goal_loc, reward=100.0)
    return agent, adapter

def create_q_agent(grid_size, walls, goal_loc, gamma=0.99):
    """Create a fresh Q-learning agent (not yet trained)."""
    env = SR_Gridworld(grid_size)
    env.set_walls(walls)
    adapter = GridworldAdapter(env, grid_size)

    goal_states = adapter.get_goal_states(goal_loc)
    C = adapter.create_goal_prior(goal_states, reward=100.0, default_cost=-0.1)

    q_agent = QLearningAgent(
        adapter=adapter,
        goal_states=goal_states,
        C=C,
        gamma=gamma,
        alpha=0.1,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
    )

    return q_agent, adapter

# ==================== Experiment ====================

def run_goal_revaluation_experiment(args):
    """Main experiment: train on Goal A, switch to Goal B, compare adaptation.

    Returns:
        Dict of all result arrays
    """
    n_budgets = len(args.retrain_budgets)

    # Phase 1 results (shape: n_runs)
    p1_reward_sr_hier = np.zeros(args.n_runs)
    p1_reward_sr_flat = np.zeros(args.n_runs)
    p1_reward_q = np.zeros(args.n_runs)
    p1_steps_sr_hier = np.zeros(args.n_runs)
    p1_steps_sr_flat = np.zeros(args.n_runs)
    p1_steps_q = np.zeros(args.n_runs)

    # Phase 2 results
    p2_reward_sr_hier = np.zeros(args.n_runs)
    p2_reward_sr_flat = np.zeros(args.n_runs)
    p2_reward_q_transfer = np.zeros((args.n_runs, n_budgets))
    p2_reward_q_scratch = np.zeros((args.n_runs, n_budgets))
    p2_steps_sr_hier = np.zeros(args.n_runs)
    p2_steps_sr_flat = np.zeros(args.n_runs)
    p2_steps_q_transfer = np.zeros((args.n_runs, n_budgets))
    p2_steps_q_scratch = np.zeros((args.n_runs, n_budgets))
    p2_reached_sr_hier = np.zeros(args.n_runs)
    p2_reached_sr_flat = np.zeros(args.n_runs)
    p2_reached_q_transfer = np.zeros((args.n_runs, n_budgets))
    p2_reached_q_scratch = np.zeros((args.n_runs, n_budgets))

    layout = getattr(args, 'layout', 'serpentine')
    video_dir = f"figures/eval/goal_revaluation/{layout}"

    for n in range(args.n_runs):
        print("\n" + "x" * 60)
        print(f"Run {n + 1}/{args.n_runs}")
        print("x" * 60)

        # ---- Phase 1: Train on Goal A ----
        print(f"\n--- Phase 1: Training on Goal A {args.goal_A} ---")

        # SR agent (one agent, two eval modes)
        sr_agent, sr_adapter = create_sr_agent(
            args.grid_size, args.n_macro, args.walls,
            args.goal_A, args.initial_train_episodes, args.gamma,
        )

        # Q-learning agent
        q_agent, q_adapter = create_q_agent(
            args.grid_size, args.walls, args.goal_A, args.gamma,
        )
        q_agent.learn(args.initial_train_episodes)

        # Evaluate Phase 1
        print("\nEvaluating on Goal A...")
        res = evaluate_agent(sr_agent, 0, args.max_steps, args.n_eval, "hier")
        p1_reward_sr_hier[n] = res["reward"]
        p1_steps_sr_hier[n] = res["steps"]
        print(f"  SR Hier:  reward={res['reward']:.1f}, steps={res['steps']:.0f}, "
              f"reached={res['reached']:.0%}")

        res = evaluate_agent(sr_agent, 0, args.max_steps, args.n_eval, "flat")
        p1_reward_sr_flat[n] = res["reward"]
        p1_steps_sr_flat[n] = res["steps"]
        print(f"  SR Flat:  reward={res['reward']:.1f}, steps={res['steps']:.0f}, "
              f"reached={res['reached']:.0%}")

        res = evaluate_agent(q_agent, 0, args.max_steps, args.n_eval, "q")
        p1_reward_q[n] = res["reward"]
        p1_steps_q[n] = res["steps"]
        print(f"  Q-Learn:  reward={res['reward']:.1f}, steps={res['steps']:.0f}, "
              f"reached={res['reached']:.0%}")

        # Generate Phase 1 videos (first run only)
        if n == 0:
            os.makedirs(video_dir, exist_ok=True)
            # SR on Goal A
            sr_agent.reset_episode(init_state=0)
            sr_agent.run_episode_flat(max_steps=args.max_steps)
            sr_agent.show_video(
                save_path=f"{video_dir}/phase1_sr_goal_A.mp4",
                init_loc=args.init_loc, goal_loc=args.goal_A)

        # ---- Phase 2: Switch to Goal B ----
        print(f"\n--- Phase 2: Switch to Goal B {args.goal_B} ---")

        # SR: instant replanning
        sr_agent.set_goal(args.goal_B, reward=100.0)
        sr_agent._compute_macro_preference()  # Needed for hierarchical planning

        res = evaluate_agent(sr_agent, 0, args.max_steps, args.n_eval, "hier")
        p2_reward_sr_hier[n] = res["reward"]
        p2_steps_sr_hier[n] = res["steps"]
        p2_reached_sr_hier[n] = res["reached"]
        print(f"  SR Hier (0 retrain): reward={res['reward']:.1f}, "
              f"steps={res['steps']:.0f}, reached={res['reached']:.0%}")

        res = evaluate_agent(sr_agent, 0, args.max_steps, args.n_eval, "flat")
        p2_reward_sr_flat[n] = res["reward"]
        p2_steps_sr_flat[n] = res["steps"]
        p2_reached_sr_flat[n] = res["reached"]
        print(f"  SR Flat (0 retrain): reward={res['reward']:.1f}, "
              f"steps={res['steps']:.0f}, reached={res['reached']:.0%}")

        # Generate Phase 2 SR video (first run only) — shows instant replanning
        if n == 0:
            sr_agent.reset_episode(init_state=0)
            sr_agent.run_episode_flat(max_steps=args.max_steps)
            sr_agent.show_video(
                save_path=f"{video_dir}/phase2_sr_goal_B_0_retrain.mp4",
                init_loc=args.init_loc, goal_loc=args.goal_B)

        # Q-learning: test at various retraining budgets
        for bi, budget in enumerate(args.retrain_budgets):
            # --- Transfer condition ---
            q_transfer = copy.deepcopy(q_agent)
            switch_q_goal(q_transfer, q_adapter, args.goal_B, reset_q=False)
            if budget > 0:
                q_transfer.learn(budget)

            res = evaluate_agent(q_transfer, 0, args.max_steps, args.n_eval, "q")
            p2_reward_q_transfer[n, bi] = res["reward"]
            p2_steps_q_transfer[n, bi] = res["steps"]
            p2_reached_q_transfer[n, bi] = res["reached"]
            print(f"  Q Transfer ({budget:>5d} retrain): reward={res['reward']:.1f}, "
                  f"steps={res['steps']:.0f}, reached={res['reached']:.0%}")

            # --- From-scratch condition ---
            q_scratch = copy.deepcopy(q_agent)
            switch_q_goal(q_scratch, q_adapter, args.goal_B, reset_q=True)
            if budget > 0:
                q_scratch.learn(budget)

            res = evaluate_agent(q_scratch, 0, args.max_steps, args.n_eval, "q")
            p2_reward_q_scratch[n, bi] = res["reward"]
            p2_steps_q_scratch[n, bi] = res["steps"]
            p2_reached_q_scratch[n, bi] = res["reached"]
            print(f"  Q Scratch ({budget:>5d} retrain): reward={res['reward']:.1f}, "
                  f"steps={res['steps']:.0f}, reached={res['reached']:.0%}")

            # Generate Q-learning videos (first run, key budgets only)
            if n == 0 and budget in (0, args.retrain_budgets[-1]):
                generate_q_video(
                    q_transfer, q_adapter, 0, args.goal_B, args.max_steps,
                    save_path=f"{video_dir}/phase2_q_transfer_goal_B_{budget}_retrain.mp4",
                    init_loc=args.init_loc)

    return {
        # Phase 1
        "p1_reward_sr_hier": p1_reward_sr_hier,
        "p1_reward_sr_flat": p1_reward_sr_flat,
        "p1_reward_q": p1_reward_q,
        "p1_steps_sr_hier": p1_steps_sr_hier,
        "p1_steps_sr_flat": p1_steps_sr_flat,
        "p1_steps_q": p1_steps_q,
        # Phase 2
        "p2_reward_sr_hier": p2_reward_sr_hier,
        "p2_reward_sr_flat": p2_reward_sr_flat,
        "p2_reward_q_transfer": p2_reward_q_transfer,
        "p2_reward_q_scratch": p2_reward_q_scratch,
        "p2_steps_sr_hier": p2_steps_sr_hier,
        "p2_steps_sr_flat": p2_steps_sr_flat,
        "p2_steps_q_transfer": p2_steps_q_transfer,
        "p2_steps_q_scratch": p2_steps_q_scratch,
        "p2_reached_sr_hier": p2_reached_sr_hier,
        "p2_reached_sr_flat": p2_reached_sr_flat,
        "p2_reached_q_transfer": p2_reached_q_transfer,
        "p2_reached_q_scratch": p2_reached_q_scratch,
    }

# ==================== Plotting ====================

def _plot_retraining_axes(ax, budgets, sr_hier_mean, sr_hier_sem,
                          sr_flat_mean, sr_flat_sem,
                          qt_mean, qt_sem, qs_mean, qs_sem, ylabel,
                          title=None, compact=False):
    """Draw SR lines and Q-learning curves on a given axes.

    Args:
        compact: If True, use smaller fonts suitable for multi-panel figures.
    """
    lw = 1.8 if compact else 2.5
    ms = 5 if compact else 8
    label_fs = 14 if compact else 28
    tick_fs = 12 if compact else 26

    # SR horizontal lines with bands
    ax.axhline(sr_hier_mean, color="C0", linewidth=lw,
               label="SR Hierarchy (0 retrain)")
    ax.axhspan(sr_hier_mean - sr_hier_sem, sr_hier_mean + sr_hier_sem,
               color="C0", alpha=0.15)
    ax.axhline(sr_flat_mean, color="C1", linewidth=lw, linestyle="--",
               label="SR Flat (0 retrain)")
    ax.axhspan(sr_flat_mean - sr_flat_sem, sr_flat_mean + sr_flat_sem,
               color="C1", alpha=0.15)

    # Q-learning curves
    ax.plot(budgets, qt_mean, "s-", color="C2", linewidth=lw, markersize=ms,
            label="Q-Learning (transfer)")
    ax.fill_between(budgets, qt_mean - qt_sem, qt_mean + qt_sem,
                    color="C2", alpha=0.3)
    ax.plot(budgets, qs_mean, "D-", color="C3", linewidth=lw, markersize=ms,
            label="Q-Learning (from scratch)")
    ax.fill_between(budgets, qs_mean - qs_sem, qs_mean + qs_sem,
                    color="C3", alpha=0.3)

    ax.set_ylabel(ylabel, fontsize=label_fs)
    ax.tick_params(axis='y', labelsize=tick_fs)
    if title:
        ax.set_title(title, fontsize=label_fs + 2)


def plot_retraining_curve(args, data_dir, save_dir, metric="reward", layout_name=None):
    """Plot SR (flat lines) vs Q-learning (rising curves) after goal switch.

    Generates three variants:
        1. symlog  — log-spaced x-axis (handles budget=0), all points visible
        2. linear  — standard linear x-axis, full range
        3. zoomed  — linear x-axis cropped to the rising region only

    Args:
        metric: 'reward', 'steps', or 'reached'
        layout_name: Optional layout name for the plot title
    """
    os.makedirs(save_dir, exist_ok=True)
    budgets = args.retrain_budgets

    # Load data
    sr_hier = np.load(os.path.join(data_dir, f"p2_{metric}_sr_hier.npy"))
    sr_flat = np.load(os.path.join(data_dir, f"p2_{metric}_sr_flat.npy"))
    q_transfer = np.load(os.path.join(data_dir, f"p2_{metric}_q_transfer.npy"))
    q_scratch = np.load(os.path.join(data_dir, f"p2_{metric}_q_scratch.npy"))

    # SR: flat horizontal lines (same value at every x-position)
    sr_hier_mean = float(np.mean(sr_hier))
    sr_hier_sem = float(np.std(sr_hier) / np.sqrt(len(sr_hier)))
    sr_flat_mean = float(np.mean(sr_flat))
    sr_flat_sem = float(np.std(sr_flat) / np.sqrt(len(sr_flat)))

    # Q-learning: curves across retraining budgets
    qt_mean = np.mean(q_transfer, axis=0)
    qt_sem = np.std(q_transfer, axis=0) / np.sqrt(q_transfer.shape[0])
    qs_mean = np.mean(q_scratch, axis=0)
    qs_sem = np.std(q_scratch, axis=0) / np.sqrt(q_scratch.shape[0])

    ylabel = {
        "reward": "Total Reward",
        "steps": "Steps to Goal",
        "reached": "Goal Reached (fraction)",
    }.get(metric, metric)

    metric_label = {"reward": "Reward", "steps": "Steps", "reached": "Goal Reached"}.get(metric, metric)
    title = f"Goal Revaluation — {layout_name.capitalize()} ({metric_label})" if layout_name else None

    common = dict(sr_hier_mean=sr_hier_mean, sr_hier_sem=sr_hier_sem,
                  sr_flat_mean=sr_flat_mean, sr_flat_sem=sr_flat_sem,
                  qt_mean=qt_mean, qt_sem=qt_sem,
                  qs_mean=qs_mean, qs_sem=qs_sem, ylabel=ylabel,
                  title=title)

    # --- 1. Symlog (default) ---
    fig, ax = plt.subplots(figsize=(14, 10))
    _plot_retraining_axes(ax, budgets, **common)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets], fontsize=20, rotation=45)
    ax.set_xlabel("Retraining Episodes After Goal Switch", fontsize=28)
    ax.legend(fontsize=20, loc="best")
    fig.tight_layout()
    fname = f"goal_revaluation_{metric}.png"
    fig.savefig(os.path.join(save_dir, fname), format="png")
    plt.close(fig)
    print(f"  Saved {save_dir}/{fname}")

    # --- 2. Linear (full range) ---
    fig, ax = plt.subplots(figsize=(14, 10))
    _plot_retraining_axes(ax, budgets, **common)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets], fontsize=20, rotation=45)
    ax.set_xlabel("Retraining Episodes After Goal Switch", fontsize=28)
    ax.legend(fontsize=20, loc="best")
    fig.tight_layout()
    fname = f"goal_revaluation_{metric}_linear.png"
    fig.savefig(os.path.join(save_dir, fname), format="png")
    plt.close(fig)
    print(f"  Saved {save_dir}/{fname}")

    # --- 3. Zoomed to rising region ---
    # Find the range where Q-learning is climbing: from the last budget
    # where the slower curve (transfer) is still near its minimum to the
    # first budget where it reaches the SR level.
    sr_level = max(sr_hier_mean, sr_flat_mean)
    qt_min = qt_mean[0]

    # Rising region: from first budget to the first budget where transfer
    # reaches >= 90% of SR performance (or end of budgets).
    threshold = qt_min + 0.9 * (sr_level - qt_min)
    zoom_end_idx = len(budgets) - 1
    for i, v in enumerate(qt_mean):
        if v >= threshold:
            zoom_end_idx = min(i + 1, len(budgets) - 1)  # one past
            break

    # Start one index before the rise begins for context
    zoom_start_idx = 0

    zi = slice(zoom_start_idx, zoom_end_idx + 1)
    z_budgets = budgets[zi] if isinstance(budgets, np.ndarray) else budgets[zi]

    fig, ax = plt.subplots(figsize=(14, 10))
    _plot_retraining_axes(ax, z_budgets,
                          sr_hier_mean=sr_hier_mean, sr_hier_sem=sr_hier_sem,
                          sr_flat_mean=sr_flat_mean, sr_flat_sem=sr_flat_sem,
                          qt_mean=qt_mean[zi], qt_sem=qt_sem[zi],
                          qs_mean=qs_mean[zi], qs_sem=qs_sem[zi],
                          ylabel=ylabel, title=title)
    ax.set_xticks(z_budgets)
    ax.set_xticklabels([str(b) for b in z_budgets], fontsize=20, rotation=45)
    ax.set_xlabel("Retraining Episodes After Goal Switch", fontsize=28)
    ax.legend(fontsize=20, loc="best")
    fig.tight_layout()
    fname = f"goal_revaluation_{metric}_zoomed.png"
    fig.savefig(os.path.join(save_dir, fname), format="png")
    plt.close(fig)
    print(f"  Saved {save_dir}/{fname}")

def plot_phase1_baseline(args, data_dir, save_dir):
    """Bar chart showing all agents perform well on Goal A (baseline)."""
    os.makedirs(save_dir, exist_ok=True)

    p1_sr_hier = np.load(os.path.join(data_dir, "p1_reward_sr_hier.npy"))
    p1_sr_flat = np.load(os.path.join(data_dir, "p1_reward_sr_flat.npy"))
    p1_q = np.load(os.path.join(data_dir, "p1_reward_q.npy"))

    labels = ["SR Hierarchy", "SR Flat", "Q-Learning"]
    means = [np.mean(p1_sr_hier), np.mean(p1_sr_flat), np.mean(p1_q)]
    sems = [
        np.std(p1_sr_hier) / np.sqrt(len(p1_sr_hier)),
        np.std(p1_sr_flat) / np.sqrt(len(p1_sr_flat)),
        np.std(p1_q) / np.sqrt(len(p1_q)),
    ]
    colors = ["C0", "C1", "C2"]

    fig = plt.figure(figsize=(10, 8))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=sems, capsize=8, color=colors)
    plt.xticks(x, labels, fontsize=22)
    plt.yticks(fontsize=26)
    plt.ylabel("Reward on Goal A (baseline)", fontsize=24)
    plt.title("Phase 1: All Agents Trained on Goal A", fontsize=24)
    plt.tight_layout()

    fname = "goal_revaluation_baseline.png"
    plt.savefig(os.path.join(save_dir, fname), format="png")
    plt.close()
    print(f"  Saved {save_dir}/{fname}")

def _draw_grid_on_ax(ax, grid_size, walls, init_loc, goal_A, goal_B, title,
                     goal_C=None):
    """Draw a gridworld layout diagram on a matplotlib axes."""
    grid = np.ones((grid_size, grid_size, 3))  # white background
    for w in walls:
        grid[w[0], w[1]] = [0.25, 0.25, 0.25]  # dark gray walls

    # Transpose so x=column, y=row with origin at top-left (matching video convention)
    ax.imshow(grid.transpose(1, 0, 2), origin='upper', aspect='equal')

    # Draw grid lines
    for i in range(grid_size + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)

    # Markers
    ax.plot(init_loc[0], init_loc[1], 'o', color='limegreen',
            markersize=14, markeredgecolor='black', markeredgewidth=1.5,
            label='Start', zorder=5)
    ax.plot(goal_A[0], goal_A[1], 's', color='red',
            markersize=14, markeredgecolor='black', markeredgewidth=1.5,
            label='Goal A', zorder=5)
    ax.plot(goal_B[0], goal_B[1], '^', color='dodgerblue',
            markersize=14, markeredgecolor='black', markeredgewidth=1.5,
            label='Goal B', zorder=5)
    if goal_C is not None:
        ax.plot(goal_C[0], goal_C[1], 'D', color='darkorange',
                markersize=14, markeredgecolor='black', markeredgewidth=1.5,
                label='Goal C', zorder=5)

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(grid_size - 0.5, -0.5)  # invert y so (0,0) is top-left
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=16, fontweight='bold')


def plot_grid_layout(args, save_dir):
    """Save a standalone gridworld layout diagram."""
    os.makedirs(save_dir, exist_ok=True)
    layout_name = getattr(args, 'layout', 'gridworld')

    fig, ax = plt.subplots(figsize=(6, 6))
    walls = [tuple(w) for w in args.walls] if args.walls else []
    init_loc = tuple(args.init_loc)
    goal_A = tuple(args.goal_A)
    goal_B = tuple(args.goal_B)
    goal_C = tuple(args.goal_C) if hasattr(args, 'goal_C') and args.goal_C is not None else None

    _draw_grid_on_ax(ax, args.grid_size, walls, init_loc, goal_A, goal_B,
                     f"{layout_name.capitalize()} Layout", goal_C=goal_C)
    ax.legend(fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              ncol=4, frameon=True)
    fig.tight_layout()

    fname = f"grid_layout_{layout_name}.png"
    fig.savefig(os.path.join(save_dir, fname), format="png", dpi=150)
    plt.close(fig)
    print(f"  Saved {save_dir}/{fname}")


def _load_layout_data(data_dir, metric):
    """Load Phase 2 data for a metric from a layout data directory.

    Returns:
        Dict with sr_hier, sr_flat, q_transfer, q_scratch arrays and budgets,
        or None if data is missing.
    """
    try:
        args_path = os.path.join(data_dir, "args.json")
        with open(args_path, "r") as f:
            saved = json.load(f)
        budgets = saved["retrain_budgets"]

        sr_hier = np.load(os.path.join(data_dir, f"p2_{metric}_sr_hier.npy"))
        sr_flat = np.load(os.path.join(data_dir, f"p2_{metric}_sr_flat.npy"))
        q_transfer = np.load(os.path.join(data_dir, f"p2_{metric}_q_transfer.npy"))
        q_scratch = np.load(os.path.join(data_dir, f"p2_{metric}_q_scratch.npy"))

        return dict(
            budgets=budgets, args=saved,
            sr_hier=sr_hier, sr_flat=sr_flat,
            q_transfer=q_transfer, q_scratch=q_scratch,
        )
    except (FileNotFoundError, KeyError):
        return None


def plot_combined_figure(data_base_dir, save_dir, replay_tag="replay"):
    """Create a multi-panel figure comparing all layouts side by side.

    Layout: 4 rows × N columns (grid diagram + reward + steps + reached).
    """
    layouts = []
    for name in AVAILABLE_LAYOUTS:
        d = os.path.join(data_base_dir, name, replay_tag)
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "args.json")):
            layouts.append(name)

    if len(layouts) < 2:
        print("  Skipping combined figure (need data for >= 2 layouts)")
        return

    os.makedirs(save_dir, exist_ok=True)
    n_cols = len(layouts)
    metrics = ["reward", "steps", "reached"]
    n_rows = 1 + len(metrics)  # grid row + metric rows

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 5 * n_rows),
                             gridspec_kw={'height_ratios': [1] + [1.3] * len(metrics)})
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    # --- Row 0: Grid layout diagrams ---
    for ci, layout_name in enumerate(layouts):
        data = _load_layout_data(os.path.join(data_base_dir, layout_name, replay_tag), "reward")
        if data is None:
            continue
        a = data["args"]
        walls = [tuple(w) for w in a["walls"]]
        init_loc = tuple(a["init_loc"])
        goal_A = tuple(a["goal_A"])
        goal_B = tuple(a["goal_B"])
        goal_C = tuple(a["goal_C"]) if "goal_C" in a and a["goal_C"] is not None else None
        _draw_grid_on_ax(axes[0, ci], a["grid_size"], walls,
                         init_loc, goal_A, goal_B, layout_name.capitalize(),
                         goal_C=goal_C)

    # Add a legend for the grid row markers (from first column only)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, -1].legend(handles, labels, fontsize=10, loc='upper right')

    # --- Rows 1-3: Metric curves ---
    ylabel_map = {
        "reward": "Total Reward",
        "steps": "Steps to Goal",
        "reached": "Goal Reached",
    }

    all_handles, all_labels = None, None

    for ri, metric in enumerate(metrics):
        row_idx = ri + 1
        for ci, layout_name in enumerate(layouts):
            ax = axes[row_idx, ci]
            data = _load_layout_data(os.path.join(data_base_dir, layout_name, replay_tag), metric)
            if data is None:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha='center', va='center', fontsize=14)
                continue

            budgets = data["budgets"]
            sr_hier = data["sr_hier"]
            sr_flat = data["sr_flat"]
            q_transfer = data["q_transfer"]
            q_scratch = data["q_scratch"]

            sr_hier_mean = float(np.mean(sr_hier))
            sr_hier_sem = float(np.std(sr_hier) / np.sqrt(len(sr_hier)))
            sr_flat_mean = float(np.mean(sr_flat))
            sr_flat_sem = float(np.std(sr_flat) / np.sqrt(len(sr_flat)))
            qt_mean = np.mean(q_transfer, axis=0)
            qt_sem = np.std(q_transfer, axis=0) / np.sqrt(q_transfer.shape[0])
            qs_mean = np.mean(q_scratch, axis=0)
            qs_sem = np.std(q_scratch, axis=0) / np.sqrt(q_scratch.shape[0])

            # Only show ylabel on leftmost column
            yl = ylabel_map[metric] if ci == 0 else ""
            _plot_retraining_axes(
                ax, budgets,
                sr_hier_mean=sr_hier_mean, sr_hier_sem=sr_hier_sem,
                sr_flat_mean=sr_flat_mean, sr_flat_sem=sr_flat_sem,
                qt_mean=qt_mean, qt_sem=qt_sem,
                qs_mean=qs_mean, qs_sem=qs_sem,
                ylabel=yl, compact=True,
            )

            ax.set_xscale("symlog", linthresh=10)
            ax.set_xticks(budgets)
            ax.set_xticklabels([str(b) for b in budgets], fontsize=9, rotation=45)

            # Only show x-label on bottom row
            if ri == len(metrics) - 1:
                ax.set_xlabel("Retraining Episodes", fontsize=13)

            # Capture legend handles from first subplot
            if all_handles is None:
                all_handles, all_labels = ax.get_legend_handles_labels()

    # Shared legend at the bottom
    if all_handles:
        fig.legend(all_handles, all_labels, loc='lower center',
                   ncol=4, fontsize=13, frameon=True,
                   bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Goal Revaluation: SR vs Q-Learning Across Environments",
                 fontsize=20, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])

    fname = "goal_revaluation_combined.png"
    fig.savefig(os.path.join(save_dir, fname), format="png",
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {save_dir}/{fname}")


# ==================== Multi-Phase Timeline ====================

def run_multiphase_experiment(args):
    """Run a multi-phase goal revaluation experiment on a continuous timeline.

    Protocol:
        Phase A: Train on Goal A for `phase_budget` episodes, evaluate at checkpoints.
        Phase B: Switch to Goal B. SR replans instantly. Q-learning retrains for
                 another `phase_budget` episodes, evaluated at checkpoints.
        Phase C: Switch to Goal C. Same as B.

    Returns:
        Dict with timeline arrays for plotting.
    """
    phase_budget = args.phase_budget
    checkpoints = args.phase_checkpoints
    goals = [args.goal_A, args.goal_B, args.goal_C]
    goal_labels = ["A", "B", "C"]
    n_phases = len(goals)

    # Result arrays: (n_runs, n_phases, n_checkpoints)
    reward_sr_flat = np.zeros((args.n_runs, n_phases, len(checkpoints)))
    reward_sr_hier = np.zeros((args.n_runs, n_phases, len(checkpoints)))
    reward_q = np.zeros((args.n_runs, n_phases, len(checkpoints)))
    reward_q_scratch = np.zeros((args.n_runs, n_phases, len(checkpoints)))

    for run in range(args.n_runs):
        print(f"\n{'='*50}")
        print(f"  Multi-phase run {run+1}/{args.n_runs}")
        print(f"{'='*50}")

        # --- Create Q agents (transfer keeps Q, scratch resets Q at each switch) ---
        q_agent, q_adapter = create_q_agent(
            args.grid_size, args.walls, goals[0], args.gamma,
        )
        q_scratch_agent, q_scratch_adapter = create_q_agent(
            args.grid_size, args.walls, goals[0], args.gamma,
        )

        # SR agent placeholder
        sr_agent = None

        for phase_idx, goal in enumerate(goals):
            label = goal_labels[phase_idx]
            print(f"\n  --- Phase {label}: Goal {goal} ---")

            if phase_idx == 0:
                use_replay = getattr(args, 'use_replay', True)
                incremental = getattr(args, 'incremental', False)

                if incremental:
                    # Incremental mode: ONE agent per mode, trained with
                    # delta episodes at each checkpoint — matches legacy
                    # learn_env_likelikood() where partially-learned M
                    # creates hier vs flat divergence.
                    # Learning rates match legacy: hier=0.05, flat=0.1
                    sr_hier_agent, _ = create_sr_agent_untrained(
                        args.grid_size, args.n_macro, args.walls,
                        goal, args.gamma, use_replay=use_replay,
                        learning_rate=0.05,
                    )
                    sr_flat_agent, _ = create_sr_agent_untrained(
                        args.grid_size, args.n_macro, args.walls,
                        goal, args.gamma, use_replay=use_replay,
                        learning_rate=0.1,
                    )

                cumulative = 0
                for ci, ckpt in enumerate(checkpoints):
                    train_amount = ckpt - cumulative
                    if train_amount > 0:
                        q_agent.learn(train_amount)
                        q_scratch_agent.learn(train_amount)

                    if ckpt == 0:
                        # No training yet — random walk times out
                        untrained_reward = -0.1 * args.max_steps
                        reward_sr_flat[run, phase_idx, ci] = untrained_reward
                        reward_sr_hier[run, phase_idx, ci] = untrained_reward
                    elif incremental:
                        # Incremental: train the same agents with delta episodes
                        delta = ckpt - cumulative
                        if delta > 0:
                            if cumulative == 0:
                                # First checkpoint: full learn_environment
                                sr_hier_agent.learn_environment(ckpt)
                                sr_flat_agent.learn_environment(ckpt, flat_only=True)
                            else:
                                sr_hier_agent.learn_environment_incremental(delta)
                                sr_flat_agent.learn_environment_incremental(delta, flat_only=True)

                        res = evaluate_agent(sr_hier_agent, 0, args.max_steps, args.n_eval, "hier")
                        reward_sr_hier[run, phase_idx, ci] = res["reward"]

                        res = evaluate_agent(sr_flat_agent, 0, args.max_steps, args.n_eval, "flat")
                        reward_sr_flat[run, phase_idx, ci] = res["reward"]
                    else:
                        # Fresh-agent mode (default): create independent
                        # agents from scratch at each checkpoint.
                        # Hier: lr=0.05 with adjacency/macro overhead
                        sr_hier_agent, _ = create_sr_agent(
                            args.grid_size, args.n_macro, args.walls,
                            goal, ckpt, args.gamma,
                            learn_from_experience=True,
                            use_replay=use_replay,
                            learning_rate=0.05,
                        )
                        res = evaluate_agent(sr_hier_agent, 0, args.max_steps, args.n_eval, "hier")
                        reward_sr_hier[run, phase_idx, ci] = res["reward"]

                        # Flat: lr=0.1, ALL episodes to SR (no adjacency)
                        sr_flat_agent, _ = create_sr_agent(
                            args.grid_size, args.n_macro, args.walls,
                            goal, ckpt, args.gamma,
                            learn_from_experience=True,
                            use_replay=use_replay,
                            learning_rate=0.1,
                            flat_only=True,
                        )
                        res = evaluate_agent(sr_flat_agent, 0, args.max_steps, args.n_eval, "flat")
                        reward_sr_flat[run, phase_idx, ci] = res["reward"]

                    cumulative = ckpt

                    res = evaluate_agent(q_agent, 0, args.max_steps, args.n_eval, "q")
                    reward_q[run, phase_idx, ci] = res["reward"]

                    res = evaluate_agent(q_scratch_agent, 0, args.max_steps, args.n_eval, "q")
                    reward_q_scratch[run, phase_idx, ci] = res["reward"]

                    print(f"    ckpt {ckpt}: SR_flat={reward_sr_flat[run, phase_idx, ci]:.0f}, "
                          f"SR_hier={reward_sr_hier[run, phase_idx, ci]:.0f}, "
                          f"Q_xfer={reward_q[run, phase_idx, ci]:.0f}, "
                          f"Q_scratch={reward_q_scratch[run, phase_idx, ci]:.0f}")

                # Ensure sr_agent is fully trained for subsequent phases
                if incremental:
                    sr_agent = sr_hier_agent
                else:
                    sr_agent, _ = create_sr_agent(
                        args.grid_size, args.n_macro, args.walls,
                        goal, phase_budget, args.gamma,
                    )
            else:
                # Phase B/C: switch goal
                # SR: instant replan
                sr_agent.set_goal(goal, reward=100.0)
                sr_agent._compute_macro_preference()

                # Q transfer: keep old Q-values
                switch_q_goal(q_agent, q_adapter, goal, reset_q=False)
                # Q scratch: zero out Q-table
                switch_q_goal(q_scratch_agent, q_scratch_adapter, goal, reset_q=True)

                cumulative = 0
                for ci, ckpt in enumerate(checkpoints):
                    train_amount = ckpt - cumulative
                    if train_amount > 0:
                        q_agent.learn(train_amount)
                        q_scratch_agent.learn(train_amount)
                    cumulative = ckpt

                    res = evaluate_agent(sr_agent, 0, args.max_steps, args.n_eval, "flat")
                    reward_sr_flat[run, phase_idx, ci] = res["reward"]

                    res = evaluate_agent(sr_agent, 0, args.max_steps, args.n_eval, "hier")
                    reward_sr_hier[run, phase_idx, ci] = res["reward"]

                    res = evaluate_agent(q_agent, 0, args.max_steps, args.n_eval, "q")
                    reward_q[run, phase_idx, ci] = res["reward"]

                    res = evaluate_agent(q_scratch_agent, 0, args.max_steps, args.n_eval, "q")
                    reward_q_scratch[run, phase_idx, ci] = res["reward"]

                    print(f"    ckpt {ckpt}: SR_flat={reward_sr_flat[run, phase_idx, ci]:.0f}, "
                          f"SR_hier={reward_sr_hier[run, phase_idx, ci]:.0f}, "
                          f"Q_xfer={reward_q[run, phase_idx, ci]:.0f}, "
                          f"Q_scratch={reward_q_scratch[run, phase_idx, ci]:.0f}")

    return {
        "reward_sr_flat": reward_sr_flat,
        "reward_sr_hier": reward_sr_hier,
        "reward_q": reward_q,
        "reward_q_scratch": reward_q_scratch,
        "checkpoints": np.array(checkpoints),
        "goals": [list(g) for g in goals],
        "goal_labels": goal_labels,
    }


def plot_multiphase_timeline(data_dir, save_dir, args):
    """Plot the multi-phase timeline showing all goal switches on one x-axis."""
    os.makedirs(save_dir, exist_ok=True)

    reward_sr_flat = np.load(os.path.join(data_dir, "mp_reward_sr_flat.npy"))
    reward_sr_hier = np.load(os.path.join(data_dir, "mp_reward_sr_hier.npy"))
    reward_q = np.load(os.path.join(data_dir, "mp_reward_q.npy"))

    # Q scratch may not exist in older data
    q_scratch_path = os.path.join(data_dir, "mp_reward_q_scratch.npy")
    reward_q_scratch = np.load(q_scratch_path) if os.path.exists(q_scratch_path) else None

    with open(os.path.join(data_dir, "mp_args.json"), "r") as f:
        mp_args = json.load(f)

    checkpoints = mp_args["checkpoints"]
    goals = mp_args["goals"]
    goal_labels = mp_args["goal_labels"]
    phase_budget = mp_args["phase_budget"]
    n_phases = len(goals)
    layout_name = mp_args.get("layout", "")

    # Build continuous x-axis: phase 0 checkpoints, then offset by phase_budget per phase
    x_all = []
    for p in range(n_phases):
        x_all.extend([c + p * phase_budget for c in checkpoints])
    x_all = np.array(x_all)

    # Flatten the (n_runs, n_phases, n_checkpoints) -> (n_runs, n_phases * n_checkpoints)
    n_runs = reward_sr_flat.shape[0]
    n_ckpts = len(checkpoints)
    sr_flat_all = reward_sr_flat.reshape(n_runs, -1)
    sr_hier_all = reward_sr_hier.reshape(n_runs, -1)
    q_all = reward_q.reshape(n_runs, -1)

    # Mean and SEM
    sr_flat_mean = np.mean(sr_flat_all, axis=0)
    sr_flat_sem = np.std(sr_flat_all, axis=0) / np.sqrt(n_runs)
    sr_hier_mean = np.mean(sr_hier_all, axis=0)
    sr_hier_sem = np.std(sr_hier_all, axis=0) / np.sqrt(n_runs)
    q_mean = np.mean(q_all, axis=0)
    q_sem = np.std(q_all, axis=0) / np.sqrt(n_runs)

    if reward_q_scratch is not None:
        qs_all = reward_q_scratch.reshape(n_runs, -1)
        qs_mean = np.mean(qs_all, axis=0)
        qs_sem = np.std(qs_all, axis=0) / np.sqrt(n_runs)

    def _decorate_multiphase_ax(ax):
        """Add phase shading, goal-switch lines, and labels with goal markers."""
        y_lo, y_hi = -40, 120
        ax.set_ylim(y_lo, y_hi)
        # Goal marker styles matching _draw_grid_on_ax
        goal_markers = ['s', '^', 'D']       # square, triangle, diamond
        goal_colors = ['red', 'dodgerblue', 'darkorange']
        phase_colors = ['#e8e8e8', '#f5f5f5', '#e8e8e8']
        for p in range(n_phases):
            x_start = p * phase_budget
            x_end = (p + 1) * phase_budget
            ax.axvspan(x_start, x_end, alpha=0.3,
                       color=phase_colors[p % len(phase_colors)], zorder=0)
            if p > 0:
                ax.axvline(x_start, color='black', linewidth=2, linestyle='--',
                           alpha=0.7, zorder=4)
                # "Switch to Goal X" with marker inline via annotate
                switch_x = x_start + phase_budget * 0.05
                ax.plot(switch_x, y_hi - 8, goal_markers[p],
                        color=goal_colors[p], markersize=11,
                        markeredgecolor='black', markeredgewidth=1.5,
                        zorder=6, clip_on=False)
                ax.annotate(f'Switch to\nGoal {goal_labels[p]}',
                            xy=(switch_x, y_hi - 8),
                            xytext=(14, 0), textcoords='offset points',
                            fontsize=13, fontweight='bold', color='black',
                            ha='left', va='center')

        ax.set_xlabel('Cumulative Training Episodes', fontsize=20)
        ax.set_ylabel('Total Reward', fontsize=20)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_xlim(0, n_phases * phase_budget)

    title = f'Multi-Phase Goal Revaluation — {layout_name.capitalize()}' if layout_name else \
            'Multi-Phase Goal Revaluation'

    # --- Plot 1: SR + Q transfer only ---
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(x_all, sr_hier_mean, '-', color='C0', linewidth=2.5,
            label='Hierarchy', zorder=3)
    ax.fill_between(x_all, sr_hier_mean - sr_hier_sem, sr_hier_mean + sr_hier_sem,
                    color='C0', alpha=0.15)
    ax.plot(x_all, sr_flat_mean, '--', color='C1', linewidth=2.5,
            label='Flat', zorder=3)
    ax.fill_between(x_all, sr_flat_mean - sr_flat_sem, sr_flat_mean + sr_flat_sem,
                    color='C1', alpha=0.15)
    ax.plot(x_all, q_mean, 's-', color='C2', linewidth=2, markersize=6,
            label='Q-Learning', zorder=3)
    ax.fill_between(x_all, q_mean - q_sem, q_mean + q_sem,
                    color='C2', alpha=0.2)
    _decorate_multiphase_ax(ax)
    ax.legend(fontsize=16, loc='center right')
    fig.tight_layout()
    fname = f"goal_revaluation_multiphase_{layout_name}.png" if layout_name else \
            "goal_revaluation_multiphase.png"
    fig.savefig(os.path.join(save_dir, fname), format="png", dpi=150)
    plt.close(fig)
    print(f"  Saved {save_dir}/{fname}")

    # --- Plot 2: All agents (including Q from scratch) ---
    if reward_q_scratch is not None:
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(x_all, sr_hier_mean, '-', color='C0', linewidth=2.5,
                label='Hierarchy', zorder=3)
        ax.fill_between(x_all, sr_hier_mean - sr_hier_sem, sr_hier_mean + sr_hier_sem,
                        color='C0', alpha=0.15)
        ax.plot(x_all, sr_flat_mean, '--', color='C1', linewidth=2.5,
                label='Flat', zorder=3)
        ax.fill_between(x_all, sr_flat_mean - sr_flat_sem, sr_flat_mean + sr_flat_sem,
                        color='C1', alpha=0.15)
        ax.plot(x_all, q_mean, 's-', color='C2', linewidth=2, markersize=6,
                label='Q-Learning', zorder=3)
        ax.fill_between(x_all, q_mean - q_sem, q_mean + q_sem,
                        color='C2', alpha=0.2)
        ax.plot(x_all, qs_mean, 'D-', color='C3', linewidth=2, markersize=5,
                label='Q-Learning (scratch)', zorder=3)
        ax.fill_between(x_all, qs_mean - qs_sem, qs_mean + qs_sem,
                        color='C3', alpha=0.2)
        _decorate_multiphase_ax(ax)
        ax.legend(fontsize=16, loc='center right')
        fig.tight_layout()
        fname = f"goal_revaluation_multiphase_all_{layout_name}.png" if layout_name else \
                "goal_revaluation_multiphase_all.png"
        fig.savefig(os.path.join(save_dir, fname), format="png", dpi=150)
        plt.close(fig)
        print(f"  Saved {save_dir}/{fname}")


# ==================== Main ====================

if __name__ == "__main__":
    grid_size = 9
    gamma = 0.99
    init_loc = (0, 0)

    # Experiment parameters
    initial_train_episodes = 2000
    retrain_budgets = [0, 50, 100, 200, 500, 1000, 2000, 5000]
    n_eval = 10
    max_steps = 200
    nruns = 10

    parser = argparse.ArgumentParser(
        description="Goal Revaluation Eval: SR vs Q-Learning"
    )
    parser.add_argument("--train", action="store_true", help="Run experiments")
    parser.add_argument("--quick", action="store_true", help="Quick test")
    parser.add_argument("--multiphase", action="store_true",
                        help="Run multi-phase timeline experiment (A->B->C)")
    parser.add_argument("--n_runs", type=int, default=nruns)
    parser.add_argument("--layout", type=str, default="serpentine",
                        choices=AVAILABLE_LAYOUTS,
                        help="Wall layout (default: serpentine)")
    parser.add_argument("--no_replay", action="store_true",
                        help="Disable experience replay for SR (raw TD only)")
    parser.add_argument("--incremental", action="store_true",
                        help="Use incremental training (same agent, delta episodes) "
                             "instead of fresh agents at each checkpoint. "
                             "Matches legacy behavior and shows hier vs flat divergence.")
    args_cli = parser.parse_args()

    if args_cli.quick:
        retrain_budgets = [0, 500, 2000]
        nruns = 3
        initial_train_episodes = 1000
        n_eval = 5

    # Get layout-specific configuration
    _layout = get_layout(args_cli.layout, grid_size)
    WALLS = _layout.walls
    goal_A = _layout.default_goal
    goal_B = _layout.alt_goal
    goal_C = _layout.third_goal
    n_macro = _layout.n_clusters

    args = argparse.Namespace(
        grid_size=grid_size,
        n_macro=n_macro,
        gamma=gamma,
        init_loc=init_loc,
        goal_A=goal_A,
        goal_B=goal_B,
        goal_C=goal_C,
        walls=WALLS,
        layout=args_cli.layout,
        initial_train_episodes=initial_train_episodes,
        retrain_budgets=retrain_budgets,
        n_eval=n_eval,
        max_steps=max_steps,
        n_runs=args_cli.n_runs if not args_cli.quick else nruns,
    )

    use_replay = not args_cli.no_replay
    replay_tag = "replay" if use_replay else "no_replay"
    data_dir = f"data/eval/goal_revaluation/{args_cli.layout}/{replay_tag}"
    save_dir = f"figures/eval/goal_revaluation/{args_cli.layout}/{replay_tag}"

    if args_cli.train:
        os.makedirs(data_dir, exist_ok=True)

        # Save args (convert tuples/lists for JSON)
        args_save = vars(args).copy()
        args_save["walls"] = [list(w) for w in args_save["walls"]]
        args_save["init_loc"] = list(args_save["init_loc"])
        args_save["goal_A"] = list(args_save["goal_A"])
        args_save["goal_B"] = list(args_save["goal_B"])
        with open(os.path.join(data_dir, "args.json"), "w") as f:
            json.dump(args_save, f, indent=2)

        print("=" * 60)
        print(f"GOAL REVALUATION EVAL: SR vs Q-Learning ({args_cli.layout})")
        print("=" * 60)
        print(f"Layout: {args_cli.layout}, Grid: {grid_size}x{grid_size}, "
              f"Macro clusters: {n_macro}")
        print(f"Goal A: {args.goal_A}, Goal B: {args.goal_B}")
        print(f"Initial training: {args.initial_train_episodes} eps")
        print(f"Retrain budgets: {args.retrain_budgets}")
        print(f"Runs: {args.n_runs}, Eval episodes: {args.n_eval}")
        print(f"Replay: {use_replay}")

        t0 = time.time()
        results = run_goal_revaluation_experiment(args)
        elapsed = time.time() - t0
        print(f"\nExperiment completed in {elapsed:.0f}s")

        # Save all results
        for key, arr in results.items():
            np.save(os.path.join(data_dir, f"{key}.npy"), arr)
        print(f"\nSaved all data to {data_dir}/")

    # ---- Multi-phase experiment ----
    if args_cli.multiphase and args_cli.train:
        if not use_replay:
            # No replay: slower convergence, need wider checkpoint range.
            # Fine-grained in 0-750 to capture hier vs flat divergence,
            # extended to 8000 so flat SR converges on harder layouts
            # (serpentine flat needs ~6000 episodes without replay).
            if args_cli.quick:
                phase_budget = 5000
                phase_checkpoints = [0, 100, 200, 500, 750, 1000, 2000, 3000, 5000]
                mp_n_runs = 3
            else:
                phase_budget = 8000
                phase_checkpoints = [0, 50, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 8000]
                mp_n_runs = args.n_runs
        else:
            # With replay: both SR converge very fast (~100-200 eps).
            # Tighter budget so the plot zooms into the interesting region.
            if args_cli.quick:
                phase_budget = 1000
                phase_checkpoints = [0, 50, 100, 150, 200, 300, 400, 500, 750, 1000]
                mp_n_runs = 3
            else:
                phase_budget = 1500
                phase_checkpoints = [0, 25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500]
                mp_n_runs = args.n_runs

        mp_args = argparse.Namespace(
            grid_size=grid_size, n_macro=n_macro, gamma=gamma,
            init_loc=init_loc, walls=WALLS, layout=args_cli.layout,
            goal_A=goal_A, goal_B=goal_B, goal_C=goal_C,
            phase_budget=phase_budget, phase_checkpoints=phase_checkpoints,
            n_eval=n_eval, max_steps=max_steps, n_runs=mp_n_runs,
            use_replay=use_replay,
            incremental=args_cli.incremental,
        )

        print("\n" + "=" * 60)
        print("MULTI-PHASE TIMELINE EXPERIMENT")
        print("=" * 60)
        print(f"Goals: A={goal_A}, B={goal_B}, C={goal_C}")
        print(f"Phase budget: {phase_budget}, Checkpoints: {phase_checkpoints}")
        print(f"Runs: {mp_n_runs}, Replay: {use_replay}, Incremental: {args_cli.incremental}")

        t0 = time.time()
        mp_results = run_multiphase_experiment(mp_args)
        elapsed = time.time() - t0
        print(f"\nMulti-phase experiment completed in {elapsed:.0f}s")

        # Save
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, "mp_reward_sr_flat.npy"), mp_results["reward_sr_flat"])
        np.save(os.path.join(data_dir, "mp_reward_sr_hier.npy"), mp_results["reward_sr_hier"])
        np.save(os.path.join(data_dir, "mp_reward_q.npy"), mp_results["reward_q"])
        np.save(os.path.join(data_dir, "mp_reward_q_scratch.npy"), mp_results["reward_q_scratch"])

        mp_save = {
            "checkpoints": mp_results["checkpoints"].tolist(),
            "goals": mp_results["goals"],
            "goal_labels": mp_results["goal_labels"],
            "phase_budget": phase_budget,
            "layout": args_cli.layout,
            "n_runs": mp_n_runs,
        }
        with open(os.path.join(data_dir, "mp_args.json"), "w") as f:
            json.dump(mp_save, f, indent=2)
        print(f"Saved multi-phase data to {data_dir}/")

    elif not args_cli.train:
        # Load saved args
        args_path = os.path.join(data_dir, "args.json")
        if os.path.exists(args_path):
            with open(args_path, "r") as f:
                saved = json.load(f)
                saved["walls"] = [tuple(w) for w in saved["walls"]]
                saved["init_loc"] = tuple(saved["init_loc"])
                saved["goal_A"] = tuple(saved["goal_A"])
                saved["goal_B"] = tuple(saved["goal_B"])
                args = argparse.Namespace(**saved)
            print(f"Loaded args from {args_path}")
        else:
            print("No saved args found. Run with --train first.")

    # ---- Generate plots ----
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)
    layout_name = getattr(args, 'layout', args_cli.layout)

    # Grid layout diagram
    plot_grid_layout(args, save_dir)

    # Phase 1 baseline
    if os.path.exists(os.path.join(data_dir, "p1_reward_sr_hier.npy")):
        plot_phase1_baseline(args, data_dir, save_dir)

    # Phase 2: retraining curves
    if os.path.exists(os.path.join(data_dir, "p2_reward_sr_hier.npy")):
        plot_retraining_curve(args, data_dir, save_dir, metric="reward",
                              layout_name=layout_name)

    if os.path.exists(os.path.join(data_dir, "p2_steps_sr_hier.npy")):
        plot_retraining_curve(args, data_dir, save_dir, metric="steps",
                              layout_name=layout_name)

    if os.path.exists(os.path.join(data_dir, "p2_reached_sr_hier.npy")):
        plot_retraining_curve(args, data_dir, save_dir, metric="reached",
                              layout_name=layout_name)

    # Multi-phase timeline plot (if data exists)
    if os.path.exists(os.path.join(data_dir, "mp_reward_sr_flat.npy")):
        plot_multiphase_timeline(data_dir, save_dir, args)

    # Combined cross-layout figure (uses all available layout data for this replay mode)
    data_base_dir = "data/eval/goal_revaluation"
    combined_save_dir = "figures/eval/goal_revaluation"
    plot_combined_figure(data_base_dir, combined_save_dir, replay_tag)

    print(f"\nDone! Figures saved to {save_dir}/")
