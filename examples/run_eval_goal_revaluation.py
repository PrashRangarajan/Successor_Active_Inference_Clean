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

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

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

    # Limit frames for long failed episodes (cap at 50 frames)
    n_frames = min(len(state_locs), 50)
    if len(state_locs) > 50:
        # Sample evenly
        indices = np.linspace(0, len(state_locs) - 1, 50, dtype=int)
        sampled_locs = [state_locs[i] for i in indices]
        state_locs_anim = sampled_locs
    else:
        state_locs_anim = state_locs

    # Replace state_locs for animation
    orig_state_locs = state_locs
    state_locs = state_locs_anim

    ani = animation.FuncAnimation(fig, animate_func, np.arange(len(state_locs)),
                                  init_func=init_func, interval=500, blit=True)
    ani.save(save_path)
    plt.close()
    print(f"  Video saved to {save_path}")

    # Restore
    state_locs = orig_state_locs


# ==================== Agent Factories ====================


def create_sr_agent(grid_size, n_clusters, walls, goal_loc, num_episodes, gamma=0.99):
    """Create a fresh SR agent trained on a specific goal."""
    env = SR_Gridworld(grid_size)
    env.set_walls(walls)
    adapter = GridworldAdapter(env, grid_size)

    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=n_clusters,
        gamma=gamma,
        learning_rate=0.05,
        learn_from_experience=False,  # Analytical M (known dynamics)
    )
    agent.set_goal(goal_loc, reward=100.0)
    agent.learn_environment(num_episodes)

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
                    q_scratch, q_adapter, 0, args.goal_B, args.max_steps,
                    save_path=f"{video_dir}/phase2_q_scratch_goal_B_{budget}_retrain.mp4",
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


def plot_retraining_curve(args, data_dir, save_dir, metric="reward"):
    """Plot SR (flat lines) vs Q-learning (rising curves) after goal switch.

    Args:
        metric: 'reward', 'steps', or 'reached'
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

    # Labels
    labels = {
        "reward": "Total Reward",
        "steps": "Steps to Goal",
        "reached": "Goal Reached (fraction)",
    }

    fig = plt.figure(figsize=(14, 10))

    # SR horizontal lines with bands
    plt.axhline(sr_hier_mean, color="C0", linewidth=2.5, label="SR Hierarchy (0 retrain)")
    plt.axhspan(sr_hier_mean - sr_hier_sem, sr_hier_mean + sr_hier_sem,
                color="C0", alpha=0.15)
    plt.axhline(sr_flat_mean, color="C1", linewidth=2.5, linestyle="--",
                label="SR Flat (0 retrain)")
    plt.axhspan(sr_flat_mean - sr_flat_sem, sr_flat_mean + sr_flat_sem,
                color="C1", alpha=0.15)

    # Q-learning curves
    plt.plot(budgets, qt_mean, "s-", color="C2", linewidth=2, markersize=8,
             label="Q-Learning (transfer)")
    plt.fill_between(budgets, qt_mean - qt_sem, qt_mean + qt_sem,
                     color="C2", alpha=0.3)
    plt.plot(budgets, qs_mean, "D-", color="C3", linewidth=2, markersize=8,
             label="Q-Learning (from scratch)")
    plt.fill_between(budgets, qs_mean - qs_sem, qs_mean + qs_sem,
                     color="C3", alpha=0.3)

    plt.xlabel("Retraining Episodes After Goal Switch", fontsize=28)
    plt.ylabel(labels.get(metric, metric), fontsize=28)
    plt.legend(fontsize=20, loc="best")
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()

    fname = f"goal_revaluation_{metric}.png"
    plt.savefig(os.path.join(save_dir, fname), format="png")
    plt.close()
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
    parser.add_argument("--n_runs", type=int, default=nruns)
    parser.add_argument("--layout", type=str, default="serpentine",
                        choices=AVAILABLE_LAYOUTS,
                        help="Wall layout (default: serpentine)")
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
    n_macro = _layout.n_clusters

    args = argparse.Namespace(
        grid_size=grid_size,
        n_macro=n_macro,
        gamma=gamma,
        init_loc=init_loc,
        goal_A=goal_A,
        goal_B=goal_B,
        walls=WALLS,
        layout=args_cli.layout,
        initial_train_episodes=initial_train_episodes,
        retrain_budgets=retrain_budgets,
        n_eval=n_eval,
        max_steps=max_steps,
        n_runs=args_cli.n_runs if not args_cli.quick else nruns,
    )

    data_dir = f"data/eval/goal_revaluation/{args_cli.layout}"
    save_dir = f"figures/eval/goal_revaluation/{args_cli.layout}"

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

        t0 = time.time()
        results = run_goal_revaluation_experiment(args)
        elapsed = time.time() - t0
        print(f"\nExperiment completed in {elapsed:.0f}s")

        # Save all results
        for key, arr in results.items():
            np.save(os.path.join(data_dir, f"{key}.npy"), arr)
        print(f"\nSaved all data to {data_dir}/")

    else:
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

    # Phase 1 baseline
    if os.path.exists(os.path.join(data_dir, "p1_reward_sr_hier.npy")):
        plot_phase1_baseline(args, data_dir, save_dir)

    # Phase 2: retraining curves
    if os.path.exists(os.path.join(data_dir, "p2_reward_sr_hier.npy")):
        plot_retraining_curve(args, data_dir, save_dir, metric="reward")

    if os.path.exists(os.path.join(data_dir, "p2_steps_sr_hier.npy")):
        plot_retraining_curve(args, data_dir, save_dir, metric="steps")

    if os.path.exists(os.path.join(data_dir, "p2_reached_sr_hier.npy")):
        plot_retraining_curve(args, data_dir, save_dir, metric="reached")

    print(f"\nDone! Figures saved to {save_dir}/")
