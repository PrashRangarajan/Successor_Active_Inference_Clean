"""Evaluation: POMDP Gridworld benchmark comparing Hierarchy vs Flat.

Runs repeated experiments across training checkpoints to compare
hierarchical vs flat active inference on the POMDP Gridworld environment.
The agent receives noisy observations and must maintain beliefs over states.

Additionally tracks belief accuracy (how often the agent's belief matches
its true state) as a function of training episodes.

Saves .npy data files to data/eval/pomdp_gridworld/ and figures to
figures/eval/pomdp_gridworld/.

Usage:
    # Run experiments:
    python examples/run_eval_pomdp_gridworld.py --train

    # Plot from saved data:
    python examples/run_eval_pomdp_gridworld.py

    # Quick test (2 seeds × 3 checkpoints):
    python examples/run_eval_pomdp_gridworld.py --train --quick
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import json
import time

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-poster")

from core import HierarchicalSRAgent
from environments.pomdp_gridworld import POMDPGridworldAdapter
from unified_env import StandardGridworld as SR_Gridworld


# ==================== Utilities ====================


def relative_stability_paper_style(returns, Ke=100, smooth_window=1, eps=1e-8):
    """Paper-style 'Relative Stability' (lower is better)."""
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


def compute_belief_accuracy(adapter):
    """Compute belief accuracy from episode history.

    Returns fraction of steps where belief state == true state.
    """
    true_states = adapter.state_history
    belief_states = adapter.belief_history

    if len(true_states) == 0:
        return 0.0

    n_correct = sum(1 for t, b in zip(true_states, belief_states) if t == b)
    return n_correct / len(true_states)


# ==================== Agent Factory ====================


def create_pomdp_agent(grid_size, walls, noisy_states, n_clusters,
                        goal_loc, num_episodes, gamma=0.99,
                        noise_level=0.2, noise_spread=3.0, beta=0.1,
                        test_max_steps=200):
    """Create a fresh POMDP Gridworld SR agent trained for exactly num_episodes.

    Uses analytical M (learn_from_experience=False) since POMDP gridworld
    is a discrete environment where analytical computation is exact.

    Returns:
        (agent, adapter) tuple
    """
    env = SR_Gridworld(grid_size, noise=None)
    env.set_walls(walls)

    # Compute hallway indices from noisy state coordinates
    hallway_indices = [grid_size * pos[0] + pos[1] for pos in noisy_states]

    adapter = POMDPGridworldAdapter(
        env,
        grid_size=grid_size,
        noise_level=noise_level,
        noisy_states=hallway_indices,
        noise_spread=noise_spread,
        use_true_state_for_learning=False,  # Full POMDP mode
    )

    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=n_clusters,
        gamma=gamma,
        learning_rate=0.05,
        learn_from_experience=False,  # Analytical M for discrete gridworld
    )
    agent.set_goal(goal_loc, reward=100.0)

    # Create C with information gain term
    goal_states = adapter.get_goal_states(goal_loc)
    C = adapter.create_goal_prior_with_info_gain(
        goal_states, reward=100.0, default_cost=-0.1, beta=beta,
    )
    agent.C = C

    agent.learn_environment(num_episodes)

    return agent, adapter


# ==================== Experiment ====================


def pomdp_rewards_experiment(args):
    """Main experiment: rewards across training checkpoints for Hierarchy vs Flat.

    Also tracks belief accuracy and whether the agent truly reached the goal
    (vs just believing it did).

    Returns:
        Tuple of (SR_rewards_hier, SR_rewards_flat, SR_steps_hier, SR_steps_flat,
                  SR_belief_acc_hier, SR_belief_acc_flat,
                  SR_true_goal_hier, SR_true_goal_flat)
    """
    n_trials = len(args.episodes)

    SR_rewards_hier = np.zeros((args.n_runs, n_trials))
    SR_rewards_flat = np.zeros((args.n_runs, n_trials))
    SR_steps_hier = np.zeros((args.n_runs, n_trials))
    SR_steps_flat = np.zeros((args.n_runs, n_trials))
    SR_belief_acc_hier = np.zeros((args.n_runs, n_trials))
    SR_belief_acc_flat = np.zeros((args.n_runs, n_trials))
    SR_true_goal_hier = np.zeros((args.n_runs, n_trials))  # Actually reached goal
    SR_true_goal_flat = np.zeros((args.n_runs, n_trials))

    init_state = 0  # Top-left corner

    for n in range(args.n_runs):
        print("x" * 40)
        print(f"Run: {n + 1}/{args.n_runs}")
        print("x" * 40)

        for trial in range(n_trials):
            num_episodes = args.episodes[trial]

            print()
            print("+" * 25)
            print(f"{num_episodes} training episodes")
            print("+" * 25)

            # Retry loop for rare LinAlgError during spectral clustering
            while True:
                try:
                    agent, adapter = create_pomdp_agent(
                        args.grid_size, args.walls, args.noisy_states,
                        args.n_clusters, args.goal_loc, num_episodes,
                        gamma=args.gamma,
                        noise_level=args.noise_level,
                        noise_spread=args.noise_spread,
                        beta=args.beta,
                        test_max_steps=args.test_max_steps,
                    )
                except (np.linalg.LinAlgError, ValueError) as e:
                    print(f"  Error: {e} — retrying...")
                    continue
                else:
                    break

            # Evaluate hierarchy
            print("\nHierarchy")
            agent.reset_episode(init_state=init_state)
            result_hier = agent.run_episode_hierarchical(max_steps=args.test_max_steps)

            # Check true goal achievement
            true_state_hier = adapter.get_true_state_index()
            true_reached_hier = true_state_hier in agent.goal_states
            belief_acc_hier = compute_belief_accuracy(adapter)

            SR_rewards_hier[n, trial] = result_hier["reward"]
            SR_steps_hier[n, trial] = result_hier["steps"]
            SR_belief_acc_hier[n, trial] = belief_acc_hier
            SR_true_goal_hier[n, trial] = float(true_reached_hier)

            print(f"  Hier: reward={result_hier['reward']:.1f}, "
                  f"steps={result_hier['steps']}, "
                  f"believes_goal={result_hier['reached_goal']}, "
                  f"true_goal={true_reached_hier}, "
                  f"belief_acc={belief_acc_hier:.2f}")

            # Evaluate flat (same agent, same M)
            print("\nFlat")
            agent.reset_episode(init_state=init_state)
            result_flat = agent.run_episode_flat(max_steps=args.test_max_steps)

            # Check true goal achievement
            true_state_flat = adapter.get_true_state_index()
            true_reached_flat = true_state_flat in agent.goal_states
            belief_acc_flat = compute_belief_accuracy(adapter)

            SR_rewards_flat[n, trial] = result_flat["reward"]
            SR_steps_flat[n, trial] = result_flat["steps"]
            SR_belief_acc_flat[n, trial] = belief_acc_flat
            SR_true_goal_flat[n, trial] = float(true_reached_flat)

            print(f"  Flat: reward={result_flat['reward']:.1f}, "
                  f"steps={result_flat['steps']}, "
                  f"believes_goal={result_flat['reached_goal']}, "
                  f"true_goal={true_reached_flat}, "
                  f"belief_acc={belief_acc_flat:.2f}")

    return (SR_rewards_hier, SR_rewards_flat, SR_steps_hier, SR_steps_flat,
            SR_belief_acc_hier, SR_belief_acc_flat,
            SR_true_goal_hier, SR_true_goal_flat)


# ==================== Plotting ====================


def plot_pomdp_rewards(args, data_dir="data/eval/pomdp_gridworld",
                        save_dir="figures/eval/pomdp_gridworld"):
    """Plot reward curves with confidence bands (Hierarchy vs Flat)."""
    os.makedirs(save_dir, exist_ok=True)
    eps_range = args.episodes

    hier = np.load(os.path.join(data_dir, "SR_rewards_hierarchy.npy"))[:, :len(eps_range)]
    flat = np.load(os.path.join(data_dir, "SR_rewards_flat.npy"))[:, :len(eps_range)]

    mean_hier = np.mean(hier, axis=0)
    std_hier = np.std(hier, axis=0) / np.sqrt(len(hier))
    mean_flat = np.mean(flat, axis=0)
    std_flat = np.std(flat, axis=0) / np.sqrt(len(flat))

    fig = plt.figure(figsize=(14, 10))
    plt.plot(eps_range, mean_hier, label="Hierarchy")
    plt.fill_between(eps_range, mean_hier - std_hier, mean_hier + std_hier, alpha=0.5)
    plt.plot(eps_range, mean_flat, label="Flat")
    plt.fill_between(eps_range, mean_flat - std_flat, mean_flat + std_flat, alpha=0.5)

    plt.xlabel("Number of Training Episodes", fontsize=28)
    plt.ylabel("Total Reward", fontsize=28)
    plt.title("POMDP Gridworld: Reward vs Training", fontsize=28)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pomdp_reward.png"), format="png")
    plt.close()
    print(f"  Saved {save_dir}/pomdp_reward.png")


def plot_pomdp_steps(args, data_dir="data/eval/pomdp_gridworld",
                      save_dir="figures/eval/pomdp_gridworld"):
    """Plot steps-to-goal curves (Hierarchy vs Flat)."""
    os.makedirs(save_dir, exist_ok=True)
    eps_range = args.episodes

    hier = np.load(os.path.join(data_dir, "SR_steps_hierarchy.npy"))[:, :len(eps_range)]
    flat = np.load(os.path.join(data_dir, "SR_steps_flat.npy"))[:, :len(eps_range)]

    mean_hier = np.mean(hier, axis=0)
    std_hier = np.std(hier, axis=0) / np.sqrt(len(hier))
    mean_flat = np.mean(flat, axis=0)
    std_flat = np.std(flat, axis=0) / np.sqrt(len(flat))

    fig = plt.figure(figsize=(14, 10))
    plt.plot(eps_range, mean_hier, label="Hierarchy")
    plt.fill_between(eps_range, mean_hier - std_hier, mean_hier + std_hier, alpha=0.5)
    plt.plot(eps_range, mean_flat, label="Flat")
    plt.fill_between(eps_range, mean_flat - std_flat, mean_flat + std_flat, alpha=0.5)

    plt.xlabel("Number of Training Episodes", fontsize=28)
    plt.ylabel("Steps to Goal", fontsize=28)
    plt.title("POMDP Gridworld: Steps vs Training", fontsize=28)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pomdp_steps.png"), format="png")
    plt.close()
    print(f"  Saved {save_dir}/pomdp_steps.png")


def plot_pomdp_belief_accuracy(args, data_dir="data/eval/pomdp_gridworld",
                                save_dir="figures/eval/pomdp_gridworld"):
    """Plot belief accuracy curves (Hierarchy vs Flat)."""
    os.makedirs(save_dir, exist_ok=True)
    eps_range = args.episodes

    hier = np.load(os.path.join(data_dir, "SR_belief_acc_hierarchy.npy"))[:, :len(eps_range)]
    flat = np.load(os.path.join(data_dir, "SR_belief_acc_flat.npy"))[:, :len(eps_range)]

    mean_hier = np.mean(hier, axis=0)
    std_hier = np.std(hier, axis=0) / np.sqrt(len(hier))
    mean_flat = np.mean(flat, axis=0)
    std_flat = np.std(flat, axis=0) / np.sqrt(len(flat))

    fig = plt.figure(figsize=(14, 10))
    plt.plot(eps_range, mean_hier, label="Hierarchy")
    plt.fill_between(eps_range, mean_hier - std_hier, mean_hier + std_hier, alpha=0.5)
    plt.plot(eps_range, mean_flat, label="Flat")
    plt.fill_between(eps_range, mean_flat - std_flat, mean_flat + std_flat, alpha=0.5)

    plt.xlabel("Number of Training Episodes", fontsize=28)
    plt.ylabel("Belief Accuracy", fontsize=28)
    plt.title("POMDP Gridworld: Belief Accuracy vs Training", fontsize=28)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pomdp_belief_accuracy.png"), format="png")
    plt.close()
    print(f"  Saved {save_dir}/pomdp_belief_accuracy.png")


def plot_pomdp_true_goal_rate(args, data_dir="data/eval/pomdp_gridworld",
                               save_dir="figures/eval/pomdp_gridworld"):
    """Plot true goal achievement rate (Hierarchy vs Flat)."""
    os.makedirs(save_dir, exist_ok=True)
    eps_range = args.episodes

    hier = np.load(os.path.join(data_dir, "SR_true_goal_hierarchy.npy"))[:, :len(eps_range)]
    flat = np.load(os.path.join(data_dir, "SR_true_goal_flat.npy"))[:, :len(eps_range)]

    # Goal rate = fraction of runs that actually reached goal
    rate_hier = np.mean(hier, axis=0)
    rate_flat = np.mean(flat, axis=0)

    fig = plt.figure(figsize=(14, 10))
    plt.plot(eps_range, rate_hier, 'o-', label="Hierarchy", markersize=8)
    plt.plot(eps_range, rate_flat, 's-', label="Flat", markersize=8)

    plt.xlabel("Number of Training Episodes", fontsize=28)
    plt.ylabel("True Goal Achievement Rate", fontsize=28)
    plt.title("POMDP Gridworld: True Goal Rate vs Training", fontsize=28)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pomdp_true_goal_rate.png"), format="png")
    plt.close()
    print(f"  Saved {save_dir}/pomdp_true_goal_rate.png")


def plot_pomdp_stability(args, data_dir="data/eval/pomdp_gridworld",
                          save_dir="figures/eval/pomdp_gridworld"):
    """Plot relative stability bar chart (Hierarchy vs Flat)."""
    os.makedirs(save_dir, exist_ok=True)

    hier_path = os.path.join(data_dir, "SR_relative_stability_hierarchy.npy")
    flat_path = os.path.join(data_dir, "SR_relative_stability_flat.npy")

    labels, means, sems = [], [], []

    if os.path.exists(hier_path):
        st = np.load(hier_path)
        labels.append("Hierarchy")
        means.append(float(np.mean(st)))
        sems.append(float(np.std(st) / np.sqrt(len(st))))

    if os.path.exists(flat_path):
        st = np.load(flat_path)
        labels.append("Flat")
        means.append(float(np.mean(st)))
        sems.append(float(np.std(st) / np.sqrt(len(st))))

    if len(labels) == 0:
        print("  No stability data found — skipping")
        return

    color_map = {"Hierarchy": "C0", "Flat": "C1"}
    bar_colors = [color_map.get(label, "C0") for label in labels]

    fig = plt.figure(figsize=(10, 8))
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=sems, capsize=8, color=bar_colors)
    plt.xticks(x, labels, fontsize=26)
    plt.yticks(fontsize=26)
    plt.ylabel("Relative Stability", fontsize=20)
    plt.title("POMDP Gridworld: Relative Stability", fontsize=22)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pomdp_relative_stability.png"), format="png")
    plt.close()
    print(f"  Saved {save_dir}/pomdp_relative_stability.png")


# ==================== Main ====================


if __name__ == "__main__":
    # POMDP Gridworld configuration
    grid_size = 9
    n_clusters = 4  # One per room
    gamma = 0.99
    nruns = 10
    eps = [100, 250, 500, 1000, 1500, 2500, 5000]
    test_max_steps = 200

    # POMDP-specific settings
    noise_level = 0.3   # 30% base noise — harder POMDP
    noise_spread = 3.0  # Noisy room is 3x noisier than base (90% noise)
    beta = 1.0  # Information gain weight — strong enough to reroute around noisy room

    init_loc = (0, 0)
    goal_loc = (grid_size - 1, grid_size - 1)

    # 4-rooms layout on 9x9 grid
    # Vertical wall at x=4 (rows 0-8), horizontal wall at y=4 (cols 0-8)
    # With doorways at specific positions
    mid = grid_size // 2  # 4

    # Vertical wall (x=mid for all y, except doorways)
    vertical_door_y = [1, 7]  # doorways in vertical wall
    vertical_wall = [(mid, y) for y in range(grid_size) if y not in vertical_door_y]

    # Horizontal wall (y=mid for all x, except doorways)
    horizontal_door_x = [1, 7]  # doorways in horizontal wall
    horizontal_wall = [(x, mid) for x in range(grid_size) if x not in horizontal_door_x]

    walls = vertical_wall + horizontal_wall

    # Asymmetric noise: top-right room is heavily noisy
    # Agent at (0,0), goal at (8,8). Direct route goes through top-right room.
    # With high noise there, the EFE-aware agent should prefer the longer
    # route through the bottom-left room (clean path).
    #
    # Room layout:
    #   Top-left (0-3, 0-3):     Agent start - CLEAN
    #   Top-right (5-8, 0-3):    Direct route - NOISY
    #   Bottom-left (0-3, 5-8):  Alternative  - CLEAN
    #   Bottom-right (5-8, 5-8): Goal         - CLEAN
    noisy_states = [
        (x, y)
        for x in range(mid + 1, grid_size)
        for y in range(mid)
        if (x, y) not in walls
    ]

    parser = argparse.ArgumentParser(description="POMDP Gridworld Eval: Hierarchy vs Flat")
    parser.add_argument("--train", action="store_true", help="Run experiments")
    parser.add_argument("--quick", action="store_true", help="Quick test")
    parser.add_argument("--n_runs", type=int, default=nruns)
    args_cli = parser.parse_args()

    if args_cli.quick:
        eps = [500, 2000, 5000]
        nruns = 2

    args = argparse.Namespace(
        grid_size=grid_size,
        n_clusters=n_clusters,
        gamma=gamma,
        n_runs=args_cli.n_runs if not args_cli.quick else nruns,
        episodes=eps,
        test_max_steps=test_max_steps,
        noise_level=noise_level,
        noise_spread=noise_spread,
        beta=beta,
        init_loc=init_loc,
        goal_loc=goal_loc,
        noisy_states=noisy_states,
        walls=walls,
    )

    data_dir = "data/eval/pomdp_gridworld"
    save_dir = "figures/eval/pomdp_gridworld"

    if args_cli.train:
        os.makedirs(data_dir, exist_ok=True)

        # Save args
        args_save = vars(args).copy()
        args_save["init_loc"] = list(args_save["init_loc"])
        args_save["goal_loc"] = list(args_save["goal_loc"])
        args_save["noisy_states"] = [list(s) for s in args_save["noisy_states"]]
        with open(os.path.join(data_dir, "args.json"), "w") as f:
            json.dump(args_save, f, indent=2)

        print("=" * 60)
        print("POMDP GRIDWORLD EVAL: Hierarchy vs Flat")
        print("=" * 60)
        print(f"Grid: {grid_size}x{grid_size}, States: {grid_size**2}")
        print(f"Noise level: {noise_level}, Noisy states: {noisy_states}")
        print(f"Info gain beta: {beta}")
        print(f"Runs: {args.n_runs}, Checkpoints: {args.episodes}")

        t0 = time.time()
        (SR_rewards_hier, SR_rewards_flat, SR_steps_hier, SR_steps_flat,
         SR_belief_acc_hier, SR_belief_acc_flat,
         SR_true_goal_hier, SR_true_goal_flat) = pomdp_rewards_experiment(args)
        elapsed = time.time() - t0
        print(f"\nExperiment completed in {elapsed:.0f}s")

        # Compute relative stability
        SR_rel_stability_hier = np.array([
            relative_stability_paper_style(SR_rewards_hier[i, :])
            for i in range(SR_rewards_hier.shape[0])
        ])
        SR_rel_stability_flat = np.array([
            relative_stability_paper_style(SR_rewards_flat[i, :])
            for i in range(SR_rewards_flat.shape[0])
        ])

        # Save data
        np.save(os.path.join(data_dir, "SR_rewards_hierarchy.npy"), SR_rewards_hier)
        np.save(os.path.join(data_dir, "SR_rewards_flat.npy"), SR_rewards_flat)
        np.save(os.path.join(data_dir, "SR_steps_hierarchy.npy"), SR_steps_hier)
        np.save(os.path.join(data_dir, "SR_steps_flat.npy"), SR_steps_flat)
        np.save(os.path.join(data_dir, "SR_belief_acc_hierarchy.npy"), SR_belief_acc_hier)
        np.save(os.path.join(data_dir, "SR_belief_acc_flat.npy"), SR_belief_acc_flat)
        np.save(os.path.join(data_dir, "SR_true_goal_hierarchy.npy"), SR_true_goal_hier)
        np.save(os.path.join(data_dir, "SR_true_goal_flat.npy"), SR_true_goal_flat)
        np.save(os.path.join(data_dir, "SR_relative_stability_hierarchy.npy"), SR_rel_stability_hier)
        np.save(os.path.join(data_dir, "SR_relative_stability_flat.npy"), SR_rel_stability_flat)
        print(f"\nSaved all data to {data_dir}/")

    else:
        # Load saved args
        args_path = os.path.join(data_dir, "args.json")
        if os.path.exists(args_path):
            with open(args_path, "r") as f:
                saved = json.load(f)
                saved["init_loc"] = tuple(saved["init_loc"])
                saved["goal_loc"] = tuple(saved["goal_loc"])
                saved["noisy_states"] = [tuple(s) for s in saved["noisy_states"]]
                args = argparse.Namespace(**saved)
            print(f"Loaded args: {args}")
        else:
            print("No saved args found. Run with --train first.")

    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(os.path.join(data_dir, "SR_rewards_hierarchy.npy")):
        plot_pomdp_rewards(args, data_dir=data_dir, save_dir=save_dir)

    if os.path.exists(os.path.join(data_dir, "SR_steps_hierarchy.npy")):
        plot_pomdp_steps(args, data_dir=data_dir, save_dir=save_dir)

    if os.path.exists(os.path.join(data_dir, "SR_belief_acc_hierarchy.npy")):
        plot_pomdp_belief_accuracy(args, data_dir=data_dir, save_dir=save_dir)

    if os.path.exists(os.path.join(data_dir, "SR_true_goal_hierarchy.npy")):
        plot_pomdp_true_goal_rate(args, data_dir=data_dir, save_dir=save_dir)

    plot_pomdp_stability(args, data_dir=data_dir, save_dir=save_dir)

    print(f"\nDone! Figures saved to {save_dir}/")
