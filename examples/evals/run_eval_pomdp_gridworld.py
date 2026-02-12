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

import os

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import argparse
import json
import time
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-poster")

from core import HierarchicalSRAgent
from core.eval_utils import (
    relative_stability,
    compute_stability_array,
    plot_reward_curves,
    plot_step_curves,
    plot_stability_bars,
    save_eval_data,
    load_eval_args,
)
from environments.pomdp_gridworld import POMDPGridworldAdapter
from environments.gridworld import get_layout, AVAILABLE_LAYOUTS
from examples.configs import POMDP_GRIDWORLD
from unified_env import StandardGridworld as SR_Gridworld

# ==================== Utilities ====================

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

# ==================== POMDP-Specific Plotting ====================

def plot_pomdp_belief_accuracy(args, data_dir, save_dir):
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

def plot_pomdp_true_goal_rate(args, data_dir, save_dir):
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

# ==================== Main ====================

if __name__ == "__main__":
    # POMDP Gridworld configuration (from centralized config)
    grid_size = POMDP_GRIDWORLD["grid_size"]
    gamma = POMDP_GRIDWORLD["gamma"]
    nruns = POMDP_GRIDWORLD["eval_n_runs"]
    eps = list(POMDP_GRIDWORLD["eval_episodes"])
    test_max_steps = POMDP_GRIDWORLD["test_max_steps"]

    # POMDP-specific settings
    noise_level = POMDP_GRIDWORLD["noise_level"]
    noise_spread = POMDP_GRIDWORLD["noise_spread"]
    beta = POMDP_GRIDWORLD["beta"]

    init_loc = POMDP_GRIDWORLD["init_loc"]

    parser = argparse.ArgumentParser(description="POMDP Gridworld Eval: Hierarchy vs Flat")
    parser.add_argument("--train", action="store_true", help="Run experiments")
    parser.add_argument("--quick", action="store_true", help="Quick test")
    parser.add_argument("--n_runs", type=int, default=nruns)
    parser.add_argument("--layout", type=str, default="fourrooms",
                        choices=AVAILABLE_LAYOUTS,
                        help="Wall layout (default: fourrooms)")
    args_cli = parser.parse_args()

    # Get layout-specific configuration
    _layout = get_layout(args_cli.layout, grid_size)
    n_clusters = _layout.n_clusters
    goal_loc = _layout.default_goal
    walls = _layout.walls
    wall_set = set(walls)

    # Asymmetric noise: top-right quadrant is heavily noisy.
    # Agent at (0,0), goal at bottom-right. Direct route goes through
    # top-right area. With high noise there, the EFE-aware agent should
    # prefer the longer route through cleaner areas.
    mid = grid_size // 2
    noisy_states = [
        (x, y)
        for x in range(mid + 1, grid_size)
        for y in range(mid)
        if (x, y) not in wall_set
    ]

    if args_cli.quick:
        eps = list(POMDP_GRIDWORLD["eval_quick_episodes"])
        nruns = POMDP_GRIDWORLD["eval_quick_n_runs"]

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

    data_dir = f"data/eval/pomdp_gridworld/{args_cli.layout}"
    save_dir = f"figures/eval/pomdp_gridworld/{args_cli.layout}"

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
        SR_rel_stability_hier = compute_stability_array(SR_rewards_hier)
        SR_rel_stability_flat = compute_stability_array(SR_rewards_flat)

        # Save standard eval data
        save_eval_data(data_dir, {
            "SR_rewards_hierarchy": SR_rewards_hier,
            "SR_rewards_flat": SR_rewards_flat,
            "SR_steps_hierarchy": SR_steps_hier,
            "SR_steps_flat": SR_steps_flat,
            "SR_relative_stability_hierarchy": SR_rel_stability_hier,
            "SR_relative_stability_flat": SR_rel_stability_flat,
        })

        # Save POMDP-specific metrics (belief accuracy, true goal reached)
        np.save(os.path.join(data_dir, "SR_belief_acc_hierarchy.npy"), SR_belief_acc_hier)
        np.save(os.path.join(data_dir, "SR_belief_acc_flat.npy"), SR_belief_acc_flat)
        np.save(os.path.join(data_dir, "SR_true_goal_hierarchy.npy"), SR_true_goal_hier)
        np.save(os.path.join(data_dir, "SR_true_goal_flat.npy"), SR_true_goal_flat)
        print(f"  Saved POMDP-specific metrics to {data_dir}/")

    else:
        # Load saved args
        args = load_eval_args(data_dir, tuple_keys=["init_loc", "goal_loc", "noisy_states"])
        if args is None:
            print("No saved args found. Run with --train first.")

    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(os.path.join(data_dir, "SR_rewards_hierarchy.npy")):
        data = OrderedDict()
        data["Hierarchy"] = np.load(os.path.join(data_dir, "SR_rewards_hierarchy.npy"))
        data["Flat"] = np.load(os.path.join(data_dir, "SR_rewards_flat.npy"))
        plot_reward_curves(args.episodes, data, os.path.join(save_dir, "pomdp_reward.png"))

    if os.path.exists(os.path.join(data_dir, "SR_steps_hierarchy.npy")):
        data = OrderedDict()
        data["Hierarchy"] = np.load(os.path.join(data_dir, "SR_steps_hierarchy.npy"))
        data["Flat"] = np.load(os.path.join(data_dir, "SR_steps_flat.npy"))
        plot_step_curves(args.episodes, data, os.path.join(save_dir, "pomdp_steps.png"))

    if os.path.exists(os.path.join(data_dir, "SR_belief_acc_hierarchy.npy")):
        plot_pomdp_belief_accuracy(args, data_dir=data_dir, save_dir=save_dir)

    if os.path.exists(os.path.join(data_dir, "SR_true_goal_hierarchy.npy")):
        plot_pomdp_true_goal_rate(args, data_dir=data_dir, save_dir=save_dir)

    stability_data = OrderedDict()
    hier_stab_path = os.path.join(data_dir, "SR_relative_stability_hierarchy.npy")
    flat_stab_path = os.path.join(data_dir, "SR_relative_stability_flat.npy")
    if os.path.exists(hier_stab_path):
        stability_data["Hierarchy"] = np.load(hier_stab_path)
    if os.path.exists(flat_stab_path):
        stability_data["Flat"] = np.load(flat_stab_path)
    plot_stability_bars(stability_data, os.path.join(save_dir, "pomdp_relative_stability.png"))

    print(f"\nDone! Figures saved to {save_dir}/")
