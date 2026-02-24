"""Noise-avoidance experiment for POMDP Active Inference.

Replicates the key result from the paper (Figures 14 & 15): an Active
Inference agent that penalises observation entropy (high beta) avoids
the noisy room and takes the longer clean path, while an agent that
ignores entropy (beta=0) takes the shorter path through noise.

Uses the **fiverooms** layout (9x9):

    LT ──(4,0)── RT (noisy)
    │              │
  (2,4)          (6,2)
    │              │
    LB            RM  ← goal at (6,4)
    │              │
  (4,8)          (6,6)
    │              │
          RB

  Short path (~11 steps): LT → RT (noisy) → RM
  Long  path (~20 steps): LT → LB → RB → RM

Beta semantics (as the code defines it):
  C = C_reward - beta * entropy(A)
  Higher beta → larger entropy penalty → agent avoids noisy room

Produces two figures:
  1. Two-panel trajectory comparison (high beta vs zero beta)
  2. P(noisy-room path) vs observation entropy sweep
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from unified_env import StandardGridworld as SR_Gridworld
from environments.pomdp_gridworld import POMDPGridworldAdapter
from environments.gridworld import get_layout
from core.hierarchical_agent import HierarchicalSRAgent


FIG_DIR = "figures/noise_avoidance"
GRID_SIZE = 9
START_LOC = (0, 0)   # left-top room
GOAL_LOC = (6, 4)    # right-middle room


# ==================== Entropy <-> mixing helpers ====================

def _column_entropy(alpha, N):
    """Entropy of an observation column with mixing parameter *alpha*.

    The column has P(o=s|s) = 1-alpha+alpha/N for the true state
    and P(o!=s|s) = alpha/N for each of the remaining N-1 states.
    """
    p_true = 1.0 - alpha + alpha / N
    p_other = alpha / N
    H = -p_true * np.log(p_true + 1e-15)
    if N > 1 and p_other > 0:
        H -= (N - 1) * p_other * np.log(p_other + 1e-15)
    return H


def _alpha_for_entropy(H_target, N):
    """Invert _column_entropy: find alpha that yields H_target nats."""
    H_min = _column_entropy(0.0, N)
    H_max = _column_entropy(1.0, N)
    if H_target <= H_min:
        return 0.0
    if H_target >= H_max:
        return 1.0
    return brentq(lambda a: _column_entropy(a, N) - H_target, 0.0, 1.0)


# ==================== Room geometry (fiverooms) ====================

def _get_rooms(wall_set):
    """Return dict mapping room names to sets of (x, y) cells.

    Fiverooms layout (9x9):
        LT: x in [0,3], y in [0,3]   (left-top, large)
        LB: x in [0,3], y in [5,8]   (left-bottom, large)
        RT: x in [5,8], y in [0,1]   (right-top, small -- the noisy room)
        RM: x in [5,8], y in [3,5]   (right-middle -- contains goal)
        RB: x in [5,8], y in [7,8]   (right-bottom, small)

    Doorways (not assigned to any room):
        (4,0) LT<->RT, (2,4) LT<->LB, (6,2) RT<->RM,
        (4,8) LB<->RB, (6,6) RM<->RB
    """
    rooms = {}
    rooms["LT"] = {(x, y) for x in range(4) for y in range(4)
                    if (x, y) not in wall_set}
    rooms["LB"] = {(x, y) for x in range(4) for y in range(5, GRID_SIZE)
                    if (x, y) not in wall_set}
    rooms["RT"] = {(x, y) for x in range(5, GRID_SIZE) for y in range(2)
                    if (x, y) not in wall_set}
    rooms["RM"] = {(x, y) for x in range(5, GRID_SIZE) for y in range(3, 6)
                    if (x, y) not in wall_set}
    rooms["RB"] = {(x, y) for x in range(5, GRID_SIZE) for y in range(7, GRID_SIZE)
                    if (x, y) not in wall_set}
    return rooms


# ==================== Environment Setup ====================

def setup_environment_with_entropy(target_entropy, noise_level=0.3,
                                   beta=1.0):
    """Create fiverooms POMDP where the right-top room has exactly
    *target_entropy* nats of observation entropy.

    Directly constructs the A-matrix columns for the noisy-room states
    using the mixing parameter that yields the desired entropy.
    The agent learns the environment under these noise conditions.

    Args:
        target_entropy: Desired entropy (nats) for the RT room states.
        noise_level: Base noise level for all other states.
        beta: Entropy penalty weight for C = C_reward - beta * H(A).

    Returns:
        agent, adapter, goal_states, wall_set, rooms
    """
    layout = get_layout("fiverooms", GRID_SIZE)
    walls = layout.walls
    wall_set = set(walls)
    rooms = _get_rooms(wall_set)

    env = SR_Gridworld(GRID_SIZE, noise=None)
    env.set_walls(walls)

    # Build adapter with base noise only (no extra noisy states yet)
    adapter = POMDPGridworldAdapter(
        env,
        grid_size=GRID_SIZE,
        noise_level=noise_level,
        noisy_states=None,
        use_true_state_for_learning=False,
    )

    # Compute mixing parameter for the target entropy
    N = adapter.n_states
    alpha = _alpha_for_entropy(target_entropy, N)

    # Directly set A columns for the right-top room states
    noisy_indices = [GRID_SIZE * x + y for (x, y) in rooms["RT"]]
    for s_idx in noisy_indices:
        adapter._A[:, s_idx] = alpha / N
        adapter._A[s_idx, s_idx] += (1.0 - alpha)

    # Re-normalise (should already sum to 1, but be safe)
    adapter._A = adapter._A / adapter._A.sum(axis=0, keepdims=True)

    # Push updated A into environment so observations match the model
    adapter._env.set_likelihood_dist(adapter._A)

    goal_states = adapter.get_goal_states(GOAL_LOC)

    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=layout.n_clusters,
        gamma=0.99,
        learning_rate=0.05,
        learn_from_experience=False,
    )
    agent.set_goal(GOAL_LOC, reward=100.0)

    # Set C with information-gain penalty
    C = adapter.create_goal_prior_with_info_gain(
        goal_states, reward=100.0, default_cost=-0.1, beta=beta)
    agent.C = C

    return agent, adapter, goal_states, wall_set, rooms


# ==================== Path Classification ====================

def classify_path(adapter, rooms):
    """Classify the last episode trajectory as 'noisy' or 'clean'.

    'noisy'   -- trajectory visited RT (right-top, short path)
    'clean'   -- trajectory visited LB or RB (long path)
    'neither' -- didn't clearly pass through either route
    """
    visited_rt = False
    visited_long = False   # LB or RB

    for s_idx in adapter.state_history:
        loc = adapter.render_state(s_idx)
        if loc in rooms["RT"]:
            visited_rt = True
        if loc in rooms["LB"] or loc in rooms["RB"]:
            visited_long = True

    if visited_rt and not visited_long:
        return "noisy"
    elif visited_long and not visited_rt:
        return "clean"
    elif visited_rt and visited_long:
        return "noisy"       # visited both -- count as noisy
    else:
        return "neither"


# ==================== Trajectory Comparison (Figure 1) ====================

def _draw_episode_panel(ax, adapter, agent, wall_set,
                        noisy_cells, title):
    """Draw a single trajectory panel on *ax*."""
    entropy_vals = adapter.get_observation_entropy()
    entropy_grid = entropy_vals.reshape(GRID_SIZE, GRID_SIZE).T

    ax.imshow(entropy_grid, cmap='YlOrRd', interpolation='nearest', alpha=0.7)

    # Noisy zone outline
    for (x, y) in noisy_cells:
        ax.add_patch(plt.Rectangle(
            (x - 0.5, y - 0.5), 1, 1,
            facecolor='none', edgecolor='red',
            linewidth=2, linestyle='--'))

    # Walls
    for w in adapter.get_wall_indices():
        loc = adapter.render_state(w)
        ax.add_patch(plt.Rectangle(
            (loc[0] - 0.5, loc[1] - 0.5), 1, 1,
            facecolor='black', edgecolor='white', linewidth=0.8))

    # Trajectory (true states)
    locs = [adapter.render_state(s) for s in adapter.state_history]
    xs = [loc[0] for loc in locs]
    ys = [loc[1] for loc in locs]
    ax.plot(xs, ys, 'b-o', linewidth=2, markersize=3, alpha=0.85, zorder=4)

    # Start & goal markers
    ax.scatter(xs[0], ys[0], color='blue', s=220, marker='o',
               zorder=5, edgecolors='white', linewidths=2, label='Start')
    goal_loc = adapter.render_state(agent.goal_states[0])
    ax.scatter(goal_loc[0], goal_loc[1], color='lime', s=300, marker='*',
               zorder=5, edgecolors='black', linewidths=1.5, label='Goal')

    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(GRID_SIZE - 0.5, -0.5)
    ax.set_xticks(np.arange(GRID_SIZE))
    ax.set_yticks(np.arange(GRID_SIZE))
    ax.set_xticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, GRID_SIZE, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.4)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_title(title, fontsize=13)
    ax.legend(loc='upper right', fontsize=9)


def plot_trajectory_comparison(beta_high=2.0, noise_level=0.3):
    """Produce a two-panel figure: high-beta vs zero-beta trajectories.

    Both panels share the same noise regime (same A matrix).
    M and B don't depend on A, so we learn once and vary only C.

    Panel (a): beta=0 -> agent ignores entropy -> takes short path through RT
    Panel (b): beta=beta_high -> agent penalises entropy -> avoids RT
    """
    os.makedirs(FIG_DIR, exist_ok=True)

    N = GRID_SIZE ** 2
    # High entropy for noisy room (alpha=0.85 gives ~94% of max entropy)
    H_noisy = _column_entropy(0.85, N)

    # Build environment with noisy RT room; learn once
    # (beta=0 for setup -- we'll set C ourselves for each panel)
    agent, adapter, goal_states, wall_set, rooms = \
        setup_environment_with_entropy(H_noisy, noise_level=noise_level,
                                       beta=0.0)
    print("  Learning environment dynamics...")
    agent.learn_environment(num_episodes=2000)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Panel (a): beta = 0 (no entropy penalty) ---
    C_no_info = adapter.create_goal_prior(goal_states, reward=100.0,
                                           default_cost=-0.1)
    agent.C = C_no_info
    agent.reset_episode(init_state=0)
    result_a = agent.run_episode_flat(max_steps=200)
    path_a = classify_path(adapter, rooms)
    _draw_episode_panel(axes[0], adapter, agent, wall_set,
                        rooms["RT"],
                        fr"(a) $\beta = 0$ — {path_a} path"
                        f" ({result_a['steps']} steps)")
    print(f"  Panel (a): beta=0  -> {path_a} path, {result_a['steps']} steps")

    # --- Panel (b): beta = beta_high ---
    C_info = adapter.create_goal_prior_with_info_gain(
        goal_states, reward=100.0, default_cost=-0.1, beta=beta_high)
    agent.C = C_info
    agent.reset_episode(init_state=0)
    result_b = agent.run_episode_flat(max_steps=200)
    path_b = classify_path(adapter, rooms)
    _draw_episode_panel(axes[1], adapter, agent, wall_set,
                        rooms["RT"],
                        fr"(b) $\beta = {beta_high}$ — {path_b} path"
                        f" ({result_b['steps']} steps)")
    print(f"  Panel (b): beta={beta_high}  -> {path_b} path, "
          f"{result_b['steps']} steps")

    fig.suptitle("Noise Avoidance: Trajectory Comparison (fiverooms)",
                 fontsize=16, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(FIG_DIR, "trajectory_comparison.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved to {save_path}")


# ==================== Entropy Sweep (Figure 2) ====================

def run_entropy_sweep(n_episodes=50, beta=1.0, entropies=None,
                      noise_level=0.3):
    """Sweep observation entropy in the RT room and compute
    P(noisy path) for each value.

    For each target entropy a fresh adapter+agent is built (with the
    A-matrix columns for the noisy room set to achieve that entropy),
    the agent learns the environment under those noise conditions,
    and then runs N episodes.

    Returns:
        entropies, fractions (arrays of same length)
    """
    N = GRID_SIZE ** 2

    if entropies is None:
        H_base = _column_entropy(noise_level, N)
        H_max = _column_entropy(0.95, N)
        entropies = np.linspace(H_base, H_max, 15)

    fractions = []

    for i, H_target in enumerate(entropies):
        agent, adapter, goal_states, wall_set, rooms = \
            setup_environment_with_entropy(H_target,
                                           noise_level=noise_level,
                                           beta=beta)

        # Learn environment under this noise regime
        agent.learn_environment(num_episodes=2000)

        # Verify actual entropy in noisy room
        entropy_vals = adapter.get_observation_entropy()
        noisy_indices = [GRID_SIZE * x + y for (x, y) in rooms["RT"]]
        actual_H = np.mean(entropy_vals[noisy_indices])

        noisy_count = 0
        valid_count = 0
        for _ in range(n_episodes):
            agent.reset_episode(init_state=0)
            agent.run_episode_flat(max_steps=200)
            label = classify_path(adapter, rooms)
            if label in ("noisy", "clean"):
                valid_count += 1
                if label == "noisy":
                    noisy_count += 1

        frac = noisy_count / valid_count if valid_count > 0 else np.nan
        fractions.append(frac)
        alpha = _alpha_for_entropy(H_target, N)
        print(f"  [{i+1}/{len(entropies)}] H_target={H_target:.3f}  "
              f"actual={actual_H:.3f}  alpha={alpha:.4f}  "
              f"noisy={noisy_count}/{valid_count}  P(noisy)={frac:.2f}")

    return np.array(entropies), np.array(fractions)


def plot_entropy_sweep(entropies, fractions, n_episodes=50,
                       noise_level=0.3, beta=2.0):
    """Plot P(noisy path) vs observation entropy in the RT room."""
    os.makedirs(FIG_DIR, exist_ok=True)

    # Binomial 95% CI
    se = np.sqrt(np.clip(fractions * (1 - fractions), 0, None)
                 / max(n_episodes, 1))
    ci = 1.96 * se

    # Baseline entropy (clean states)
    N = GRID_SIZE ** 2
    H_base = _column_entropy(noise_level, N)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(entropies, fractions, 'o-', color='#2c7bb6', linewidth=2,
            markersize=5, label='P(short noisy path)')
    ax.fill_between(entropies,
                    np.clip(fractions - ci, 0, 1),
                    np.clip(fractions + ci, 0, 1),
                    alpha=0.2, color='#2c7bb6')
    ax.axvline(H_base, color='green', linestyle=':', alpha=0.6,
               label=f'Baseline H = {H_base:.2f}')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    ax.set_xlabel('Observation entropy in noisy room (nats)', fontsize=13)
    ax.set_ylabel('P(path through noisy room)', fontsize=13)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(FIG_DIR, "entropy_sweep.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Entropy sweep saved to {save_path}")


# ==================== Beta Sweep (Figure 3) ====================

def run_beta_sweep(n_episodes=50, betas=None, noise_level=0.3):
    """Sweep beta (entropy penalty weight) with fixed high noise in RT.

    Builds the environment once (fixed A matrix with noisy RT room),
    learns once, then varies only C = C_reward - beta * H(A) for each beta.

    Returns:
        betas, fractions (arrays of same length)
    """
    if betas is None:
        betas = np.linspace(0, 5, 20)

    N = GRID_SIZE ** 2
    H_noisy = _column_entropy(0.85, N)  # high entropy in RT

    # Build and learn once (A and B are fixed across all betas)
    agent, adapter, goal_states, wall_set, rooms = \
        setup_environment_with_entropy(H_noisy, noise_level=noise_level,
                                       beta=0.0)
    agent.learn_environment(num_episodes=2000)

    fractions = []

    for i, beta in enumerate(betas):
        # Only recompute C for the new beta
        C = adapter.create_goal_prior_with_info_gain(
            goal_states, reward=100.0, default_cost=-0.1, beta=beta)
        agent.C = C

        noisy_count = 0
        valid_count = 0
        for _ in range(n_episodes):
            agent.reset_episode(init_state=0)
            agent.run_episode_flat(max_steps=200)
            label = classify_path(adapter, rooms)
            if label in ("noisy", "clean"):
                valid_count += 1
                if label == "noisy":
                    noisy_count += 1

        frac = noisy_count / valid_count if valid_count > 0 else np.nan
        fractions.append(frac)
        print(f"  [{i+1}/{len(betas)}] beta={beta:.2f}  "
              f"noisy={noisy_count}/{valid_count}  P(noisy)={frac:.2f}")

    return np.array(betas), np.array(fractions)


def plot_beta_sweep(betas, fractions, n_episodes=50):
    """Plot P(noisy path) vs beta (entropy penalty weight)."""
    os.makedirs(FIG_DIR, exist_ok=True)

    # Binomial 95% CI
    se = np.sqrt(np.clip(fractions * (1 - fractions), 0, None)
                 / max(n_episodes, 1))
    ci = 1.96 * se

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(betas, fractions, 'o-', color='#d7191c', linewidth=2,
            markersize=5, label='P(short noisy path)')
    ax.fill_between(betas,
                    np.clip(fractions - ci, 0, 1),
                    np.clip(fractions + ci, 0, 1),
                    alpha=0.2, color='#d7191c')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    ax.set_xlabel(r'$\beta$ (entropy penalty weight)', fontsize=13)
    ax.set_ylabel('P(path through noisy room)', fontsize=13)
    ax.set_title(r'Noise Avoidance vs $\beta$', fontsize=15)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    save_path = os.path.join(FIG_DIR, "beta_sweep.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Beta sweep saved to {save_path}")


# ==================== Main ====================

def main():
    print("=" * 60)
    print("NOISE-AVOIDANCE EXPERIMENT (fiverooms)")
    print("=" * 60)
    print(f"  Start: {START_LOC}  (left-top room)")
    print(f"  Goal:  {GOAL_LOC}  (right-middle room)")
    print(f"  Noise: RT room (x=5..8, y=0..1)")
    print(f"  Short path: LT -> RT (noisy) -> RM  (~11 steps)")
    print(f"  Long  path: LT -> LB -> RB -> RM    (~20 steps)")
    print()

    # --- Figure 1: Trajectory comparison ---
    # Panel (a): beta=0 -> no entropy penalty -> short path through noise
    # Panel (b): beta=high -> entropy penalty -> long clean path
    print("Figure 1: Trajectory comparison...")
    plot_trajectory_comparison(beta_high=2.0, noise_level=0.3)

    # --- Figure 2: Entropy sweep ---
    # x-axis = observation entropy in RT room (nats)
    # Fixed beta=2.0; each point rebuilds adapter+agent with that noise
    print("\nFigure 2: Entropy sweep (beta=2.0 fixed)...")
    N = GRID_SIZE ** 2
    H_base = _column_entropy(0.3, N)
    H_max = _column_entropy(0.95, N)
    entropies = np.linspace(H_base, H_max, 15)

    entropies, fractions = run_entropy_sweep(
        n_episodes=50,
        beta=2.0,
        entropies=entropies,
        noise_level=0.3,
    )
    plot_entropy_sweep(entropies, fractions, n_episodes=50,
                       noise_level=0.3, beta=2.0)

    # --- Figure 3: Beta sweep ---
    # x-axis = beta (entropy penalty weight)
    # Fixed high noise in RT; learn once, vary C
    print("\nFigure 3: Beta sweep (fixed high noise in RT)...")
    betas = np.linspace(0, 5, 20)
    betas, beta_fractions = run_beta_sweep(
        n_episodes=50,
        betas=betas,
        noise_level=0.3,
    )
    plot_beta_sweep(betas, beta_fractions, n_episodes=50)

    print("\nDone. Figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
