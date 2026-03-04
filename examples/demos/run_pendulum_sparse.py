"""Sparse-reward Pendulum — hierarchy vs flat comparison.

The shaped Pendulum reward C(s) = -(theta^2 + 0.1*omega^2) provides a
smooth gradient everywhere, making flat planning effective.  This variant
uses a **sparse reward**: +reward inside a small ball near upright,
slight negative elsewhere.  With sparse C the value function V = M @ C
has a much weaker gradient far from goal, making flat planning harder.
Hierarchical planning compensates because C_macro concentrates the sparse
signal at the cluster level.

Generates:
  figures/demos/pendulum_sparse/
    C_sparse_heatmap.png         -- sparse C as 2D heatmap
    value_comparison.png         -- V(sparse) vs V(shaped) side-by-side
    trajectory_comparison.png    -- hierarchical vs flat phase-space paths
    reward_accumulation.png      -- cumulative reward over episode steps
    clustering/                  -- macro-state clusters
    pendulum_sparse_stages.png   -- stage diagram (hierarchical episode)
    pendulum_sparse_episode.mp4  -- video
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import imageio

from core import HierarchicalSRAgent
from environments.pendulum import PendulumAdapter
from examples.configs import PENDULUM_SPARSE

import gymnasium as gym


# ========================== Plotting helpers ============================

def plot_C_heatmap(adapter, C, save_path):
    """Plot the sparse C vector as a 2D heatmap over (theta, omega)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    theta_centers, omega_centers = adapter.get_bin_centers()
    grid = np.zeros((adapter.n_omega_bins, adapter.n_theta_bins))
    for t in range(adapter.n_theta_bins):
        for w in range(adapter.n_omega_bins):
            idx = adapter.state_space.state_to_index((t, w))
            grid[w, t] = C[idx]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(grid, origin="lower", aspect="auto",
                   extent=[theta_centers[0], theta_centers[-1],
                           omega_centers[0], omega_centers[-1]],
                   cmap="RdYlGn")
    ax.set_xlabel(r"Angle ($\theta$)")
    ax.set_ylabel(r"Angular Velocity ($\omega$)")
    plt.colorbar(im, ax=ax, label="C(s)")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved C heatmap to {save_path}")


def plot_value_comparison(agent, adapter, C_sparse, C_shaped, save_path):
    """Side-by-side V = M @ C heatmaps for sparse vs shaped reward."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    V_sparse = adapter.multiply_M_C(agent.M, C_sparse)
    V_shaped = adapter.multiply_M_C(agent.M, C_shaped)

    theta_centers, omega_centers = adapter.get_bin_centers()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, V, title in zip(axes, [V_sparse, V_shaped],
                             ["V (sparse reward)", "V (shaped reward)"]):
        grid = np.zeros((adapter.n_omega_bins, adapter.n_theta_bins))
        for t in range(adapter.n_theta_bins):
            for w in range(adapter.n_omega_bins):
                idx = adapter.state_space.state_to_index((t, w))
                grid[w, t] = V[idx]
        im = ax.imshow(grid, origin="lower", aspect="auto",
                       extent=[theta_centers[0], theta_centers[-1],
                               omega_centers[0], omega_centers[-1]],
                       cmap="viridis")
        ax.set_xlabel(r"Angle ($\theta$)")
        ax.set_ylabel(r"Angular Velocity ($\omega$)")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved value comparison to {save_path}")


def plot_trajectory_comparison(adapter, thetas_h, omegas_h,
                               thetas_f, omegas_f, C_sparse, save_path):
    """Hierarchical vs flat trajectories in phase space with goal overlay."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, thetas, omegas, label in zip(
            axes,
            [thetas_h, thetas_f],
            [omegas_h, omegas_f],
            ["Hierarchical", "Flat"]):
        T = len(thetas)
        t = np.arange(T)
        ax.scatter(thetas, omegas, c=t, s=12, cmap="plasma", zorder=3)
        ax.plot(thetas, omegas, linewidth=0.8, color="gray", alpha=0.4)

        # Draw goal region ellipse
        # theta^2 + 0.1*omega^2 < radius^2  =>  ellipse with a=radius, b=radius/sqrt(0.1)
        radius = PENDULUM_SPARSE["sparse_radius"]
        from matplotlib.patches import Ellipse
        ell = Ellipse((0, 0), width=2 * radius,
                       height=2 * radius / np.sqrt(0.1),
                       facecolor="green", alpha=0.15, edgecolor="green",
                       linewidth=2, linestyle="--", zorder=2)
        ax.add_patch(ell)

        # Start / end markers
        ax.scatter(thetas[0], omegas[0], s=100, marker="o",
                   edgecolors="black", facecolors="blue", zorder=4,
                   label="Start")
        ax.scatter(thetas[-1], omegas[-1], s=100, marker="*",
                   edgecolors="black", facecolors="red", zorder=4,
                   label="End")

        if hasattr(adapter, 'get_bin_edges'):
            edges0, edges1 = adapter.get_bin_edges()
            ax.set_xlim(edges0[0], edges0[-1])
            ax.set_ylim(edges1[0], edges1[-1])

        ax.set_xlabel(r"Angle ($\theta$)")
        ax.set_ylabel(r"Angular Velocity ($\omega$)")
        ax.set_title(label)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved trajectory comparison to {save_path}")


def plot_reward_accumulation(rewards_h, rewards_f, save_path):
    """Cumulative reward over episode steps for hierarchy vs flat."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    cum_h = np.cumsum(rewards_h)
    cum_f = np.cumsum(rewards_f)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(cum_h, label="Hierarchical", linewidth=2)
    ax.plot(cum_f, label="Flat", linewidth=2, linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative reward")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved reward accumulation to {save_path}")


# ========================== Episode tracking ============================

def run_episode_with_tracking(agent, adapter, init_state, max_steps=200):
    """Run a flat-policy episode and record trajectory + per-step reward.

    Returns:
        (frames, thetas, omegas, actions, per_step_rewards)
    """
    frames, thetas, omegas = [], [], []
    actions_taken, step_rewards = [], []

    agent.reset_episode(init_state=init_state)
    V = adapter.multiply_M_C(agent.M, agent.C)
    stop_at_goal = not agent._shaped_goal

    steps = 0
    while steps < max_steps:
        if stop_at_goal and agent._is_at_goal():
            break

        obs = adapter.get_current_obs()
        theta, omega = adapter.obs_to_continuous(obs)
        thetas.append(theta)
        omegas.append(omega)

        frame = adapter.render()
        if frame is not None:
            frames.append(frame)

        # Record reward at this state
        s_idx = adapter.get_current_state_index()
        step_rewards.append(agent.C[s_idx])

        action = agent._select_micro_action(V)
        actions_taken.append(action)

        smooth = agent.test_smooth_steps
        if smooth <= 1:
            adapter.step(action)
            n_phys = 1
        else:
            s_before = adapter.get_current_state_index()
            n_phys = 0
            for i in range(smooth):
                result = adapter.step_with_info(action)
                if result is not None:
                    _, _, terminated, truncated, _ = result
                    done = terminated or truncated
                else:
                    adapter.step(action)
                    done = False
                n_phys += 1
                if done:
                    break
                if adapter.get_current_state_index() != s_before:
                    break

        agent.current_state = agent._get_planning_state()
        steps += n_phys

    # Record final state
    obs = adapter.get_current_obs()
    theta, omega = adapter.obs_to_continuous(obs)
    thetas.append(theta)
    omegas.append(omega)
    frame = adapter.render()
    if frame is not None:
        frames.append(frame)

    return frames, thetas, omegas, actions_taken, step_rewards


# ========================== Main ============================

def run_pendulum_sparse():
    """Run the sparse-reward Pendulum experiment."""

    cfg = PENDULUM_SPARSE
    init_state = [np.pi, 0.0]  # Hanging down
    fig_dir = "figures/demos/pendulum_sparse"
    os.makedirs(fig_dir, exist_ok=True)

    # --- Environment ---
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    adapter = PendulumAdapter(env,
                              n_theta_bins=cfg["n_theta_bins"],
                              n_omega_bins=cfg["n_omega_bins"],
                              n_torque_bins=cfg["n_torque_bins"])

    print("=" * 50)
    print("PENDULUM (SPARSE REWARD) — Hierarchy vs Flat")
    print("=" * 50)
    print(f"State space: {adapter.n_states} states  "
          f"({cfg['n_theta_bins']}×{cfg['n_omega_bins']})")
    print(f"Sparse radius: {cfg['sparse_radius']}, "
          f"reward: {cfg['sparse_reward']}, "
          f"default cost: {cfg['sparse_default_cost']}")

    # --- Agent ---
    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=cfg["n_clusters"],
        gamma=cfg["gamma"],
        learning_rate=cfg["learning_rate"],
        learn_from_experience=True,
        train_smooth_steps=cfg.get("train_smooth_steps", 1),
        test_smooth_steps=cfg.get("test_smooth_steps", 1),
    )

    # --- Sparse reward ---
    C_sparse = adapter.create_sparse_prior(
        radius=cfg["sparse_radius"],
        reward=cfg["sparse_reward"],
        default_cost=cfg["sparse_default_cost"],
    )
    agent.set_shaped_goal(C_sparse,
                          goal_threshold=cfg["sparse_goal_threshold"])

    # --- Learn (M is reward-independent) ---
    print("\n" + "=" * 50)
    print("LEARNING PHASE")
    print("=" * 50)
    agent.learn_environment(num_episodes=cfg["train_episodes"])

    # ==================== Visualisation ====================
    print("\n" + "=" * 50)
    print("VISUALISATION")
    print("=" * 50)

    # C heatmap
    plot_C_heatmap(adapter, C_sparse,
                   save_path=f"{fig_dir}/C_sparse_heatmap.png")

    # Value comparison: sparse vs shaped
    C_shaped = adapter.create_shaped_prior(scale=10.0)
    plot_value_comparison(agent, adapter, C_sparse, C_shaped,
                          save_path=f"{fig_dir}/value_comparison.png")

    # Cluster visualisation
    agent.visualize_clusters(save_dir=f"{fig_dir}/clustering")

    # ==================== Test episodes ====================
    print("\n" + "=" * 50)
    print("TEST PHASE — Hierarchical Policy")
    print("=" * 50)

    agent.reset_episode(init_state=init_state)
    result_h = agent.run_episode_hierarchical(max_steps=200)

    print(f"\nHierarchical:")
    print(f"  Steps: {result_h['steps']}")
    print(f"  Total reward: {result_h['reward']:.2f}")
    print(f"  Reached goal: {result_h['reached_goal']}")

    print("\n" + "=" * 50)
    print("TEST PHASE — Flat Policy")
    print("=" * 50)

    agent.reset_episode(init_state=init_state)
    result_f = agent.run_episode_flat(max_steps=200)

    print(f"\nFlat:")
    print(f"  Steps: {result_f['steps']}")
    print(f"  Total reward: {result_f['reward']:.2f}")
    print(f"  Reached goal: {result_f['reached_goal']}")

    print("\n" + "=" * 50)
    print("COMPARISON")
    print("=" * 50)
    print(f"Hierarchical: reward={result_h['reward']:.1f}, "
          f"goal={result_h['reached_goal']}")
    print(f"Flat:         reward={result_f['reward']:.1f}, "
          f"goal={result_f['reached_goal']}")
    delta = result_h['reward'] - result_f['reward']
    print(f"Advantage:    {delta:+.1f} reward")

    # ==================== Tracked episodes for plots ====================
    print("\n" + "=" * 50)
    print("RECORDING TRAJECTORIES")
    print("=" * 50)

    # Hierarchical trajectory
    frames_h, thetas_h, omegas_h, acts_h, rew_h = \
        run_episode_with_tracking(agent, adapter, init_state, max_steps=200)

    # Flat trajectory
    frames_f, thetas_f, omegas_f, acts_f, rew_f = \
        run_episode_with_tracking(agent, adapter, init_state, max_steps=200)

    # Trajectory comparison
    plot_trajectory_comparison(adapter, thetas_h, omegas_h,
                               thetas_f, omegas_f, C_sparse,
                               save_path=f"{fig_dir}/trajectory_comparison.png")

    # Reward accumulation
    plot_reward_accumulation(rew_h, rew_f,
                             save_path=f"{fig_dir}/reward_accumulation.png")

    # Stage diagram (hierarchical episode)
    if frames_h:
        agent.plot_stage_state_diagram(
            frames_h, thetas_h, omegas_h,
            save_path=f"{fig_dir}/pendulum_sparse_stages.png",
        )

    # Video (hierarchical)
    if frames_h:
        video_path = f"{fig_dir}/pendulum_sparse_episode.mp4"
        imageio.mimsave(video_path, frames_h, fps=30, macro_block_size=1)
        print(f"  Video saved to {video_path} ({len(frames_h)} frames)")

    # Macro-state trajectory (hierarchical)
    if thetas_h:
        agent.plot_trajectory_with_macro_states(
            thetas_h, omegas_h,
            save_path=f"{fig_dir}/trajectory_macro_state.png",
            color_by='macro_state',
        )

    env.close()

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)


if __name__ == "__main__":
    run_pendulum_sparse()
