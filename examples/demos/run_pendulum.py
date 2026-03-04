"""Example: Running hierarchical SR agent on Pendulum.

Pendulum-v1 has a continuous action space (torque) which is discretized.
The agent must learn to swing the pendulum up from hanging (θ=π) to
upright (θ=0).  This is a goal-reaching task well suited to the SR
framework: the successor matrix M captures the dynamics, and the goal
prior C encodes the upright target.

Generates cluster heatmap, macro-state trajectory, stage diagram, and video.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import imageio

from core import HierarchicalSRAgent
from environments.pendulum import PendulumAdapter
from examples.configs import PENDULUM

import gymnasium as gym

def run_episode_with_tracking(agent, adapter, init_state, max_steps=200):
    """Run an episode capturing frames, positions, velocities, and actions.

    Uses the agent's own flat policy (``_select_micro_action``) so the
    recorded trajectory matches what ``run_episode_flat`` would produce.

    For sparse goals the episode terminates on goal arrival (matching
    ``run_episode_flat``).  For shaped goals the episode runs for the
    full ``max_steps`` so the video shows both swing-up and balancing.

    Captures a frame on **every physics step** (not just per decision
    point) so the video covers the full episode smoothly.

    Returns:
        (frames, thetas, omegas, actions) tuple
    """
    frames = []
    thetas = []
    omegas = []
    actions_taken = []

    # Reset via the agent so agent.current_state is properly initialised
    agent.reset_episode(init_state=init_state)

    # Compute value function (same as run_episode_flat)
    V = adapter.multiply_M_C(agent.M, agent.C)

    # Shaped goals: run the full episode to show maintenance behaviour.
    # Sparse goals: stop when the goal is reached.
    stop_at_goal = not agent._shaped_goal

    steps = 0

    while steps < max_steps:
        if stop_at_goal and agent._is_at_goal():
            break

        # Record continuous state at the decision point
        obs = adapter.get_current_obs()
        theta, omega = adapter.obs_to_continuous(obs)
        thetas.append(theta)
        omegas.append(omega)

        # Capture frame before action
        frame = adapter.render()
        if frame is not None:
            frames.append(frame)

        # Select action using the agent's micro-level policy
        action = agent._select_micro_action(V)
        actions_taken.append(action)

        # Smooth stepping — capture a frame on every physics sub-step
        smooth = agent.test_smooth_steps
        if smooth <= 1:
            adapter.step(action)
            n_phys = 1
        else:
            s_before = adapter.get_current_state_index()
            n_phys = 0
            for i in range(smooth):
                if hasattr(adapter, 'step_with_info'):
                    _, _, terminated, truncated, _ = adapter.step_with_info(action)
                    done = terminated or truncated
                else:
                    adapter.step(action)
                    done = False
                n_phys += 1

                # Capture intermediate frame
                if i < smooth - 1:  # skip last — next loop iteration captures it
                    sub_frame = adapter.render()
                    if sub_frame is not None:
                        frames.append(sub_frame)

                if done:
                    break
                s_after = adapter.get_current_state_index()
                if s_after != s_before:
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

    return frames, thetas, omegas, actions_taken

def run_episode_with_video(agent, adapter, init_state, max_steps=200,
                           smooth_steps=None):
    """Run an episode using the agent's policy and capture frames for video.

    Uses smooth stepping to match the agent's test_smooth_steps for consistent
    dynamics.  Does NOT terminate on reaching the goal — the agent continues
    to balance at the top, demonstrating maintenance behavior.
    """
    if smooth_steps is None:
        smooth_steps = getattr(agent, 'test_smooth_steps', 5)

    frames = []

    adapter.reset(init_state)
    V = adapter.multiply_M_C(agent.M, agent.C)

    done = False
    steps = 0

    while not done and steps < max_steps:
        # Capture frame
        frame = adapter.render()
        if frame is not None:
            frames.append(frame)

        # Select action using agent's policy
        state_onehot = adapter._current_state
        V_adj = []
        for act in range(adapter.n_actions):
            s_next_dist = adapter.multiply_B_s(agent.B, state_onehot, act)
            V_adj.append(float(s_next_dist @ V))

        best_action = np.argmax(V_adj)

        # Take action with smooth stepping (match agent's test dynamics)
        for _ in range(smooth_steps):
            _, _, terminated, truncated, _ = adapter.step_with_info(best_action)
            done = terminated or truncated

            frame = adapter.render()
            if frame is not None:
                frames.append(frame)

            if done:
                break

        steps += 1

    return frames

def run_pendulum_example():
    """Run the hierarchical SR agent on Pendulum."""

    # Configuration (from centralized config)
    n_theta_bins = PENDULUM["n_theta_bins"]
    n_omega_bins = PENDULUM["n_omega_bins"]
    n_torque_bins = PENDULUM["n_torque_bins"]
    n_clusters = PENDULUM["n_clusters"]
    init_state = [np.pi, 0.0]  # Hanging down, zero velocity

    # Create gymnasium environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")

    # Wrap with adapter
    adapter = PendulumAdapter(env,
                              n_theta_bins=n_theta_bins,
                              n_omega_bins=n_omega_bins,
                              n_torque_bins=n_torque_bins)

    print("=" * 50)
    print("PENDULUM - Hierarchical Active Inference")
    print("=" * 50)
    print(f"State space: {n_theta_bins} × {n_omega_bins} = {adapter.n_states} states")
    print(f"Actions: {n_torque_bins} discrete torques from "
          f"{adapter._discrete_torques}")
    print(f"Clusters: {n_clusters}")

    # Create agent
    agent = HierarchicalSRAgent(
        adapter=adapter,
        n_clusters=n_clusters,
        gamma=PENDULUM["gamma"],
        learning_rate=PENDULUM["learning_rate"],
        learn_from_experience=True,
        train_smooth_steps=PENDULUM.get("train_smooth_steps", 5),
        test_smooth_steps=PENDULUM.get("test_smooth_steps", 5),
    )

    # Set shaped goal — mirrors Pendulum-v1 reward: -(θ² + 0.1·ω²).
    # Continuous reward landscape — no absorbing states needed.
    # The agent accumulates reward by staying near upright for the full episode.
    # Threshold at -1.0: agent is "at goal" when θ² + 0.1·ω² < ~1.0 (rescaled).
    C_shaped = adapter.create_shaped_prior(scale=10.0)
    agent.set_shaped_goal(C_shaped, goal_threshold=-1.0)

    print(f"\nHigh-reward states ({len(agent.goal_states)} states): "
          f"{[adapter.get_state_label(g) for g in agent.goal_states[:6]]}...")

    # Learn environment
    print("\n" + "=" * 50)
    print("LEARNING PHASE")
    print("=" * 50)
    agent.learn_environment(num_episodes=PENDULUM["train_episodes"])

    # ==================== Visualization ====================
    print("\n" + "=" * 50)
    print("VISUALIZATION")
    print("=" * 50)

    os.makedirs("figures/demos/pendulum", exist_ok=True)

    # Value function, policy, and overlay plots
    agent.plot_value_function(save_path="figures/demos/pendulum/value_function.png")
    agent.plot_policy(save_path="figures/demos/pendulum/policy.png")
    agent.plot_value_with_policy(save_path="figures/demos/pendulum/value_with_policy.png")

    # Cluster heatmap
    agent.visualize_clusters(save_dir="figures/demos/pendulum/clustering")

    # Macro action heatmap (target cluster at each state)
    agent.plot_macro_action_heatmap(save_path="figures/demos/pendulum/macro_actions.png")

    # Matrix visualizations
    agent.view_matrices(save_dir="figures/demos/pendulum/matrices", learned=True)
    print("  Saved matrix visualizations to figures/demos/pendulum/matrices/")

    # Macro action network (per-transition micro-level policies)
    agent.visualize_policy(save_dir="figures/demos/pendulum/macro_action_network")
    print("  Saved policy visualizations to figures/demos/pendulum/macro_action_network/")

    # Test hierarchical policy
    print("\n" + "=" * 50)
    print("TEST PHASE - Hierarchical Policy")
    print("=" * 50)

    agent.reset_episode(init_state=init_state)
    result = agent.run_episode_hierarchical(max_steps=200)

    print(f"\nHierarchical Results:")
    print(f"  Steps taken: {result['steps']}")
    print(f"  Total reward: {result['reward']:.2f}")
    print(f"  Reached goal: {result['reached_goal']}")
    print(f"  Final state: {result['final_state']}")

    # Test flat policy
    print("\n" + "=" * 50)
    print("TEST PHASE - Flat Policy")
    print("=" * 50)

    agent.reset_episode(init_state=init_state)
    result_flat = agent.run_episode_flat(max_steps=200)

    print(f"\nFlat Results:")
    print(f"  Steps taken: {result_flat['steps']}")
    print(f"  Total reward: {result_flat['reward']:.2f}")
    print(f"  Reached goal: {result_flat['reached_goal']}")
    print(f"  Final state: {result_flat['final_state']}")

    print("\n" + "=" * 50)
    print("COMPARISON")
    print("=" * 50)
    print(f"Hierarchical: {result['steps']} steps, reached goal: {result['reached_goal']}")
    print(f"Flat: {result_flat['steps']} steps, reached goal: {result_flat['reached_goal']}")

    # ==================== Record Episode + Trajectory ====================
    print("\n" + "=" * 50)
    print("RECORDING VIDEO & TRAJECTORY")
    print("=" * 50)

    frames, thetas, omegas, actions = run_episode_with_tracking(
        agent, adapter, init_state, max_steps=200,
    )

    if frames:
        video_path = "figures/demos/pendulum/pendulum_episode.mp4"
        imageio.mimsave(video_path, frames, fps=30, macro_block_size=1)
        print(f"  Video saved to: {video_path} ({len(frames)} frames)")
    else:
        print("  No frames captured")

    if thetas:
        # Trajectory colored by macro state
        agent.plot_trajectory_with_macro_states(
            thetas, omegas,
            save_path="figures/demos/pendulum/trajectory_macro_state.png",
            color_by='macro_state',
        )

        # Trajectory colored by macro action (target cluster)
        agent.plot_trajectory_with_macro_states(
            thetas, omegas,
            save_path="figures/demos/pendulum/trajectory_macro_action.png",
            color_by='macro_action',
        )

        # Trajectory colored by micro action taken
        agent.plot_trajectory_with_actions(
            thetas, omegas, actions,
            save_path="figures/demos/pendulum/trajectory_actions.png",
        )

        # Stage diagram (snapshots + phase plot)
        if frames:
            agent.plot_stage_state_diagram(
                frames, thetas, omegas,
                save_path="figures/demos/pendulum/pendulum_stages.png",
            )

            # Combined vertical video (environment + animated trajectory)
            agent.generate_combined_video(
                frames, thetas, omegas,
                save_path="figures/demos/pendulum/pendulum_combined.mp4",
                color_by='macro_action',
            )

    env.close()

    print("\n" + "=" * 50)
    print("DONE")
    print("=" * 50)

if __name__ == '__main__':
    run_pendulum_example()
