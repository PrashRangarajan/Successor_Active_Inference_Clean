"""Diagnostic script: compare flat vs hierarchical neural SR on Acrobot.

Trains a HierarchicalNeuralSRAgent on Acrobot using the known-working
two-phase config, learns the hierarchy, and evaluates BOTH flat and
hierarchical episode execution with extensive diagnostic output.

Usage:
    python examples/diagnose_hierarchical.py
"""

import os
import sys
import time
import traceback

# Ensure imports resolve from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import gymnasium as gym

from environments.acrobot import AcrobotAdapter
from core.neural.continuous_adapter import ContinuousAdapter
from core.neural.agent import NeuralSRAgent
from core.neural.hierarchical_agent import HierarchicalNeuralSRAgent
from examples.configs import NEURAL_ACROBOT

np.set_printoptions(precision=4, suppress=True, linewidth=120)


def separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def make_adapter(render_mode=None, max_episode_steps=500):
    """Create Acrobot environment + ContinuousAdapter."""
    cfg = NEURAL_ACROBOT
    kwargs = {}
    if render_mode:
        kwargs['render_mode'] = render_mode
    env = gym.make('Acrobot-v1', max_episode_steps=max_episode_steps, **kwargs)
    base = AcrobotAdapter(
        env,
        n_theta_bins=cfg["n_theta_bins"],
        n_dtheta_bins=cfg["n_dtheta_bins"],
        goal_velocity_filter=cfg["goal_velocity_filter"],
    )
    adapter = ContinuousAdapter(base)
    return adapter, base


def acrobot_height_reward(obs):
    """Dense shaped reward based on Acrobot end-effector height.

    Height = -cos(θ1) - cos(θ1 + θ2), ranges from -2 (hanging) to +2 (upright).
    Goal: height > 1.0. We normalize to [-1, 1] range.
    """
    import math
    c1, s1, c2, s2 = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
    theta1 = math.atan2(s1, c1)
    theta2 = math.atan2(s2, c2)
    height = -np.cos(theta1) - np.cos(theta1 + theta2)
    # Normalize: height in [-2, 2] → reward in [-1, 1]
    return height / 2.0


def create_agent(adapter, cls=NeuralSRAgent, **extra_kwargs):
    """Create and configure a neural SR agent."""
    cfg = NEURAL_ACROBOT
    agent = cls(
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
        **extra_kwargs,
    )
    agent.set_goal(
        goal_spec=None,
        reward=cfg["reward"],
        default_cost=cfg["default_cost"],
        use_env_reward=True,
        terminal_bonus=cfg.get("terminal_bonus", 0.0),
        reward_shaping_fn=acrobot_height_reward,
    )
    return agent


def train_agent(agent, tag=""):
    """Two-phase training: diverse + mixed."""
    cfg = NEURAL_ACROBOT
    ep_diverse = cfg["train_episodes_diverse"]
    ep_fixed = cfg["train_episodes_fixed"]
    diverse_frac = cfg.get("diverse_fraction", 0.3)

    t0 = time.time()

    print(f"Phase 1: Diverse exploration ({ep_diverse} episodes) {tag}")
    agent.learn_environment(
        num_episodes=ep_diverse,
        steps_per_episode=cfg["steps_per_episode"],
        diverse_start=True,
        log_interval=max(1, ep_diverse // 5),
    )

    print(f"\nPhase 2: Mixed training ({ep_fixed} episodes, "
          f"{diverse_frac:.0%} diverse) {tag}")
    agent.learn_environment(
        num_episodes=ep_fixed,
        steps_per_episode=cfg["steps_per_episode"],
        diverse_start=True,
        diverse_fraction=diverse_frac,
        log_interval=max(1, ep_fixed // 5),
    )

    elapsed = time.time() - t0
    print(f"\nTraining time: {elapsed:.1f}s  |  Total steps: {agent.total_steps}")
    return elapsed


def eval_flat(agent, adapter, base, n_episodes=5, label="Flat"):
    """Evaluate flat episodes and print per-episode results."""
    separator(f"Evaluation: {label}")
    agent.adapter = adapter
    agent.goal_states = base.get_goal_states()

    results = []
    for i in range(n_episodes):
        result = agent.run_episode(
            init_state=[0, 0, 0, 0],
            max_steps=NEURAL_ACROBOT["test_max_steps"],
        )
        results.append(result)
        status = "GOAL" if result['reached_goal'] else "TIMEOUT"
        obs = result['final_state']
        print(f"  [{label}] Ep {i+1}: steps={result['steps']:4d}  "
              f"reward={result['reward']:7.1f}  {status}  "
              f"final_obs={obs}")

    steps = [r['steps'] for r in results]
    successes = [r['reached_goal'] for r in results]
    print(f"\n  Success: {sum(successes)}/{len(successes)}  "
          f"Avg steps: {np.mean(steps):.1f} +/- {np.std(steps):.1f}")
    return results


def eval_hierarchical_verbose(agent, adapter, base, n_episodes=5):
    """Evaluate hierarchical episodes with detailed diagnostic output."""
    separator("Evaluation: Hierarchical (verbose)")
    agent.adapter = adapter
    agent.goal_states = base.get_goal_states()

    # Pre-compute useful info
    V_macro = agent.get_macro_values()
    goal_macros = agent._get_goal_macro_states()
    print(f"  V_macro = {V_macro}")
    print(f"  Goal macro states: {goal_macros}")
    print()

    results = []
    for i in range(n_episodes):
        print(f"  --- Hierarchical Episode {i+1} ---")
        # Manual step-by-step hierarchical episode for diagnostics
        result = _run_hierarchical_verbose(agent, adapter, base, V_macro, goal_macros)
        results.append(result)
        status = "GOAL" if result['reached_goal'] else "TIMEOUT"
        print(f"  Result: steps={result['steps']:4d}  "
              f"reward={result['reward']:7.1f}  {status}")
        print()

    steps = [r['steps'] for r in results]
    successes = [r['reached_goal'] for r in results]
    print(f"\n  Success: {sum(successes)}/{len(successes)}  "
          f"Avg steps: {np.mean(steps):.1f} +/- {np.std(steps):.1f}")
    return results


def _run_hierarchical_verbose(agent, adapter, base, V_macro, goal_macros):
    """Run one hierarchical episode with detailed prints."""
    max_steps = NEURAL_ACROBOT["test_max_steps"]
    obs = adapter.reset([0, 0, 0, 0])
    total_steps = 0
    total_reward = 0.0

    s_macro_start = agent.clustering.predict(obs)
    print(f"    Start obs: {obs}")
    print(f"    Start macro state: {s_macro_start}")

    # ---- Phase 1: Macro navigation ----
    macro_nav_iterations = 0
    while total_steps < max_steps:
        s_macro = agent.clustering.predict(obs)

        if s_macro in goal_macros:
            print(f"    [Macro] Reached goal cluster {s_macro} after {total_steps} steps.")
            break

        # Select target macro
        target_macro = agent._select_macro_target(s_macro, V_macro)
        if target_macro is None:
            print(f"    [Macro] No valid macro target from cluster {s_macro}! Adjacency: {agent.adj_list.get(s_macro, [])}")
            break

        bottleneck_key = (s_macro, target_macro)
        n_bottleneck = len(agent.bottleneck_obs.get(bottleneck_key, []))
        print(f"    [Macro] In cluster {s_macro}, targeting cluster {target_macro} "
              f"(V_macro[tgt]={V_macro[target_macro]:.4f}, "
              f"bottleneck obs: {n_bottleneck})")

        # Navigate toward target cluster
        steps_before = total_steps
        steps, reward, obs = agent._navigate_to_macro(
            s_macro, target_macro, max_steps - total_steps
        )
        total_steps += steps
        total_reward += reward

        new_macro = agent.clustering.predict(obs)
        print(f"    [Macro] Navigation took {steps} steps, now in cluster {new_macro}")

        # Check if we hit the goal during navigation
        terminal = adapter.is_terminal(obs)
        if terminal is True:
            print(f"    [Macro] Hit terminal during navigation!")
            return {
                'steps': total_steps,
                'reward': total_reward,
                'reached_goal': True,
                'final_state': obs,
            }
        if terminal is None and agent.goal_states:
            if adapter.is_in_goal_bin(agent.goal_states, obs):
                print(f"    [Macro] In goal bin during navigation!")
                return {
                    'steps': total_steps,
                    'reward': total_reward,
                    'reached_goal': True,
                    'final_state': obs,
                }

        macro_nav_iterations += 1
        if macro_nav_iterations > 20:
            print(f"    [Macro] Too many macro navigation iterations, breaking.")
            break

    # ---- Phase 2: Micro goal-seeking ----
    print(f"    [Micro] Starting micro goal-seeking at step {total_steps}")
    micro_start_macro = agent.clustering.predict(obs)
    print(f"    [Micro] Current cluster: {micro_start_macro}")

    while total_steps < max_steps:
        action = agent.select_action(obs, greedy=True)
        next_obs, env_reward, terminated, truncated, info = adapter.step(action)

        total_reward += env_reward
        total_steps += 1

        terminal = adapter.is_terminal(next_obs)
        if terminal is True:
            print(f"    [Micro] Reached terminal at step {total_steps}")
            return {
                'steps': total_steps,
                'reward': total_reward,
                'reached_goal': True,
                'final_state': next_obs,
            }
        if terminal is None and agent.goal_states:
            if adapter.is_in_goal_bin(agent.goal_states, next_obs):
                print(f"    [Micro] Reached goal bin at step {total_steps}")
                return {
                    'steps': total_steps,
                    'reward': total_reward,
                    'reached_goal': True,
                    'final_state': next_obs,
                }

        # Environment terminated (e.g., Acrobot height threshold) = goal success
        if terminated:
            print(f"    [Micro] Environment terminated (goal reached) at step {total_steps}")
            return {
                'steps': total_steps,
                'reward': total_reward,
                'reached_goal': True,
                'final_state': next_obs,
            }
        if truncated:
            print(f"    [Micro] Truncated at step {total_steps}")
            break

        obs = next_obs

    print(f"    [Micro] Timed out at {total_steps} steps")
    return {
        'steps': total_steps,
        'reward': total_reward,
        'reached_goal': False,
        'final_state': adapter.get_current_obs(),
    }


def diagnose_hierarchy(agent, base):
    """Print detailed hierarchy diagnostics."""
    separator("Hierarchy Diagnostics")

    # Cluster sizes
    stats = agent.clustering.get_cluster_stats()
    print(f"Number of clusters: {stats['n_clusters']}")
    print(f"Total samples used for clustering: {stats['n_samples']}")
    print(f"Cluster sizes:")
    for c, size in sorted(stats['cluster_sizes'].items()):
        print(f"  Cluster {c}: {size} samples ({100*size/stats['n_samples']:.1f}%)")

    # Adjacency
    print(f"\nAdjacency graph:")
    for macro in range(agent.n_clusters):
        neighbors = agent.adj_list.get(macro, [])
        bottleneck_counts = {}
        for nb in neighbors:
            key = (macro, nb)
            bottleneck_counts[nb] = len(agent.bottleneck_obs.get(key, []))
        print(f"  Cluster {macro} -> {neighbors}  (bottleneck obs: {bottleneck_counts})")

    # Goal clusters
    goal_macros = agent._get_goal_macro_states()
    print(f"\nGoal macro states: {goal_macros}")

    # Count goal observations per cluster
    goal_states_set = set(base.get_goal_states())
    n_goal_per_cluster = {}
    for obs_i, label in zip(agent.clustering.observations, agent.clustering.labels):
        discrete = base.discretize_obs(obs_i)
        idx = base.state_space.state_to_index(discrete)
        if idx in goal_states_set:
            n_goal_per_cluster[int(label)] = n_goal_per_cluster.get(int(label), 0) + 1
    print(f"Goal observations per cluster: {n_goal_per_cluster}")

    # C_macro and V_macro
    print(f"\nC_macro (per-cluster expected reward):")
    print(f"  {agent.C_macro}")

    V_macro = agent.get_macro_values()
    print(f"\nV_macro = M_macro @ C_macro:")
    print(f"  {V_macro}")

    # M_macro
    print(f"\nM_macro (macro successor matrix):")
    print(agent.M_macro)

    # B_macro
    print(f"\nB_macro (macro transition matrix, shape {agent.B_macro.shape}):")
    for a in range(agent.B_macro.shape[2]):
        print(f"  Action {a}:")
        print(f"  {agent.B_macro[:,:,a]}")

    # Check: what does the agent think the best macro target is from each cluster?
    print(f"\nBest macro target from each cluster (via V_macro):")
    for s in range(agent.n_clusters):
        target = agent._select_macro_target(s, V_macro)
        neighbors = agent.adj_list.get(s, [])
        print(f"  Cluster {s}: neighbors={neighbors}, selected target={target}, "
              f"V_macro[neighbors]={[V_macro[n] for n in neighbors] if neighbors else 'N/A'}")

    # Cluster centers in observation space
    print(f"\nCluster centers (observation space):")
    for c in range(agent.n_clusters):
        center = agent.clustering.cluster_centers[c]
        print(f"  Cluster {c}: {center}")

    # Check: what cluster does [0,0,0,0] map to?
    init_obs = np.array([np.cos(0), np.sin(0), np.cos(0), np.sin(0), 0, 0], dtype=np.float32)
    init_cluster = agent.clustering.predict(init_obs)
    print(f"\nInit state [0,0,0,0] obs={init_obs} -> Cluster {init_cluster}")

    # Check w vector
    print(f"\nReward weight vector w (norm={agent.w.norm().item():.4f}):")
    w_np = agent.w.detach().cpu().numpy()
    print(f"  min={w_np.min():.4f}  max={w_np.max():.4f}  mean={w_np.mean():.4f}  std={w_np.std():.4f}")

    # Check Q-values at init state
    q_init = agent.get_q_values(init_obs)
    print(f"\nQ-values at init state [0,0,0,0]: {q_init}")
    print(f"  Best action: {q_init.argmax()}")

    # Check Q-values with bottleneck w for a sample transition
    print(f"\nBottleneck w diagnostics:")
    for (src, tgt), bottleneck_obs_list in sorted(agent.bottleneck_obs.items()):
        if len(bottleneck_obs_list) > 0:
            w_temp = agent._compute_bottleneck_w(bottleneck_obs_list)
            w_temp_np = w_temp.detach().cpu().numpy()
            # Q-values at init state with this temporary w
            with torch.no_grad():
                obs_t = torch.as_tensor(init_obs, dtype=torch.float32).unsqueeze(0)
                all_sf = agent.sf_net(obs_t)
                q_temp = (all_sf * w_temp).sum(dim=-1).squeeze(0).cpu().numpy()
            print(f"  ({src}->{tgt}): {len(bottleneck_obs_list)} obs, "
                  f"w_temp norm={w_temp.norm().item():.4f}, "
                  f"Q(init, w_temp)={q_temp}")

    return goal_macros, V_macro


def main():
    separator("DIAGNOSTIC: Flat vs Hierarchical Neural SR on Acrobot")
    cfg = NEURAL_ACROBOT
    print(f"Config: {cfg}")

    # ==================== Part 1: Train the agent ====================
    separator("Part 1: Training HierarchicalNeuralSRAgent")

    adapter_train, base_train = make_adapter()
    print(f"Observation dim: {adapter_train.obs_dim}")
    print(f"Actions: {adapter_train.n_actions}")
    print(f"Goal states (discrete): {len(base_train.get_goal_states())} bins")
    print()

    agent = create_agent(
        adapter_train,
        cls=HierarchicalNeuralSRAgent,
        n_clusters=4,
        cluster_method='kmeans',
        cluster_on='observations',
        n_cluster_samples=5000,
        adjacency_episodes=500,
        adjacency_episode_length=50,
    )

    train_time = train_agent(agent, tag="[Hierarchical Agent]")

    # ==================== Part 2: Evaluate FLAT ====================
    adapter_test, base_test = make_adapter()
    flat_results = eval_flat(agent, adapter_test, base_test, n_episodes=5, label="Flat (before hierarchy)")

    # ==================== Part 3: Learn Hierarchy ====================
    separator("Part 3: Learning Hierarchy")
    # Use training adapter for hierarchy learning (needs random exploration)
    agent.adapter = adapter_train
    agent.goal_states = base_train.get_goal_states()
    agent.learn_hierarchy()

    # ==================== Part 4: Diagnose Hierarchy ====================
    diagnose_hierarchy(agent, base_train)

    # ==================== Part 5: Evaluate both flat and hierarchical ====================
    adapter_test2, base_test2 = make_adapter()

    # Flat evaluation (after hierarchy learned -- should be same as before)
    flat_results2 = eval_flat(agent, adapter_test2, base_test2, n_episodes=5, label="Flat (after hierarchy)")

    # Hierarchical evaluation
    adapter_test3, base_test3 = make_adapter()
    hier_results = eval_hierarchical_verbose(agent, adapter_test3, base_test3, n_episodes=5)

    # ==================== Summary ====================
    separator("SUMMARY")

    def summarize(label, results):
        steps = [r['steps'] for r in results]
        successes = [r['reached_goal'] for r in results]
        sr = sum(successes)
        print(f"  {label}: {sr}/{len(successes)} success  "
              f"avg_steps={np.mean(steps):.1f}+/-{np.std(steps):.1f}")

    summarize("Flat (before hierarchy)", flat_results)
    summarize("Flat (after hierarchy) ", flat_results2)
    summarize("Hierarchical           ", hier_results)

    print("\nDone.")


if __name__ == "__main__":
    main()
