"""Centralized hyperparameter configurations for all environments.

Each environment has a single canonical config dict used by both the
standalone run_*.py demo and the run_eval_*.py benchmark. This ensures
parameter consistency and makes tuning easy — change one place, both
scripts stay in sync.

Usage:
    from examples.configs import ACROBOT, MOUNTAINCAR

    n_theta_bins = ACROBOT["n_theta_bins"]
    gamma        = ACROBOT["gamma"]
"""

import numpy as np

# =====================================================================
# Gridworld
# =====================================================================
GRIDWORLD = {
    "grid_size": 9,
    "n_clusters": 4,           # Overridden per layout (fourrooms=4, maze=6, etc.)
    "gamma": 0.99,
    "learning_rate": 0.05,
    "learn_from_experience": False,  # Analytical M for known dynamics
    "train_episodes": 1500,
    "test_max_steps": 200,

    # Eval-specific
    "eval_n_runs": 20,
    "eval_episodes": [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000, 7000],
    "eval_quick_episodes": [500, 2000, 5000],
    "eval_quick_n_runs": 2,
}

# =====================================================================
# Key Gridworld
# =====================================================================
KEY_GRIDWORLD = {
    "grid_size": 5,
    "n_clusters": 5,
    "gamma": 0.99,
    "learning_rate": 0.05,
    "learn_from_experience": True,
    "train_episodes": 1500,
    "test_max_steps": 100,
    "has_pickup_action": True,

    # Layout
    "init_loc": (0, 0),
    "key_loc": (3, 0),

    # Eval-specific
    "eval_n_runs": 10,
    "eval_episodes": [100, 200, 500, 750, 1000, 1500, 2000, 3000, 4000],
    "eval_quick_episodes": [500, 1500, 3000],
    "eval_quick_n_runs": 2,
}

# =====================================================================
# Mountain Car
# =====================================================================
MOUNTAINCAR = {
    "n_pos_bins": 10,
    "n_vel_bins": 10,
    "n_clusters": 5,
    "gamma": 0.95,
    "learning_rate": 0.05,
    "learn_from_experience": True,
    "train_episodes": 5000,
    "test_max_steps": 500,
    "train_smooth_steps": 10,
    "test_smooth_steps": 1,

    # Agent
    "reward": 100.0,
    "default_cost": -1.0,

    # Eval-specific
    "eval_n_runs": 5,
    "eval_episodes": [500, 1000, 2000, 4000, 6000, 8000, 10000],
    "eval_quick_episodes": [1000, 4000, 8000],
    "eval_quick_n_runs": 2,
}

# =====================================================================
# Acrobot
# =====================================================================
ACROBOT = {
    "n_theta_bins": 6,
    "n_dtheta_bins": 5,
    "n_clusters": 4,
    "gamma": 0.95,
    "learning_rate": 0.05,
    "learn_from_experience": True,
    "train_episodes": 10000,
    "test_max_steps": 1000,
    "train_smooth_steps": 10,
    "test_smooth_steps": 10,
    "goal_velocity_filter": True,

    # Agent
    "reward": 100.0,
    "default_cost": -1.0,

    # Eval-specific
    "eval_n_runs": 5,
    "eval_episodes": [1000, 2000, 5000, 10000, 15000, 20000],
    "eval_quick_episodes": [1000, 5000, 10000],
    "eval_quick_n_runs": 2,
}

# =====================================================================
# Pendulum
# =====================================================================
PENDULUM = {
    "n_theta_bins": 21,         # Odd → goal region centered on θ=0
    "n_omega_bins": 21,         # Odd → goal region centered on ω=0
    "n_torque_bins": 7,         # Includes 0-torque; balanced resolution
    "n_clusters": 4,
    "gamma": 0.97,              # Slightly higher for longer-horizon swing-up
    "learning_rate": 0.05,
    "learn_from_experience": True,
    "train_episodes": 9000,
    "test_max_steps": 200,
    "train_smooth_steps": 1,    # No smooth stepping — shaped reward provides gradient
    "test_smooth_steps": 1,     # Match train; finer control → better balancing

    # Agent
    "reward": 100.0,
    "default_cost": -1.0,

    # Eval-specific
    "eval_n_runs": 5,
    "eval_episodes": [500, 1000, 2000, 3000, 5000, 8000],
    "eval_quick_episodes": [1000, 3000, 8000],
    "eval_quick_n_runs": 2,
}

# =====================================================================
# Pendulum (Sparse Reward)
# =====================================================================
# Same dynamics as PENDULUM but with a step-function reward: +reward
# inside a small ball near upright, slight negative elsewhere.
# Tests whether hierarchy helps when V = M @ C has a weak gradient.
PENDULUM_SPARSE = {
    **PENDULUM,

    # Sparse reward parameters
    "sparse_radius": 0.5,            # θ² + 0.1·ω² < radius² → goal (~2.5% of states)
    "sparse_reward": 10.0,           # reward inside the ball
    "sparse_default_cost": -0.1,     # slight penalty outside
    "sparse_goal_threshold": 0.0,    # C value cutoff for _is_at_goal
}

# =====================================================================
# CartPole (experimental — survival task, not goal-reaching)
# =====================================================================
CARTPOLE = {
    "n_pos_bins": 6,
    "n_vel_bins": 6,
    "n_angle_bins": 8,
    "n_ang_vel_bins": 6,
    "n_clusters": 6,
    "gamma": 0.99,
    "learning_rate": 0.05,
    "learn_from_experience": True,
    "train_episodes": 6000,
    "test_max_steps": 500,

    # Sparse reward (survival task — same pattern as pendulum sparse)
    "sparse_radius": 0.5,
    "sparse_reward": 10.0,
    "sparse_default_cost": -0.1,
    "sparse_goal_threshold": 0.0,

    # Eval-specific
    "eval_n_runs": 5,
    "eval_episodes": [1000, 2000, 4000, 6000, 8000, 10000],
    "eval_quick_episodes": [1000, 4000, 8000],
    "eval_quick_n_runs": 2,
}

# =====================================================================
# PointMaze (gymnasium-robotics)
# =====================================================================
POINTMAZE = {
    "n_x_bins": 20,
    "n_y_bins": 20,
    "n_clusters": 4,            # UMaze has ~3-4 natural rooms
    "gamma": 0.95,
    "learning_rate": 0.05,
    "learn_from_experience": True,
    "train_episodes": 5000,
    "test_max_steps": 5000,

    # Smooth stepping: the point mass moves ~0.0024 units/physics-step.
    # With 20 bins over 5 units, each bin is 0.25 units -> need ~100 steps
    # to cross one bin.  _step_with_smooth breaks early once the discrete
    # state changes.
    # Train: 200 gives margin for exploration.
    # Test:  100 so each action covers ~1 bin.  With max_steps=5000
    #        the agent gets ~50 actions = ~12 units of movement, enough
    #        to traverse the U-maze (~6 units path length).
    "train_smooth_steps": 200,
    "test_smooth_steps": 100,

    # Agent
    "reward": 100.0,
    "default_cost": -1.0,

    # Maze variant
    "maze_id": "PointMaze_UMaze-v3",

    # Eval-specific
    "eval_n_runs": 5,
    "eval_episodes": [500, 1000, 2000, 4000, 6000, 8000],
    "eval_quick_episodes": [1000, 4000, 8000],
    "eval_quick_n_runs": 2,
}

# =====================================================================
# POMDP Gridworld
# =====================================================================
POMDP_GRIDWORLD = {
    "grid_size": 9,
    "gamma": 0.99,
    "test_max_steps": 200,

    # POMDP-specific
    "noise_level": 0.3,
    "noise_spread": 3.0,
    "beta": 1.0,

    # Layout
    "init_loc": (0, 0),

    # Eval-specific
    "eval_n_runs": 10,
    "eval_episodes": [100, 250, 500, 1000, 1500, 2500, 5000],
    "eval_quick_episodes": [500, 2000, 5000],
    "eval_quick_n_runs": 2,
}

# =====================================================================
# Shared defaults (replay, Q-learning)
# =====================================================================
SHARED = {
    "use_replay": True,
    "n_replay_epochs": 10,

    # Q-learning baseline
    "q_epsilon_decay": 0.999,
    "q_epsilon_end": 0.1,
}
