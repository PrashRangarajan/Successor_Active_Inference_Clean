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
    "n_clusters": 6,
    "gamma": 0.95,
    "learning_rate": 0.05,
    "learn_from_experience": True,
    "train_episodes": 6000,
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
    "n_theta_bins": 20,
    "n_omega_bins": 20,
    "n_torque_bins": 5,
    "n_clusters": 4,
    "gamma": 0.95,
    "learning_rate": 0.05,
    "learn_from_experience": True,
    "train_episodes": 5000,
    "test_max_steps": 200,

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

    # Agent
    "reward": 100.0,
    "default_cost": -1.0,

    # Eval-specific
    "eval_n_runs": 5,
    "eval_episodes": [1000, 2000, 4000, 6000, 8000, 10000],
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
# Neural Acrobot (deep successor features)
# =====================================================================
NEURAL_ACROBOT = {
    # Environment (bins only used for goal checking, not state representation)
    "n_theta_bins": 6,
    "n_dtheta_bins": 5,
    "goal_velocity_filter": True,

    # Neural architecture
    "sf_dim": 64,
    "hidden_sizes": (128, 128),

    # Training
    "gamma": 0.99,
    "lr": 1e-3,
    "lr_w": 1e-3,
    "batch_size": 128,
    "buffer_size": 100_000,
    "target_update_freq": 500,
    "tau": 0.01,
    "steps_per_episode": 500,

    # Two-phase training: diverse exploration then fixed-start
    "train_episodes_diverse": 1000,   # Phase 1: build SF representation
    "train_episodes_fixed": 2000,     # Phase 2: learn task from fixed start

    # Exploration
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay_steps": 30_000,

    # Test
    "test_max_steps": 500,
    "reward": 1.0,
    "default_cost": 0.0,

    # Eval-specific
    "eval_n_runs": 5,
    "eval_episodes": [200, 500, 1000, 2000, 3000, 5000],
    "eval_quick_episodes": [500, 1000, 2000],
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
