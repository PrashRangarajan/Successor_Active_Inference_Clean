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
    "train_episodes": 1500,
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

    # Replanning experiment: 5 sequential goals across all rooms of UMaze
    "replan_goals": [
        {"cell": [3, 1], "label": "Bottom-left"},
        {"cell": [1, 3], "label": "Top-right"},
        {"cell": [3, 3], "label": "Bottom-right"},
        {"cell": [1, 1], "label": "Top-left"},
        {"cell": [2, 3], "label": "U-bend"},
    ],

    # Eval-specific
    "eval_n_runs": 5,
    "eval_episodes": [25, 50, 100, 200, 300, 500, 750, 1000, 1500],
    "eval_quick_episodes": [50, 200, 500, 1000],
    "eval_quick_n_runs": 2,
}

# =====================================================================
# PointMaze — Medium (8×8 grid, ~416 navigable bins)
# =====================================================================
POINTMAZE_MEDIUM = {
    **POINTMAZE,
    "n_x_bins": 32,             # 8 cols × 4 bins/cell
    "n_y_bins": 32,             # 8 rows × 4 bins/cell
    "n_clusters": 8,            # Medium has ~6-8 natural rooms
    "train_episodes": 5000,     # ~416 navigable states, 5k is sufficient
    "test_max_steps": 10000,    # Longer paths
    "maze_id": "PointMaze_Medium-v3",
    "render_width": 720,        # 8×8 → square, higher res for clarity
    "render_height": 720,

    # Replanning: 5 goals spanning the four corners + center corridor
    # Top-left is NOT first because the agent spawns in that cell;
    # short-distance navigation on a discretized value plateau is unreliable.
    "replan_goals": [
        {"cell": [6, 6], "label": "Bot-right"},
        {"cell": [1, 6], "label": "Top-right"},
        {"cell": [1, 1], "label": "Top-left"},
        {"cell": [6, 1], "label": "Bot-left"},
        {"cell": [3, 3], "label": "Center"},
    ],

    # Eval-specific (override UMaze — Medium needs more training)
    "eval_episodes": [100, 250, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000],
    "eval_quick_episodes": [250, 1000, 3000, 5000],
}

# =====================================================================
# PointMaze — Large (9×12 grid, ~414 navigable bins)
# =====================================================================
POINTMAZE_LARGE = {
    **POINTMAZE,
    "n_x_bins": 36,             # 12 cols × 3 bins/cell
    "n_y_bins": 27,             # 9 rows × 3 bins/cell
    "n_clusters": 12,           # Large has many rooms/corridors; 12 keeps within-cluster paths short
    "train_episodes": 15000,    # ~414 navigable states, complex layout needs more
    "test_max_steps": 20000,    # Longest paths in the maze
    "maze_id": "PointMaze_Large-v3",
    "render_width": 640,        # 12:9 aspect ratio → 4:3
    "render_height": 480,

    # Replanning: 6 goals spanning quadrants + center + mid-corridor
    "replan_goals": [
        {"cell": [1, 1], "label": "Top-left"},
        {"cell": [1, 10], "label": "Top-right"},
        {"cell": [7, 1], "label": "Bot-left"},
        {"cell": [7, 10], "label": "Bot-right"},
        {"cell": [3, 5], "label": "Center"},
        {"cell": [5, 7], "label": "Mid-right"},
    ],

    # Eval-specific (override UMaze — Large needs the most training)
    "eval_episodes": [250, 500, 1000, 2000, 3000, 4000, 5000, 7000, 10000],
    "eval_quick_episodes": [500, 2000, 5000, 10000],
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

    # Training — lower LR and softer target updates for stability
    # (aligned with HalfCheetah/InvertedPendulum which converge well)
    "gamma": 0.99,
    "lr": 3e-4,                       # was 1e-3 — too aggressive for reach tasks
    "lr_w": 3e-4,                     # was 1e-3
    "batch_size": 256,                # was 128 — reduce gradient variance
    "buffer_size": 300_000,           # was 200_000 — smooth distribution shift
    "target_update_freq": 1000,       # was 500 — less frequent target updates
    "tau": 0.005,                     # was 0.01 — softer Polyak averaging
    "steps_per_episode": 500,

    # Training schedule — three-phase gradual transition
    # (avoids hard distribution shift that caused reward crash at ep 2000)
    "train_episodes_phase1": 1500,    # Phase 1: 100% diverse (build SF representation)
    "train_episodes_phase2": 1500,    # Phase 2: 60% diverse (gradual shift)
    "train_episodes_phase3": 2000,    # Phase 3: 30% diverse (task-focused)
    "diverse_fraction_phase2": 0.6,   # intermediate diversity
    "diverse_fraction_phase3": 0.3,   # final diversity
    "buffer_keep_phase2": 0.3,        # keep 30% of buffer at Phase 1→2
    "buffer_keep_phase3": 0.5,        # keep 50% of buffer at Phase 2→3
    "lr_phase2_fraction": 0.5,        # Phase 2 LR = 50% of initial
    "lr_phase3_fraction": 0.25,       # Phase 3 LR = 25% of initial

    # Exploration — phase-aware epsilon resets at distribution shifts
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 120_000,          # Phase 1 decay
    "epsilon_phase2_start": 0.3,             # Bump at Phase 1→2 boundary
    "epsilon_phase2_decay_steps": 80_000,    # Decay within Phase 2
    "epsilon_phase3_start": 0.15,            # Smaller bump at Phase 2→3
    "epsilon_phase3_decay_steps": 50_000,    # Decay within Phase 3

    # Test
    "test_max_steps": 500,
    "reward": 1.0,
    "default_cost": 0.0,
    "terminal_bonus": 100.0,  # Bonus when env terminates (goal reached)

    # Eval-specific
    "eval_n_runs": 8,
    "eval_episodes": [500, 1000, 2000, 3000, 5000],
    "eval_quick_episodes": [500, 1000, 2000],
    "eval_quick_n_runs": 5,

    # Staged learning — consolidate SF before goal-focused phase
    "staged_learning": True,
    "consolidation_episodes": 500,
    "consolidation_episodic_replay": 10,

    # Training frequency — skip some gradient updates to reduce overhead
    "train_every": 4,

    # Replay buffer — PER + episodic replay
    "use_per": True,
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
    "per_beta_end": 1.0,
    "use_episodic_replay": True,
    "episodic_replay_episodes": 5,
}

# =====================================================================
# Neural InvertedPendulum (MuJoCo — deep successor features)
# =====================================================================
NEURAL_INVERTED_PENDULUM = {
    # Environment
    "n_force_bins": 7,              # Discrete forces in [-3, 3]

    # Neural architecture
    "sf_dim": 64,
    "hidden_sizes": (128, 128),

    # Training — lower learning rates for stability on survival tasks
    "gamma": 0.99,
    "lr": 3e-4,
    "lr_w": 3e-4,
    "batch_size": 128,
    "buffer_size": 200_000,
    "target_update_freq": 1000,     # Less frequent target updates
    "tau": 0.005,                   # Softer target updates
    "steps_per_episode": 500,

    # Training schedule — more episodes for this task
    "train_episodes_diverse": 2000,   # Phase 1: build SF representation
    "train_episodes_fixed": 3000,     # Phase 2: mixed training
    "diverse_fraction": 0.3,

    # Exploration — moderate decay; with reward shaping, early episodes
    # are longer so steps accumulate faster than without shaping.
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 10_000,

    # Test
    "test_max_steps": 1000,
    "reward": 1.0,
    "default_cost": 0.0,
    "terminal_bonus": -5.0,          # Penalty for pole falling (termination = failure)

    # Eval-specific
    "eval_n_runs": 5,
    "eval_episodes": [500, 1000, 2000, 3000, 5000],
    "eval_quick_episodes": [500, 1000, 2000],
    "eval_quick_n_runs": 2,

    # Replay buffer — PER + episodic replay (opt-in)
    "use_per": False,
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
    "per_beta_end": 1.0,
    "use_episodic_replay": False,
    "episodic_replay_episodes": 2,
}

# =====================================================================
# Neural HalfCheetah (MuJoCo — deep successor features, action-conditioned)
# =====================================================================
NEURAL_HALF_CHEETAH = {
    # Environment
    "n_bins_per_joint": 3,          # 3^6 = 729 discrete actions

    # Neural architecture — action-conditioned network for large action space
    "sf_dim": 128,
    "hidden_sizes": (256, 256),
    "sf_network_cls": "action_conditioned",

    # Training
    "gamma": 0.99,
    "lr": 3e-4,
    "lr_w": 3e-4,
    "batch_size": 256,
    "buffer_size": 500_000,
    "target_update_freq": 1000,
    "tau": 0.005,
    "steps_per_episode": 500,

    # Training schedule
    "train_episodes_diverse": 2000,   # Phase 1: build SF representation
    "train_episodes_fixed": 3000,     # Phase 2: mixed training
    "diverse_fraction": 0.3,
    "buffer_keep_phase2": 0.3,        # keep 30% of buffer at Phase 1→2
    "lr_phase2_fraction": 0.5,        # Phase 2 LR = 50% of initial

    # Exploration — phase-aware epsilon resets
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 650_000,           # 65% of Phase 1 (2000×500=1M steps)
    "epsilon_phase2_start": 0.3,
    "epsilon_phase2_decay_steps": 975_000,   # 65% of Phase 2 (3000×500=1.5M steps)

    # Test
    "test_max_steps": 1000,
    "reward": 1.0,
    "default_cost": 0.0,
    "terminal_bonus": 0.0,

    # Eval-specific
    "eval_n_runs": 5,
    "eval_episodes": [500, 1000, 2000, 3000, 5000],
    "eval_quick_episodes": [500, 1000, 2000],
    "eval_quick_n_runs": 3,

    # Replay buffer — PER + episodic replay (opt-in)
    "use_per": False,
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
    "per_beta_end": 1.0,
    "use_episodic_replay": False,
    "episodic_replay_episodes": 2,
}

# =====================================================================
# Neural PointMaze (gymnasium-robotics — deep successor features)
# =====================================================================
NEURAL_POINTMAZE = {
    # Environment — same bins as tabular for goal checking
    "maze_id": "PointMaze_UMaze-v3",
    "n_x_bins": 20,
    "n_y_bins": 20,

    # Neural architecture
    "sf_dim": 128,
    "hidden_sizes": (256, 256),

    # Training
    "gamma": 0.99,
    "lr": 3e-4,
    "lr_w": 3e-4,
    "batch_size": 256,
    "buffer_size": 500_000,
    "target_update_freq": 1000,
    "tau": 0.005,
    "steps_per_episode": 300,

    # Two-phase training schedule
    "train_episodes_phase1": 2000,    # Phase 1: 100% diverse (build SF map)
    "train_episodes_phase2": 3000,    # Phase 2: 30% diverse (goal-focused)
    "diverse_fraction_phase2": 0.3,
    "buffer_keep_phase2": 0.3,
    "lr_phase2_fraction": 0.5,

    # Exploration — phase-aware epsilon resets
    # Decay spans ~65% of each phase so exploration covers most of training.
    # Phase 1: 2000 eps × 300 spe = 600K steps → decay over 390K (floor at ep 1300)
    # Phase 2: 3000 eps × 300 spe = 900K steps → decay over 585K (floor at ep 1950)
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 390_000,
    "epsilon_phase2_start": 0.3,
    "epsilon_phase2_decay_steps": 585_000,

    # Smooth stepping — the point mass moves slowly, so we repeat
    # each action for multiple physics steps to cover meaningful distance.
    # Train: 200 steps gives margin for exploration.
    # Test: 100 steps for finer control.
    "train_smooth_steps": 200,
    "test_smooth_steps": 100,

    # Test
    "test_max_steps": 5000,
    "reward": 1.0,
    "default_cost": 0.0,
    "terminal_bonus": 50.0,

    # Eval-specific
    "eval_n_runs": 5,
    "eval_episodes": [500, 1000, 2000, 3000, 5000],
    "eval_quick_episodes": [500, 1000, 2000],
    "eval_quick_n_runs": 3,

    # Replay buffer — PER + episodic replay
    "use_per": True,
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
    "per_beta_end": 1.0,
    "use_episodic_replay": True,
    "episodic_replay_episodes": 5,

    # Hindsight Experience Replay (HER)
    # Relabels failed trajectories with achieved goals, turning every
    # episode into useful training data for goal-conditioned learning.
    "use_her": True,
    "her_k": 4,                    # future goals per transition
    "her_goal_indices": [4, 6],    # obs[4:6] = [goal_x, goal_y]

    # Staged learning — Phase 2 consolidation before goal-focused Phase 3
    # In Phase 2, freeze w to consolidate φ/ψ with extra replay,
    # then in Phase 3, freeze φ and only train w for goal adaptation.
    "staged_learning": True,
    "consolidation_episodes": 1000,  # Phase 2: SF consolidation
    "consolidation_episodic_replay": 10,  # replay intensity during consolidation

    # Training frequency — update networks every N env steps.
    # Reduces CPU/GPU overhead ~4x with minimal impact on learning.
    "train_every": 4,
}

# =====================================================================
# Neural AntMaze (gymnasium-robotics — deep successor features, action-conditioned)
# =====================================================================
NEURAL_ANTMAZE = {
    # Environment
    "maze_id": "AntMaze_UMaze-v5",
    "n_bins_per_joint": 3,           # 3^8 = 6561 discrete actions

    # Neural architecture — action-conditioned network for large action space
    "sf_dim": 128,
    "hidden_sizes": (256, 256),
    "sf_network_cls": "action_conditioned",

    # Training
    "gamma": 0.99,
    "lr": 3e-4,
    "lr_w": 3e-4,
    "batch_size": 256,
    "buffer_size": 500_000,
    "target_update_freq": 1000,
    "tau": 0.005,
    "steps_per_episode": 500,

    # Training schedule — two-phase
    "train_episodes_diverse": 2000,   # Phase 1: build SF representation
    "train_episodes_fixed": 3000,     # Phase 2: mixed training
    "diverse_fraction": 0.3,
    "buffer_keep_phase2": 0.3,
    "lr_phase2_fraction": 0.5,

    # Exploration — phase-aware epsilon resets
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 200_000,
    "epsilon_phase2_start": 0.3,
    "epsilon_phase2_decay_steps": 100_000,

    # Test
    "test_max_steps": 1000,
    "reward": 1.0,
    "default_cost": 0.0,
    "terminal_bonus": 50.0,

    # Eval-specific
    "eval_n_runs": 5,
    "eval_episodes": [500, 1000, 2000, 3000, 5000],
    "eval_quick_episodes": [500, 1000, 2000],
    "eval_quick_n_runs": 3,

    # Replay buffer — PER + episodic replay (opt-in)
    "use_per": False,
    "per_alpha": 0.6,
    "per_beta_start": 0.4,
    "per_beta_end": 1.0,
    "use_episodic_replay": False,
    "episodic_replay_episodes": 2,
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
