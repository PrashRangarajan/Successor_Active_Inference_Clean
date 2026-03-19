"""Experiment H: Larger sf_dim (256) with best settings from Exp F.

Same as Exp F (ε floor 0.20, 4000 eps, 500 steps, Phase 1 only) but with
sf_dim=256 instead of 128. Tests whether more SF dimensions help the linear
Q=φᵀw decomposition express the reward gradient through the U-bend.

hidden_sizes stays at (256, 256) to isolate the sf_dim variable.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.configs import NEURAL_POINTMAZE

# Same Phase 1 as Exp F
NEURAL_POINTMAZE["train_episodes_phase1"] = 4000
NEURAL_POINTMAZE["steps_per_episode"] = 500
NEURAL_POINTMAZE["epsilon_decay_steps"] = 1_300_000

# Higher epsilon floor (from Exp F)
NEURAL_POINTMAZE["epsilon_end"] = 0.20

# Larger SF dimension
NEURAL_POINTMAZE["sf_dim"] = 256

# No Phase 2/3
NEURAL_POINTMAZE["consolidation_episodes"] = 0
NEURAL_POINTMAZE["train_episodes_phase2"] = 0

import examples.configs
examples.configs.NEURAL_POINTMAZE = NEURAL_POINTMAZE

save_dir = sys.argv[1] if len(sys.argv) > 1 else "data/neural_point_maze_exp_h"
device = sys.argv[2] if len(sys.argv) > 2 else "cuda"

sys.argv = [
    "run_neural_point_maze.py",
    "--save-dir", save_dir,
    "--device", device,
    "--no-staging",
]

from examples.run_neural_point_maze import main
main()
