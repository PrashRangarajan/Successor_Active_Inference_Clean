"""Experiment G: Pure random exploration (epsilon=1.0 throughout).

Tests whether the SF representation can be learned from purely random
transitions. At eval time the greedy policy (epsilon=0) is used.

Same Phase 1 length as Exp C/F (4000 eps, 500 steps). No Phase 2/3.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.configs import NEURAL_POINTMAZE

# Same Phase 1 as Exp C/F
NEURAL_POINTMAZE["train_episodes_phase1"] = 4000
NEURAL_POINTMAZE["steps_per_episode"] = 500
NEURAL_POINTMAZE["epsilon_decay_steps"] = 1_300_000

# Pure random: epsilon never decays
NEURAL_POINTMAZE["epsilon_start"] = 1.0
NEURAL_POINTMAZE["epsilon_end"] = 1.0

# No Phase 2/3
NEURAL_POINTMAZE["consolidation_episodes"] = 0
NEURAL_POINTMAZE["train_episodes_phase2"] = 0

import examples.configs
examples.configs.NEURAL_POINTMAZE = NEURAL_POINTMAZE

save_dir = sys.argv[1] if len(sys.argv) > 1 else "data/neural_point_maze_exp_g"
device = sys.argv[2] if len(sys.argv) > 2 else "cuda"

sys.argv = [
    "run_neural_point_maze.py",
    "--save-dir", save_dir,
    "--device", device,
    "--no-staging",
]

from examples.run_neural_point_maze import main
main()
