"""Experiment F: Higher epsilon floor (0.20) with long Phase 1 only.

Same as Exp C's Phase 1 (4000 eps, 500 steps) but with epsilon_end=0.20
instead of 0.05. The agent never stops exploring, keeping diverse transitions
flowing into the buffer throughout training.

No Phase 2 or 3 — Phase 1 only based on finding that later phases don't help.
Uses vectorized envs (8 parallel) for speed.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.configs import NEURAL_POINTMAZE

# Same Phase 1 as Exp C
NEURAL_POINTMAZE["train_episodes_phase1"] = 4000
NEURAL_POINTMAZE["steps_per_episode"] = 500
NEURAL_POINTMAZE["epsilon_decay_steps"] = 1_300_000

# Higher epsilon floor
NEURAL_POINTMAZE["epsilon_end"] = 0.20

# No Phase 2/3
NEURAL_POINTMAZE["consolidation_episodes"] = 0
NEURAL_POINTMAZE["train_episodes_phase2"] = 0

import examples.configs
examples.configs.NEURAL_POINTMAZE = NEURAL_POINTMAZE

save_dir = sys.argv[1] if len(sys.argv) > 1 else "data/neural_point_maze_exp_f"
device = sys.argv[2] if len(sys.argv) > 2 else "cuda"
vec_envs = sys.argv[3] if len(sys.argv) > 3 else "8"

sys.argv = [
    "run_neural_point_maze.py",
    "--save-dir", save_dir,
    "--device", device,
    "--no-staging",
    "--vec-envs", vec_envs,
]

from examples.run_neural_point_maze import main
main()
