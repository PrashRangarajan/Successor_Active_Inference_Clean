"""Experiment I: More training data with best settings from Exp F.

Same as Exp F (sf_dim=128, ε floor 0.20, 500 steps, Phase 1 only) but with
8000 episodes instead of 4000. Tests whether more training data helps the SF
representation learn the Q-gradient through the U-bend.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.configs import NEURAL_POINTMAZE

# Double the Phase 1 episodes
NEURAL_POINTMAZE["train_episodes_phase1"] = 8000
NEURAL_POINTMAZE["steps_per_episode"] = 500
NEURAL_POINTMAZE["epsilon_decay_steps"] = 2_600_000  # scale with episodes

# Higher epsilon floor (from Exp F)
NEURAL_POINTMAZE["epsilon_end"] = 0.20

# sf_dim=128 (same as Exp F)
# No Phase 2/3
NEURAL_POINTMAZE["consolidation_episodes"] = 0
NEURAL_POINTMAZE["train_episodes_phase2"] = 0

import examples.configs
examples.configs.NEURAL_POINTMAZE = NEURAL_POINTMAZE

save_dir = sys.argv[1] if len(sys.argv) > 1 else "data/neural_point_maze_exp_i"
device = sys.argv[2] if len(sys.argv) > 2 else "cuda"

sys.argv = [
    "run_neural_point_maze.py",
    "--save-dir", save_dir,
    "--device", device,
    "--no-staging",
]

from examples.run_neural_point_maze import main
main()
