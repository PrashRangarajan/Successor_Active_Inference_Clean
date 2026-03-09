"""Experiment D: Long Phase 1 + skip Phase 2 (consolidation) + Phase 3.

Same as Exp C (4000 ep Phase 1, 500 steps, joint fine-tune Phase 3)
but without the Phase 2 consolidation step, which appeared to slightly
degrade performance in Exp C.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.configs import NEURAL_POINTMAZE

# Same Phase 1 overrides as Exp C
NEURAL_POINTMAZE["train_episodes_phase1"] = 4000
NEURAL_POINTMAZE["steps_per_episode"] = 500
NEURAL_POINTMAZE["epsilon_decay_steps"] = 1_300_000  # 65% of 4000*500

# Disable consolidation by setting episodes to 0
NEURAL_POINTMAZE["consolidation_episodes"] = 0

import examples.configs
examples.configs.NEURAL_POINTMAZE = NEURAL_POINTMAZE

save_dir = sys.argv[1] if len(sys.argv) > 1 else "data/neural_point_maze_exp_d"
device = sys.argv[2] if len(sys.argv) > 2 else "cuda"

sys.argv = [
    "run_neural_point_maze.py",
    "--save-dir", save_dir,
    "--device", device,
    "--no-freeze-phase3",
]

from examples.run_neural_point_maze import main
main()
