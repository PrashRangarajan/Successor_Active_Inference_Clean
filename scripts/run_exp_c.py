"""Experiment C: More Phase 1 exploration for better bottom-corridor SF features.

Changes from default NEURAL_POINTMAZE:
- Phase 1: 4000 episodes (was 2000) — 2x more diverse exploration
- steps_per_episode: 500 (was 300) — longer episodes for full maze traversals
- epsilon_decay_steps: 1_300_000 (65% of 4000*500=2M) — match longer Phase 1
- No SF freeze in Phase 3 (joint fine-tuning, as in Exp A)

Hypothesis: The Q-value gradient breaks at the U-bend because φ features
in the bottom corridor are too uniform. More diverse exploration with longer
episodes should force more bottom→U-bend→top traversals, building richer
SF features that differentiate positions along the path.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.configs import NEURAL_POINTMAZE

# Override config for this experiment
NEURAL_POINTMAZE["train_episodes_phase1"] = 4000
NEURAL_POINTMAZE["steps_per_episode"] = 500
NEURAL_POINTMAZE["epsilon_decay_steps"] = 1_300_000  # 65% of 4000*500

# Patch the config module so run_neural_point_maze picks up changes
import examples.configs
examples.configs.NEURAL_POINTMAZE = NEURAL_POINTMAZE

# Patch sys.argv before importing main
save_dir = sys.argv[1] if len(sys.argv) > 1 else "data/neural_point_maze_exp_c"
device = sys.argv[2] if len(sys.argv) > 2 else "cuda"

sys.argv = [
    "run_neural_point_maze.py",
    "--save-dir", save_dir,
    "--device", device,
    "--no-freeze-phase3",
]

from examples.run_neural_point_maze import main
main()
