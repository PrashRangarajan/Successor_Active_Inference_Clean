"""Experiment I: Multi-seed replication of Exp F (ε=0.20, Phase 1 only).

Runs the best config (Exp F) across multiple seeds to measure variance.
Each seed gets a separate save directory: data/neural_point_maze_exp_i/seed_{N}/

Usage:
    # Run seeds 0,2,4 on GPU 0:
    CUDA_VISIBLE_DEVICES=0 conda run -n sai python scripts/run_exp_i_seeds.py cuda 0 2 4

    # Run seeds 1,3 on GPU 1:
    CUDA_VISIBLE_DEVICES=1 conda run -n sai python scripts/run_exp_i_seeds.py cuda 1 3
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch


def run_seed(seed, device="cuda"):
    """Run one seed of Exp F config."""
    # Set all random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Re-import config fresh (run_neural_point_maze modifies global state)
    import importlib
    import examples.configs
    importlib.reload(examples.configs)
    from examples.configs import NEURAL_POINTMAZE

    # Exp F config: ε=0.20, 4000 eps, Phase 1 only
    NEURAL_POINTMAZE["train_episodes_phase1"] = 4000
    NEURAL_POINTMAZE["steps_per_episode"] = 500
    NEURAL_POINTMAZE["epsilon_decay_steps"] = 1_300_000
    NEURAL_POINTMAZE["epsilon_end"] = 0.20

    # No Phase 2/3
    NEURAL_POINTMAZE["consolidation_episodes"] = 0
    NEURAL_POINTMAZE["train_episodes_phase2"] = 0

    examples.configs.NEURAL_POINTMAZE = NEURAL_POINTMAZE

    save_dir = f"data/neural_point_maze_exp_i/seed_{seed}"

    sys.argv = [
        "run_neural_point_maze.py",
        "--save-dir", save_dir,
        "--device", device,
        "--no-staging",
    ]

    # Reload the main module to pick up fresh config
    import examples.run_neural_point_maze
    importlib.reload(examples.run_neural_point_maze)
    examples.run_neural_point_maze.main()


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    seeds = [int(s) for s in sys.argv[2:]] if len(sys.argv) > 2 else [0, 1, 2, 3, 4]

    print(f"=== Experiment I: Multi-seed (Exp F config) ===")
    print(f"Device: {device}")
    print(f"Seeds: {seeds}")
    print()

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}\n")
        run_seed(seed, device)

    print(f"\n{'='*60}")
    print(f"All seeds complete. Results in data/neural_point_maze_exp_i/seed_*/")
    print(f"{'='*60}")
