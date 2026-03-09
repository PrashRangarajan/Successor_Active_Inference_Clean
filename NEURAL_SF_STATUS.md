# Neural Successor Feature Experiments — Status & Context

> **Read this first** when continuing work on the neural SF agent experiments.
> Last updated: 2026-03-09

## Project Overview

This project implements **Successor Representation (SR) / Successor Features (SF)** for reinforcement learning, extending the tabular SR framework to continuous state spaces using neural networks.

Core equation: `Q(s, a) = φ(s, a)ᵀ · w`
- `φ(s, a)` = successor features (learned via TD, analogous to tabular M matrix)
- `w` = reward weight vector (learned via reward regression: `r(s) ≈ ψ(s)ᵀ · w`)
- `ψ(s)` = reward features from a separate small network

**Key advantage**: When the goal changes, only `w` needs retraining — φ captures task-independent dynamics.

**Note on "linear decomposition"**: Although Q = φᵀw is linear in w, both φ(s,a) and ψ(s) are neural networks — nonlinear in the state. The representation capacity should be sufficient; the real challenges are optimization and exploration.

## Architecture

```
core/neural/
  agent.py          - NeuralSRAgent (main agent class)
                      Includes select_actions_batch() and learn_phase1_vectorized()
  networks.py       - SFNetwork, ActionConditionedSFNetwork, RewardFeatureNetwork
  losses.py         - sf_td_loss, reward_prediction_loss
  replay_buffer.py  - ReplayBuffer with PER, HER, episodic replay, add_batch()
  continuous_adapter.py - ContinuousAdapter wrapping BinnedContinuousAdapter

environments/
  point_maze/       - PointMazeAdapter (gymnasium-robotics PointMaze_UMaze-v3)
  point_maze/wrappers.py - Gymnasium wrappers for vectorized training
  acrobot/          - AcrobotAdapter

examples/
  configs.py        - All hyperparameter configs (NEURAL_POINTMAZE, NEURAL_ACROBOT, etc.)
  run_neural_point_maze.py  - PointMaze training script (supports --vec-envs N)
  run_neural_acrobot.py     - Acrobot training script (4-phase staged learning)

scripts/
  visualize_qvalues.py      - Q-value heatmap visualization for PointMaze
  run_exp_c.py through run_exp_f.py - Experiment scripts
```

## PointMaze UMaze — Current Status

### Environment
- Observation: 6D `[x, y, vx, vy, goal_x, goal_y]`, 8 discrete directional actions
- Coordinate range: (-2.5, 2.5) x (-2.5, 2.5), 20x20 bins (112 navigable)
- U-shaped maze: top corridor → U-bend (right) → bottom corridor
- Goal: typically in top-left; fixed start: bottom corridor

### Training Pipeline

Originally 3-phase staged learning, but experiments show **Phase 1 alone is sufficient**:
1. **Phase 1** — Diverse exploration (100% random starts): Build SF representation
2. ~~Phase 2 — SF consolidation (w frozen)~~: Slightly degrades performance
3. ~~Phase 3 — Goal-focused (30% diverse)~~: Doesn't improve success rate over Phase 1 alone

### Results

| Run | Config | Success Rate | Notes |
|-----|--------|-------------|-------|
| Original | Staged + SF frozen Phase 3 | 2/10 (20%) | SF frozen → w can't adapt |
| Exp A | Staged + joint fine-tune Phase 3 | 3/10 (30%) | Best reward curve (-15) |
| Exp B | No staging (simple 2-phase) | 3/10 (30%) | Worst rewards (-142) |
| **Exp C** | **4000 ep Phase 1, staged** | **5/10 (50%)** | **Best so far** |
| Exp C P1 only | Just Phase 1 checkpoint evaluated | 5/10 (50%) | Phase 2/3 add nothing |
| Exp D | Exp C config, skip Phase 2 | Pending results | Still running |
| Exp E | Exp C + 4000 more Phase 1 at ε=0.05 | 1/10 (10%) | Overfitting — low ε hurts |
| **Exp F** | **ε floor 0.20, 8 vec envs, Phase 1 only** | **Running** | Testing higher exploration |

**Checkpoints**: `data/neural_point_maze/`, `data/neural_point_maze_exp_{a,b,c,d,e,f}/`

### Key Findings

1. **More high-ε exploration helps** — Exp C's 4000 eps with ε decaying 1.0→0.05 got 50%
2. **More low-ε episodes hurts** — Exp E's additional 4000 eps at ε=0.05 dropped to 10%
3. **Phases 2 and 3 don't improve success rate** — Phase 1 alone = 50%, full pipeline = 50%
4. **Q-value gradient improved in Exp C** — wider range (-17.6 to +6.6 vs ~-6 to +3), more differentiation in bottom corridor, but still flat near U-bend entrance

### Diagnosed Problem (via Q-value visualization)

**The Q-value gradient doesn't fully propagate through the U-bend into the bottom corridor.**

See `figures/neural_point_maze/qvalue_comparison.png`:
- Top corridor: smooth Q-value gradient toward goal, correct policy arrows ✓
- Bottom corridor: improved in Exp C but still partially flat near U-bend ✗
- Starts near top succeed fast (avg 67 steps), starts in bottom time out

### Currently Running

**Experiment D** — Phase 1 + Phase 3 (no Phase 2), same config as Exp C otherwise.
- GPU 0, PID 145925
- Phase 1 done, in Phase 3

**Experiment F** — Higher epsilon floor (ε: 1.0→0.20), Phase 1 only, 8 vectorized envs.
- GPU 1, PID 187664
- Hypothesis: persistent 20% exploration keeps diverse transitions flowing

### Next Steps

1. **Evaluate Exp D and F** when they finish
2. If higher ε floor helps: try even higher (0.30) or longer training
3. **Increase sf_dim** (256 or 512) for more expressive SF features
4. If all else fails: nonlinear Q head Q(s,a) = f(φ(s,a)) instead of φᵀw

### Key Hyperparameters (configs.py → NEURAL_POINTMAZE)
- sf_dim: 128, hidden: (256, 256), γ: 0.99, lr: 3e-4
- Buffer: 500K, PER + episodic replay (5 eps) + HER (k=4)
- train_every: 4 (skip 3/4 gradient steps for speed)

## Vectorized Phase 1 (Removed)

Attempted vectorized Phase 1 using `gymnasium.vector.AsyncVectorEnv` with N parallel
MuJoCo environments. Removed because GPU training — not CPU stepping — is the actual
bottleneck. With `train_every=4`, more envs just increase GPU load proportionally,
negating CPU-side gains. No wall-clock speedup observed (Exp F: 8 envs ≈ same time as
Exp D: 1 env). Code preserved in git history (commit 81008bb) for reference.

## HalfCheetah — Completed ✓

**Result: Neural SF works well on HalfCheetah.**
- Avg reward: ~6000 at 5000 episodes (5 runs)
- Clean learning curves, consistent across seeds
- Checkpoints/video/figures in `data/neural_half_cheetah/` and `figures/eval/neural_half_cheetah/`

## Acrobot — Parked

**Result: SR/SF framework struggles with Acrobot.**
- Even tabular SR fails (1000-step timeouts)
- The real issues are likely optimization difficulty and long credit assignment, not representation capacity
- Only Q-learning gets somewhat close (min 586 steps)
- Would need investigation into exploration strategy, not just architecture changes

## Environment Setup

- **Conda env**: `sai` (Python 3.10, PyTorch 2.10, CUDA 12.8, gymnasium 1.2.3)
- **Activate**: `conda activate sai` or `conda run -n sai <command>`
- **2x NVIDIA RTX 4090** (24GB each), shared machine
- **CPU**: AMD Threadripper PRO 5955WX (16 cores / 32 threads), 128GB RAM

## Branch
- `neural-successor-representation` (off main)

## Useful Commands
```bash
# Run PointMaze
CUDA_VISIBLE_DEVICES=0 conda run -n sai python examples/run_neural_point_maze.py \
    --device cuda --save-dir data/neural_point_maze_exp_X

# Run with experiment flags
conda run -n sai python examples/run_neural_point_maze.py \
    --device cuda --no-freeze-phase3 --no-staging \
    --save-dir data/neural_point_maze_exp_X

# Evaluate only
conda run -n sai python examples/run_neural_point_maze.py \
    --eval-only --checkpoint data/neural_point_maze_exp_X/checkpoint.pt --device cuda

# Visualize Q-values (compares all checkpoints)
conda run -n sai python scripts/visualize_qvalues.py

# Check GPU usage
nvidia-smi

# Check running experiments
ps aux | grep run_exp | grep python | grep -v grep
```
