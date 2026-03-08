# Implementation Plan: HER + Staged Learning + Camera Fix

## Problem
Neural SF agent fails on PointMaze (SF loss rises, rewards decline) while tabular SR succeeds.
Root causes identified from tabular vs neural comparison:
1. No HER — failed trajectories provide zero goal-directed learning signal
2. No staged learning — φ, ψ, w learned simultaneously (unstable)
3. Camera doesn't track cheetah in video rendering

## Changes Overview

### Step 1: HER for the SF Framework
**Files**: `core/neural/replay_buffer.py`, `core/neural/agent.py`

**replay_buffer.py** — Add HER relabeling storage:
- Add `add_her_transitions(episode_obs, episode_actions, episode_dones, goal_xy, reward_fn)` method
- After each episode, take the trajectory and relabel goals:
  - For each transition (s, a, s') in the episode, pick `k=4` future states as substitute goals (strategy: "future")
  - For each substitute goal g', recompute the 6D obs by replacing obs[4:6] with g'
  - Recompute reward using the reward_shaping_fn with the new goal
  - Mark as terminal if achieved position is within 0.45 of substitute goal
  - Add these synthetic transitions to the buffer
- This multiplies effective data by ~4x without extra environment steps

**agent.py** — Integrate HER into learning loop:
- Add `use_her: bool`, `her_k: int = 4`, `her_goal_strategy: str = "future"` params to __init__
- At end of each episode in `learn_environment()`, after `buffer.end_episode()`:
  - If `use_her` is enabled, collect the episode's transitions
  - Call `buffer.add_her_transitions(...)` to relabel and store
- Need to pass the reward_shaping_fn or a modified version that can recompute reward for any goal

**Key insight**: For SFs, HER is natural. The reward features ψ(s) describe the state, not the goal. Only the reward function changes with the goal. So we:
1. Keep the same ψ(s') in the SF target (state features don't depend on goal)
2. Recompute the reward for the relabeled goal
3. The relabeled transition trains the same φ network with a different reward signal

### Step 2: Staged Learning (Freeze φ, Then Learn w)
**Files**: `core/neural/agent.py`, `examples/run_neural_point_maze.py`

**agent.py** — Add staged learning mode:
- Add `freeze_reward_weights()` method: sets `requires_grad=False` on w and detaches it from optimizer
- Add `unfreeze_reward_weights()` method: re-enables gradient
- Add `freeze_sf()` method: freezes sf_net parameters
- Add `unfreeze_sf()` method: unfreezes sf_net parameters

**run_neural_point_maze.py** — Use 3-phase training:
- Phase 1 (unchanged): Diverse exploration, learn φ + ψ + w together (building SF map)
- Phase 2 (new): Freeze w, continue training φ + ψ only with more replay intensity
  - This consolidates the SF representation without chasing a moving reward signal
  - Use higher episodic replay (10 episodes) to mirror tabular's 10-epoch consolidation
- Phase 3 (current Phase 2): Unfreeze w, freeze φ, goal-focused training
  - Now only w adapts to the goal reward — much more stable

### Step 3: Episode Buffer for HER
**Files**: `core/neural/replay_buffer.py`

Add episode-level storage to support HER relabeling:
- Track `_current_episode_obs`, `_current_episode_actions`, `_current_episode_rewards`, `_current_episode_dones` as lists
- In `add()`: append to current episode lists
- In `end_episode()`: finalize and store episode for HER processing
- `get_last_episode()`: return the most recent complete episode data

### Step 4: HalfCheetah Camera Tracking Fix
**Files**: `environments/mujoco/half_cheetah.py` or `examples/run_neural_half_cheetah.py`

In `record_episode_video()`, configure the MuJoCo camera to track the cheetah torso:
- After creating the render env, access `env.unwrapped.mujoco_renderer`
- Set camera to track body ID 0 (torso) using `cam.trackbodyid` and `cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING`
- This makes the camera follow the cheetah as it runs forward

### Step 5: Config & Wiring
**Files**: `examples/configs.py`, `examples/run_neural_point_maze.py`

**configs.py** — Add HER config to NEURAL_POINTMAZE:
```python
"use_her": True,
"her_k": 4,
"her_goal_strategy": "future",
```

**run_neural_point_maze.py** — Wire HER + 3-phase training:
- Pass HER config to agent
- Implement 3-phase training schedule
- Create a goal-relabeling reward function that accepts arbitrary goal positions

### Step 6: Test & Validate
- Run quick smoke test (--quick flag) to verify no crashes
- Launch full training on GPU
