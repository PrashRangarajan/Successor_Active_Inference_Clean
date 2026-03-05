"""Run the legacy experiment_hierarchy_flat.py with serpentine config."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-poster')

from experiment_hierarchy_flat import SR_rewards_values

grid_size = 9

# Serpentine walls (from the legacy __main__ block)
WALLS = [(1, x) for x in range(grid_size // 2 + 2)] + \
        [(3, x) for x in range(grid_size // 2 - 2, grid_size)] + \
        [(5, x) for x in range(grid_size // 2 + 2)] + \
        [(7, x) for x in range(grid_size // 2 - 2, grid_size)]

n_macro = 4
init_loc = (0, 0)
goal_loc = (grid_size - 1, grid_size - 1)
goal_val = 100
n_runs = 5
# Full checkpoints from the commented-out line in legacy code
episodes = [50, 100, 200, 300, 400, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]

args = argparse.Namespace(
    grid_size=grid_size,
    n_macro=n_macro,
    init_loc=init_loc,
    goal_loc=goal_loc,
    goal_val=goal_val,
    n_runs=n_runs,
    walls=WALLS,
    episodes=episodes,
    train=True,
    save_data=True,
)

os.makedirs("data", exist_ok=True)
os.makedirs("figures/eval", exist_ok=True)
os.makedirs("figures/clustering", exist_ok=True)
os.makedirs("figures/matrices", exist_ok=True)
os.makedirs("figures/Macro Action Network", exist_ok=True)

print(f"Running legacy SR_rewards_values with {n_runs} runs")
print(f"Serpentine {grid_size}x{grid_size}, goal={goal_loc}, init={init_loc}")
print(f"Episodes: {episodes}")
print(f"Walls: {WALLS}")

SR_vals, SR_vals2, SR_succ, SR_succ2, SR_succ_macro, SR_rewards, SR_rewards2 = SR_rewards_values(args)

np.save("data/SR_values_hierarchy.npy", SR_vals)
np.save("data/SR_values_flat.npy", SR_vals2)
np.save("data/SR_succ_hierarchy.npy", SR_succ)
np.save("data/SR_succ_flat.npy", SR_succ2)
np.save("data/SR_succ_macro.npy", SR_succ_macro)
np.save("data/SR_rewards_hierarchy.npy", SR_rewards)
np.save("data/SR_rewards_flat.npy", SR_rewards2)

print("\nData saved. Now plotting...")

# Plot rewards
eps_range = episodes
mean_SR_rewards = np.mean(SR_rewards, axis=0)
std_SR_rewards = np.std(SR_rewards, axis=0) / np.sqrt(n_runs)
mean_SR_rewards2 = np.mean(SR_rewards2, axis=0)
std_SR_rewards2 = np.std(SR_rewards2, axis=0) / np.sqrt(n_runs)

fig, ax = plt.subplots(figsize=(14, 10))
ax.plot(eps_range, mean_SR_rewards, label="Hierarchy", linewidth=2)
ax.fill_between(eps_range, mean_SR_rewards - std_SR_rewards,
                mean_SR_rewards + std_SR_rewards, alpha=0.3)
ax.plot(eps_range, mean_SR_rewards2, label="Flat", linewidth=2)
ax.fill_between(eps_range, mean_SR_rewards2 - std_SR_rewards2,
                mean_SR_rewards2 + std_SR_rewards2, alpha=0.3)
ax.set_xlabel("Number of Training Episodes", fontsize=28)
ax.set_ylabel("Total Reward", fontsize=28)
ax.legend(fontsize=26)
ax.tick_params(labelsize=22)
plt.tight_layout()
save_path = "figures/eval/legacy_hier_vs_flat_serpentine.png"
plt.savefig(save_path, format="png", dpi=150)
plt.close()
print(f"Plot saved to {save_path}")

# Also print summary table
print("\n" + "="*60)
print(f"{'Episodes':>10} | {'Hier Reward':>14} | {'Flat Reward':>14}")
print("-"*60)
for i, ep in enumerate(eps_range):
    print(f"{ep:>10} | {mean_SR_rewards[i]:>14.1f} | {mean_SR_rewards2[i]:>14.1f}")
print("="*60)
