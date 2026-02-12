import numpy as np
from utils import *
from env import *
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-poster')

def plot_SR_times(args):
    SR_times = np.load("data/SR_times_hierarchy.npy")
    mean_SR_times = np.mean(SR_times, axis=0)
    std_SR_times = np.std(SR_times, axis=0) / np.sqrt(len(SR_times))
    
    SR_times2 = np.load("data/SR_times_flat.npy")
    mean_SR_times2 = np.mean(SR_times2, axis=0)
    std_SR_times2 = np.std(SR_times2, axis=0) / np.sqrt(len(SR_times2))
    eps_range = args.episodes
    fig = plt.figure(figsize=(14,10))
    plt.plot(eps_range,mean_SR_times, label="Hierarchy")
    plt.fill_between(eps_range, mean_SR_times - std_SR_times, mean_SR_times + std_SR_times, alpha=0.5)
    plt.plot(eps_range,mean_SR_times2, label="Flat")
    plt.fill_between(eps_range, mean_SR_times2 - std_SR_times2, mean_SR_times2 + std_SR_times2, alpha=0.5)
    #plt.title("Processing Time for different number of training episodes",fontsize=30)
    plt.xlabel("Training Episodes", fontsize=28)
    plt.ylabel("Processing Time (s)", fontsize=28)
    plt.legend(fontsize=40,loc="upper left")
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig("figures/eval/processing_times.png", format="png")
    plt.close()

def plot_SR_steps(args):
    SR_steps = np.load("data/SR_steps_hierarchy.npy")
    SR_steps2 = np.load("data/SR_steps_flat.npy")
    mean_SR_steps = np.mean(SR_steps, axis=0)
    mean_SR_steps2 = np.mean(SR_steps2, axis=0)
    std_SR_steps = np.std(SR_steps, axis=0) / np.sqrt(len(SR_steps))
    std_SR_steps2 = np.std(SR_steps2, axis=0) / np.sqrt(len(SR_steps2))
    eps_range = args.episodes
    fig = plt.figure(figsize=(14,10))
    plt.plot(eps_range,mean_SR_steps,label="Hierarchy")
    plt.fill_between(eps_range, mean_SR_steps - std_SR_steps, mean_SR_steps + std_SR_steps, alpha=0.5)
    plt.plot(eps_range,mean_SR_steps2,label="Flat")
    plt.fill_between(eps_range, mean_SR_steps2 - std_SR_steps2, mean_SR_steps2 + std_SR_steps2, alpha=0.5)

    plt.title("Number of Steps to Success",fontsize=30)
    plt.xlabel("Training Episodes", fontsize=28)
    plt.ylabel("Mean Steps", fontsize=28)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig("figures/eval/num_steps.png", format="png")
    plt.close()

    print(SR_steps,SR_steps2)


def plot_SR_rewards(args):
    eps_range = args.episodes
    SR_rewards = np.load("data/SR_rewards_hierarchy.npy")[:,:len(eps_range)]
    mean_SR_rewards = np.mean(SR_rewards, axis=0)
    std_SR_rewards = np.std(SR_rewards, axis=0) / np.sqrt(len(SR_rewards))
    
    SR_rewards2 = np.load("data/SR_rewards_flat.npy")[:,:len(eps_range)]
    mean_SR_rewards2 = np.mean(SR_rewards2, axis=0)
    std_SR_rewards2 = np.std(SR_rewards2, axis=0) / np.sqrt(len(SR_rewards2))

    print(SR_rewards)
    print(SR_rewards2)
    
    
    fig = plt.figure(figsize=(14,10))
    plt.plot(eps_range,mean_SR_rewards, label="Hierarchy")
    plt.fill_between(eps_range, mean_SR_rewards - std_SR_rewards, mean_SR_rewards + std_SR_rewards, alpha=0.5)
    plt.plot(eps_range,mean_SR_rewards2,label="Flat")
    plt.fill_between(eps_range, mean_SR_rewards2 - std_SR_rewards2, mean_SR_rewards2 + std_SR_rewards2, alpha=0.5)

    # plt.title("Reward Obtained for different number of training episodes",fontsize=30)
    plt.xlabel("Number of Training Episodes", fontsize=28)
    plt.ylabel("Total Reward", fontsize=28)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig("figures/eval/reward_obtained.png", format="png")
    plt.close()
    
    
    
def plot_SR_distances(args, GOALS):
    SR_dists = np.load("data/SR_dists_hierarchy.npy")
    SR_dists2 = np.load("data/SR_dists_flat.npy")
    mean_SR_dists = np.mean(SR_dists, axis=0)
    mean_SR_dists2 = np.mean(SR_dists2, axis=0)
    std_SR_dists = np.std(SR_dists, axis=0) / np.sqrt(len(SR_dists))
    std_SR_dists2 = np.std(SR_dists2, axis=0) / np.sqrt(len(SR_dists2))
    grid_size= args.grid_size
    env = SR_Gridworld(grid_size)
    env.set_walls(args.walls)

    xs = [get_distance(env, (0,0),g) for g in GOALS]

    fig = plt.figure(figsize=(14,10))
    plt.plot(xs,mean_SR_dists,label="Hierarchy")
    plt.fill_between(xs, mean_SR_dists - std_SR_dists, mean_SR_dists + std_SR_dists, alpha=0.5)
    plt.plot(xs,mean_SR_dists2,label="Flat")
    plt.fill_between(xs, mean_SR_dists2 - std_SR_dists2, mean_SR_dists2 + std_SR_dists2, alpha=0.5)

    # plt.title("Planning Steps with different distances from goal",fontsize=30)

    plt.xlabel("Distance From Goal", fontsize=28)
    plt.ylabel("Number of Planning Steps", fontsize=28)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig("figures/eval/num_dist.png", format="png")
    plt.close()

    print(SR_dists,SR_dists2)
    
def plot_SR_values(args):
    eps_range = args.episodes
    SR_values = np.load("data/SR_values_hierarchy.npy")[:,:len(eps_range)]
    SR_values2 = np.load("data/SR_values_flat.npy")[:,:len(eps_range)]
    SR_succ = np.load("data/SR_succ_hierarchy.npy")[:,:len(eps_range)]
    SR_succ2 = np.load("data/SR_succ_flat.npy")[:,:len(eps_range)]
    SR_succ_macro = np.load("data/SR_succ_macro.npy")[:,:len(eps_range)]
    
    mean_SR_values = np.mean(SR_values, axis=0)
    mean_SR_values2 = np.mean(SR_values2, axis=0)
    std_SR_values = np.std(SR_values, axis=0) / np.sqrt(len(SR_values))
    std_SR_values2 = np.std(SR_values2, axis=0) / np.sqrt(len(SR_values2))
    
    fig = plt.figure(figsize=(14,10))
    plt.plot(eps_range,mean_SR_values,label="Macro")
    plt.fill_between(eps_range, mean_SR_values - std_SR_values, mean_SR_values + std_SR_values, alpha=0.5)
    plt.plot(eps_range,mean_SR_values2,label="Micro")
    plt.fill_between(eps_range, mean_SR_values2 - std_SR_values2, mean_SR_values2 + std_SR_values2, alpha=0.5)

    # plt.title("Distance of successor values from true successor",fontsize=30)
    plt.xlabel("Number of Training Episodes", fontsize=28)
    plt.ylabel("Value distance", fontsize=28)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig("figures/eval/goal_values.png", format="png")
    plt.close()

    # print(SR_values,SR_values2)
    
    mean_SR_succ = np.mean(SR_succ, axis=0)
    mean_SR_succ2 = np.mean(SR_succ2, axis=0)
    mean_SR_succ_macro = np.mean(SR_succ_macro, axis=0)
    std_SR_succ = np.std(SR_succ, axis=0) / np.sqrt(len(SR_succ))
    std_SR_succ2 = np.std(SR_succ2, axis=0) / np.sqrt(len(SR_succ2))
    std_SR_succ_macro = np.std(SR_succ_macro, axis=0) / np.sqrt(len(SR_succ_macro))
    fig = plt.figure(figsize=(14,10))
    # plt.plot(eps_range,mean_SR_succ,label="Hierarchy Micro")
    # plt.fill_between(eps_range, mean_SR_succ - std_SR_succ, mean_SR_succ + std_SR_succ, alpha=0.5)
    plt.plot(eps_range,mean_SR_succ2,label="Micro")
    plt.fill_between(eps_range, mean_SR_succ2 - std_SR_succ2, mean_SR_succ2 + std_SR_succ2, alpha=0.5)
    plt.plot(eps_range,mean_SR_succ_macro,label="Macro")
    plt.fill_between(eps_range, mean_SR_succ_macro - std_SR_succ_macro, mean_SR_succ_macro + std_SR_succ_macro, alpha=0.5)

    print("SR_succ",SR_succ)
    # print("SR_succ2",SR_succ2)
    print("SR_succ_macro",SR_succ_macro)
    
    # plt.title("Distance of successor matrix from true successor",fontsize=30)
    plt.xlabel("Number of Training Episodes", fontsize=28)
    plt.ylabel("Successor distance", fontsize=28)
    plt.legend(fontsize=26)
    plt.xticks(fontsize=26)
    plt.yticks(fontsize=26)
    plt.tight_layout()
    plt.savefig("figures/eval/successor_values.png", format="png")
    plt.close()