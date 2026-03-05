import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from utils import *
from env import *
from plot_hierarchy import *
from matplotlib.cm import get_cmap
from copy import deepcopy
import os
import argparse
import json


from hierarchy import SR_Agent_Hierarchy, SR_performance_Hierarchy
from flat import SR_Agent_Flat, SR_performance_Flat


def SR_distances(args, GOALS):
    SR_dists = []
    SR_dists2 = []
    grid_size = args.grid_size
    
    print('Comparing High and Low for different goal distances')
    for n in range(args.n_runs):
        print(f'Run: {n+1}')
        SR_dist = []
        SR_dist2 = []
        num_episodes = 1500#args.episodes[-1]

        # SR agent
        inverse_factor = 1
        if grid_size > 7:
            inverse_factor = 4

        for goal_loc in GOALS:
            print(f'Goal Loc = {goal_loc}')
            grid_env = SR_Gridworld(grid_size)         
            grid_env.set_walls(args.walls)

            grid_env2 = SR_Gridworld(grid_size)         
            grid_env2.set_walls(args.walls)
            
            agent = SR_Agent_Hierarchy(grid_env, inverse_factor=inverse_factor,n_clust = args.n_macro, 
                                  init_loc = args.init_loc, goal_loc = goal_loc, 
                                  goal_val=args.goal_val, learn_likelikood=True)    
            agent2 = SR_Agent_Flat(grid_env2, inverse_factor=inverse_factor, 
                                  init_loc = args.init_loc, goal_loc = goal_loc, 
                                  goal_val=args.goal_val, learn_likelikood=True)
            
            print('\nHierarchy')
            agent.learn_env_likelikood(num_episodes)
            N_steps, total_r,total_time, end_state = agent.run_episode()
            print('\nFlat')
            agent2.learn_env_likelikood(num_episodes)
            N_steps2, total_r2,total_time2, end_state2 = agent2.run_episode()


            SR_dist.append(deepcopy(N_steps))
            SR_dist2.append(deepcopy(N_steps2))

        SR_dists.append(np.array(SR_dist))
        SR_dists2.append(np.array(SR_dist2))

    SR_dists = np.array(SR_dists)
    SR_dists2 = np.array(SR_dists2) 
    return SR_dists, SR_dists2


def SR_values_old(args):
    SR_vals = []
    SR_vals2 = []
    
    print('Comparing Successor matrices during training')
    
    inverse_factor = 1
    grid_size = args.grid_size
    if grid_size > 7:
        inverse_factor = 4
    print('\nTest run for true values\n')
    grid_env = SR_Gridworld(grid_size)         
    grid_env.set_walls(args.walls)
    true_num_episodes = args.episodes[-1]*2
    agent = SR_Agent_Hierarchy(grid_env, inverse_factor=inverse_factor,n_clust = args.n_macro, 
                                  init_loc = args.init_loc, goal_loc = args.goal_loc, goal_val=args.goal_val,
                                  num_episodes=true_num_episodes, learn_likelikood=True) 
    true_V = agent.M @ agent.C
    true_M = agent.M
    true_M_macro = agent.M_macro
    print(true_M_macro)
    exit()
    n_trials = len(args.episodes)
    SR_vals = np.zeros((args.n_runs, n_trials))
    SR_vals2 = np.zeros((args.n_runs, n_trials))
    
    SR_succ = np.zeros((args.n_runs, n_trials))
    SR_succ2 = np.zeros((args.n_runs, n_trials))
    
    SR_succ_macro = np.zeros((args.n_runs, n_trials))
    
    for n in range(args.n_runs):
        print('x'*20)
        print(f'Run: {n+1}')
        print('x'*20)

        for trial in range(n_trials):
            num_episodes = args.episodes[trial]
            print()
            print('+'*25)
            print(f'{num_episodes} training episodes')
            print('+'*25)
            print()
            grid_env = SR_Gridworld(grid_size)         
            grid_env.set_walls(args.walls)
            
            grid_env2 = SR_Gridworld(grid_size)         
            grid_env2.set_walls(args.walls)
            
            agent = agent = SR_Agent_Hierarchy(grid_env, inverse_factor=inverse_factor,n_clust = args.n_macro, 
                                  init_loc = args.init_loc, goal_loc = args.goal_loc, goal_val=args.goal_val,
                                  num_episodes=num_episodes, learn_likelikood=True)    
            agent2 = SR_Agent_Flat(grid_env, inverse_factor=inverse_factor, 
                                  init_loc = args.init_loc, goal_loc = args.goal_loc, goal_val=args.goal_val,
                                  num_episodes=num_episodes, learn_likelikood=True)

            
            V = agent.M @ agent.C
            V2 = agent2.M @ agent2.C
            print('\nHierarchy')
            N_steps, total_r,total_time, end_state = agent.run_episode()
            print('\nFlat')
            N_steps2, total_r2,total_time2, end_state2 = agent2.run_episode()
            
            SR_vals[n,trial] = deepcopy(np.linalg.norm(V-true_V))
            SR_vals2[n, trial] = deepcopy(np.linalg.norm(V2-true_V))
            
            SR_succ[n, trial] = deepcopy(np.linalg.norm(agent.M-true_M)/np.linalg.norm(true_M))
            SR_succ2[n, trial] = deepcopy(np.linalg.norm(agent2.M-true_M)/np.linalg.norm(true_M))
            SR_succ_macro[n, trial] = deepcopy(np.linalg.norm(agent.M_macro-true_M_macro)/np.linalg.norm(true_M_macro))
                   

    return SR_vals, SR_vals2, SR_succ, SR_succ2, SR_succ_macro

def SR_values(args):
    SR_vals = []
    SR_vals2 = []
    print('Comparing Successor matrices during training')
    
    grid_size = args.grid_size

    inverse_factor = 1
    if grid_size > 7:
        inverse_factor = 4
   
    n_trials = len(args.episodes)
    SR_vals = np.zeros((args.n_runs, n_trials))
    SR_vals2 = np.zeros((args.n_runs, n_trials))
    
    SR_succ = np.zeros((args.n_runs, n_trials))
    SR_succ2 = np.zeros((args.n_runs, n_trials))
    
    SR_succ_macro = np.zeros((args.n_runs, n_trials))
    
    for n in range(args.n_runs):
        print('x'*20)
        print(f'Run: {n+1}')
        print('x'*20)

        V_store = []
        V2_store = []
        M_store = []
        M2_store = []
        M_macro_store = []

        grid_env = SR_Gridworld(grid_size)         
        grid_env.set_walls(args.walls)
        
        agent = SR_Agent_Hierarchy(grid_env, inverse_factor=inverse_factor,n_clust = args.n_macro, 
                                    init_loc = args.init_loc, goal_loc = args.goal_loc, goal_val=args.goal_val,
                                    learn_likelikood=True) 

        grid_env2 = SR_Gridworld(grid_size)         
        grid_env2.set_walls(args.walls)

        agent2 = SR_Agent_Flat(grid_env, inverse_factor=inverse_factor, 
                                    init_loc = args.init_loc, goal_loc = args.goal_loc, goal_val=args.goal_val,
                                    learn_likelikood=True)

        for trial in range(n_trials):
            if trial == 0:
                num_episodes = args.episodes[trial]
            else:
                num_episodes = args.episodes[trial] - args.episodes[trial-1]
            print()
            print('+'*25)
            print(f'{args.episodes[trial]} training episodes')
            print('+'*25)
            print()

            agent.learn_env_likelikood(num_episodes)
            V = agent.M @ agent.C
            V_store.append(deepcopy(V))
            M_store.append(deepcopy(agent.M))
            M_macro_store.append(deepcopy(agent.M_macro))
            print('\nHierarchy')
            N_steps, total_r,total_time,_ = agent.run_episode()

            agent2.learn_env_likelikood(num_episodes)
            V2 = agent2.M @ agent2.C
            V2_store.append(deepcopy(V2))
            M2_store.append(deepcopy(agent2.M))

            print('\nFlat')
            N_steps2, total_r2,total_time2, _ = agent2.run_episode()

        agent.learn_env_likelikood()
        true_V = agent.M @ agent.C
        true_M = agent.M
        true_M_macro = agent.M_macro
        agent2.learn_env_likelikood()
        true_V2 = agent2.M @ agent2.C
        true_M2 = agent2.M
  
        for trial in range(n_trials):   
            SR_vals[n,trial] = deepcopy(np.linalg.norm(V_store[trial]-true_V))
            SR_vals2[n, trial] = deepcopy(np.linalg.norm(V2_store[trial]-true_V2))
            
            SR_succ[n, trial] = deepcopy(np.linalg.norm(M_store[trial]-true_M)/np.linalg.norm(true_M))
            SR_succ2[n, trial] = deepcopy(np.linalg.norm(M2_store[trial]-true_M2)/np.linalg.norm(true_M2))
            SR_succ_macro[n, trial] = deepcopy(np.linalg.norm(M_macro_store[trial]-true_M_macro)/np.linalg.norm(true_M_macro))
            

    return SR_vals, SR_vals2, SR_succ, SR_succ2, SR_succ_macro

    
def SR_rewards_values(args):
    SR_vals = []
    SR_vals2 = []
    SR_rewards = []
    print('Comparing Successor matrices during training')
    
    grid_size = args.grid_size
    
    inverse_factor = 1
    if grid_size > 7:
        inverse_factor = 4

    
    n_trials = len(args.episodes)
    SR_vals = np.zeros((args.n_runs, n_trials))
    SR_vals2 = np.zeros((args.n_runs, n_trials))
    
    SR_succ = np.zeros((args.n_runs, n_trials))
    SR_succ2 = np.zeros((args.n_runs, n_trials))
    
    SR_succ_macro = np.zeros((args.n_runs, n_trials))

    SR_rewards = np.zeros((args.n_runs, n_trials))
    SR_rewards2 = np.zeros((args.n_runs, n_trials))
    
    for n in range(args.n_runs):
        print('x'*20)
        print(f'Run: {n+1}')
        print('x'*20)

        V_store = []
        V2_store = []
        M_store = []
        M2_store = []
        M_macro_store = []

        grid_env = SR_Gridworld(grid_size)         
        grid_env.set_walls(args.walls)

        while True:
            try:
                agent = SR_Agent_Hierarchy(grid_env, inverse_factor=inverse_factor,n_clust = args.n_macro, 
                                            init_loc = args.init_loc, goal_loc = args.goal_loc, goal_val=args.goal_val,
                                            learn_likelikood=True) 

                grid_env2 = SR_Gridworld(grid_size)         
                grid_env2.set_walls(args.walls)

                agent2 = SR_Agent_Flat(grid_env, inverse_factor=inverse_factor, 
                                            init_loc = args.init_loc, goal_loc = args.goal_loc, goal_val=args.goal_val,
                                            learn_likelikood=True)

                for trial in range(n_trials):
                    if trial == 0:
                        num_episodes = args.episodes[trial]
                    else:
                        num_episodes = args.episodes[trial] - args.episodes[trial-1]
                    print()
                    print('+'*25)
                    print(f'{args.episodes[trial]} training episodes')
                    print('+'*25)
                    print()

                    agent.learn_env_likelikood(num_episodes)
                    V = agent.M @ agent.C
                    V_store.append(deepcopy(V))
                    M_store.append(deepcopy(agent.M))
                    M_macro_store.append(deepcopy(agent.M_macro))
                    print('\nHierarchy')
                    N_steps, total_r,total_time,_ = agent.run_episode()

                    agent2.learn_env_likelikood(num_episodes)
                    V2 = agent2.M @ agent2.C
                    V2_store.append(deepcopy(V2))
                    M2_store.append(deepcopy(agent2.M))

                    print('\nFlat')
                    N_steps2, total_r2,total_time2, _ = agent2.run_episode()

                    SR_rewards[n,trial] = deepcopy(total_r)
                    SR_rewards2[n,trial] = deepcopy(total_r2)
            except np.linalg.LinAlgError:
                continue
            else:
                break
        agent.learn_env_likelikood()
        true_V = agent.M @ agent.C
        true_M = agent.M
        true_M_macro = agent.M_macro
        agent2.learn_env_likelikood()
        true_V2 = agent2.M @ agent2.C
        true_M2 = agent2.M
  
        for trial in range(n_trials):   
            SR_vals[n,trial] = deepcopy(np.linalg.norm(V_store[trial]-true_V))
            SR_vals2[n, trial] = deepcopy(np.linalg.norm(V2_store[trial]-true_V2))
            
            SR_succ[n, trial] = deepcopy(np.linalg.norm(M_store[trial]-true_M)/np.linalg.norm(true_M))
            SR_succ2[n, trial] = deepcopy(np.linalg.norm(M2_store[trial]-true_M2)/np.linalg.norm(true_M2))
            SR_succ_macro[n, trial] = deepcopy(np.linalg.norm(M_macro_store[trial]-true_M_macro)/np.linalg.norm(true_M_macro))
            

    return SR_vals, SR_vals2, SR_succ, SR_succ2, SR_succ_macro, SR_rewards, SR_rewards2
if __name__ == '__main__':
    grid_size = 9
    n_macro = 4
    init_loc = (0,0)
    goal_loc = (grid_size-1,grid_size-1)
    nruns = 2
    eps = [50,100]#[50, 100, 200, 300, 400, 500, 750, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000] 
    GOALS = [(0,3),(0,6), (2,4), (3,0), (4,4), (5,8), (6,3), (7,0),(8,4), (8,8)]
    goal_val = 100
    train = True
    save_data = True
    WALLS = [(1,x) for x in range(grid_size//2+2)] + \
            [(3,x) for x in range(grid_size//2-2,grid_size)] +\
            [(5,x) for x in range(grid_size//2+2)] + \
            [(7,x) for x in range(grid_size//2-2,grid_size)]
            
    # WALLS = [(1,x) for x in range(grid_size-1)] + \
    #         [(4,x) for x in range(1,grid_size)]
    
    parser = argparse.ArgumentParser(description="Arguments for hierarchical model")
    parser.add_argument("grid_size", nargs='?',type=int, default=grid_size, help="Grid size of environment")
    parser.add_argument("n_macro", nargs='?',type=int, default=n_macro, help="Number of macro states")
    parser.add_argument("init_loc", nargs='?',type=int, default=init_loc, help="Initial location of agent")
    parser.add_argument("goal_loc", nargs='?',type=int, default=goal_loc, help="Goal location of agent")
    parser.add_argument("goal_val", nargs='?',type=int, default=goal_val, help="Goal value of agent")
    parser.add_argument("n_runs", nargs='?', type=int, default=nruns, help="Number of runs for the experiment")
    parser.add_argument("walls", nargs='?', type=list, default=WALLS, help="Range of walls")
    parser.add_argument("episodes", nargs='?', type=tuple, default=eps, help="Range of number of episodes")
    parser.add_argument("train", nargs='?', type=bool, default=train, help="Train mode")
    parser.add_argument("save_data", nargs='?', type=bool, default=save_data, help="Save data")
    args = parser.parse_args()


    if train:
        if save_data:
            if not os.path.exists("data/"):
                os.makedirs("data/")
            with open("data/args.json", "w") as f:
                json.dump(vars(args), f)

        # print('Hierarchical Model')
        # print('*'*50)
        # SR_steps, SR_rewards, SR_times= SR_performance_Hierarchy(args)
        # print('*'*50)
        
        # print('Flat Model')
        # print('*'*50)
        # SR_steps2, SR_rewards2, SR_times2= SR_performance_Flat(args)
        # print('*'*50)
        
        # if save_data:
            # np.save("data/SR_steps_hierarchy.npy", SR_steps)
            # np.save("data/SR_rewards_hierarchy.npy", SR_rewards)
            # np.save("data/SR_times_hierarchy.npy", SR_times)
            
            # np.save("data/SR_steps_flat.npy", SR_steps2)
            # np.save("data/SR_rewards_flat.npy", SR_rewards2)
            # np.save("data/SR_times_flat.npy", SR_times2)
        
        
        # Getting both reward and succesor matrices
        SR_vals, SR_vals2, SR_succ, SR_succ2, SR_succ_macro, SR_rewards, SR_rewards2 = SR_rewards_values(args)
        if save_data:
            np.save("data/SR_values_hierarchy.npy", SR_vals)
            np.save("data/SR_values_flat.npy", SR_vals2)
            np.save("data/SR_succ_hierarchy.npy", SR_succ)
            np.save("data/SR_succ_flat.npy", SR_succ2) 
            np.save("data/SR_succ_macro.npy", SR_succ_macro)
            np.save("data/SR_rewards_hierarchy.npy", SR_rewards)
            np.save("data/SR_rewards_flat.npy", SR_rewards2)

        # SR_dists, SR_dists2 = SR_distances(args, GOALS)
        # if save_data:
        #     np.save("data/SR_dists_hierarchy.npy", SR_dists)
        #     np.save("data/SR_dists_flat.npy", SR_dists2)

    
    # print(SR_vals, SR_vals2)
    # print()
    # print(SR_succ, SR_succ2)
    else:
        with open("data/args.json", "r") as f:
            args = json.load(f)
            args = argparse.Namespace(**args)
            print(args)
    
    # plot_SR_times(args)
    # plot_SR_steps(args)
    plot_SR_rewards(args)
    plot_SR_values(args)
    # plot_SR_distances(args, GOALS)