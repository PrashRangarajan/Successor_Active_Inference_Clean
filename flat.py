
'''
Successor+Active inference, no hierarchy
'''
import numpy as np
from utils import *
from env import *
import matplotlib.pyplot as plt

import matplotlib.animation as animation
from copy import deepcopy
import time
import random
import os
from matplotlib.colors import LogNorm
import argparse


class SR_Agent_Flat(object):
    def __init__(self, env, A=None,B=None,C=None, discount_factor=0.99,inverse_factor=1, goal_val=100,
                 init_loc = None, goal_loc = None, num_episodes=500, learn_likelikood = False):
        if A is None:
            self.A = env.get_likelihood_dist()
        else:
            self.A = A
        self.env = env
        self.B = B
        self.B_macro = None
        self.M = None
        
        if init_loc is None:          
            self.init_loc = (0, 0)
        else:
            self.init_loc = init_loc
            
        if goal_loc is None:
            goal_loc = (env.grid_size,env.grid_size)
            self.goal_idx = env.grid_size**2 - 1
        else:
            self.goal_idx = loc_to_idx(goal_loc,env.grid_size)
            
        if C is not None:
            self.C = C
        else:
            self.C = np.ones(env.grid_size**2) * -0.1
            self.C[self.goal_idx] = goal_val
        self.C_macro = None
        self.gamma = discount_factor
        self.inverse_factor = inverse_factor
        
        self.env.get_image(self.init_loc, goal_loc)
        self.s = None
        self.s_array = None
        self.a_array = []
        self.max_s = idx_to_onehot(np.argmax(self.C), self.env.grid_size)
        self.learn_likelikood = learn_likelikood
        self.n_actions = 4
        if num_episodes is None:
            self.num_episodes = 0 # Keep track of number of episodes
        else:
            self.num_episodes = num_episodes
        self.learning_rate = 0.1

    def view_matrices(self):
        if self.learn_likelikood:
            learn_str = 'estimated'
        else:
            learn_str = 'actual'
        B_avg = np.sum(self.B, axis=2) / self.B.shape[-1]
        plt.figure()
        plt.title("Default Policy B Matrix",fontsize=20)
        plt.imshow(B_avg)#, norm=LogNorm(vmin=B_avg.min(), vmax=B_avg.max()))
        plt.xticks(list(range(0,self.env.grid_size**2)),fontsize=12)
        plt.yticks(list(range(0, self.env.grid_size**2)),fontsize=12)
        plt.savefig(f"figures/matrices/B_matrix_flat.png", format="png")
        plt.close()

        plt.figure()
        plt.imshow(self.M, aspect='equal', cmap='cividis')#, norm=LogNorm(vmin=self.M.min(), vmax=self.M.max()))
        plt.xticks(list(range(0,self.env.grid_size**2)),fontsize=12)
        plt.yticks(list(range(0,self.env.grid_size**2)),fontsize=12)
        plt.title(f"Successor Matrix M ({learn_str})", fontsize=20)
        plt.savefig(f"figures/matrices/m_{learn_str}_flat.png", format="png")
        plt.close()
        np.save("data/M.npy", self.M)

        M_orig = self.M[0,:].reshape(self.env.grid_size,self.env.grid_size).T
        plt.figure()
        plt.imshow(M_orig, aspect='equal', cmap='copper', norm=LogNorm(vmax=M_orig.max()))
        plt.xticks(list(range(0,self.env.grid_size)),fontsize=12)
        plt.yticks(list(range(0,self.env.grid_size)),fontsize=12)
        plt.title(f"Successor Matrix M from start({learn_str})", fontsize=20)
        plt.savefig(f"figures/matrices/m_origin_{learn_str}_flat.png", format="png")
        plt.close()
        


    def update_sr(self, current_exp, next_exp, M):
        # SARSA TD learning rule
        # Modifies M
        
        grid_size =  self.env.grid_size
        s_1 = current_exp[0]
        #a_1 = current_exp[1]
        s_2 = current_exp[2]
        #a_2 = next_exp[1]
        r = current_exp[3]
        done = current_exp[4]
        I = idx_to_onehot(s_1, grid_size)
        if done:            
            td_error = (I + self.gamma * idx_to_onehot(s_2, grid_size) - M[s_1,:])
        else:
            td_error = (I + self.gamma * M[s_2, :] - M[s_1,:])
        M[s_1,:] += self.learning_rate * td_error
        return td_error


    def actual_successor_transition_matrix(self):
        env = self.env
        grid_size = self.env.grid_size

        self.B = env.get_transition_dist()
        I = np.identity(grid_size**2)
        B_avg = np.sum(self.B, axis=2) / 4
        self.M = np.linalg.inv(I - (0.95 * B_avg))
        return self.B, self.M

    def show_video(self):
        #print(self.s_array)
        gs = self.env.grid_size 

        fig = plt.figure()
        grid = np.zeros((gs, gs))
        wall_idx = tuple(np.array(self.env.walls).T) # Convert to right format
        start_idx = (0,0)
        goal_idx = idx_to_loc(self.goal_idx, gs)
        grid[start_idx] = 1
        grid[goal_idx] = 0.5
        grid[wall_idx] = 0.25
        im = plt.imshow(grid.T, aspect='equal', cmap='magma')
        plt.title(f'Gridworld of size {gs}')
        
        ax = plt.gca()
        
        # Major ticks
        ax.set_xticks(np.arange(0, gs, 1))
        ax.set_yticks(np.arange(0, gs, 1))

        # Labels for major ticks
        ax.set_xticklabels(np.arange(gs, 1))
        ax.set_yticklabels(np.arange(gs, 1))

        # Minor ticks
        ax.set_xticks(np.arange(-.5, gs, 1), minor=True)
        ax.set_yticks(np.arange(-.5, gs, 1), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

        # Remove minor ticks
        ax.tick_params(which='minor', bottom=False, left=False)

        def init():
            grid = np.zeros((gs, gs))
            grid[wall_idx] = 0.25
            grid[start_idx] = 1
            grid[goal_idx] = 0.5
            im.set_data(grid)
            return im,

        def animate(i):
            for txt in ax.texts[::-1]:
                txt.remove()
            s = self.s_array[i]
            grid = np.zeros((gs, gs))
            grid[wall_idx] = 0.25
            s_idx = idx_to_loc(onehot_to_idx(s), gs)
            grid[s_idx] = 1
            grid[goal_idx] = 0.5
            im.set_data(grid.T)

            ax.text(goal_idx[0], goal_idx[1], 'Goal', fontsize = 10, ha="center", va="center", color="w")
            ax.text(s_idx[0], s_idx[1], 'Agent', fontsize = 10, ha="center", va="center", color="b")
            for idx in self.env.walls:
                ax.text(idx[0], idx[1], 'Wall', fontsize = 10, ha="center", va="center", color="w")

            return im,
        ani = animation.FuncAnimation(fig, animate, np.arange(len(self.s_array)), init_func=init,
                              interval=500, blit=True)
        ani.save(f'figures/env.mp4')
        plt.close()
    
    def show_actions(self):

        gs = self.env.grid_size
        fig = plt.figure()
        grid = np.zeros((gs, gs))
        if len(self.env.walls) > 0:
            wall_idx = tuple(np.array(self.env.walls).T)
        start_loc = self.init_loc
        
        if isinstance(self.goal_idx, int):
            goal_loc = [idx_to_loc(self.goal_idx, gs)]
        else:   
            goal_loc = [idx_to_loc(gidx, gs) for gidx in self.goal_idx]
        grid[start_loc] = 1
        grid[wall_idx] = 2
        grid[tuple(zip(*goal_loc))] = 0.5

        arrows = {1:(1,0), 0:(-1,0),3:(0,1),2:(0,-1)}
        scale = 0.25
        arrows_grid = np.full((gs, gs),-1)
        s_idxs = [onehot_to_loc(s, gs) for s in self.s_array]
        for i in range(1,len(self.a_array)):
            arrows_grid[s_idxs[i]] = self.a_array[i]
        arrows_grid = arrows_grid.T
        
        fig, ax = plt.subplots(figsize=(gs,gs))
        for r, row in enumerate(arrows_grid):
            for c, cell in enumerate(row):
                if cell in arrows:
                    plt.arrow(c-scale*arrows[cell][0], r-scale*arrows[cell][1], scale*arrows[cell][0], scale*arrows[cell][1], head_width=0.15,color='w')

        cmap = colors.ListedColormap(['black','purple','yellow','white'])

        im = plt.imshow(grid.T, aspect='equal', cmap=cmap)
        # plt.title(f'Gridworld of size {gs}')
        ax = plt.gca()

        for goal in goal_loc:
                ax.text(goal[0], goal[1], 'Goal', fontsize = 18, ha="center", va="center", color="w")
        ax.text(start_loc[0], start_loc[1], 'Agent', fontsize = 18, ha="center", va="center", color="b")
        # for widx in self.env.walls:
        #     ax.text(widx[0], widx[1], 'Wall', fontsize = 10, ha="center", va="center", color="w")

        # Major ticks
        ax.set_xticks(np.arange(0, gs, 1))
        ax.set_yticks(np.arange(0, gs, 1))

        # Labels for major ticks
        ax.set_xticklabels(range(0, gs))
        ax.set_xticklabels(range(0, gs))

        # Minor ticks
        ax.set_xticks(np.arange(-.5, gs, 1), minor=True)
        ax.set_yticks(np.arange(-.5, gs, 1), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

        # Remove minor ticks
        ax.tick_params(which='minor', bottom=False, left=False)

        # plt.title(f"Actions taken", fontsize=20)
        plt.savefig(f"figures/Actions_taken_flat.png", format="png")
        plt.close()
         
        
    def env_reset(self,init=None, verbose=False):
        self.s = self.env.reset(init)
        self.s_array = [self.s]
        if verbose:
            print("Init S: ", self.env.get_state_loc())
            #print("max_s: ", self.max_s)
            print('walls',self.env.walls)
        return self.s
    
    def learn_successor_transition_matrix(self, num_episodes):
        env = self.env
        grid_size = self.env.grid_size
        N = grid_size**2

        if self.M is None:
            self.B = np.zeros((N,N,self.n_actions))
            self.M = np.zeros((N,N))
        

        max_s = np.argmax(self.C)
        # num_episodes = self.num_episodes
        train_episode_length = 40
        experiences = []
        lifetime_td_errors = []

        for ep in range(num_episodes):
            s = onehot_to_idx(self.env_reset())
            episodic_error = []
            for j in range(train_episode_length):
                action = random.randrange(self.n_actions)
                s_next = onehot_to_idx(env.step(action))
                done = False #(s == max_s)      
                self.B[s_next, s, action] += 1   # Update B  
                experiences.append([s, action, s_next, self.C[s_next], done])
                # print(experiences)
                s = s_next
                if (j > 1):
                    td_sr = self.update_sr(experiences[-2], experiences[-1],self.M)
                    episodic_error.append(np.mean(np.abs(td_sr)))
                if done:
                    td_sr = self.update_sr(experiences[-1], experiences[-1],self.M)
                    episodic_error.append(np.mean(np.abs(td_sr)))
                    break
            lifetime_td_errors.append(np.mean(episodic_error))

        # Handling transitions not encountered by manually adding a P(s_t+1=a|s_t=a)=1
        for col in range(N):
            for action in range(self.n_actions):
                col_sum = np.sum(self.B[:,col,action])
                if col_sum == 0:
                    self.B[col,col,action] = 1
        # Normalize matrix
        self.B = self.B/self.B.sum(axis=0, keepdims=True)
        self.B[max_s, max_s, :] = 1
        return self.B, self.M
          

    def learn_env_likelikood(self, num_episodes=None):
        if num_episodes is None:
            num_episodes = 500
        self.num_episodes += num_episodes
        if self.learn_likelikood:
            self.B, self.M = self.learn_successor_transition_matrix(num_episodes)
            
        else:
            self.B, self.M = self.actual_successor_transition_matrix()

    
    def run_episode(self, time_episode = True):       
        if time_episode == True:
            start_time = time.time()
        grid_size = self.env.grid_size
        # self.view_matrices()

        self.env_reset(init=loc_to_idx(self.init_loc, grid_size) , verbose=True)

        # run the episode until discovered the reward
        N_steps = 0
        total_r = 0
        
        
        goal = idx_to_onehot(self.goal_idx, grid_size)
        V = self.M @ self.C
        print('Current state',self.env.get_state_loc())
        
        self.s = self.env.get_state()
        
        while (self.env.get_state_idx() != self.goal_idx):
            V_adj = []
            for act in range(self.n_actions):
                s_adj = self.B[:,:,act] @ self.env.get_state()
                V_adj.append(V[onehot_to_idx(s_adj)])         
            ba_gen = (ele for ele in np.argsort(V_adj)[::-1])
            
            while True:
                try:
                    best_action = next(ba_gen)
                except StopIteration:
                    best_action = random.randrange(self.n_actions)
                    
                s_new = onehot_to_idx(self.env.step(best_action))
                if s_new != onehot_to_idx(self.s):
                    break
                
            self.s = idx_to_onehot(s_new, grid_size)
            self.s_array.append(self.s)
            self.a_array.append(best_action)
            print('s:',self.env.get_state_loc())
            r = self.C[s_new]
            total_r += r

            N_steps +=1
            if N_steps > 5*self.env.grid_size:
                break

        end_state = self.env.get_state_loc()
        print('End state', end_state)
        
        self.show_video()
        if time_episode == True:
            total_time = time.time() - start_time
            return N_steps, total_r, total_time, end_state
        return N_steps, total_r, 0, end_state
    
    
def SR_performance_Flat(args):
    SR_steps = []
    SR_rewards = []
    SR_times = []

    grid_size = args.grid_size
 

    for n in range(args.n_runs):
        print('x'*20)
        print(f"Run {n}")
        print('x'*20)
        SR_step = []
        SR_reward = []
        SR_time = []

        grid_env = SR_Gridworld(grid_size)         
        grid_env.set_walls(args.walls)
        walls = grid_env.get_walls()
        print('walls',walls)

        # SR agent
        inverse_factor = 1
        if grid_size > 7:
            inverse_factor = 4
        agent = SR_Agent_Flat(grid_env, inverse_factor=inverse_factor, goal_val=args.goal_val,
                                init_loc = args.init_loc, goal_loc = args.goal_loc, 
                                learn_likelikood=True)                


        for num in range(len(args.episodes)):
            if num == 0:
                num_episodes = args.episodes[num]
            else:
                num_episodes = args.episodes[num] - args.episodes[num-1]
            print()
            print('+'*25)
            print(f'{args.episodes[num]} training episodes')
            print('+'*25)
            print()
            
            agent.learn_env_likelikood(num_episodes)
            N_steps, total_r,total_time,_ = agent.run_episode()
            # agent.show_video()
            # agent.show_actions()
            
            SR_step.append(deepcopy(N_steps))
            SR_reward.append(deepcopy(total_r))
            SR_time.append(deepcopy(total_time))
            


        SR_step = np.array(SR_step)
        SR_reward= np.array(SR_reward)
        SR_time = np.array(SR_time)

        SR_steps.append(SR_step)
        SR_rewards.append(SR_reward)
        SR_times.append(SR_time)

    SR_steps = np.array(SR_steps)
    SR_rewards= np.array(SR_rewards)
    SR_times = np.array(SR_times)
    
    return SR_steps, SR_rewards, SR_times

    
if __name__ == '__main__':
    grid_size = 9
    init_loc = (0,0)
    goal_loc = (grid_size-1,grid_size-1)
    nruns = 1
    eps = [200,8000]
    goal_val = 100
    WALLS = [(1,x) for x in range(grid_size//2+2)] + \
            [(3,x) for x in range(grid_size//2-2,grid_size)] +\
            [(5,x) for x in range(grid_size//2+2)] + \
            [(7,x) for x in range(grid_size//2-2,grid_size)]
    
    parser = argparse.ArgumentParser(description="Arguments for hierarchical model")
    parser.add_argument("grid_size", nargs='?',type=int, default=grid_size, help="Grid size") 
    parser.add_argument("init_loc", nargs='?',type=int, default=init_loc, help="Initial location of agent")
    parser.add_argument("goal_loc", nargs='?',type=int, default=goal_loc, help="Goal location of agent")
    parser.add_argument("goal_val", nargs='?',type=int, default=goal_val, help="Goal value of agent")
    parser.add_argument("n_runs", nargs='?', type=int, default=nruns, help="Range of number of runs")
    parser.add_argument("walls", nargs='?', type=list, default=WALLS, help="Range of walls")
    parser.add_argument("episodes", nargs='?', type=tuple, default=eps, help="Range of number of episodes")
    args = parser.parse_args()
    
    print('Flat Model')
    print('*'*50)
    SR_steps, SR_rewards, SR_times= SR_performance_Flat(args)
    print('*'*50)
    
    
    if not os.path.exists("data/"):
        os.makedirs("data/")
        
    np.save("data/SR_steps.npy", SR_steps)
    np.save("data/SR_rewards.npy", SR_rewards)
    np.save("data/SR_times.npy", SR_times)
