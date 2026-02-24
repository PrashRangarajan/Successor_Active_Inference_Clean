'''
Hierarhy of SR agents
The agent can go to multiple goals and must pick the best one
The environment is the scenario where the agent must navigate 4 walls
Noise is also added to the hierarchy so now the agent incorporates the noise in the lower level
into the higher level planning
'''

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
from scipy import stats
from utils import *
from env import *
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from sklearn.cluster import SpectralClustering
from sklearn.manifold import SpectralEmbedding

import matplotlib.animation as animation
import matplotlib.patches as mpatches
from copy import deepcopy
import time
import random
from matplotlib.colors import LogNorm
from collections import Counter, defaultdict
import os
import glob

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from nn import *

import argparse
import sys
np.set_printoptions(threshold=sys.maxsize)


# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


class SR_Agent_Hierarchy(object):
    def __init__(self, env, A=None,B=None,C=None, discount_factor=0.99,inverse_factor=1, n_clust = 4, state_size = None,
                 beta=1, init_loc = None, goal_locs = None, goal_vals = 100, learn_likelikood = False, use_nn = False):
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
            
        if goal_locs is None:
            goal_locs = [(env.grid_size,env.grid_size)]
            self.goal_idx = [env.grid_size**2 - 1]
        else:
            self.goal_idx = [loc_to_idx(goal,env.grid_size) for goal in goal_locs]
        if C is not None:
            self.C = C
        else:
            self.C = np.ones(env.grid_size**2) * -0.1
            self.C[self.goal_idx] = goal_vals
        
        self.C_macro = None
        self.gamma = discount_factor
        self.beta = beta
        self.inverse_factor = inverse_factor
        
        self.env.get_image(self.init_loc, goal_locs)
        self.s = None
        self.s_array = None
        self.a_array = None
        self.b = None
        self.b_idx = None
        self.o_array = None
        self.b_array = None
        self.state_size = state_size
        
        # self.max_s = idx_to_onehot(np.argmax(self.C), self.env.grid_size)
        
        self.learn_likelikood = learn_likelikood
        self.n_clust = n_clust
        self.n_actions = 4
        self.n_macro_actions = None
        self.num_episodes = 0 #Keep track of number of episodes
        self.learning_rate = 0.05 #for learning successor
        
        self.use_nn = use_nn
        if self.use_nn:
            self.train_nn()
            self.visualize_nn()

    def env_reset(self,init=None, verbose=False):
        '''
        init: idx
        Reset the environment state
        Also resets the array storing state history    
        '''
        N = self.env.grid_size**2
        
        # Excluding walls, the belief about initial location can be anywhere
        wall_idx = [loc_to_idx(wall, self.env.grid_size) for wall in self.env.get_walls()]
        allowed_idx = list(set(range(self.env.grid_size**2-1)) - set(wall_idx))
        prior_b = np.zeros(N)
        prior_b[allowed_idx] = 1/len(allowed_idx)
        
        self.s = self.env.reset(init)
        self.s_array = [self.onehot_to_idx(self.s)]
        o_new = self.env.get_obs_idx()
        self.b = infer_states(o_new, self.A, prior_b)
        self.b_idx = self.onehot_to_idx(self.b)
        self.b_array = [self.b_idx]
        self.o_array = [o_new]
        self.a_array = []
        
        if verbose:
            print('Micro State')
            print("Init State: ", self.env.get_state_loc())
            print("Init Obs: ", self.env.get_obs_idx())
            print("Init Belief: ", self.b_idx)
            print(f"Goal State(s): {[idx_to_loc(gidx, self.env.grid_size) for gidx in self.goal_idx]}")
            print('walls',self.env.walls)
        return self.b, self.b_idx
    
    def onehot_to_idx(self, b):
        #Modify utils function to convert one hot to index
        #Excluding walls, the belief about initial location can be anywhere
        walls = [loc_to_idx(wall, self.env.grid_size) for wall in self.env.get_walls()]
        b_idx_arr = [onehot_to_idx(b) for _ in range(3)]
        
        # get mode to avoid errors
        #such as agent randomly believing it is in a far away state
        b_idx = max(set(b_idx_arr), key=b_idx_arr.count) 
        
        # Extra insurance that agent doesn't spawn in a wall
        while b_idx in walls:
            b_idx = onehot_to_idx(b)
        return b_idx
    
    def actual_successor_transition_matrix(self):
        env = self.env
        grid_size = self.env.grid_size

        self.B = env.get_transition_dist()
        I = np.identity(grid_size**2)
        B_avg = np.sum(self.B, axis=2) / self.n_actions
        self.M = np.linalg.inv(I - (0.95 * B_avg))
        return self.B, self.M


    def update_sr(self, current_exp, M):
        # SARSA TD learning rule
        # Modifies M
        grid_size =  self.env.grid_size
        s_1 = current_exp[0]
        s_2 = current_exp[2]
        done = current_exp[4]
        I = idx_to_onehot(s_1, grid_size)
        if done:            
            td_error = (I + self.gamma * idx_to_onehot(s_2, grid_size) - M[s_1,:])
        else:
            td_error = (I + self.gamma * M[s_2, :] - M[s_1,:])
        M[s_1,:] += self.learning_rate * td_error
        return td_error
          
    
    def learn_successor_transition_matrix(self, num_episodes):
        env = self.env
        grid_size = self.env.grid_size
        N = grid_size**2
        
        if self.M is None:
            self.B = np.zeros((N,N,self.n_actions))
            self.M = np.zeros((N,N))
        
               
        train_episode_length = 40
        experiences = []
        lifetime_td_errors = []

        for ep in range(num_episodes):
            self.env_reset()
            s_idx = self.onehot_to_idx(self.s)  

            episodic_error = []
            for j in range(train_episode_length):
                action = random.randrange(self.n_actions)
                s_next_idx = self.onehot_to_idx(env.step(action))
                o_next_idx = self.env.get_obs_idx()
                prior_b = self.B[:,:,action] @ self.b
                self.b = infer_states(o_next_idx, self.A, prior_b)
                b_next_idx = self.onehot_to_idx(self.b)
                done = False#(s_next_idx == self.goal_idx) 
                # if j < 5: print('entropy', j, entropy(self.b))   
                # plot_beliefs(self.b, f'{self.env.get_state_loc()},{action}, {self.env.get_obs_idx()} {entropy(self.b)}')
                self.B[:, self.b_idx, action] += self.b  # Update B  
                experiences.append([self.b_idx, action, b_next_idx, self.C[s_next_idx], done])
                self.b_idx = b_next_idx
                if (j > 1):
                    td_sr = self.update_sr(experiences[-2], self.M)
                    episodic_error.append(np.mean(np.abs(td_sr)))
                if done:
                    td_sr = self.update_sr(experiences[-1], self.M)
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
        self.B[self.goal_idx, self.goal_idx, :] = 1
        print('Learning Complete')
        return self.B, self.M
    
    def learn_macro_clusters(self):
        grid_size = self.env.grid_size
        N = self.env.grid_size**2
        mask = np.full(N,True)
        walls = [loc_to_idx(wall, grid_size) for wall in self.env.get_walls()]
        mask[walls] = False
        Mf = self.M[mask,:][:,mask]
        
        mask_ind = [i for i in range(len(mask)) if mask[i]]
        
        n_clust=self.n_clust
        Mf = np.maximum(Mf, Mf.transpose()) # To fix symmetry issues

        pos = SpectralEmbedding(n_components=2, affinity='precomputed').fit_transform(Mf)
        pos_walls = np.zeros((N,2))
        pos_walls[mask] = pos

        labels = np.ones(N) * n_clust  
        sc = SpectralClustering(n_clust, affinity='precomputed', n_init=100,
                        assign_labels='discretize')
  
        labels[mask_ind] = sc.fit_predict(Mf)
        labels_grid = labels.reshape(grid_size,grid_size).T
        
        self.lg = labels_grid
        
        # Last cluster is set of walls which isn't added to state list s in the end
        counts = Counter(labels)
        R = defaultdict(list)
        for i,n in enumerate(labels):
            R[n].append(i)
        s = [R[i] for i in range(n_clust)] 
        
        inv_s = {}
        for k, seq in enumerate(s):
            for idx in seq:
                inv_s[idx] = k
        print('Clustering complete')
        
        plt.figure()
        im = plt.imshow(labels_grid,cmap='gist_heat')
        colours = im.cmap(im.norm(np.unique(labels_grid))) # Colours including walls as separate colour (white)
        plt.xticks(np.arange(grid_size),fontsize=12)
        plt.yticks(np.arange(grid_size),fontsize=12)
        plt.title(f"Macro state clusters", fontsize=20)
        patches = [ mpatches.Patch(color=colours[i], label=f'{i}' ) for i in range(len(colours)-1) ]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig(f"figures/clustering/Macro_s_noisy.png", format="png")
        plt.close()
        
        plt.figure()
        for i, lst in enumerate(s):
            plt.scatter(pos_walls[lst,0], pos_walls[lst,1], label=f'{i}',color=colours[i])
        for i, state in enumerate(np.arange(N)[mask]):
            plt.annotate(state, pos[i,:])
        plt.legend()
        plt.title('Micro states clusters')
        plt.savefig('figures/clustering/macro_state_viz_noisy.png')
        plt.close()
        return s, inv_s
     
    def learn_adjacency(self, num_episodes):
        env = self.env
        grid_size = self.env.grid_size
        
        train_episode_length = 50
        experiences = []
        lifetime_td_errors = []

        adj_list = defaultdict(set)
        bottleneck_states = defaultdict(lambda: np.zeros(grid_size**2))
        # Learn which macro states are adjacent, and also improver M further using the new info we have
        for ep in range(num_episodes):
            self.env_reset()
            s_idx = self.onehot_to_idx(self.s) 
            # self.b_idx = s_idx 
            # self.b = self.s
            episodic_error = []
            for j in range(train_episode_length):
                action = random.randrange(self.n_actions)
                s_next_idx = self.onehot_to_idx(env.step(action))
                o_next_idx = self.env.get_obs_idx()
                prior_b = self.B[:,:,action] @ self.b
                self.b = infer_states(o_next_idx, self.A, prior_b)
                b_next_idx = self.onehot_to_idx(self.b)
                done = False#(s_next == self.goal_idx)      
                # self.B[s_next, s, action] += 1   # Update B  
                experiences.append([self.b_idx, action, b_next_idx, self.C[s_next_idx], done])
                if j>2:
                    # if j <= 2: print(j, entropy(self.b))
                    b_macro = self.micro_to_macro[self.b_idx]
                    b_next_macro = self.micro_to_macro[b_next_idx]
                    # print(s_idx, s_next_idx, self.b_idx, b_next_idx, end = ' ')
                    if b_macro != b_next_macro:
                        adj_list[b_macro].add(b_next_macro)
                        bottleneck_states[(b_macro,b_next_macro)] += self.b

                self.b_idx = b_next_idx
                s_idx = s_next_idx
                
                if (j > 1):
                    td_sr = self.update_sr(experiences[-2],self.M)
                    episodic_error.append(np.mean(np.abs(td_sr)))
                if done:
                    td_sr = self.update_sr(experiences[-1],self.M)
                    episodic_error.append(np.mean(np.abs(td_sr)))
                    break
            lifetime_td_errors.append(np.mean(episodic_error))
        adj_list = {k:list(v) for k,v in adj_list.items()}
        self.n_macro_actions = max(len(v) for k,v in adj_list.items())            
        return adj_list, bottleneck_states
    
    def learn_successor_transition_matrix_macro(self):
        N = self.n_clust
        action_list = self.adj_list
        A = self.n_macro_actions   
        B = np.zeros((N,N,A))
        M = np.zeros((N,N))

        for state in range(N):
            if state not in action_list:
                B[state,state,0] += 1
                continue
            for ind,val in enumerate(action_list[state]):
                B[val,state,ind] += 1
                
        I = np.eye(N)

        # Handling transitions not encountered by manually adding a P(s_t+1=a|s_t=a)=1
        for col in range(N):
            for action in range(A):
                col_sum = np.sum(B[:,col,action])
                if col_sum == 0:
                    B[col,col,action] = 1
        #Normalize matrix
        B = B/B.sum(axis=0, keepdims=True)
        B_avg = np.sum(B, axis=2)
        B_avg = B_avg/np.sum(B_avg, axis=1)[:,np.newaxis] 
        M = np.linalg.inv(I - (0.9 * B_avg))
        return B, M
    
    def learn_env_likelikood(self, num_episodes=None):
        if num_episodes is None:
            num_episodes = 500
        self.num_episodes += num_episodes
        if self.learn_likelikood:
            self.B, self.M = self.learn_successor_transition_matrix(num_episodes - num_episodes//4)
            
        else:
            self.B, self.M = self.actual_successor_transition_matrix()
        self.macro_state_list, self.micro_to_macro = self.learn_macro_clusters()
        print(self.macro_state_list, self.micro_to_macro)
        
        # Create C macro now that we have macro states
        self.C_macro = np.zeros(self.n_clust)
        G = self.C-self.beta*entropy(self.A)
        for i in range(self.n_clust):
            self.C_macro[i] = np.sum(G[self.macro_state_list[i]])
        print('C_macro',self.C_macro)
        self.adj_list, self.bottleneck_states = self.learn_adjacency(num_episodes//4)
        print('adj', self.adj_list)
        print('bs',{key:idx_to_loc(np.argmax(val), self.env.grid_size) for key, val in self.bottleneck_states.items()})
        self.B_macro, self.M_macro =  self.learn_successor_transition_matrix_macro()
        
    #### Finished Learning both Micro and Macro States ####
        
    def train_nn(self):
        env = self.env
        grid_size = env.grid_size
        
        print('Current state',self.env.get_state_loc())
        N_steps = 0
        total_r = 0
        state_size = self.state_size
        action_size = self.n_actions
        macro_action_size = self.n_macro_actions
        print('Number of macro actions:',self.n_macro_actions)
        input_size = state_size + macro_action_size
        # Instantiate the model with hyperparameters
        self.macro_action_nn = Macro_Action_NN(input_size=input_size, output_size=action_size, hidden_dim=384)
        # We'll also set the model to the device that we defined earlier (default is CPU)
        self.macro_action_nn.to(device)


        for filename in glob.glob("figures/train loss/train_loss_*"):
            os.remove(filename) 

        # Define hyperparameters
        n_epochs_1 = 25
        n_epochs_2 = 25
        lr=1e-3

        # Define Loss, Optimizer
        # criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.macro_action_nn.parameters(), lr=lr)

        for i in range(n_epochs_1):
            for macro_state in range(self.n_clust):
                for macro_action, macro_final_state in enumerate(self.adj_list[macro_state]):
                    final_state = random.choice(self.bottleneck_states[(macro_state, macro_final_state)])
                    print('Initial Macro state', macro_state)
                    print('Final Macro state', macro_final_state)
                    print('Final state', final_state)
                    #final_state_one_hot = F.one_hot(torch.tensor([[final_state]]),state_size).to(torch.float)
                    macro_action = F.one_hot(torch.tensor([[macro_action]]),macro_action_size).to(torch.float)
                    losses = []
                    cnt = 0
                    for epoch in range(n_epochs_2):
                        print(f'Epoch: {epoch}')
                        init_state = random.choice(self.macro_state_list[macro_state])
                        self.env_reset(init = init_state)
                        self.b_idx = init_state

                        C = idx_to_onehot(final_state, grid_size)
                        V = self.M @ (C - self.beta*entropy(self.A))
                        
                        print('Current state',self.env.get_state_loc())
                        N_steps = 0
                        total_r = 0
                            
                        
                        loss = 0
                        adv_list = []
                        prob_dist_list = []
                        while (self.b_idx != final_state):  
                            V_adj = []
                            for act in range(self.n_actions):
                                b_adj = self.B[:,:,act] @ self.b
                                V_adj.append(V[self.onehot_to_idx(b_adj)])
                            greedy_prob_dist = Categorical(logits=torch.tensor(V_adj))

                            best_action = greedy_prob_dist.sample()
                            
                            input_state = torch.tensor([[self.b]]).to(torch.float)
                            s_new = self.onehot_to_idx(env.step(best_action))
                            o_new = self.env.get_obs_idx()
                            self.b = infer_states(o_new, self.A, self.B[:,:,best_action] @ self.b)
                            b_new = self.onehot_to_idx(self.b)

                            print('best action', best_action.item(),end=' ')
                            
                            next_action_logit= self.macro_action_nn(input_state, macro_action)
                            next_action_logit = torch.squeeze(next_action_logit)
                            prob_dist = Categorical(logits=next_action_logit)
                            
                            V_old = V[self.b_idx]
                            V_new = V[b_new]
                            r = self.C[s_new]
                            total_r += r
                            
                            # loss = -torch.log(torch.squeeze(next_action_logit[best_action])) * (r + self.gamma*V_new - V_old)
                            # loss += torch.norm(F.one_hot(torch.tensor([[best_action_actual]]),action_size).to(torch.float)-next_action_logit)*r/10
                            
                            advantage = (r + self.gamma*V_new - V_old)
                            adv_list.append(advantage)
                            prob_dist_list.append(prob_dist.log_prob(best_action))
                            # loss = -prob_dist.log_prob(best_action) * advantage
                            
                            print('Reward:',r, 'Advantage', advantage)
                            print('V(t):',V_old, 'V(t+1):', V_new)
                            print('Logits: ',next_action_logit.detach())
                            # print('loss=', loss.item())

                            self.b_idx = b_new 
                            self.b_array.append(self.b_idx)
                                            
                            
                            print('s:',self.env.get_state_loc())
                            N_steps +=1
                            if N_steps > 4*self.env.grid_size:
                                break
                            
                            
                            # if epoch%100==0:
                            #     print('embed',self.macro_action_nn.embedding.weight.grad)
                            #     print('fc',self.macro_action_nn.fc.weight.grad)
                                
                        # print(self.b_idx, final_state, prob_dist_list)
                        prob_dist_list=torch.stack(prob_dist_list)
                        adv_list = np.array(adv_list)

                        if r == 1:
                            cnt += 1
                            
                        adv_list = (adv_list - np.mean(adv_list))/(np.std(adv_list)+1e-5)
                        adv_list = torch.tensor(adv_list,dtype=torch.float32)
                        loss = -torch.dot(prob_dist_list,adv_list)
                        # print('loss',loss )

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        print("Loss: {:.4f}".format(loss.item()))
                        losses.append(loss.item())
                    plt.figure()
                    plt.plot(losses)
                    plt.savefig(f'figures/train loss/train_loss_{macro_state}_{macro_final_state}.png')
                    plt.close()
                    print('Successful episodes', cnt)
                    
                    
    def execute_macro_action(self, init_state, final_state, use_nn):
        env = self.env
        grid_size = env.grid_size
        N = grid_size**2
        #for bottleneck_state in self.bottleneck_states[(init_state, final_state)]: break# Obtain element from set of size 1
        bottleneck_state = self.onehot_to_idx(self.bottleneck_states[(init_state, final_state)])
        C = idx_to_onehot(bottleneck_state,grid_size)
        pred_uncertainty = np.array([kl_divergence(self.A[:,i],C) for i in range(N)])
        pred_divergence = entropy(self.A) 
        G = pred_divergence - self.beta*pred_uncertainty
        
        
        print('Current state',self.env.get_state_loc())
        print('bottleneck', idx_to_loc(bottleneck_state,grid_size))
        N_steps = 0
        total_r = 0
        
        if not use_nn:
            V = self.M @ (C-self.beta*entropy(self.A))#(-G)
            # print(V.reshape(grid_size,grid_size).T)
            
            while (self.b_idx != bottleneck_state) and (self.b_idx not in self.goal_idx):          
                V_adj = []
                for act in range(self.n_actions):
                    b_adj = self.B[:,:,act] @ self.b
                    q_adj  = self.A @ b_adj
                    # print(self.onehot_to_idx(q_adj))
                    # pred_uncertainty = kl_divergence(q_adj,C)
                    # pred_divergence = entropy(self.A) @ b_adj
                    # G_adj = pred_divergence - self.beta*pred_uncertainty              
                    V_adj.append(V[self.onehot_to_idx(q_adj)])   
                
                # best_action = self.onehot_to_idx(softmax(V_adj)) 
                ba_gen = (ele for ele in np.argsort(V_adj)[::-1])
                try:
                    best_action = next(ba_gen) #Greedy
                except StopIteration:
                    print('stuck')
                    best_action = self.onehot_to_idx(softmax(V_adj)) #Weights based on probabilitiy
                
                # so now we actually take the next action
                s_new = self.onehot_to_idx(env.step(best_action))
                o_new = self.env.get_obs_idx()
                next_prior = self.B[:,:,best_action] @ self.b
                self.b = infer_states(o_new, self.A, next_prior)
                b_new = self.onehot_to_idx(self.b)  
                # print(best_action, s_new, b_new, o_new)
                self.b_idx = b_new 
                # s = s_new
            
                self.s_array.append(s_new)
                self.o_array.append(o_new)
                self.b_array.append(b_new)
                self.a_array.append(best_action)
                r = self.C[s_new]
                total_r += r
                
                # if (self.b_idx != b_new): break

                print('s:',self.env.get_state_loc(), 'b:', idx_to_loc(self.b_idx,grid_size), 'a', best_action )
                N_steps +=1
                if self.micro_to_macro[self.b_idx] not in [init_state, final_state]: 
                    print('Not in correct macro state.')
                    break
                if N_steps > 4*self.env.grid_size :
                    break
        else:
            with torch.no_grad():
                macro_action = F.one_hot(torch.tensor([[self.adj_list[init_state].index(final_state)]]),self.n_macro_actions).to(torch.float)
                while (self.b_idx != bottleneck_state) and (self.b_idx not in self.goal_idx):
                    state = torch.tensor(self.b).to(torch.float)
                    next_action = torch.squeeze(self.macro_action_nn(state, macro_action))
                    print('Next action Logits:',next_action,end=', ')
                    prob_dist = Categorical(logits=next_action)
                    best_action = prob_dist.sample()
                    # best_action = torch.argmax(next_action).item()
                    print('Best action:', best_action)
                    
                    s_new = self.onehot_to_idx(env.step(best_action))
                    o_new = self.env.get_obs_idx()
                    self.b = infer_states(o_new, self.A, self.B[:,:,best_action] @ self.b)
                    b_new = self.onehot_to_idx(self.b)
                    
                    self.b_idx = b_new 
                    
                    self.s_array.append(s_new)
                    self.o_array.append(o_new)
                    self.b_array.append(b_new)
                    self.a_array.append(best_action)
                    
                    r = self.C[s_new]
                    total_r += r

                    print('s:',self.env.get_state_loc(), 'b:', idx_to_loc(self.b_idx,grid_size))
                    N_steps +=1
                    if N_steps > 4*self.env.grid_size:
                        break
            
        print('End state',self.env.get_state_loc())
        return N_steps, total_r

    #### Neural Network components done ####

    def run_episode(self, time_episode = True):       
        if time_episode == True:
            start_time = time.time()
        grid_size = self.env.grid_size
        
        self.view_matrices()

        self.env_reset(init=loc_to_idx(self.init_loc, grid_size) , verbose=True)

        # run the episode until discovered the reward
        N = self.n_clust
        N_steps = 0
        total_r = 0
        
        b_macro = self.micro_to_macro[self.b_idx]
        b_macro_max = [self.micro_to_macro[gidx] for gidx in self.goal_idx] #last square in grid

        if time_episode == True:
            ckpt_time = time.time()
        
        while b_macro != b_macro_max:
            V_macro = self.M_macro @ self.C_macro
            # print('vm',V_macro)
            b_adj = []
            for action in range(self.n_macro_actions):
                #Note: We use the macro belief here, so we use onehot_to_idx instead of self.onehot_to_idx
                b_adj.append(onehot_to_idx(self.B_macro[:,:,action] @ np.eye(N)[b_macro]))
            V_adj = [V_macro[s] for s in b_adj]
            # print('vadj',V_adj)
            # print('sadj',s_adj)

            ba_gen = (ele for ele in np.argsort(V_adj)[::-1])
            best_action = next(ba_gen)
            b_macro_new = b_adj[best_action]
            print(f's macro: {b_macro}->{b_macro_new}')
            
            while b_macro == b_macro_new:
                try:
                    best_action = next(ba_gen)
                    b_macro_new = b_adj[best_action]
                    print('sm next attempt:',b_macro_new)
                except StopIteration:
                    break
            if b_macro == b_macro_new: break #There were no macro actions available for that state, so we move to planning in lower level.
            
            # if entropy(self.b) < 1:
            N_steps_action, total_r_action = self.execute_macro_action(b_macro,b_macro_new,self.use_nn)
            total_r += total_r_action
            N_steps += 1 #N_steps_action
            
            b_macro = self.micro_to_macro[self.b_idx]
            print('ms: ',b_macro)
            print('--\n')
            if N_steps > 2*self.n_clust: 
                break

        if time_episode == True:
            ckpt_time2 = time.time()

        V = self.M @ (self.C - self.beta*entropy(self.A))
        
        print('Current state',self.env.get_state_loc())
        
        while (self.env.get_state_idx() not in self.goal_idx):
            V_adj = []
            for act in range(self.n_actions):
                b_adj = self.B[:,:,act] @ self.b
                V_adj.append(V[self.onehot_to_idx(b_adj)])   
            ba_gen = (ele for ele in np.argsort(V_adj)[::-1])
            
            while True:
                try:
                    best_action = next(ba_gen)
                except StopIteration:
                    best_action = random.randrange(self.n_actions)
                    
                # so now we actually take the next action
                s_new = self.onehot_to_idx(self.env.step(best_action))
                o_new = self.env.get_obs_idx()
                self.b = infer_states(o_new, self.A, self.B[:,:,best_action] @ self.b)
                b_new = self.onehot_to_idx(self.b)
                # print(best_action, s_new, b_new, o_new)
                if (self.b_idx != b_new): break

            self.b_idx = b_new
            
            self.s_array.append(s_new)
            self.o_array.append(o_new)
            self.b_array.append(b_new)
            self.a_array.append(best_action)
            
            r = self.C[s_new]
            total_r += r
            print('s:',self.env.get_state_loc(), 'b:', idx_to_loc(self.b_idx,grid_size))
            N_steps +=1
            if N_steps > 3*self.env.grid_size:
                break
        
        end_state = self.env.get_state_loc()
        print('End state',end_state)
        
        if time_episode == True:
            total_time = (time.time() - ckpt_time2) + (ckpt_time - start_time)
            return N_steps, total_r, total_time, end_state
        return N_steps, total_r, 0, end_state
    
    
  
  ######VISUALIZATION#######    
  
    def visualize_nn(self):
        
        # Clear out old images
        for filename in glob.glob("figures/Macro Action Network/Macro_Action_Network_*"):
            os.remove(filename) 
            
        with torch.no_grad():
            for macro_state in range(self.n_clust):
                for macro_action, macro_final_state in enumerate(self.adj_list[macro_state]):
                    final_state = self.bottleneck_states[(macro_state, macro_final_state)][0]
                    print(f'\nInitial state: Micro - {macro_state}')
                    print(f'Final state: Micro - {final_state}, Macro - {macro_final_state}')
                    macro_action = F.one_hot(torch.tensor([[macro_action]]),self.n_macro_actions).to(torch.float)
                    arrows_grid = np.full((grid_size,grid_size),-1)
                    for i in self.macro_state_list[macro_state]:
                        self.env_reset(init=i)
                        state = F.one_hot(torch.tensor([[i]]),args.state_size).to(torch.float)
                        print(self.env.get_state_loc())
                        next_action = torch.squeeze(self.macro_action_nn(state, macro_action))
                        print('Next action Logits:',next_action,end=', ')
                        next_action = torch.argmax(next_action).item()
                        print('Best action:', next_action)
                        arrows_grid[self.env.get_state_loc()] = next_action
                        
                        
                    arrows_grid = arrows_grid.T
                    
                    plt.figure()
                    arrows = {1:(1,0), 0:(-1,0),3:(0,1),2:(0,-1)}
                    scale = 0.25
            
                    plt.xticks(np.arange(grid_size),fontsize=12)
                    plt.yticks(np.arange(grid_size),fontsize=12)
                    fig, ax = plt.subplots(figsize=(grid_size,grid_size))
                    for r, row in enumerate(arrows_grid):
                        for c, cell in enumerate(row):
                            if cell in arrows:
                                plt.arrow(c, r, scale*arrows[cell][0], scale*arrows[cell][1], head_width=0.1,color='w')
                    im = plt.imshow(self.lg,cmap='gist_heat')
                    colours = im.cmap(im.norm(np.unique(self.lg)))
                    
                    #Just to add labels for the macro actions in the figure
                    patches = [ mpatches.Patch(color=colours[i], label=f'{i}' ) for i in range(len(colours)-1) ]
                    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
                    plt.title(f"Actions predicted: Macro state {macro_state} to {macro_final_state}", fontsize=20)
                    plt.savefig(f"figures/Macro Action Network/Macro_Action_Network_{macro_state}_{macro_final_state}.png", format="png")
                    plt.close()
  
    def view_matrices(self):
        if self.learn_likelikood:
            learn_str = 'estimated'
        else:
            learn_str = 'actual'
        images_loc = 'figures/matrices/'
        B_avg = np.sum(self.B, axis=2) / self.B.shape[-1]
        grid_size = self.env.grid_size
        plt.figure()
        if self.env.noise is None:
            plt.title(f"Default Policy B Matrix",fontsize=20)
        else:
            plt.title(f"Default Policy B Matrix with noise {self.env.noise}",fontsize=20)
        plt.imshow(B_avg)#, norm=LogNorm(vmin=B_avg.min(), vmax=B_avg.max()))
        plt.xticks(list(range(0,grid_size**2)),fontsize=12)
        plt.yticks(list(range(0, grid_size**2)),fontsize=12)
        plt.colorbar()
        plt.savefig(images_loc+f"B_matrix_micro.png", format="png")
        plt.close()
        
        
        plt.figure()
        plt.imshow(self.A, aspect='equal', cmap='cividis')
        plt.xticks(list(range(0,grid_size**2)),fontsize=12)
        plt.yticks(list(range(0,grid_size**2)),fontsize=12)
        plt.title(f"Sensory Likelihood Matrix A", fontsize=20)
        plt.savefig(images_loc+f"A_{learn_str}_micro.png", format="png")
        plt.colorbar()
        plt.close()
        np.save("data/A.npy", self.A)

        plt.figure()
        plt.imshow(self.M, aspect='equal', cmap='cividis')#, norm=LogNorm(vmin=self.M.min(), vmax=self.M.max()))
        plt.xticks(list(range(0,grid_size**2)),fontsize=12)
        plt.yticks(list(range(0,grid_size**2)),fontsize=12)
        plt.title(f"Successor Matrix M ({learn_str})", fontsize=20)
        plt.colorbar()
        plt.savefig(images_loc+f"m_{learn_str}_micro.png", format="png")
        plt.close()
        np.save("data/M.npy", self.M)
        
        V = self.M @ self.C
        plt.figure()
        plt.imshow(V.reshape(grid_size,grid_size).T)
        plt.xticks(list(range(0,grid_size)),fontsize=12)
        plt.yticks(list(range(0,grid_size)),fontsize=12)
        plt.title("Estimated Value Function",fontsize=20)
        plt.colorbar()
        plt.savefig(images_loc+f"{learn_str}_value_function_old.png", format="png")
        plt.close()
        np.save("data/V.npy", V)
        
        plt.figure()
        V = self.M @ (entropy(self.A))
        plt.imshow(V.reshape(grid_size,grid_size).T)
        plt.xticks(list(range(0,grid_size)),fontsize=12)
        plt.yticks(list(range(0,grid_size)),fontsize=12)
        plt.title("Estimated Value Function gain",fontsize=20)
        plt.colorbar()
        plt.savefig(images_loc+f"{learn_str}_value_function_gain.png", format="png")
        plt.close()
        
        pred_uncertainty = np.array([kl_divergence(self.A[:,i],softmax(self.C)) for i in range(grid_size**2)])
        pred_divergence = entropy(self.A)
        G = pred_divergence - 10*pred_uncertainty
        V = self.M @ G
        plt.figure()
        plt.imshow(V.reshape(grid_size,grid_size).T)
        plt.xticks(list(range(0,grid_size)),fontsize=12)
        plt.yticks(list(range(0,grid_size)),fontsize=12)
        plt.title("Estimated Value Function",fontsize=20)
        plt.colorbar()
        plt.savefig(images_loc+f"{learn_str}_value_function.png", format="png")
        plt.close()
        np.save("data/V.npy", V)
        

        M_orig = self.M[0,:].reshape(grid_size,grid_size).T
        plt.figure()
        plt.imshow(M_orig, aspect='equal', cmap='copper', norm=LogNorm(vmax=M_orig.max()))
        plt.xticks(list(range(0,grid_size)),fontsize=12)
        plt.yticks(list(range(0,grid_size)),fontsize=12)
        plt.title(f"Successor Matrix M from start({learn_str})", fontsize=20)
        plt.colorbar()
        plt.savefig(images_loc+f"m_origin_{learn_str}.png", format="png")
        plt.close()
        
        B_macro_avg = np.sum(self.B_macro, axis=2) / self.n_macro_actions
        plt.figure()
        plt.title("Default Policy B Matrix",fontsize=20)
        plt.imshow(B_macro_avg)#, norm=LogNorm(vmin=B_avg.min(), vmax=B_avg.max()))
        plt.xticks(list(range(0,self.n_clust)),fontsize=12)
        plt.yticks(list(range(0, self.n_clust)),fontsize=12)
        for i in range(len(B_macro_avg)):
            for j in range(len(B_macro_avg)):
                plt.text(j, i, f'{B_macro_avg[i, j]:.2f}',
                            ha="center", va="center", color="w")
        plt.savefig(images_loc+f"B_matrix_macro.png", format="png")
        plt.close()

        plt.figure() 
        plt.imshow(self.M_macro, aspect='equal', cmap='cividis')#, norm=LogNorm(vmin=self.M.min(), vmax=self.M.max()))
        plt.xticks(list(range(0,self.n_clust)),fontsize=12)
        plt.yticks(list(range(0,self.n_clust)),fontsize=12)
        plt.title(f"Successor Matrix M ({learn_str})", fontsize=20)
        for i in range(len(self.M_macro)):
            for j in range(len(self.M_macro)):
                plt.text(j, i, f'{self.M_macro[i, j]:.2f}',
                            ha="center", va="center", color="w")
        plt.savefig(images_loc+f"m_{learn_str}_macro.png", format="png")
        plt.close()    

    def show_actions(self):

        gs = self.env.grid_size
        fig = plt.figure()
        grid = np.zeros((gs, gs))
        if len(self.env.walls) > 0:
            wall_idx = tuple(np.array(self.env.walls).T)
        start_loc = self.init_loc
        goal_loc = [idx_to_loc(gidx, gs) for gidx in self.goal_idx]
        grid[start_loc] = 1
        grid[wall_idx] = 2
        grid[tuple(zip(*goal_loc))] = 0.5

        arrows = {1:(1,0), 0:(-1,0),3:(0,1),2:(0,-1)}
        scale = 0.25
        arrows_grid = np.full((gs, gs),-1)
        s_idxs = [idx_to_loc(s, gs) for s in self.s_array]
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
                ax.text(goal[0], goal[1], 'Goal', fontsize = 10, ha="center", va="center", color="w")
        ax.text(start_loc[0], start_loc[1], 'Agent', fontsize = 10, ha="center", va="center", color="b")
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
        plt.savefig(f"figures/Actions_taken.png", format="png")
        plt.close()

        
    def show_video(self, vid_type='s', title=''):
        #print(self.s_array)
        gs = self.env.grid_size

        fig = plt.figure()
        grid = np.zeros((gs, gs))
        if len(self.env.walls) > 0:
            wall_idx = tuple(np.array(self.env.walls).T)
        start_idx = self.init_loc
        goal_idx = [idx_to_loc(gidx, gs) for gidx in self.goal_idx]
        grid[start_idx] = 1
        grid[goal_idx] = 0.5
        grid[wall_idx] = 0.25

        im = plt.imshow(grid.T, aspect='equal', cmap='magma')
        if vid_type == 'o':
            plt.title(f'Observations for grid world of size {gs}')
        elif vid_type == 's':
            plt.title(f'True states for grid world of size {gs}')
        elif vid_type == 'b':
            plt.title(f'Belief states for grid world of size {gs}')
        else:
            raise ValueError('Invalid video type')
        
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
        
        past_idx_array = []


        def init():
            grid = np.zeros((gs, gs))
            grid[start_idx] = 1
            grid[wall_idx] = 0.25  
            grid[tuple(zip(*goal_idx))] = 0.5
            im.set_data(grid)
            return im,


        def animate(i):
            for txt in ax.texts[::-1]:
                txt.remove()
            
            
            if vid_type == 'o':
                o = self.o_array[i]
                idx = idx_to_loc(o, gs)
                
            elif vid_type == 's':
                s = self.s_array[i]
                idx = idx_to_loc(s, gs)
                
            elif vid_type == 'b':
                b = self.b_array[i]
                idx = idx_to_loc(b, gs)
            
            grid = np.zeros((gs, gs))
            grid[wall_idx] = 0.25
            grid[tuple(zip(*goal_idx))] = 0.5
            
            #Colour the previous steps in path
            for past_idx in past_idx_array:
                grid[past_idx] = 0.75
            grid[idx] = 1
            past_idx_array.append(idx)
            im.set_data(grid.T)
            for goal in goal_idx:
                ax.text(goal[0], goal[1], 'Goal', fontsize = 10, ha="center", va="center", color="w")
            ax.text(idx[0], idx[1], 'Agent', fontsize = 10, ha="center", va="center", color="b")
            for widx in self.env.walls:
                ax.text(widx[0], widx[1], 'Wall', fontsize = 10, ha="center", va="center", color="w")

            return im,
        ani = animation.FuncAnimation(fig, animate, np.arange(len(self.b_array)), init_func=init,
                              interval=500, blit=True)
        ani.save(f'videos/env_micro_{vid_type}{title}.mp4')
        plt.close()    
        
    def show_video_combined(self):
        #print(self.s_array)
        gs = self.env.grid_size

        fig = plt.figure()
        grid = np.zeros((gs, gs)) 
        start_idx = self.init_loc
        goal_idx = [idx_to_loc(gidx, gs) for gidx in self.goal_idx]
        if len(self.env.walls) > 0:
            wall_idx = tuple(np.array(self.env.walls).T)
        grid[start_idx] = 1
        grid[goal_idx] = 0.5
        grid[wall_idx] = 0.25
        im = plt.imshow(grid.T, aspect='equal', cmap='magma')
        plt.title(f'Params for Grid size {gs} and noise {self.env.noise}')
            
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

        s_idx_array = []
        o_idx_array = []
        b_idx_array = []


        def init():
            grid = np.zeros((gs, gs))
            grid[start_idx] = 1
            grid[wall_idx] = 0.25  
            grid[goal_idx] = 0.5
            im.set_data(grid)
            return im,


        def animate(i):
            for txt in ax.texts[::-1]:
                txt.remove()
            grid = np.zeros((gs, gs))
            grid[wall_idx] = 0.25  
            grid[goal_idx] = 0.5
        
            o = self.o_array[i]
            o_idx = idx_to_loc(o, gs)
            s = self.s_array[i]
            s_idx = idx_to_loc(s, gs)
            b = self.b_array[i]
            b_idx = idx_to_loc(b, gs)

            
            #Colour the previous steps in path
            for past_idx in s_idx_array:
                grid[past_idx] += 0.1
            for past_idx in o_idx_array:
                grid[past_idx] += 0.2
            for past_idx in b_idx_array:
                grid[past_idx] += 0.4
            grid[s_idx] = 1
            grid[b_idx] = 1
            grid[o_idx] = 1
            s_idx_array.append(s_idx)
            o_idx_array.append(o_idx)
            b_idx_array.append(b_idx)

            im.set_data(grid.T)


            ax.text(s_idx[0], s_idx[1], 'State', fontsize = 10, ha="center", va="center", color="b")
            ax.text(o_idx[0], o_idx[1], 'Obs', fontsize = 10, ha="center", va="center", color="b")
            ax.text(b_idx[0], b_idx[1], 'Belief', fontsize = 10, ha="center", va="center", color="b")
            ax.text(goal_idx[0], goal_idx[1], 'Goal', fontsize = 10, ha="center", va="center", color="w")
            for widx in self.env.walls:
                ax.text(widx[0], widx[1], 'Wall', fontsize = 10, ha="center", va="center", color="w")

            return im,
        ani = animation.FuncAnimation(fig, animate, np.arange(len(self.o_array)), init_func=init,
                              interval=500, blit=True)
        ani.save(f'videos/params.mp4')
        plt.close()  


##### All main helper functions are done #### 

def SR_performance_Hierarchy(args):
    SR_steps = []
    SR_rewards = []
    SR_times = []

    n_macro_states = args.n_macro
    grid_size = args.grid_size
    print('Number of macro states: ',n_macro_states)
 

    for n in range(args.n_runs):
        print('x'*20)
        print(f"Run {n}")
        print('x'*20)
        SR_step = []
        SR_reward = []
        SR_time = []

        grid_env = SR_Gridworld(grid_size,noise=args.noise)    
        A = np.identity(grid_size**2)    
        A[:,72] =  stats.gamma.pdf(np.arange(A.shape[1]), a=n+1, scale=1)  
        A[:,74] =  stats.gamma.pdf(np.arange(A.shape[1]), a=n+1, scale=1)
        grid_env.set_likelihood_dist(A)         

        # grid_env.set_walls(create_walls(grid_size, wall_size=wall_size))
        grid_env.set_walls(args.walls)


        walls = grid_env.get_walls()
        print('walls',walls)
        

        # SR agent
        inverse_factor = 1
        if grid_size > 7:
            inverse_factor = 4
        agent = SR_Agent_Hierarchy(grid_env, inverse_factor=inverse_factor,n_clust = n_macro_states, beta=args.beta,
                                state_size = args.state_size, init_loc = args.init_loc, goal_locs = args.goal_locs,
                                goal_vals=args.goal_vals, learn_likelikood=True, use_nn=args.use_nn) 
        
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
            SR_step.append(deepcopy(N_steps))
            SR_reward.append(deepcopy(total_r))
            SR_time.append(deepcopy(total_time))

            agent.show_actions()
            agent.show_video('s')
            

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


def SR_performance_Noise(args):
    '''
    Vary the noise for the 2 goals and see how the agent performs
    
    '''
    
    n_trials = len(args.episodes)
    SR_final_beliefs = np.zeros((n_trials,args.n_runs))
    SR_entropies = np.zeros((args.n_runs,))

    n_macro_states = args.n_macro
    grid_size = args.grid_size
    print('Number of macro states: ',n_macro_states)
    
    goal_idxs = [loc_to_idx(goal,grid_size) for goal in goal_locs]
 

    for n in range(args.n_runs):
        print('x'*20)
        print(f"Run {n}")
        print('x'*20)

        A = np.identity(grid_size**2)    
        A[:,goal_idxs[0]] =  stats.gamma.pdf(np.arange(A.shape[1]), a=n+1, scale=1)   
        grid_env = SR_Gridworld(grid_size,noise=args.noise)  
        grid_env.set_likelihood_dist(A)   
        SR_entropies[n] = entropy(A)[goal_idxs[0]]

        # grid_env.set_walls(create_walls(grid_size, wall_size=wall_size))
        grid_env.set_walls(args.walls)


        walls = grid_env.get_walls()
        print('walls',walls)
        

        # SR agent
        inverse_factor = 1
        if grid_size > 7:
            inverse_factor = 4
        agent = SR_Agent_Hierarchy(grid_env, inverse_factor=inverse_factor,n_clust = n_macro_states, beta=args.beta,
                                state_size = args.state_size, init_loc = args.init_loc, goal_locs = args.goal_locs,
                                goal_vals=args.goal_vals, learn_likelikood=True, use_nn=args.use_nn) 
        
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
            agent.run_episode()
            
            final_state = loc_to_idx(grid_env.get_state_loc(),grid_size)
            SR_final_beliefs[num, n] = deepcopy(final_state)
    
    return SR_final_beliefs, SR_entropies

    
if __name__ == '__main__':
    
    grid_size = 9
    n_macro = 5
    init_loc = (0,0)
    goal_locs = [(grid_size-1, grid_size-1)]#[(0,grid_size-1), (grid_size-1,0)]
    use_nn = False
    noise = None
    nruns = 1
    eps = [1500]
    goal_vals = [5]#[1.75,.5]
    # WALLS = [(1,x) for x in range(grid_size//2+2)] + \
    #         [(3,x) for x in range(grid_size//2-2,grid_size)] +\
    #         [(5,x) for x in range(grid_size//2+2)] + \
    #         [(7,x) for x in range(grid_size//2-2,grid_size)]
    beta = 1

    WALLS = [(4,x) for x in range(grid_size) if x not in [2,6]] + \
            [(x,4) for x in range(grid_size) if x not in [2,6]]
    
    parser = argparse.ArgumentParser(description="Arguments for hierarchical model")
    parser.add_argument("grid_size", nargs='?',type=int, default=grid_size, help="Grid size of environment")
    parser.add_argument("state_size", nargs='?',type=int, default=grid_size**2, help="Size of state embedding")
    parser.add_argument("n_macro", nargs='?',type=int, default=n_macro, help="Number of macro states")
    parser.add_argument("init_loc", nargs='?',type=int, default=init_loc, help="Initial location of agent")
    parser.add_argument("goal_locs", nargs='?',type=int, default=goal_locs, help="Goal location of agent")
    parser.add_argument("goal_vals", nargs='?',type=int, default=goal_vals, help="Goal value of agent")
    parser.add_argument("noise", nargs='?', type=bool, default=noise, help="Noise level (0-1)")
    parser.add_argument("beta", nargs='?', type=bool, default=beta, help="Beta value for entropy regularization")
    parser.add_argument("n_runs", nargs='?', type=int, default=nruns, help="Number of runs for the experiment")
    parser.add_argument("use_nn", nargs='?', type=bool, default=use_nn, help="Train an RNN for the policy?")
    parser.add_argument("walls", nargs='?', type=list, default=WALLS, help="Range of walls")
    parser.add_argument("episodes", nargs='?', type=list, default=eps, help="Range of number of episodes")
    args = parser.parse_args()
    
    print('Hierarchical SR Model')
    print('*'*50)
    SR_steps, SR_rewards, SR_times= SR_performance_Hierarchy(args)
    print('*'*50)
    exit()
    print('Hierarchical SR Model')
    print('*'*50)
    SR_final_beliefs, SR_entropies= SR_performance_Noise(args)
    print('*'*50)
    print(SR_final_beliefs)
    print(SR_entropies)
    probs = np.mean(SR_final_beliefs==loc_to_idx(goal_locs[0],grid_size),axis=0)
    
    
    plt.figure()
    plt.plot(SR_entropies, probs)
    plt.xlabel("Entropy of likelihood")
    plt.ylabel(f"Probability of reaching goal {goal_locs[0]}")
    plt.savefig("figures/noise/final_beliefs_entropy.png")
    plt.close()
    
    
    if not os.path.exists("data/"):
        os.makedirs("data/")
        
    np.save("data/SR_final_beliefs.npy", SR_final_beliefs)

