import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import random
import itertools
from random import choices


def loc_to_idx(pos,grid_size):
    return (pos[0] * grid_size) + pos[1]

def idx_to_loc(idx, grid_size):
    x,y = divmod(idx, grid_size)
    return x, y

def idx_to_onehot(idx, grid_size):
    state = np.zeros(grid_size**2)
    state[idx] = 1
    return state

def onehot_to_idx(s):
    return choices(np.arange(len(s)),s)[0]

def onehot_to_loc(s,grid_size):
    return idx_to_loc(onehot_to_idx(s), grid_size)

def loc_to_onehot(pos, grid_size):
    return idx_to_onehot(loc_to_idx(pos,grid_size), grid_size)
        
# hierarchy.py - fixed macro states
def micro_to_macro(pos,grid_sq_len):
    return (pos[0]//grid_sq_len, pos[1]//grid_sq_len)

def micro_to_macro_idx(pos, grid_size, grid_sq_len):
    return loc_to_idx(micro_to_macro(idx_to_loc(pos, grid_size), grid_sq_len),grid_size//grid_sq_len)

        
def create_walls(grid_size, wall_size=3):
    '''
    grid_size: Side length of grid
    wall_size: Number of wall positions
    Returns a list of wall positions
    '''
    walls = random.sample(list(itertools.product(range(grid_size), repeat = 2)), wall_size)
    while (0,0) in walls or (grid_size-1,grid_size-1) in walls:
        walls = random.sample(list(itertools.product(range(grid_size), repeat = 2)), wall_size)
    return walls

def plot_beliefs(belief_dist, title_str=""):
    """
    Plot a categorical distribution or belief distribution, stored in the 1-D numpy vector `belief_dist`
    """

    # if not np.isclose(belief_dist.sum(), 1.0):
    #   raise ValueError("Distribution not normalized! Please normalize")
    plt.grid(zorder=0)
    plt.bar(range(belief_dist.shape[0]), belief_dist, color='r', zorder=3)
    # plt.xticks(range(belief_dist.shape[0]))
    for i, v in enumerate(belief_dist):
        if v > 0.01:
            plt.text(i, v, i, ha='center', va='bottom')
    plt.yscale('log')
    plt.title(title_str)
    plt.show()
  
def generate_B_matrices(grid_size, walls=[]):
    N = grid_size**2
    N_size = (N,N)
    B_up = np.zeros(N_size)
    B_down = np.zeros(N_size)
    B_L = np.zeros(N_size)
    B_R = np.zeros(N_size)

    B = np.zeros((N,N,4))
    for i in range(N):
        start_x, start_y = idx_to_loc(i, grid_size)
        if (start_x, start_y) in walls:
            B_L[i,i] = 1
            B_R[i,i] = 1
            B_up[i,i] = 1
            B_down[i,i] = 1
            continue

        for j in range(N):         
            end_x, end_y = idx_to_loc(j, grid_size)

            # left matrix
            if start_x == 0:
                if start_x == end_x and start_y == end_y:
                    B_L[i,j] = 1
            if start_x != 0:
                if end_x == start_x - 1 and start_y == end_y:
                    if (end_x, end_y) in walls:
                        B_L[i,i] = 1
                    else:
                        B_L[i,j] = 1
            # right matrix
            if start_x == grid_size-1:
                if start_x == end_x and start_y == end_y:
                    B_R[i,j] = 1
            if start_x != grid_size - 1:
                if end_x == start_x + 1 and start_y == end_y:
                    if (end_x, end_y) in walls:
                        B_R[i,i] = 1
                    else:
                        B_R[i,j] = 1
            # up matrix
            if start_y == 0:
                if start_y == end_y and start_x == end_x:
                    B_up[i,j] = 1
            if start_y != 0:
                if end_y == start_y - 1 and start_x == end_x:
                    if (end_x, end_y) in walls:
                        B_up[i,i] = 1
                    else:
                        B_up[i,j] = 1
            # down matrix
            if start_y == grid_size-1:
                if start_y == end_y and start_x == end_x:
                    B_down[i,j] = 1
            if start_y != grid_size - 1:
                if end_y == start_y + 1 and start_x == end_x:
                    if (end_x, end_y) in walls:
                        B_down[i,i] = 1
                    else:
                        B_down[i,j] = 1
    B[:,:,0] = B_L.T
    B[:,:,1] = B_R.T
    B[:,:,2] = B_up.T
    B[:,:,3] = B_down.T
    return B
        
        
def log_stable(x):
    return np.log(x + 1e-3)

def entropy(A):
    H_A = - (A * log_stable(A)).sum(axis=0)
    return H_A

def softmax(dist):
    """ 
    Computes the softmax function on a set of values
    """

    output = dist - dist.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    return output

def kl_divergence(q1, q2):
  """ Compute the Kullback-Leibler divergence between two 1-D categorical distributions"""
  
  return (log_stable(q1) - log_stable(q2)) @ q1

def infer_states(observation_index, A, prior):

  """ Implement inference here -- NOTE: prior is already passed in, so you don't need to do anything with the B matrix. """
  """ This function has already been given P(s_t). The conditional expectation that creates "today's prior", using "yesterday's posterior", will happen *before calling* this function"""
  
  log_likelihood = log_stable(A[observation_index,:])

  log_prior = log_stable(prior)

  qs = softmax(log_likelihood + log_prior)
   
  return qs