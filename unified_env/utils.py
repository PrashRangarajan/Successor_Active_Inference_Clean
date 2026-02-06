"""Consolidated utility functions for gridworld environments.

This module provides common utilities for state representation conversion,
mathematical operations, and environment helpers.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import itertools
from random import choices


# =============================================================================
# Basic State Conversions (for standard gridworld)
# =============================================================================

def loc_to_idx(pos, grid_size):
    """Convert (x, y) location to flat index."""
    return (pos[0] * grid_size) + pos[1]


def idx_to_loc(idx, grid_size):
    """Convert flat index to (x, y) location."""
    x, y = divmod(idx, grid_size)
    return x, y


def idx_to_onehot(idx, grid_size):
    """Convert flat index to one-hot vector."""
    state = np.zeros(grid_size**2)
    state[idx] = 1
    return state


def onehot_to_idx(s):
    """Convert one-hot vector to index (probabilistic sampling)."""
    return choices(np.arange(len(s)), s)[0]


def onehot_to_loc(s, grid_size):
    """Convert one-hot vector to (x, y) location."""
    return idx_to_loc(onehot_to_idx(s), grid_size)


def loc_to_onehot(pos, grid_size):
    """Convert (x, y) location to one-hot vector."""
    return idx_to_onehot(loc_to_idx(pos, grid_size), grid_size)


# =============================================================================
# Augmented State Conversions (for key gridworld)
# These handle states of the form (location, has_key) or (x, y, has_key)
# =============================================================================

def loc_to_idloc(loc, grid_size):
    """Convert (x, y) location to location index (same as loc_to_idx)."""
    return (loc[0] * grid_size) + loc[1]


def idloc_to_loc(idloc, grid_size):
    """Convert location index to (x, y) (same as idx_to_loc)."""
    x, y = divmod(idloc, grid_size)
    return x, y


def augmented_idx_to_val(idx, grid_size):
    """Convert (idloc, idkey) to single value index.

    Value range: 0 to 2*grid_size^2-1
    States without key come first, then states with key.
    """
    return idx[1] * grid_size**2 + idx[0]


def val_to_augmented_idx(val, grid_size):
    """Convert single value to (idloc, idkey)."""
    idkey, idloc = divmod(val, grid_size**2)
    return idloc, idkey


def augmented_idx_to_state(idx, grid_size):
    """Convert (idloc, idkey) to (x, y, key)."""
    idloc, idkey = idx
    return idloc_to_loc(idloc, grid_size) + (idkey,)


def state_to_augmented_val(pos, grid_size):
    """Convert (x, y, key) to single value index."""
    loc = pos[:-1]
    key = pos[-1]
    idloc = loc_to_idloc(loc, grid_size)
    return augmented_idx_to_val((idloc, key), grid_size)


def val_to_augmented_state(val, grid_size):
    """Convert single value to (x, y, key)."""
    idloc, idkey = val_to_augmented_idx(val, grid_size)
    return idloc_to_loc(idloc, grid_size) + (idkey,)


def augmented_val_to_onehot(val, grid_size):
    """Convert single value to (N, 2) one-hot matrix."""
    state = np.zeros((grid_size**2, 2))
    state[val_to_augmented_idx(val, grid_size)] = 1
    return state


def augmented_onehot_to_val(s):
    """Convert (N, 2) one-hot matrix to single value."""
    s_flat = s.flatten('F')
    return choices(np.arange(len(s_flat)), s_flat)[0]


def augmented_idx_to_onehot(idx, grid_size):
    """Convert (idloc, idkey) to (N, 2) one-hot matrix."""
    state = np.zeros((grid_size**2, 2))
    state[idx] = 1
    return state


def augmented_onehot_to_idx(s):
    """Convert (N, 2) one-hot matrix to (idloc, idkey)."""
    N = len(s)
    s_flat = s.flatten('F')
    idx_loc = choices(np.arange(len(s_flat)), s_flat)[0]
    key, loc = divmod(idx_loc, N)
    return loc, key


def augmented_onehot_to_state(s, grid_size):
    """Convert (N, 2) one-hot matrix to (x, y, key)."""
    loc, key = augmented_onehot_to_idx(s)
    return idloc_to_loc(loc, grid_size) + (key,)


def augmented_state_to_onehot(pos, grid_size):
    """Convert (x, y, key) to (N, 2) one-hot matrix."""
    return augmented_idx_to_onehot(
        state_to_augmented_val(pos, grid_size),
        grid_size
    )


# =============================================================================
# Tensor Operations for Augmented State Space
# =============================================================================

def multiply_B_s(B, state, act):
    """Multiply transition tensor B with state for augmented state space.

    Args:
        B: Transition tensor of shape (N, 2, N, 2, N_ACT)
        state: State tensor of shape (N, 2)
        act: Action index or None for all actions

    Returns:
        Next state of shape (N, 2)
    """
    if act is None:
        return np.tensordot(B[:, :, :, :, :], state, axes=([2, 3], [0, 1]))
    else:
        return np.tensordot(B[:, :, :, :, act], state, axes=([2, 3], [0, 1]))


def multiply_M_C(M, C):
    """Multiply successor tensor M with reward C for augmented state space.

    Args:
        M: Successor tensor of shape (N, 2, N, 2)
        C: Reward tensor of shape (N, 2)

    Returns:
        Value vector of shape (2N,)
    """
    return np.tensordot(M, C, axes=([2, 3], [0, 1])).flatten('F')


def flatten_transition(B):
    """Flatten augmented transition tensor to 2D matrix.

    Args:
        B: Transition tensor of shape (N, 2, N, 2, N_A)

    Returns:
        Flattened matrix of shape (2N, 2N)
    """
    return np.hstack([
        np.vstack([B[:, j, :, i, :] for j in [0, 1]])
        for i in [0, 1]
    ])


def flatten_successor(M):
    """Flatten augmented successor tensor to 2D matrix.

    Args:
        M: Successor tensor of shape (N, 2, N, 2)

    Returns:
        Flattened matrix of shape (2N, 2N)
    """
    return np.hstack([
        np.vstack([M[:, j, :, i] for j in [0, 1]])
        for i in [0, 1]
    ])


# =============================================================================
# Macro State Conversions (for hierarchical planning)
# =============================================================================

def micro_to_macro(pos, grid_sq_len):
    """Convert micro state position to macro state position."""
    return (pos[0] // grid_sq_len, pos[1] // grid_sq_len)


def micro_to_macro_idx(pos, grid_size, grid_sq_len):
    """Convert micro state index to macro state index."""
    return loc_to_idx(
        micro_to_macro(idx_to_loc(pos, grid_size), grid_sq_len),
        grid_size // grid_sq_len
    )


# =============================================================================
# Mathematical Utilities
# =============================================================================

def log_stable(x):
    """Numerically stable log function."""
    return np.log(x + 1e-5)


def entropy(A):
    """Compute entropy of distribution A."""
    H_A = -(A * log_stable(A)).sum(axis=0)
    return H_A


def softmax(dist):
    """Compute softmax function on a set of values."""
    output = dist - dist.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    return output


def kl_divergence(q1, q2):
    """Compute KL divergence between two categorical distributions."""
    return (log_stable(q1) - log_stable(q2)) @ q1


def infer_states(observation_index, A, prior):
    """Bayesian state inference given observation and prior.

    Args:
        observation_index: Index of observed state
        A: Likelihood matrix
        prior: Prior belief over states

    Returns:
        Posterior belief over states
    """
    log_likelihood = log_stable(A[observation_index, :])
    log_prior = log_stable(prior)
    qs = softmax(log_likelihood + log_prior)
    return qs


# =============================================================================
# Environment Helpers
# =============================================================================

def create_walls(grid_size, wall_size=3):
    """Create random wall positions for gridworld.

    Args:
        grid_size: Side length of grid
        wall_size: Number of wall positions

    Returns:
        List of (x, y) wall positions
    """
    walls = random.sample(
        list(itertools.product(range(grid_size), repeat=2)),
        wall_size
    )
    # Ensure start and goal positions are not walls
    while (0, 0) in walls or (grid_size - 1, grid_size - 1) in walls:
        walls = random.sample(
            list(itertools.product(range(grid_size), repeat=2)),
            wall_size
        )
    return walls


def plot_beliefs(belief_dist, title_str=""):
    """Plot a categorical belief distribution.

    Args:
        belief_dist: 1-D numpy vector of beliefs
        title_str: Title for the plot
    """
    plt.grid(zorder=0)
    plt.bar(range(belief_dist.shape[0]), belief_dist, color='r', zorder=3)
    for i, v in enumerate(belief_dist):
        if v > 0.01:
            plt.text(i, v, i, ha='center', va='bottom')
    plt.yscale('log')
    plt.title(title_str)
    plt.show()
