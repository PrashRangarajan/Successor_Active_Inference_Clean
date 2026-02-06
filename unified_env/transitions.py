"""Transition matrix generation for gridworld environments.

This module provides functions to generate transition matrices B
for different gridworld variants.
"""

import numpy as np
from .utils import idx_to_loc, idloc_to_loc


def generate_standard_B(grid_size, walls=[]):
    """Generate transition matrices for standard gridworld.

    Args:
        grid_size: Side length of the grid
        walls: List of (x, y) wall positions

    Returns:
        B: Transition tensor of shape (N, N, 4) where N = grid_size^2
           Actions: 0=left, 1=right, 2=up, 3=down
    """
    N = grid_size**2
    N_size = (N, N)
    B_up = np.zeros(N_size)
    B_down = np.zeros(N_size)
    B_L = np.zeros(N_size)
    B_R = np.zeros(N_size)

    B = np.zeros((N, N, 4))

    for i in range(N):
        start_x, start_y = idx_to_loc(i, grid_size)

        # If starting from a wall, stay in place
        if (start_x, start_y) in walls:
            B_L[i, i] = 1
            B_R[i, i] = 1
            B_up[i, i] = 1
            B_down[i, i] = 1
            continue

        for j in range(N):
            end_x, end_y = idx_to_loc(j, grid_size)

            # Left matrix
            if start_x == 0:
                if start_x == end_x and start_y == end_y:
                    B_L[i, j] = 1
            if start_x != 0:
                if end_x == start_x - 1 and start_y == end_y:
                    if (end_x, end_y) in walls:
                        B_L[i, i] = 1
                    else:
                        B_L[i, j] = 1

            # Right matrix
            if start_x == grid_size - 1:
                if start_x == end_x and start_y == end_y:
                    B_R[i, j] = 1
            if start_x != grid_size - 1:
                if end_x == start_x + 1 and start_y == end_y:
                    if (end_x, end_y) in walls:
                        B_R[i, i] = 1
                    else:
                        B_R[i, j] = 1

            # Up matrix
            if start_y == 0:
                if start_y == end_y and start_x == end_x:
                    B_up[i, j] = 1
            if start_y != 0:
                if end_y == start_y - 1 and start_x == end_x:
                    if (end_x, end_y) in walls:
                        B_up[i, i] = 1
                    else:
                        B_up[i, j] = 1

            # Down matrix
            if start_y == grid_size - 1:
                if start_y == end_y and start_x == end_x:
                    B_down[i, j] = 1
            if start_y != grid_size - 1:
                if end_y == start_y + 1 and start_x == end_x:
                    if (end_x, end_y) in walls:
                        B_down[i, i] = 1
                    else:
                        B_down[i, j] = 1

    B[:, :, 0] = B_L.T
    B[:, :, 1] = B_R.T
    B[:, :, 2] = B_up.T
    B[:, :, 3] = B_down.T

    return B


def generate_key_B(grid_size, key_loc, walls=[], has_pickup_action=False):
    """Generate transition matrices for key gridworld.

    In key gridworld, the state space is augmented with a binary key indicator.
    Moving to the key location automatically picks up the key (unless
    has_pickup_action=True, in which case a separate pickup action is needed).

    Args:
        grid_size: Side length of the grid
        key_loc: (x, y) location of the key
        walls: List of (x, y) wall positions
        has_pickup_action: If True, add a 5th action for picking up the key

    Returns:
        B: Transition tensor of shape (N, 2, N, 2, n_actions)
           where N = grid_size^2, n_actions = 4 or 5
           Dimensions: (end_loc, end_key, start_loc, start_key, action)
           Actions: 0=left, 1=right, 2=up, 3=down, [4=pickup]
    """
    if has_pickup_action:
        return _generate_key_B_with_pickup(grid_size, key_loc, walls)
    else:
        return _generate_key_B_auto_pickup(grid_size, key_loc, walls)


def _generate_key_B_auto_pickup(grid_size, key_loc, walls):
    """Generate B matrices where entering key location auto-picks up key."""
    N = grid_size**2
    N_size = (N, 2, N, 2)
    B_up = np.zeros(N_size)
    B_down = np.zeros(N_size)
    B_L = np.zeros(N_size)
    B_R = np.zeros(N_size)

    B = np.zeros((N, 2, N, 2, 4))

    for i in range(N):
        start_x, start_y = idloc_to_loc(i, grid_size)

        # If starting from a wall, stay in place
        if (start_x, start_y) in walls:
            for k in [0, 1]:
                B_L[i, k, i, k] = 1
                B_R[i, k, i, k] = 1
                B_up[i, k, i, k] = 1
                B_down[i, k, i, k] = 1
            continue

        for j in range(N):
            end_x, end_y = idloc_to_loc(j, grid_size)

            if (end_x, end_y) != key_loc:
                # Not entering key location - key state unchanged
                for k in [0, 1]:
                    _fill_direction_matrices(
                        B_L, B_R, B_up, B_down,
                        i, j, k, k,  # key stays same
                        start_x, start_y, end_x, end_y,
                        grid_size, walls, key_loc
                    )
            else:
                # Entering key location - always end with key=1
                _fill_key_pickup_transitions(
                    B_L, B_R, B_up, B_down,
                    i, j, start_x, start_y, end_x, end_y,
                    grid_size
                )

    B[:, :, :, :, 0] = B_L
    B[:, :, :, :, 1] = B_R
    B[:, :, :, :, 2] = B_up
    B[:, :, :, :, 3] = B_down

    return B


def _generate_key_B_with_pickup(grid_size, key_loc, walls):
    """Generate B matrices with separate pickup action."""
    N = grid_size**2
    N_size = (N, 2, N, 2)
    B_up = np.zeros(N_size)
    B_down = np.zeros(N_size)
    B_L = np.zeros(N_size)
    B_R = np.zeros(N_size)
    B_P = np.zeros(N_size)

    B = np.zeros((N, 2, N, 2, 5))

    for i in range(N):
        start_x, start_y = idloc_to_loc(i, grid_size)

        # If starting from a wall, stay in place
        if (start_x, start_y) in walls:
            for k in [0, 1]:
                B_L[i, k, i, k] = 1
                B_R[i, k, i, k] = 1
                B_up[i, k, i, k] = 1
                B_down[i, k, i, k] = 1
                B_P[i, k, i, k] = 1
            continue

        # Pickup action
        if (start_x, start_y) == key_loc:
            B_P[i, 1, i, :] = 1  # Pick up key regardless of current key state
        else:
            B_P[i, 0, i, 0] = 1  # No key -> no key
            B_P[i, 1, i, 1] = 1  # Has key -> has key

        # Movement actions - key state unchanged
        for j in range(N):
            end_x, end_y = idloc_to_loc(j, grid_size)
            for k in [0, 1]:
                _fill_movement_simple(
                    B_L, B_R, B_up, B_down,
                    i, j, k,
                    start_x, start_y, end_x, end_y,
                    grid_size, walls
                )

    B[:, :, :, :, 0] = B_L
    B[:, :, :, :, 1] = B_R
    B[:, :, :, :, 2] = B_up
    B[:, :, :, :, 3] = B_down
    B[:, :, :, :, 4] = B_P

    return B


def _fill_direction_matrices(B_L, B_R, B_up, B_down,
                              i, j, start_k, end_k,
                              start_x, start_y, end_x, end_y,
                              grid_size, walls, key_loc):
    """Fill direction matrices for a state transition."""
    # Left
    if start_x == 0:
        if start_x == end_x and start_y == end_y:
            B_L[j, end_k, i, start_k] = 1
    if start_x != 0:
        if end_x == start_x - 1 and start_y == end_y:
            if (end_x, end_y) in walls:
                # Bounce back - check if bounce location is key
                if (start_x, start_y) == key_loc:
                    B_L[i, 1, i, start_k] = 1
                else:
                    B_L[i, start_k, i, start_k] = 1
            else:
                B_L[j, end_k, i, start_k] = 1

    # Right
    if start_x == grid_size - 1:
        if start_x == end_x and start_y == end_y:
            B_R[j, end_k, i, start_k] = 1
    if start_x != grid_size - 1:
        if end_x == start_x + 1 and start_y == end_y:
            if (end_x, end_y) in walls:
                if (start_x, start_y) == key_loc:
                    B_R[i, 1, i, start_k] = 1
                else:
                    B_R[i, start_k, i, start_k] = 1
            else:
                B_R[j, end_k, i, start_k] = 1

    # Up
    if start_y == 0:
        if start_y == end_y and start_x == end_x:
            B_up[j, end_k, i, start_k] = 1
    if start_y != 0:
        if end_y == start_y - 1 and start_x == end_x:
            if (end_x, end_y) in walls:
                if (start_x, start_y) == key_loc:
                    B_up[i, 1, i, start_k] = 1
                else:
                    B_up[i, start_k, i, start_k] = 1
            else:
                B_up[j, end_k, i, start_k] = 1

    # Down
    if start_y == grid_size - 1:
        if start_y == end_y and start_x == end_x:
            B_down[j, end_k, i, start_k] = 1
    if start_y != grid_size - 1:
        if end_y == start_y + 1 and start_x == end_x:
            if (end_x, end_y) in walls:
                if (start_x, start_y) == key_loc:
                    B_down[i, 1, i, start_k] = 1
                else:
                    B_down[i, start_k, i, start_k] = 1
            else:
                B_down[j, end_k, i, start_k] = 1


def _fill_key_pickup_transitions(B_L, B_R, B_up, B_down,
                                  i, j, start_x, start_y, end_x, end_y,
                                  grid_size):
    """Fill transitions when entering key location (always pick up key)."""
    # j is key location, so can't be a wall
    # Left
    if start_x == 0:
        if start_x == end_x and start_y == end_y:
            B_L[j, 1, i, :] = 1
    if start_x != 0:
        if end_x == start_x - 1 and start_y == end_y:
            B_L[j, 1, i, :] = 1

    # Right
    if start_x == grid_size - 1:
        if start_x == end_x and start_y == end_y:
            B_R[j, 1, i, :] = 1
    if start_x != grid_size - 1:
        if end_x == start_x + 1 and start_y == end_y:
            B_R[j, 1, i, :] = 1

    # Up
    if start_y == 0:
        if start_y == end_y and start_x == end_x:
            B_up[j, 1, i, :] = 1
    if start_y != 0:
        if end_y == start_y - 1 and start_x == end_x:
            B_up[j, 1, i, :] = 1

    # Down
    if start_y == grid_size - 1:
        if start_y == end_y and start_x == end_x:
            B_down[j, 1, i, :] = 1
    if start_y != grid_size - 1:
        if end_y == start_y + 1 and start_x == end_x:
            B_down[j, 1, i, :] = 1


def _fill_movement_simple(B_L, B_R, B_up, B_down,
                          i, j, k,
                          start_x, start_y, end_x, end_y,
                          grid_size, walls):
    """Fill movement matrices without key pickup logic."""
    # Left
    if start_x == 0:
        if start_x == end_x and start_y == end_y:
            B_L[j, k, i, k] = 1
    if start_x != 0:
        if end_x == start_x - 1 and start_y == end_y:
            if (end_x, end_y) in walls:
                B_L[i, k, i, k] = 1
            else:
                B_L[j, k, i, k] = 1

    # Right
    if start_x == grid_size - 1:
        if start_x == end_x and start_y == end_y:
            B_R[j, k, i, k] = 1
    if start_x != grid_size - 1:
        if end_x == start_x + 1 and start_y == end_y:
            if (end_x, end_y) in walls:
                B_R[i, k, i, k] = 1
            else:
                B_R[j, k, i, k] = 1

    # Up
    if start_y == 0:
        if start_y == end_y and start_x == end_x:
            B_up[j, k, i, k] = 1
    if start_y != 0:
        if end_y == start_y - 1 and start_x == end_x:
            if (end_x, end_y) in walls:
                B_up[i, k, i, k] = 1
            else:
                B_up[j, k, i, k] = 1

    # Down
    if start_y == grid_size - 1:
        if start_y == end_y and start_x == end_x:
            B_down[j, k, i, k] = 1
    if start_y != grid_size - 1:
        if end_y == start_y + 1 and start_x == end_x:
            if (end_x, end_y) in walls:
                B_down[i, k, i, k] = 1
            else:
                B_down[j, k, i, k] = 1
