"""Unified environment framework for gridworld variants."""

from .base import BaseGridworld
from .standard import StandardGridworld
from .key_gridworld import KeyGridworld
from .utils import (
    # Basic conversions
    loc_to_idx,
    idx_to_loc,
    idx_to_onehot,
    onehot_to_idx,
    onehot_to_loc,
    loc_to_onehot,
    # Math utilities
    log_stable,
    entropy,
    softmax,
    kl_divergence,
    # Misc
    create_walls,
    plot_beliefs,
)

__all__ = [
    # Environment classes
    'BaseGridworld',
    'StandardGridworld',
    'KeyGridworld',
    # Utilities
    'loc_to_idx',
    'idx_to_loc',
    'idx_to_onehot',
    'onehot_to_idx',
    'onehot_to_loc',
    'loc_to_onehot',
    'log_stable',
    'entropy',
    'softmax',
    'kl_divergence',
    'create_walls',
    'plot_beliefs',
]
