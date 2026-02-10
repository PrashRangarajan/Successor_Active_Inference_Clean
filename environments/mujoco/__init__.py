"""MuJoCo environment adapters for neural SR agents.

Provides direct continuous-state adapters (no binning) for MuJoCo
environments. These adapters implement the same interface as
ContinuousAdapter but without the BinnedContinuousAdapter layer,
since the neural agent operates entirely in continuous space.
"""

from .base_adapter import MuJoCoAdapter
from .inverted_pendulum import InvertedPendulumAdapter
from .half_cheetah import HalfCheetahAdapter

__all__ = [
    'MuJoCoAdapter',
    'InvertedPendulumAdapter',
    'HalfCheetahAdapter',
]
