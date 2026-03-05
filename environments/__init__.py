"""Environment adapters for Hierarchical SR Active Inference."""

from .binned_continuous_adapter import BinnedContinuousAdapter
from .gridworld import GridworldAdapter
from .mountain_car import MountainCarAdapter
from .key_gridworld import KeyGridworldAdapter
from .pomdp_gridworld import POMDPGridworldAdapter
from .acrobot import AcrobotAdapter
from .pendulum import PendulumAdapter
from .cartpole import CartPoleAdapter
from .point_maze import PointMazeAdapter

__all__ = [
    'BinnedContinuousAdapter',
    'GridworldAdapter',
    'MountainCarAdapter',
    'KeyGridworldAdapter',
    'POMDPGridworldAdapter',
    'AcrobotAdapter',
    'PendulumAdapter',
    'CartPoleAdapter',
    'PointMazeAdapter',
]

# MuJoCo adapters (optional — requires mujoco package)
try:
    from .mujoco import MuJoCoAdapter, InvertedPendulumAdapter, HalfCheetahAdapter
    __all__ += ['MuJoCoAdapter', 'InvertedPendulumAdapter', 'HalfCheetahAdapter']
except ImportError:
    pass
