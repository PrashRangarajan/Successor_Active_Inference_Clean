"""Neural Successor Representation modules.

Provides neural network-based successor feature learning as an alternative
to the tabular SR approach, enabling scaling to continuous and high-dimensional
state spaces.
"""

from .networks import SFNetwork, RewardFeatureNetwork, StateEncoder
from .replay_buffer import ReplayBuffer
from .losses import sf_td_loss, reward_prediction_loss
from .utils import soft_update, hard_update
from .clustering import SFClustering
from .hierarchical_agent import HierarchicalNeuralSRAgent

__all__ = [
    "SFNetwork",
    "RewardFeatureNetwork",
    "StateEncoder",
    "ReplayBuffer",
    "sf_td_loss",
    "reward_prediction_loss",
    "soft_update",
    "hard_update",
    "SFClustering",
    "HierarchicalNeuralSRAgent",
]
