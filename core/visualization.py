"""Visualization methods for Hierarchical SR Agent.

This module provides plotting and video generation capabilities for
visualizing the agent's learned representations and behavior.

The implementation is split across sub-mixins in core/viz/ and composed
here via multiple inheritance.
"""

from core.viz.matrices import MatrixVizMixin
from core.viz.trajectories import TrajectoryVizMixin
from core.viz.pomdp import POMDPVizMixin


class VisualizationMixin(MatrixVizMixin, TrajectoryVizMixin, POMDPVizMixin):
    """Composed visualization mixin.

    This mixin adds plotting capabilities for:
    - Transition and successor matrices (B, M)
    - Macro state clusters
    - Action trajectories
    - Policy visualization
    - Episode videos
    - Value functions and POMDP diagnostics

    Requires the agent to have:
    - self.adapter: Environment adapter
    - self.B, self.M: Transition and successor matrices
    - self.B_macro, self.M_macro: Macro-level matrices
    - self.macro_state_list, self.micro_to_macro: Clustering results
    - self.state_history, self.action_history: Episode tracking
    - self.C, self.goal_states: Goal information
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional attributes for visualization
        self.labels_grid = None  # Grid of macro state labels
        self.spectral_positions = None  # Spectral embedding positions
