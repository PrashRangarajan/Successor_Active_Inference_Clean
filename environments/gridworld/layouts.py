"""Predefined gridworld layout configurations.

Each layout defines walls, default goals, and recommended clustering
parameters for use across example and evaluation scripts.

Usage:
    from environments.gridworld import get_layout, AVAILABLE_LAYOUTS

    layout = get_layout("fourrooms", grid_size=9)
    walls = layout.walls
    goal  = layout.default_goal
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class GridworldLayout:
    """Configuration for a named gridworld layout."""
    walls: List[Tuple[int, int]]
    default_goal: Tuple[int, int]   # primary goal (single-goal scripts)
    alt_goal: Tuple[int, int]       # secondary goal (goal-revaluation)
    n_clusters: int                 # recommended number of macro clusters
    third_goal: Tuple[int, int] = None  # optional third goal for multi-phase experiments


AVAILABLE_LAYOUTS = ["serpentine", "fourrooms", "fiverooms"]


def get_layout(name: str, grid_size: int = 9) -> GridworldLayout:
    """Return layout configuration for a named gridworld layout.

    Args:
        name: One of 'serpentine', 'fourrooms', 'fiverooms'
        grid_size: Side length of the grid (default 9)

    Returns:
        GridworldLayout with walls, goals, and cluster count

    Raises:
        ValueError: If name is not a recognized layout
    """
    if name == "serpentine":
        walls = (
            [(1, x) for x in range(grid_size // 2 + 2)]
            + [(3, x) for x in range(grid_size // 2 - 2, grid_size)]
            + [(5, x) for x in range(grid_size // 2 + 2)]
            + [(7, x) for x in range(grid_size // 2 - 2, grid_size)]
        )
        return GridworldLayout(
            walls=walls,
            default_goal=(grid_size - 1, grid_size - 1),
            alt_goal=(0, grid_size - 1),
            n_clusters=4,
            third_goal=(grid_size - 1, 0),
        )
    elif name == "fourrooms":
        walls = (
            [(4, y) for y in range(grid_size) if y not in [2, 6]]
            + [(x, 4) for x in range(grid_size) if x not in [2, 6]]
        )
        return GridworldLayout(
            walls=walls,
            default_goal=(grid_size - 1, grid_size - 1),
            alt_goal=(grid_size - 1, 0),
            n_clusters=4,
            third_goal=(0, grid_size - 1),
        )
    elif name == "fiverooms":
        # Two large rooms on top separated by a vertical wall at col 4,
        # a horizontal wall at row 4 spanning most of the width, and
        # three smaller rooms on the bottom separated by vertical walls
        # at cols 2 and 6.
        walls = (
            [(x, 2) for x in range(grid_size) if x in [4, 5, 7, 8]]
            + [(x, 6) for x in range(grid_size) if x in [4, 5, 7, 8]]
            + [(x, 4) for x in range(grid_size) if x in [0, 1, 3]]
            + [(4, x) for x in range(grid_size) if x not in [0, 8]]
        )
        return GridworldLayout(
            walls=walls,
            default_goal=(grid_size - 1, grid_size - 1),
            alt_goal=(grid_size - 1, 0),
            n_clusters=5,
            third_goal=(0, grid_size - 1),
        )
    else:
        raise ValueError(
            f"Unknown layout: {name}. Choose from: {', '.join(AVAILABLE_LAYOUTS)}"
        )
