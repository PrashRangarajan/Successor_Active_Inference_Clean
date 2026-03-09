"""PointMaze environment adapter."""

from .adapter import PointMazeAdapter
from .wrappers import make_vec_env

__all__ = ['PointMazeAdapter', 'make_vec_env']
