"""Utilities for neural SR training.

Includes target network update functions and observation normalization.
"""

import numpy as np
import torch
import torch.nn as nn


def soft_update(target_net: nn.Module, online_net: nn.Module, tau: float = 0.005):
    """Polyak averaging: θ_target = τ·θ_online + (1-τ)·θ_target.

    Args:
        target_net: Target network to update.
        online_net: Online network to copy from.
        tau: Interpolation coefficient (small = slow update).
    """
    for tp, op in zip(target_net.parameters(), online_net.parameters()):
        tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)


def hard_update(target_net: nn.Module, online_net: nn.Module):
    """Copy all parameters from online to target network.

    Args:
        target_net: Target network to overwrite.
        online_net: Source network.
    """
    target_net.load_state_dict(online_net.state_dict())


class RunningMeanStd:
    """Running mean and standard deviation for observation normalization.

    Useful for environments with varying observation scales (e.g., MuJoCo).

    Args:
        shape: Shape of the observation to normalize.
        epsilon: Small constant for numerical stability.
    """

    def __init__(self, shape: tuple = (), epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
        self._epsilon = epsilon

    def update(self, batch: np.ndarray):
        """Update running statistics with a batch of observations.

        Args:
            batch: Array of observations, shape (batch_size, *shape).
        """
        batch_mean = np.mean(batch, axis=0)
        batch_var = np.var(batch, axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        """Normalize an observation using running statistics.

        Args:
            obs: Observation to normalize.

        Returns:
            Normalized observation with zero mean and unit variance.
        """
        return (obs - self.mean) / np.sqrt(self.var + self._epsilon)
