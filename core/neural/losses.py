"""Loss functions for Successor Feature learning.

The SF Bellman equation: φ(s,a) = ψ(s) + γ · φ(s', a')
where ψ(s) is the immediate reward feature and φ(s,a) is the successor feature.

This is the neural analog of the tabular TD update:
    M(s) += lr * [I(s') + γ · M(s') - M(s)]
where ψ(s') replaces I(s') and the loss gradient replaces the explicit TD update.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def sf_td_loss(
    sf_current: torch.Tensor,
    reward_features: torch.Tensor,
    sf_next_target: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """TD loss for successor feature learning.

    Computes the MSE between current SFs and their Bellman targets:
        target = ψ(s') + γ · (1 - done) · φ_target(s', a')
        loss = MSE(φ(s, a), target)

    Args:
        sf_current: Current successor features φ(s, a), shape (batch, sf_dim).
        reward_features: Reward features ψ(s') for next states, shape (batch, sf_dim).
        sf_next_target: Target successor features φ_target(s', a'),
            shape (batch, sf_dim).
        dones: Terminal flags, shape (batch,).
        gamma: Discount factor.

    Returns:
        Scalar MSE loss.
    """
    target = reward_features + gamma * (1.0 - dones.unsqueeze(-1)) * sf_next_target
    return F.mse_loss(sf_current, target.detach())


def sf_td_loss_per_sample(
    sf_current: torch.Tensor,
    reward_features: torch.Tensor,
    sf_next_target: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """TD loss for SF learning with per-sample errors and IS weighting.

    Returns both the (optionally weighted) scalar loss for backprop and
    the per-sample TD error norms for priority updates.

    Args:
        sf_current: Current successor features, shape (batch, sf_dim).
        reward_features: Reward features for next states, shape (batch, sf_dim).
        sf_next_target: Target successor features, shape (batch, sf_dim).
        dones: Terminal flags, shape (batch,).
        gamma: Discount factor.
        weights: Importance sampling weights, shape (batch,). If None,
            uniform weighting (equivalent to standard MSE).

    Returns:
        Tuple of (loss, td_error_norms):
            loss: Scalar loss for backprop.
            td_error_norms: Per-sample L2 norms of TD error vectors,
                shape (batch,). Detached from computation graph.
    """
    target = reward_features + gamma * (1.0 - dones.unsqueeze(-1)) * sf_next_target
    td_errors = sf_current - target.detach()  # (batch, sf_dim)

    # Per-sample L2 norm for priority computation
    td_error_norms = td_errors.detach().norm(dim=-1)  # (batch,)

    # Per-sample MSE (mean across sf_dim to match F.mse_loss scale)
    per_sample_loss = td_errors.pow(2).mean(dim=-1)  # (batch,)

    if weights is not None:
        loss = (weights * per_sample_loss).mean()
    else:
        loss = per_sample_loss.mean()

    return loss, td_error_norms


def reward_prediction_loss(
    predicted: torch.Tensor,
    actual: torch.Tensor,
) -> torch.Tensor:
    """Loss for reward weight learning: r(s) ≈ ψ(s)ᵀ · w.

    Args:
        predicted: Predicted rewards ψ(s)ᵀ · w, shape (batch,).
        actual: Actual rewards r(s), shape (batch,).

    Returns:
        Scalar MSE loss.
    """
    return F.mse_loss(predicted, actual)
