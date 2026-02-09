"""Loss functions for Successor Feature learning.

The SF Bellman equation: φ(s,a) = ψ(s) + γ · φ(s', a')
where ψ(s) is the immediate reward feature and φ(s,a) is the successor feature.

This is the neural analog of the tabular TD update:
    M(s) += lr * [I(s') + γ · M(s') - M(s)]
where ψ(s') replaces I(s') and the loss gradient replaces the explicit TD update.
"""

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
