"""Neural network architectures for Successor Feature learning.

The core factorization is Q(s,a) = φ(s,a)ᵀ · w where:
- φ(s,a) are successor features learned by SFNetwork
- w is a reward weight vector learned to satisfy r(s) ≈ ψ(s)ᵀ · w
- ψ(s) are reward features learned by RewardFeatureNetwork

This replaces the tabular M @ C computation with neural function approximation.
"""

from typing import Tuple

import torch
import torch.nn as nn


def _build_mlp(input_dim: int, output_dim: int,
               hidden_sizes: Tuple[int, ...] = (256, 256),
               activation: str = 'relu') -> nn.Sequential:
    """Build a multi-layer perceptron.

    Args:
        input_dim: Input dimensionality.
        output_dim: Output dimensionality.
        hidden_sizes: Tuple of hidden layer widths.
        activation: Activation function ('relu' or 'tanh').

    Returns:
        nn.Sequential MLP module.
    """
    act_fn = nn.ReLU if activation == 'relu' else nn.Tanh
    layers = []
    prev_dim = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(act_fn())
        prev_dim = h
    layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class SFNetwork(nn.Module):
    """Successor Feature network for discrete action spaces.

    Maps an observation to successor features for each action:
        obs → shared encoder → per-action SF heads → [φ_a1, ..., φ_aK]

    Each φ_ai ∈ R^sf_dim. The Q-value for action ai is Q(s, ai) = φ(s, ai)ᵀ · w.

    This is the neural analog of a row lookup M[s, :] in the tabular SR,
    extended to produce action-conditioned features.

    Args:
        obs_dim: Dimensionality of raw observations.
        n_actions: Number of discrete actions.
        sf_dim: Dimensionality of successor features.
        hidden_sizes: Hidden layer sizes for the shared encoder.
    """

    def __init__(self, obs_dim: int, n_actions: int, sf_dim: int = 128,
                 hidden_sizes: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.sf_dim = sf_dim

        # Shared feature encoder
        self.encoder = _build_mlp(obs_dim, hidden_sizes[-1], hidden_sizes[:-1])

        # Per-action successor feature heads
        self.sf_heads = nn.ModuleList([
            nn.Linear(hidden_sizes[-1], sf_dim) for _ in range(n_actions)
        ])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute successor features for all actions.

        Args:
            obs: Observations, shape (batch, obs_dim).

        Returns:
            Successor features, shape (batch, n_actions, sf_dim).
        """
        h = self.encoder(obs)
        sf_list = [head(h) for head in self.sf_heads]
        return torch.stack(sf_list, dim=1)

    def get_sf(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Get successor features for specific actions.

        Args:
            obs: Observations, shape (batch, obs_dim).
            actions: Action indices, shape (batch,) with dtype long.

        Returns:
            Successor features for selected actions, shape (batch, sf_dim).
        """
        all_sf = self.forward(obs)  # (batch, n_actions, sf_dim)
        batch_idx = torch.arange(obs.shape[0], device=obs.device)
        return all_sf[batch_idx, actions]


class RewardFeatureNetwork(nn.Module):
    """Reward feature network: maps observations to reward features ψ(s).

    The reward at state s is approximated as r(s) ≈ ψ(s)ᵀ · w, where w is
    a learned reward weight vector.

    In the tabular TD update M(s) += lr * [I(s') + γM(s') - M(s)], the
    indicator I(s') picks out which state was visited. In the neural version,
    ψ(s') plays this role — it provides the immediate "identity" signal that
    the successor features accumulate over time.

    Args:
        obs_dim: Dimensionality of raw observations.
        feature_dim: Output dimensionality (must match sf_dim).
        hidden_sizes: Hidden layer sizes.
    """

    def __init__(self, obs_dim: int, feature_dim: int,
                 hidden_sizes: Tuple[int, ...] = (128, 128)):
        super().__init__()
        self.obs_dim = obs_dim
        self.feature_dim = feature_dim
        self.net = _build_mlp(obs_dim, feature_dim, hidden_sizes)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute reward features.

        Args:
            obs: Observations, shape (batch, obs_dim).

        Returns:
            Reward features ψ(s), shape (batch, feature_dim).
        """
        return self.net(obs)


class StateEncoder(nn.Module):
    """General-purpose state encoder for observation embedding.

    Maps raw observations to a latent space. Can be used as a shared backbone
    or independently for clustering successor feature embeddings into macro
    states (Phase 3).

    Args:
        obs_dim: Dimensionality of raw observations.
        latent_dim: Dimensionality of the latent representation.
        hidden_sizes: Hidden layer sizes.
    """

    def __init__(self, obs_dim: int, latent_dim: int,
                 hidden_sizes: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.net = _build_mlp(obs_dim, latent_dim, hidden_sizes)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations into latent space.

        Args:
            obs: Observations, shape (batch, obs_dim).

        Returns:
            Latent representations, shape (batch, latent_dim).
        """
        return self.net(obs)
