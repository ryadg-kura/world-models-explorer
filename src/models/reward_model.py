"""MLP reward model predicting scalar reward from the latent state (h, z)."""

import torch
import torch.nn as nn


class RewardModel(nn.Module):
    """Predicts scalar reward from the concatenated deterministic and stochastic latent state."""

    def __init__(self, hidden_dim: int = 200, latent_dim: int = 32) -> None:
        super().__init__()
        input_dim = hidden_dim + latent_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Predict reward from latent state (h, z), returning shape (..., 1)."""
        latent = torch.cat([h, z], dim=-1)
        return self.network(latent)
