"""MLP decoder reconstructing observations from the latent state (h, z)."""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """Reconstructs CartPole observations from the concatenated deterministic and stochastic state."""

    def __init__(
        self,
        hidden_dim: int = 200,
        latent_dim: int = 32,
        obs_dim: int = 4,
    ) -> None:
        super().__init__()
        input_dim = hidden_dim + latent_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, obs_dim),
        )

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Decode latent state (h, z) into a reconstructed observation."""
        latent = torch.cat([h, z], dim=-1)
        return self.network(latent)
