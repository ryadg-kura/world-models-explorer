"""MLP encoder mapping raw CartPole observations to a latent embedding."""

import torch
import torch.nn as nn
from typing import Tuple


class Encoder(nn.Module):
    """Maps raw CartPole observations (4D) to a dense embedding vector."""

    def __init__(self, obs_dim: int = 4, embedding_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a batch of observations into embeddings."""
        return self.network(obs)
