"""Recurrent State Space Model (RSSM) combining deterministic GRU and stochastic Gaussian states."""

import torch
import torch.nn as nn
from typing import Dict, Tuple


State = Dict[str, torch.Tensor]


class RSSM(nn.Module):
    """
    RSSM with a GRU deterministic path and a Gaussian stochastic latent variable.

    State dict keys:
        h: deterministic hidden state  (batch, hidden_dim)
        z: stochastic sample           (batch, latent_dim)
    """

    def __init__(
        self,
        hidden_dim: int = 200,
        latent_dim: int = 32,
        embedding_dim: int = 64,
        action_dim: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # GRU input: [z_{t-1}, a_{t-1}]
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        # Prior p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2),
        )

        # Posterior q(z_t | h_t, e_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2),
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _split_stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Split a (..., 2*latent_dim) tensor into (mean, std) pairs."""
        mean, log_std = x.chunk(2, dim=-1)
        std = torch.nn.functional.softplus(log_std) + 1e-5
        return mean, std

    def _sample(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Draw a reparameterised sample from N(mean, std)."""
        return mean + std * torch.randn_like(mean)

    def initial_state(self, batch_size: int, device: torch.device) -> State:
        """Return a zeroed initial (h, z) state."""
        h = torch.zeros(batch_size, self.hidden_dim, device=device)
        z = torch.zeros(batch_size, self.latent_dim, device=device)
        return {"h": h, "z": z}

    # ------------------------------------------------------------------
    # core computations
    # ------------------------------------------------------------------

    def prior(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute prior distribution p(z|h) and sample z. Returns (z, mean, std)."""
        stats = self.prior_net(h)
        mean, std = self._split_stats(stats)
        z = self._sample(mean, std)
        return z, mean, std

    def posterior(
        self, h: torch.Tensor, embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute posterior q(z|h, e) and sample z. Returns (z, mean, std)."""
        stats = self.posterior_net(torch.cat([h, embedding], dim=-1))
        mean, std = self._split_stats(stats)
        z = self._sample(mean, std)
        return z, mean, std

    def step(
        self,
        prev_state: State,
        action: torch.Tensor,
        embedding: torch.Tensor,
    ) -> Tuple[State, Dict[str, torch.Tensor]]:
        """
        Single RSSM step with a real observation embedding (posterior).

        Returns updated state and a dict of distribution statistics for loss computation.
        """
        h_prev, z_prev = prev_state["h"], prev_state["z"]

        # One-hot encode discrete action
        a = action
        gru_input = torch.cat([z_prev, a], dim=-1)
        h = self.gru(gru_input, h_prev)

        z_post, post_mean, post_std = self.posterior(h, embedding)
        _, prior_mean, prior_std = self.prior(h)

        state = {"h": h, "z": z_post}
        stats = {
            "post_mean": post_mean,
            "post_std": post_std,
            "prior_mean": prior_mean,
            "prior_std": prior_std,
        }
        return state, stats

    def imagine(
        self,
        action: torch.Tensor,
        prev_state: State,
        steps: int,
    ) -> Tuple[list[State], list[torch.Tensor]]:
        """
        Imagine future states for `steps` using only the prior (no observations).

        `action` shape: (batch, action_dim) — repeated each step for simplicity.
        Returns list of states and list of prior z samples.
        """
        states: list[State] = []
        z_samples: list[torch.Tensor] = []

        state = prev_state
        for _ in range(steps):
            h_prev, z_prev = state["h"], state["z"]
            gru_input = torch.cat([z_prev, action], dim=-1)
            h = self.gru(gru_input, h_prev)
            z, _, _ = self.prior(h)
            state = {"h": h, "z": z}
            states.append(state)
            z_samples.append(z)

        return states, z_samples
