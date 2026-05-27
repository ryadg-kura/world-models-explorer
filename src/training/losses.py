"""Loss functions for RSSM world-model training."""

import torch
import torch.nn.functional as F
from typing import Dict


def reconstruction_loss(obs_pred: torch.Tensor, obs_target: torch.Tensor) -> torch.Tensor:
    """MSE between decoded observations and ground-truth observations."""
    return F.mse_loss(obs_pred, obs_target)


def kl_loss(
    post_mean: torch.Tensor,
    post_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
    free_nats: float = 3.0,
) -> torch.Tensor:
    """KL divergence KL(posterior || prior) with free-nats lower bound."""
    # Analytical KL between two Gaussians
    kl = (
        torch.log(prior_std / post_std)
        + (post_std**2 + (post_mean - prior_mean) ** 2) / (2 * prior_std**2)
        - 0.5
    )
    kl = kl.sum(dim=-1)  # sum over latent dims
    # Free nats: only penalise KL above the free-nats threshold
    kl = torch.clamp(kl, min=free_nats)
    return kl.mean()


def reward_loss(reward_pred: torch.Tensor, reward_target: torch.Tensor) -> torch.Tensor:
    """MSE between predicted and ground-truth rewards."""
    return F.mse_loss(reward_pred.squeeze(-1), reward_target)


def total_loss(
    obs_pred: torch.Tensor,
    obs_target: torch.Tensor,
    post_mean: torch.Tensor,
    post_std: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_std: torch.Tensor,
    reward_pred: torch.Tensor,
    reward_target: torch.Tensor,
    kl_weight: float = 1.0,
    free_nats: float = 3.0,
) -> Dict[str, torch.Tensor]:
    """Compute and return all losses as a dict with keys: total, recon, kl, reward."""
    recon = reconstruction_loss(obs_pred, obs_target)
    kl = kl_loss(post_mean, post_std, prior_mean, prior_std, free_nats)
    rew = reward_loss(reward_pred, reward_target)
    total = recon + kl_weight * kl + rew
    return {"total": total, "recon": recon, "kl": kl, "reward": rew}
