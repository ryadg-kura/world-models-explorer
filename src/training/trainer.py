"""Trainer orchestrating data collection, world-model training, and checkpointing."""

import os
import torch
import numpy as np
import gymnasium as gym
from typing import Dict, Optional

from src.models import RSSM, Encoder, Decoder, RewardModel
from src.training.replay_buffer import ReplayBuffer
from src.training.losses import total_loss


class Trainer:
    """Manages the full training loop: data collection, gradient updates, and checkpoints."""

    def __init__(
        self,
        hidden_dim: int = 200,
        latent_dim: int = 32,
        embedding_dim: int = 64,
        action_dim: int = 2,
        obs_dim: int = 4,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        seq_len: int = 50,
        kl_weight: float = 1.0,
        free_nats: float = 3.0,
        buffer_capacity: int = 10_000,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.kl_weight = kl_weight
        self.free_nats = free_nats
        self.action_dim = action_dim

        # Models
        self.encoder = Encoder(obs_dim=obs_dim, embedding_dim=embedding_dim).to(self.device)
        self.rssm = RSSM(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            embedding_dim=embedding_dim,
            action_dim=action_dim,
        ).to(self.device)
        self.decoder = Decoder(hidden_dim=hidden_dim, latent_dim=latent_dim, obs_dim=obs_dim).to(self.device)
        self.reward_model = RewardModel(hidden_dim=hidden_dim, latent_dim=latent_dim).to(self.device)

        params = (
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_model.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)

        self.buffer = ReplayBuffer(
            capacity=buffer_capacity,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        self.loss_history: Dict[str, list] = {
            "total": [],
            "recon": [],
            "kl": [],
            "reward": [],
        }

    # ------------------------------------------------------------------
    # data collection
    # ------------------------------------------------------------------

    def collect_data(self, env: gym.Env, n_steps: int = 1000) -> None:
        """Roll out a random policy for n_steps and store transitions in the replay buffer."""
        obs, _ = env.reset()
        for _ in range(n_steps):
            action_idx = env.action_space.sample()
            action_onehot = np.zeros(self.action_dim, dtype=np.float32)
            action_onehot[action_idx] = 1.0

            next_obs, reward, terminated, truncated, _ = env.step(action_idx)
            done = terminated or truncated
            self.buffer.add(obs, action_onehot, float(reward), done)

            obs = next_obs if not done else env.reset()[0]

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Forward pass, loss computation, backward pass, and gradient clipping for one batch."""
        obs_seq = batch["obs"]       # (T, B, obs_dim)
        act_seq = batch["actions"]   # (T, B, action_dim)
        rew_seq = batch["rewards"]   # (T, B)

        T, B, _ = obs_seq.shape

        state = self.rssm.initial_state(B, self.device)

        all_obs_pred, all_rew_pred = [], []
        all_post_mean, all_post_std = [], []
        all_prior_mean, all_prior_std = [], []

        for t in range(T):
            embedding = self.encoder(obs_seq[t])
            state, stats = self.rssm.step(state, act_seq[t], embedding)

            obs_pred = self.decoder(state["h"], state["z"])
            rew_pred = self.reward_model(state["h"], state["z"])

            all_obs_pred.append(obs_pred)
            all_rew_pred.append(rew_pred)
            all_post_mean.append(stats["post_mean"])
            all_post_std.append(stats["post_std"])
            all_prior_mean.append(stats["prior_mean"])
            all_prior_std.append(stats["prior_std"])

        obs_pred_t = torch.stack(all_obs_pred)       # (T, B, obs_dim)
        rew_pred_t = torch.stack(all_rew_pred)        # (T, B, 1)
        post_mean_t = torch.stack(all_post_mean)      # (T, B, latent_dim)
        post_std_t = torch.stack(all_post_std)
        prior_mean_t = torch.stack(all_prior_mean)
        prior_std_t = torch.stack(all_prior_std)

        # Flatten T and B dims for loss
        def flat(x: torch.Tensor) -> torch.Tensor:
            return x.reshape(-1, x.shape[-1])

        losses = total_loss(
            obs_pred=flat(obs_pred_t),
            obs_target=flat(obs_seq),
            post_mean=flat(post_mean_t),
            post_std=flat(post_std_t),
            prior_mean=flat(prior_mean_t),
            prior_std=flat(prior_std_t),
            reward_pred=rew_pred_t.reshape(-1, 1),
            reward_target=rew_seq.reshape(-1),
            kl_weight=self.kl_weight,
            free_nats=self.free_nats,
        )

        self.optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.rssm.parameters())
            + list(self.decoder.parameters())
            + list(self.reward_model.parameters()),
            max_norm=100.0,
        )
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def train(self, n_epochs: int = 50, steps_per_epoch: int = 100) -> None:
        """Main training loop: run steps_per_epoch gradient steps per epoch."""
        for epoch in range(1, n_epochs + 1):
            epoch_losses: Dict[str, list] = {"total": [], "recon": [], "kl": [], "reward": []}

            for _ in range(steps_per_epoch):
                batch = self.buffer.sample(self.batch_size, self.seq_len, device=self.device)
                step_losses = self.train_step(batch)
                for k, v in step_losses.items():
                    epoch_losses[k].append(v)

            for k in self.loss_history:
                mean_val = float(np.mean(epoch_losses[k]))
                self.loss_history[k].append(mean_val)

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:3d}/{n_epochs} | "
                    f"Loss: {self.loss_history['total'][-1]:.4f} | "
                    f"Recon: {self.loss_history['recon'][-1]:.4f} | "
                    f"KL: {self.loss_history['kl'][-1]:.4f} | "
                    f"Reward: {self.loss_history['reward'][-1]:.4f}"
                )

    # ------------------------------------------------------------------
    # checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save all model weights, optimizer state, and loss history to a file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "rssm": self.rssm.state_dict(),
                "decoder": self.decoder.state_dict(),
                "reward_model": self.reward_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "loss_history": self.loss_history,
            },
            path,
        )
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model weights, optimizer state, and loss history from a checkpoint file."""
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.rssm.load_state_dict(ckpt["rssm"])
        self.decoder.load_state_dict(ckpt["decoder"])
        self.reward_model.load_state_dict(ckpt["reward_model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.loss_history = ckpt["loss_history"]
        print(f"Checkpoint loaded from {path}")
