"""Fixed-capacity replay buffer storing sequences for RSSM training."""

import numpy as np
import torch
from typing import Dict, Optional


class ReplayBuffer:
    """Stores (obs, action, reward, done) transitions and samples fixed-length sequences."""

    def __init__(self, capacity: int = 10_000, obs_dim: int = 4, action_dim: int = 2) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._ptr = 0
        self._size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        """Add a single transition to the buffer, overwriting oldest entry when full."""
        self.obs[self._ptr] = obs
        self.actions[self._ptr] = action
        self.rewards[self._ptr] = reward
        self.dones[self._ptr] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int, seq_len: int, device: Optional[torch.device] = None
    ) -> Dict[str, torch.Tensor]:
        """Sample a batch of non-overlapping sequences; returns tensors of shape (seq_len, batch, ...)."""
        assert self._size >= seq_len, "Not enough data in buffer to sample a sequence."

        max_start = self._size - seq_len
        starts = np.random.randint(0, max_start, size=batch_size)

        obs_batch = np.stack([self.obs[s : s + seq_len] for s in starts], axis=1)
        act_batch = np.stack([self.actions[s : s + seq_len] for s in starts], axis=1)
        rew_batch = np.stack([self.rewards[s : s + seq_len] for s in starts], axis=1)
        done_batch = np.stack([self.dones[s : s + seq_len] for s in starts], axis=1)

        def to_tensor(x: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(x)
            return t.to(device) if device is not None else t

        return {
            "obs": to_tensor(obs_batch),           # (seq_len, batch, obs_dim)
            "actions": to_tensor(act_batch),        # (seq_len, batch, action_dim)
            "rewards": to_tensor(rew_batch),        # (seq_len, batch)
            "dones": to_tensor(done_batch),         # (seq_len, batch)
        }

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return self._size
