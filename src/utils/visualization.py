"""Visualization utilities for training losses and imagination comparisons."""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List

OUTPUT_DIR = "./outputs"


def _ensure_output_dir() -> None:
    """Create the outputs directory if it does not exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_losses(loss_history: Dict[str, List[float]], filename: str = "losses.png") -> None:
    """Plot total and component losses over training epochs and save to outputs/."""
    _ensure_output_dir()
    epochs = range(1, len(loss_history["total"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    titles = ["Total Loss", "Reconstruction Loss", "KL Loss", "Reward Loss"]
    keys = ["total", "recon", "kl", "reward"]

    for ax, title, key in zip(axes.flat, titles, keys):
        ax.plot(epochs, loss_history[key], linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    fig.suptitle("World Model Training Losses", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Loss plot saved to {save_path}")


def plot_imagination(
    real_traj: np.ndarray,
    imagined_traj: np.ndarray,
    filename: str = "imagination.png",
) -> None:
    """
    Plot real vs imagined CartPole state trajectories side by side.

    real_traj / imagined_traj: arrays of shape (T, 4) — [cart_pos, cart_vel, pole_angle, pole_vel].
    """
    _ensure_output_dir()

    feature_names = ["Cart Position", "Cart Velocity", "Pole Angle", "Pole Angular Velocity"]
    T_real = real_traj.shape[0]
    T_imag = imagined_traj.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for i, (ax, name) in enumerate(zip(axes.flat, feature_names)):
        ax.plot(range(T_real), real_traj[:, i], label="Real", linewidth=2, color="steelblue")
        ax.plot(
            range(T_imag),
            imagined_traj[:, i],
            label="Imagined",
            linewidth=2,
            linestyle="--",
            color="darkorange",
        )
        ax.set_title(name)
        ax.set_xlabel("Timestep")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Real vs Imagined Trajectories", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Imagination plot saved to {save_path}")
