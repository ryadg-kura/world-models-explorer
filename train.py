"""Entry point: loads config, collects data, trains the RSSM world model, saves checkpoint."""

import yaml
import torch
import gymnasium as gym

from src.training.trainer import Trainer
from src.utils.visualization import plot_losses


def main() -> None:
    """Load config, run data collection and training, then save artefacts."""
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = gym.make("CartPole-v1")

    trainer = Trainer(
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        embedding_dim=cfg["embedding_dim"],
        action_dim=env.action_space.n,
        obs_dim=env.observation_space.shape[0],
        learning_rate=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        seq_len=cfg["seq_len"],
        kl_weight=cfg["kl_weight"],
        free_nats=cfg["free_nats"],
        buffer_capacity=cfg["buffer_capacity"],
        device=device,
    )

    print(f"Collecting {cfg['collect_steps']} steps of random experience …")
    trainer.collect_data(env, n_steps=cfg["collect_steps"])
    print(f"Buffer size: {len(trainer.buffer)} transitions")

    print(f"\nTraining for {cfg['n_epochs']} epochs …\n")
    trainer.train(n_epochs=cfg["n_epochs"], steps_per_epoch=100)

    trainer.save_checkpoint("outputs/checkpoint.pt")
    plot_losses(trainer.loss_history)

    env.close()
    print("\nDone. Checkpoint and plots saved to ./outputs/")


if __name__ == "__main__":
    main()
