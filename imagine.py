"""Load a trained checkpoint and compare a real CartPole trajectory with an imagined one."""

import torch
import numpy as np
import gymnasium as gym
import yaml

from src.training.trainer import Trainer
from src.utils.visualization import plot_imagination

IMAGINE_STEPS = 30
WARMUP_STEPS = 5  # number of real steps used to seed the RSSM hidden state


def collect_real_trajectory(env: gym.Env, n_steps: int = IMAGINE_STEPS + WARMUP_STEPS) -> tuple:
    """Roll out a random policy and return (obs_list, action_list)."""
    obs_list, act_list = [], []
    obs, _ = env.reset()
    for _ in range(n_steps):
        action_idx = env.action_space.sample()
        obs_list.append(obs.copy())
        act_list.append(action_idx)
        obs, _, terminated, truncated, _ = env.step(action_idx)
        if terminated or truncated:
            obs, _ = env.reset()
    return np.array(obs_list, dtype=np.float32), act_list


def main() -> None:
    """Load checkpoint, collect a real trajectory, imagine the future, and save a comparison plot."""
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("CartPole-v1")
    action_dim = env.action_space.n

    trainer = Trainer(
        hidden_dim=cfg["hidden_dim"],
        latent_dim=cfg["latent_dim"],
        embedding_dim=cfg["embedding_dim"],
        action_dim=action_dim,
        obs_dim=env.observation_space.shape[0],
        device=device,
    )
    trainer.load_checkpoint("outputs/checkpoint.pt")

    trainer.encoder.eval()
    trainer.rssm.eval()
    trainer.decoder.eval()

    obs_arr, act_list = collect_real_trajectory(env)
    real_traj = obs_arr[WARMUP_STEPS:]  # keep only the part after warmup

    with torch.no_grad():
        # Warm up the RSSM on the first WARMUP_STEPS real observations
        state = trainer.rssm.initial_state(1, device)
        for t in range(WARMUP_STEPS):
            obs_t = torch.from_numpy(obs_arr[t]).unsqueeze(0).to(device)
            action_idx = act_list[t]
            action_onehot = torch.zeros(1, action_dim, device=device)
            action_onehot[0, action_idx] = 1.0
            embedding = trainer.encoder(obs_t)
            state, _ = trainer.rssm.step(state, action_onehot, embedding)

        # Imagine forward using the last action repeated
        last_action = torch.zeros(1, action_dim, device=device)
        last_action[0, act_list[WARMUP_STEPS - 1]] = 1.0

        imagined_states, _ = trainer.rssm.imagine(last_action, state, steps=IMAGINE_STEPS)

        imagined_obs = []
        for s in imagined_states:
            obs_pred = trainer.decoder(s["h"], s["z"])
            imagined_obs.append(obs_pred.squeeze(0).cpu().numpy())

    imagined_traj = np.array(imagined_obs, dtype=np.float32)
    plot_imagination(real_traj, imagined_traj)

    env.close()
    print("Imagination complete. Plot saved to ./outputs/imagination.png")


if __name__ == "__main__":
    main()
