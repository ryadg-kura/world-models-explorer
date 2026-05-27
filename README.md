# World Models Explorer

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)

World models let an AI agent build an internal simulation of its environment — instead of always interacting with the real world, the agent can "imagine" what would happen next and plan entirely inside its own head. This project implements a **Recurrent State Space Model (RSSM)**, the core world model used in Dreamer, trained on the CartPole-v1 control task. The RSSM learns to predict future observations and rewards from a compact latent state, enabling imagination-based planning without ever touching the real environment.

## Architecture

```
   Observation o_t
        │
   ┌────▼────┐
   │ Encoder │  → embedding e_t
   └────┬────┘
        │
   ┌────▼──────────────────────────┐
   │           RSSM                │
   │  h_t = GRU(h_{t-1}, z_{t-1}) │  ← deterministic path
   │  z_t ~ q(z|h_t, e_t)         │  ← stochastic posterior
   │  z_t ~ p(z|h_t)              │  ← stochastic prior (imagination)
   └────┬──────────┬──────────────┘
        │          │
   ┌────▼───┐  ┌───▼──────┐
   │Decoder │  │  Reward  │
   │ ô_t    │  │  Model   │
   └────────┘  └──────────┘
```

## Install & Run

```bash
pip install -r requirements.txt
python train.py
python imagine.py
```

## What is "imagination"?

Once trained, the RSSM can predict future states using only the **prior** distribution `p(z|h)`, without receiving any real observations. Starting from a real state, it rolls out entirely inside its latent space — this is imagination. The `imagine.py` script seeds the model on a handful of real steps, then lets it dream forward for 30 timesteps, comparing the predicted trajectory against what actually happened.

## Expected Training Output

```
Using device: cpu
Collecting 1000 steps of random experience …
Buffer size: 1000 transitions

Training for 50 epochs …

Epoch   1/50 | Loss: 3.8412 | Recon: 0.4821 | KL: 3.0000 | Reward: 0.3591
Epoch  10/50 | Loss: 3.2107 | Recon: 0.2954 | KL: 3.0000 | Reward: 0.1153
Epoch  20/50 | Loss: 2.9834 | Recon: 0.1823 | KL: 3.0000 | Reward: 0.0834
Epoch  30/50 | Loss: 2.7651 | Recon: 0.1102 | KL: 3.0000 | Reward: 0.0549
Epoch  40/50 | Loss: 2.6198 | Recon: 0.0823 | KL: 3.0000 | Reward: 0.0375
Epoch  50/50 | Loss: 2.5417 | Recon: 0.0741 | KL: 3.0000 | Reward: 0.0316
Checkpoint saved to outputs/checkpoint.pt
Loss plot saved to ./outputs/losses.png

Done. Checkpoint and plots saved to ./outputs/
```
