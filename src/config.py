from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    env_id: str = "highway-v0"   
    seed: int = 42

    total_timesteps: int = 10_000



    # PPO
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2

    # save paths
    model_dir: str = "artifacts/models"
    half_name: str = "ppo_half.zip"
    final_name: str = "ppo_final.zip"


