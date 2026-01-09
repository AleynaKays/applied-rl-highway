from __future__ import annotations
import gymnasium as gym
import highway_env  # registers envs

from src.config import TrainConfig
from src.envs.reward_wrapper import CustomRewardWrapper


def make_env(cfg: TrainConfig, render_mode: str | None = None) -> gym.Env:
    env = gym.make(cfg.env_id, render_mode=render_mode)
    env = CustomRewardWrapper(env)
    return env

