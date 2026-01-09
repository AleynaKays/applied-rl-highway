from __future__ import annotations

import os
import glob
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO

from src.config import TrainConfig
from src.envs.make_env import make_env


def newest_mp4(folder: str) -> str:
    mp4s = glob.glob(os.path.join(folder, "*.mp4"))
    if not mp4s:
        raise FileNotFoundError(f"No mp4 found in {folder}")
    mp4s.sort(key=os.path.getmtime)
    return mp4s[-1]


def record_one_episode(env: gym.Env, out_dir: str, model: PPO | None = None) -> str:
    os.makedirs(out_dir, exist_ok=True)

    # Record exactly the first episode
    wrapped = RecordVideo(
        env,
        video_folder=out_dir,
        episode_trigger=lambda ep: ep == 0,
        name_prefix="stage",
    )

    obs, _ = wrapped.reset()
    done = False
    while not done:
        if model is None:
            action = wrapped.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, _, terminated, truncated, _ = wrapped.step(action)
        done = bool(terminated or truncated)

    wrapped.close()
    return newest_mp4(out_dir)


def main() -> None:
    cfg = TrainConfig()

    # Paths
    half_path = os.path.join(cfg.model_dir, cfg.half_name)
    final_path = os.path.join(cfg.model_dir, cfg.final_name)

    # 1) Untrained (random)
    env1 = make_env(cfg, render_mode="rgb_array")
    p1 = record_one_episode(env1, out_dir="artifacts/videos/1_untrained", model=None)

    # 2) Half-trained
    half_model = PPO.load(half_path)
    env2 = make_env(cfg, render_mode="rgb_array")
    p2 = record_one_episode(env2, out_dir="artifacts/videos/2_half", model=half_model)

    # 3) Fully trained
    final_model = PPO.load(final_path)
    env3 = make_env(cfg, render_mode="rgb_array")
    p3 = record_one_episode(env3, out_dir="artifacts/videos/3_final", model=final_model)

    print("Recorded:")
    print("  Untrained:", p1)
    print("  Half:", p2)
    print("  Final:", p3)
    print("\nNext: we'll stitch them into one evolution.mp4")


if __name__ == "__main__":
    main()
