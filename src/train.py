from __future__ import annotations

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.config import TrainConfig
from src.envs.make_env import make_env


class HalfSaveCallback(BaseCallback):
    def __init__(self, cfg: TrainConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.saved = False

    def _on_step(self) -> bool:
        if (not self.saved) and self.num_timesteps >= self.cfg.total_timesteps // 2:
            os.makedirs(self.cfg.model_dir, exist_ok=True)
            half_path = os.path.join(self.cfg.model_dir, self.cfg.half_name)
            self.model.save(half_path)
            self.saved = True
            print(f"[Saved half model] {half_path}")
        return True


def main() -> None:
    cfg = TrainConfig()

    env = make_env(cfg)
    os.makedirs("artifacts/logs", exist_ok=True)
    env = Monitor(env, filename="artifacts/logs/monitor.csv")


    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        verbose=1,
        seed=cfg.seed,
    )

    cb = HalfSaveCallback(cfg)
    model.learn(total_timesteps=cfg.total_timesteps, callback=cb)

    os.makedirs(cfg.model_dir, exist_ok=True)
    final_path = os.path.join(cfg.model_dir, cfg.final_name)
    model.save(final_path)
    print(f"[Saved final model] {final_path}")

    env.close()


if __name__ == "__main__":
    main()
