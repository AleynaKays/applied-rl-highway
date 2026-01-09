from __future__ import annotations

from typing import Any, Tuple

import gymnasium as gym
import numpy as np


class CustomRewardWrapper(gym.Wrapper):
    """
    Custom reward:
    R = alpha * speed_norm + beta * right_lane - gamma * collision - delta * lane_change
    """

    def __init__(
        self,
        env: gym.Env,
        alpha_speed: float = 1.0,
        beta_right_lane: float = 0.2,
        gamma_collision: float = 5.0,
        delta_lane_change: float = 0.05,
        v_min: float = 20.0,
        v_max: float = 40.0,
    ) -> None:
        super().__init__(env)
        self.alpha = alpha_speed
        self.beta = beta_right_lane
        self.gamma = gamma_collision
        self.delta = delta_lane_change
        self.v_min = v_min
        self.v_max = v_max
        self._prev_lane: int | None = None

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._prev_lane = self._get_lane_index()
        return obs, info

    def step(self, action: Any):
        obs, _, terminated, truncated, info = self.env.step(action)

        speed = float(self._get_speed())
        speed_norm = (speed - self.v_min) / (self.v_max - self.v_min)
        speed_norm = float(np.clip(speed_norm, 0.0, 1.0))

        crashed = bool(info.get("crashed", False))

        lane_now = self._get_lane_index()
        lane_change = 0.0
        if self._prev_lane is not None and lane_now is not None:
            lane_change = 1.0 if lane_now != self._prev_lane else 0.0
        self._prev_lane = lane_now

        right_lane = float(self._is_right_lane())

        reward = (
            self.alpha * speed_norm
            + self.beta * right_lane
            - self.gamma * float(crashed)
            - self.delta * lane_change
        )

        return obs, reward, terminated, truncated, info

    def _get_speed(self) -> float:
        veh = getattr(self.env.unwrapped, "vehicle", None)
        if veh is None:
            return 0.0
        return float(getattr(veh, "speed", 0.0))

    def _get_lane_index(self) -> int | None:
        veh = getattr(self.env.unwrapped, "vehicle", None)
        if veh is None:
            return None

        lane_index = getattr(veh, "lane_index", None)

        if isinstance(lane_index, tuple) and len(lane_index) >= 2:
            return int(lane_index[1])

        if isinstance(lane_index, int):
            return int(lane_index)

        return None

    def _is_right_lane(self) -> int:
        road = getattr(self.env.unwrapped, "road", None)
        veh = getattr(self.env.unwrapped, "vehicle", None)
        if road is None or veh is None:
            return 0

        lanes = getattr(road, "lanes", None)
        if lanes is None or len(lanes) == 0:
            return 0

        max_lane = len(lanes) - 1
        lane_idx = self._get_lane_index()
        if lane_idx is None:
            return 0

        return 1 if lane_idx == max_lane else 0


