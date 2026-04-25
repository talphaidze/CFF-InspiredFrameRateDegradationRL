"""Observation and action wrappers shared across all agent groups.

Group A (35 Hz baseline) uses only the wrappers defined here. Group B
(Stroboscopic, 5 Hz) will add `StroboscopicWrapper` here — a frame-hold
wrapper with k=7 — and Group C will add an action-space extension for
STOP_AND_LOOK. Keep this module the single home for frame-rate wrappers
so the three groups differ in exactly one import.
"""
from __future__ import annotations

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Grayscale64Wrapper(gym.ObservationWrapper):
    """RGB -> (64, 64) grayscale, uint8."""

    def __init__(self, env: gym.Env, size: int = 64):
        super().__init__(env)
        self.size = size
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(size, size), dtype=np.uint8
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)


class FrameStack4Wrapper(gym.ObservationWrapper):
    """Stack last 4 grayscale frames along a leading channel axis: (4, H, W)."""

    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        self.k = k
        h, w = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(k, h, w), dtype=np.uint8
        )
        self._frames: list[np.ndarray] = []

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._frames = [obs.copy() for _ in range(self.k)]
        return self._stack(), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._frames.pop(0)
        self._frames.append(obs)
        return self._stack()

    def _stack(self) -> np.ndarray:
        return np.stack(self._frames, axis=0)


class ActionFilterWrapper(gym.ActionWrapper):
    """Expose a reduced discrete action space.

    The underlying MiniWorld action space is Discrete(8). For the static maze
    (Regime 1) we use only {turn_left, turn_right, move_forward}. This wrapper
    maps {0, 1, 2} to the corresponding MiniWorld action indices.
    """

    def __init__(self, env: gym.Env, action_map: list[int]):
        super().__init__(env)
        self.action_map = action_map
        self.action_space = spaces.Discrete(len(action_map))

    def action(self, action: int) -> int:
        return self.action_map[int(action)]
