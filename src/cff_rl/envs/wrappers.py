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

class StroboscopicWrapper(gym.ObservationWrapper):
    """Repeat each grayscale frame for k steps before refreshing it."""

    def __init__(self, env: gym.Env, k: int = 7):
        super().__init__(env)
        self.k = int(k)
        self._hold_counter = 0
        self._last_obs: np.ndarray | None = None
        self.observation_space = env.observation_space

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs.copy()
        self._hold_counter = self.k - 1
        return self._last_obs, info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        if self._hold_counter == 0:
            self._last_obs = obs.copy()
            self._hold_counter = self.k - 1
        else:
            self._hold_counter -= 1
        return self._last_obs

class VideoCompositeWrapper(gym.Wrapper):
    """Override env.render() with a high-res top-down + first-person composite,
    suitable for RecordVideo. Does not affect the observation pipeline.

    The underlying MiniWorld env must have been built with render_mode='rgb_array'
    and a `vis_fb` framebuffer (any window_width/window_height). We call
    render_top_view() and render_obs() directly on the unwrapped env, then
    paste them side-by-side into a single uint8 frame.
    """

    def __init__(self, env: gym.Env, render_fps: int = 4):
        super().__init__(env)
        # RecordVideo reads metadata["render_fps"]; lower it so each discrete
        # action (90 deg turn or 0.5 forward) is visible for a moment.
        self.metadata = {**self.env.metadata, "render_fps": render_fps}

    def render(self):  # type: ignore[override]
        inner = self.env.unwrapped
        td = inner.render_top_view(inner.vis_fb)  # (H, W, 3)
        fp = inner.render_obs(inner.vis_fb)        # (H, W, 3)
        h = max(td.shape[0], fp.shape[0])

        def _pad(img: np.ndarray) -> np.ndarray:
            if img.shape[0] == h:
                return img
            pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=img.dtype)
            return np.concatenate([img, pad], axis=0)

        return np.concatenate([_pad(td), _pad(fp)], axis=1)


class ProprioWrapper(gym.Wrapper):
    """Augment image obs with NavA3C-style proprioceptive scalars.

    Observation becomes a dict::

        {"image":  same shape/dtype as the inner env,
         "extras": (n_actions + 1 + 2,) float32}

    Extras layout: [prev_action one-hot, prev_reward, sin(heading), cos(heading)].
    Heading is read from the unwrapped MiniWorld agent's `dir` (radians).
    Extras are always fresh — `StroboscopicWrapper` only throttles the
    visual stream, not proprioception.
    """

    def __init__(self, env: gym.Env, n_actions: int):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box)
        self._image_space = env.observation_space
        self.n_actions = int(n_actions)
        self.n_extras = self.n_actions + 1 + 2
        self.observation_space = spaces.Dict(
            {
                "image": self._image_space,
                "extras": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.n_extras,),
                    dtype=np.float32,
                ),
            }
        )
        self._prev_action: int | None = None
        self._prev_reward: float = 0.0

    def _heading(self) -> tuple[float, float]:
        d = float(self.env.unwrapped.agent.dir)
        return float(np.sin(d)), float(np.cos(d))

    def _build_extras(self) -> np.ndarray:
        extras = np.zeros(self.n_extras, dtype=np.float32)
        if self._prev_action is not None:
            extras[int(self._prev_action)] = 1.0
        extras[self.n_actions] = float(self._prev_reward)
        s, c = self._heading()
        extras[self.n_actions + 1] = s
        extras[self.n_actions + 2] = c
        return extras

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_action = None
        self._prev_reward = 0.0
        return {"image": obs, "extras": self._build_extras()}, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._prev_action = int(action)
        self._prev_reward = float(reward)
        return (
            {"image": obs, "extras": self._build_extras()},
            reward,
            terminated,
            truncated,
            info,
        )


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
