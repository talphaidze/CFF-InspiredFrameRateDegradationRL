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

class ActiveGatingWrapper(gym.Wrapper):
    """Agent C: stroboscopic-by-default with an added STOP_AND_LOOK action.
 
    Observation frequency
    ---------------------
    - Default mode  : stroboscopic ~5 Hz — each fresh frame is held for
                      ``k`` steps before the next env observation is consumed
                      (identical behaviour to ``StroboscopicWrapper``).
    - High-freq mode: triggered by STOP_AND_LOOK.  Fresh 35 Hz observations
                      are returned for the next ``high_freq_steps`` steps,
                      then the wrapper reverts automatically to stroboscopic.
 
    Action space
    ------------
    The inner env is expected to already have a *filtered* Discrete action
    space of size ``n_base_actions`` (e.g. 3 for {turn_left, turn_right,
    move_forward}).  This wrapper appends one extra action:
 
        action == n_base_actions  →  STOP_AND_LOOK
            • Sets high-freq mode for ``high_freq_steps`` steps.
            • Passes ``null_action`` to the inner env for this single step
              (turn_left by default — the least-disruptive option available
              since MiniWorld has no explicit idle/no-op action).
 
    Placement
    ---------
    Stack order (Agent C)::
 
        gym.make(...)
        └─ ActionFilterWrapper        # reduces to 3 movement actions
           └─ Grayscale64Wrapper      # RGB → (64,64) uint8
              └─ ActiveGatingWrapper  # ← here; manages freq + action space
                 └─ FrameStack4Wrapper
                    └─ ProprioWrapper (optional)
 
    Do NOT also apply ``StroboscopicWrapper`` — this wrapper fully subsumes it.
    """
 
    def __init__(
        self,
        env: gym.Env,
        n_base_actions: int,
        k: int = 7,
        high_freq_steps: int = 35,
        null_action: int = 0,
    ):
        """
        Parameters
        ----------
        env            : inner env (after Grayscale64Wrapper).
        n_base_actions : number of movement actions already exposed by the
                         inner env (3 for the static maze).
        k              : stroboscopic hold length in steps (default 7 → ~5 Hz
                         at 35 Hz physics tick).
        high_freq_steps: number of consecutive fresh-obs steps after a
                         STOP_AND_LOOK (default 35 → 1 second at 35 Hz).
        null_action    : inner-env action executed when STOP_AND_LOOK is
                         chosen (default 0 = turn_left).
        """
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError(
                "ActiveGatingWrapper expects a Box observation space "
                f"(got {type(env.observation_space).__name__}). "
                "Ensure Grayscale64Wrapper is applied first."
            )
        self.k = int(k)
        self.high_freq_steps = int(high_freq_steps)
        self.null_action = int(null_action)
        self.n_base_actions = int(n_base_actions)
        self.STOP_AND_LOOK: int = n_base_actions  # index of the new action
 
        # Extend the action space by one slot for STOP_AND_LOOK.
        self.action_space = spaces.Discrete(n_base_actions + 1)
 
        # Internal state — reset on every episode.
        self._hold_counter: int = 0
        self._last_obs: np.ndarray | None = None
        self._high_freq_remaining: int = 0
 
    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------
 
    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs.copy()
        self._hold_counter = self.k - 1
        self._high_freq_remaining = 0
        return self._last_obs.copy(), info
 
    def step(self, action: int):  # type: ignore[override]
        is_sal = int(action) == self.STOP_AND_LOOK
 
        if is_sal:
            # Arm the high-frequency window and execute a null movement.
            self._high_freq_remaining = self.high_freq_steps
            inner_action = self.null_action
        else:
            inner_action = int(action)
 
        obs, reward, terminated, truncated, info = self.env.step(inner_action)
 
        # ---- observation gating logic ----------------------------------
        if self._high_freq_remaining > 0:
            # High-freq mode: deliver every fresh frame (35 Hz).
            self._last_obs = obs.copy()
            self._high_freq_remaining -= 1
            if self._high_freq_remaining == 0:
                # Window just closed; start a fresh stroboscopic cycle so the
                # very next non-SAL step begins a full k-step hold.
                self._hold_counter = self.k - 1
        else:
            # Stroboscopic mode: hold the last frame for k steps.
            if self._hold_counter == 0:
                self._last_obs = obs.copy()
                self._hold_counter = self.k - 1
            else:
                self._hold_counter -= 1
        # ---------------------------------------------------------------
 
        info["stop_and_look"] = is_sal
        info["high_freq_remaining"] = self._high_freq_remaining
        return self._last_obs.copy(), reward, terminated, truncated, info
   
class ActiveVisionWrapper(gym.Wrapper):
    """Agent C v2: per-action perception mode selection.

    Instead of a separate STOP_AND_LOOK action, the agent chooses between
    two perception regimes on every step by selecting from a doubled action
    space:

        actions 0 .. n-1  →  base movement with 5 Hz stroboscopic vision
        actions n .. 2n-1 →  same movement with 35 Hz fresh-frame vision

    The agent always moves — there is no freezing.  It simply decides
    *how well it wants to see* while moving, mirroring the active-vision
    idea that organisms modulate perception during locomotion.

    Placement (same position as ActiveGatingWrapper)::

        gym.make(...)
        └─ ActionFilterWrapper
           └─ Grayscale64Wrapper
              └─ ActiveVisionWrapper   # ← here
                 └─ FrameStack4Wrapper
                    └─ ProprioWrapper (optional)
    """

    def __init__(
        self,
        env: gym.Env,
        n_base_actions: int,
        k: int = 7,
    ):
        """
        Parameters
        ----------
        env            : inner env (after Grayscale64Wrapper).
        n_base_actions : number of movement actions (3 for static maze).
        k              : stroboscopic hold length for low-freq actions
                         (default 7 → ~5 Hz at 35 Hz physics tick).
        """
        super().__init__(env)
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError(
                "ActiveVisionWrapper expects a Box observation space "
                f"(got {type(env.observation_space).__name__}). "
                "Ensure Grayscale64Wrapper is applied first."
            )
        self.k = int(k)
        self.n_base_actions = int(n_base_actions)

        # actions [0, n) = low-freq, actions [n, 2n) = high-freq
        self.action_space = spaces.Discrete(2 * n_base_actions)

        self._hold_counter: int = 0
        self._last_obs: np.ndarray | None = None

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs.copy()
        self._hold_counter = self.k - 1
        return self._last_obs.copy(), info

    def step(self, action: int):  # type: ignore[override]
        action = int(action)
        high_freq = action >= self.n_base_actions
        inner_action = action % self.n_base_actions

        obs, reward, terminated, truncated, info = self.env.step(inner_action)

        if high_freq:
            # 35 Hz: always deliver a fresh frame.
            self._last_obs = obs.copy()
            # Reset stroboscopic counter so next low-freq action starts
            # a full k-step hold.
            self._hold_counter = self.k - 1
        else:
            # 5 Hz stroboscopic: hold frame for k steps.
            if self._hold_counter == 0:
                self._last_obs = obs.copy()
                self._hold_counter = self.k - 1
            else:
                self._hold_counter -= 1

        info["high_freq"] = high_freq
        return self._last_obs.copy(), reward, terminated, truncated, info


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
