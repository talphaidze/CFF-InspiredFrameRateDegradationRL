"""Regime 1 (static) environment: MiniWorld OneRoom with 3 discrete actions,
fixed forward speed, egocentric 64x64 grayscale, 4-frame stack.

Agent A (35 Hz) uses this env directly. Groups B and C will wrap the same
env with StroboscopicWrapper / STOP_AND_LOOK action.
"""
from __future__ import annotations

import os
import sys

# Enable pyglet's EGL-based headless mode before importing miniworld on Linux
# (Izar compute nodes have no X display). On macOS / Windows, pyglet's EGL
# backend isn't available — fall back to the platform's native windowing.
# Must run before `import miniworld`.
_HEADLESS = sys.platform.startswith("linux")
if _HEADLESS:
    os.environ.setdefault("PYGLET_HEADLESS", "true")
import pyglet  # noqa: E402

if _HEADLESS:
    pyglet.options["headless"] = True

import gymnasium as gym  # noqa: E402
import miniworld  # noqa: F401, E402 — registers MiniWorld envs on import
from miniworld.miniworld import MiniWorldEnv  # noqa: E402

from cff_rl.envs.wrappers import (
    ActionFilterWrapper,
    FrameStack4Wrapper,
    Grayscale64Wrapper,
    StroboscopicWrapper,
)

# MiniWorld default action indices we care about for the static task.
# From miniworld.miniworld.MiniWorldEnv.Actions:
#   0 = turn_left, 1 = turn_right, 2 = move_forward, 3 = move_back, ...
STATIC_ACTIONS = [0, 1, 2]

# 90-degree turns keep state transitions discrete (proposal § Method, Regime 1).
TURN_STEP_DEG = 45
# Fixed, moderate forward speed — the proposal specifies constant low speed to
# remove velocity dynamics from Regime 1.
FORWARD_STEP = 0.5

OBS_SIZE = 64
FRAME_STACK = 8
MAX_EPISODE_STEPS = 500


def make_static_env(
    env_id: str = "MiniWorld-FourRooms-v0",
    seed: int | None = None,
    render_mode: str | None = None,
    use_stroboscopic: bool = False,
    strobe_k: int = 7,
    frame_stack: int = FRAME_STACK,
) -> gym.Env:
    """Build the Regime 1 environment for Agent A (35 Hz baseline)."""
    extra: dict = {}
    if render_mode == "rgb_array":
        # Bigger framebuffer so the recorded video isn't tiny. Doesn't affect
        # the policy obs (still OBS_SIZE x OBS_SIZE, set via obs_width/height).
        extra.update(window_width=512, window_height=512)
    env = gym.make(
        env_id,
        obs_width=OBS_SIZE,
        obs_height=OBS_SIZE,
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=render_mode,
        **extra,
    )
    # Override motion params on the unwrapped MiniWorldEnv so turns are 90°
    # and forward velocity is constant. `params.set(name, default, min, max)`.
    inner: MiniWorldEnv = env.unwrapped  # type: ignore[assignment]
    inner.params.set("turn_step", TURN_STEP_DEG, TURN_STEP_DEG, TURN_STEP_DEG)
    inner.params.set("forward_step", FORWARD_STEP, FORWARD_STEP, FORWARD_STEP)

    env = ActionFilterWrapper(env, STATIC_ACTIONS)
    env = Grayscale64Wrapper(env, size=OBS_SIZE)
    if use_stroboscopic:
        env = StroboscopicWrapper(env, k=strobe_k)
    env = FrameStack4Wrapper(env, k=frame_stack)

    if seed is not None:
        env.reset(seed=seed)
    return env


if __name__ == "__main__":
    # Smoke test: random policy across a few episodes, verify obs shape.
    env = make_static_env(seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (FRAME_STACK, OBS_SIZE, OBS_SIZE), obs.shape
    episodes = successes = 0
    for _ in range(2000):
        a = env.action_space.sample()
        obs, r, terminated, truncated, _ = env.step(a)
        if terminated or truncated:
            episodes += 1
            successes += int(terminated)
            obs, _ = env.reset()
    print(
        f"OK — env={env.spec.id if env.spec else '?'} obs {obs.shape} "
        f"actions={env.action_space} episodes={episodes} random_success={successes}"
    )
    env.close()
