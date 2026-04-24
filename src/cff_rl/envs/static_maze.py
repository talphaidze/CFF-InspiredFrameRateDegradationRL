"""Regime 1 (static) environment: MiniWorld OneRoom with 3 discrete actions,
fixed forward speed, egocentric 64x64 grayscale, 4-frame stack.

Agent A (35 Hz) uses this env directly. Groups B and C will wrap the same
env with StroboscopicWrapper / STOP_AND_LOOK action.
"""
from __future__ import annotations

import gymnasium as gym
import miniworld  # noqa: F401 — registers MiniWorld envs on import
from miniworld.miniworld import MiniWorldEnv

from cff_rl.envs.wrappers import (
    ActionFilterWrapper,
    FrameStack4Wrapper,
    Grayscale64Wrapper,
)

# MiniWorld default action indices we care about for the static task.
# From miniworld.miniworld.MiniWorldEnv.Actions:
#   0 = turn_left, 1 = turn_right, 2 = move_forward, 3 = move_back, ...
STATIC_ACTIONS = [0, 1, 2]

# 90-degree turns keep state transitions discrete (proposal § Method, Regime 1).
TURN_STEP_DEG = 90
# Fixed, moderate forward speed — the proposal specifies constant low speed to
# remove velocity dynamics from Regime 1.
FORWARD_STEP = 0.5

OBS_SIZE = 64
FRAME_STACK = 4
MAX_EPISODE_STEPS = 500


def make_static_env(
    env_id: str = "MiniWorld-OneRoom-v0",
    seed: int | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Build the Regime 1 environment for Agent A (35 Hz baseline)."""
    env = gym.make(
        env_id,
        obs_width=OBS_SIZE,
        obs_height=OBS_SIZE,
        max_episode_steps=MAX_EPISODE_STEPS,
        render_mode=render_mode,
    )
    # Override motion params on the unwrapped MiniWorldEnv so turns are 90°
    # and forward velocity is constant. `params.set(name, default, min, max)`.
    inner: MiniWorldEnv = env.unwrapped  # type: ignore[assignment]
    inner.params.set("turn_step", TURN_STEP_DEG, TURN_STEP_DEG, TURN_STEP_DEG)
    inner.params.set("forward_step", FORWARD_STEP, FORWARD_STEP, FORWARD_STEP)

    env = ActionFilterWrapper(env, STATIC_ACTIONS)
    env = Grayscale64Wrapper(env, size=OBS_SIZE)
    env = FrameStack4Wrapper(env, k=FRAME_STACK)

    if seed is not None:
        env.reset(seed=seed)
    return env


if __name__ == "__main__":
    # Smoke test: random policy for 100 steps, verify obs shape.
    env = make_static_env(seed=0)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (FRAME_STACK, OBS_SIZE, OBS_SIZE), obs.shape
    for _ in range(100):
        a = env.action_space.sample()
        obs, r, terminated, truncated, _ = env.step(a)
        if terminated or truncated:
            obs, _ = env.reset()
    print(f"OK — obs shape {obs.shape}, action space {env.action_space}")
    env.close()
