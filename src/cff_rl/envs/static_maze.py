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
    ActiveGatingWrapper,
    ActiveVisionWrapper,
    FrameStack4Wrapper,
    Grayscale64Wrapper,
    MotionModulationWrapper,
    ProprioWrapper,
    StroboscopicWrapper,
)

# MiniWorld default action indices we care about for the static task.
# From miniworld.miniworld.MiniWorldEnv.Actions:
#   0 = turn_left, 1 = turn_right, 2 = move_forward, 3 = move_back, ...
STATIC_ACTIONS = [0, 1, 2]

# MiniWorld default action indices we care about for the static task.
# 90-degree turns keep state transitions discrete (proposal § Method, Regime 1).
TURN_STEP_DEG = 45
# Fixed, moderate forward speed — the proposal specifies constant low speed to
# remove velocity dynamics from Regime 1.
FORWARD_STEP = 0.5

OBS_SIZE = 64
FRAME_STACK = 8
MAX_EPISODE_STEPS = 1000


def _get_preset_config(preset: str) -> dict:
    """Return action list and motion parameters for the given preset.
    
    Returns dict with keys:
        - actions: list of MiniWorld action indices
        - turn_step: turn_step_deg parameter
        - forward_steps: dict mapping action_idx -> forward_step value
    """
    if preset == "baseline_10":
        return {
            "actions": [0, 1, 2],  # turn_left, turn_right, move_forward
            "turn_step": 10,
            "forward_steps": {2: 0.5},  # all forwards use standard speed
        }
    elif preset == "baseline_1":
        return {
            "actions": [0, 1, 2],  # turn_left, turn_right, move_forward
            "turn_step": 1,
            "forward_steps": {2: 0.5},  # all forwards use standard speed
        }
    elif preset == "baseline_45":
        return {
            "actions": [0, 1, 2],  # turn_left, turn_right, move_forward
            "turn_step": 45,
            "forward_steps": {2: 0.5},  # all forwards use standard speed
        }
    elif preset == "fine_turns":
        # 10° and 45° turns + forward
        # MiniWorld: 0=turn_left, 1=turn_right
        # Actions: [left-10°, left-45°, right-10°, right-45°, forward]
        return {
            "actions": [0, 0, 1, 1, 2],  # left-10°, left-45°, right-10°, right-45°, forward
            "turn_step": 10,  # will be overridden per-action via wrapper
            "forward_steps": {4: 0.5},
            "turn_angles": {0: 10, 1: 45, 2: 10, 3: 45},  # action_idx -> turn angle in degrees
        }
    elif preset == "speed_var":
        # 3 forward speeds + turn left/right
        # Actions: [slow_fwd, normal_fwd, fast_fwd, turn_left, turn_right]
        return {
            "actions": [2, 2, 2, 0, 1],  # forward (x3), left, right
            "turn_step": 45,
            "forward_steps": {0: 0.25, 1: 0.5, 2: 1},  # action_idx -> speed
        }
    elif preset == "fine_speed":
        # Combined preset: fine turns + speed variation.
        # Actions: [left-10°, left-45°, right-10°, right-45°, slow_fwd, normal_fwd, fast_fwd]
        return {
            "actions": [0, 0, 1, 1, 2, 2, 2],
            "turn_step": 10,
            "turn_angles": {0: 10, 1: 45, 2: 10, 3: 45},
            "forward_steps": {4: 0.25, 5: 0.5, 6: 1},
        }
    else:
        raise ValueError(
            f"Unknown action_preset: {preset}. "
            "Choose from: 'baseline_1', 'baseline_10', 'baseline_45', 'fine_turns', 'speed_var', 'fine_speed'"
        )



def make_static_env(
    env_id: str = "MiniWorld-FourRooms-v0",
    seed: int | None = None,
    render_mode: str | None = None,
    use_stroboscopic: bool = False,
    use_active_gating: bool = False,
    use_active_vision: bool = False,
    vision_cost: float = 0.01,
    strobe_k: int = 7,
    high_freq_steps: int = 35,
    frame_stack: int = FRAME_STACK,
    use_proprio: bool = False,
    turn_step_deg: int = TURN_STEP_DEG,
    action_preset: str = "baseline_45",
) -> gym.Env:
    """Build the Regime 1 environment for Agent A, B, C with configurable action presets.
    
    Parameters
    ----------
    action_preset : str
        - "baseline_10": 3 actions (left, right, forward)
        - "baseline_45": 3 actions (left, right, forward)
        - "fine_turns": 5 actions (left-10°, left-45°, right-10°, right-45°, forward)
        - "speed_var": 5 actions (slow_fwd, normal_fwd, fast_fwd, left, right)
        - "fine_speed": 7 actions (fine turns + slow/normal/fast forward)
    """

    print(f"make_static_env called with: preset={action_preset}, {locals()}")

    n_exclusive = sum([use_stroboscopic, use_active_gating, use_active_vision])
    if n_exclusive > 1:
        raise ValueError(
            "use_stroboscopic, use_active_gating, and use_active_vision are "
            "mutually exclusive — pick exactly one (or none for Agent A)."
        )
    
    # Get action configuration for this preset
    preset_config = _get_preset_config(action_preset)
    base_actions = preset_config["actions"]
    turn_step = preset_config["turn_step"]
    forward_steps = preset_config["forward_steps"]
    turn_angles = preset_config.get("turn_angles", None)
    
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
    
    # Configure motion parameters on the unwrapped MiniWorld env
    inner: MiniWorldEnv = env.unwrapped  # type: ignore[assignment]
    inner.params.set("turn_step", turn_step, turn_step, turn_step)
    
    # Determine which forward_step to use as default
    # For presets with per-action modulation, use the first forward_step value
    default_forward = FORWARD_STEP
    if forward_steps:
        default_forward = list(forward_steps.values())[0] if forward_steps else FORWARD_STEP
    
    inner.params.set("forward_step", default_forward, default_forward, default_forward)

    # ActionFilterWrapper maps reduced action space to MiniWorld actions
    env = ActionFilterWrapper(env, base_actions)
    env = Grayscale64Wrapper(env, size=OBS_SIZE)
    
    # For presets with per-action modulation, wrap after grayscale
    if turn_angles or forward_steps:
        env = MotionModulationWrapper(
            env,
            turn_angles=turn_angles,
            forward_steps=forward_steps,
            default_turn_step=turn_step,
            default_forward_step=default_forward,
        )
    
    if use_stroboscopic:                          # Agent B
        env = StroboscopicWrapper(env, k=strobe_k)
    elif use_active_gating:                       # Agent C v1 (STOP_AND_LOOK)
        env = ActiveGatingWrapper(
            env,
            n_base_actions=len(base_actions),
            k=strobe_k,
            high_freq_steps=high_freq_steps,
        )
    elif use_active_vision:                       # Agent C v2 (6 actions)
        env = ActiveVisionWrapper(
            env,
            n_base_actions=len(base_actions),
            k=strobe_k,
            vision_cost=vision_cost,
        )
    
    env = FrameStack4Wrapper(env, k=frame_stack)
    if use_proprio:
        n_actions = len(base_actions)
        if use_active_vision:
            n_actions = 2 * len(base_actions)
        elif use_active_gating:
            n_actions = len(base_actions) + 1
        env = ProprioWrapper(env, n_actions=n_actions)

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
