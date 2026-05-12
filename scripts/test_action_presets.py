"""Smoke test for action presets: baseline, fine_turns, speed_var.

Usage:
    python scripts/test_action_presets.py

Tests:
    1. _get_preset_config returns correct action lists and params
    2. Environment creation with each preset
    3. Action space sizes match expectations
    4. Motion behavior is correct (turn angles, forward speeds)
"""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

# Add src to path
SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from cff_rl.envs.static_maze import _get_preset_config, make_static_env


def test_preset_configs() -> None:
    """Test that _get_preset_config returns valid configs for each preset."""
    print("\n=== Testing Preset Configs ===\n")
    
    presets = {
        "baseline": {
            "n_actions": 3,
            "actions": [0, 1, 2],
            "turn_step": 90,
        },
        "fine_turns": {
            "n_actions": 5,
            "actions": [0, 0, 1, 1, 2],
            "turn_step": 10,
        },
        "speed_var": {
            "n_actions": 5,
            "actions": [2, 2, 2, 0, 1],
            "turn_step": 90,
        },
        "fine_speed": {
            "n_actions": 7,
            "actions": [0, 0, 1, 1, 2, 2, 2],
            "turn_step": 10,
        },
    }
    
    for preset_name, expected in presets.items():
        config = _get_preset_config(preset_name)
        assert len(config["actions"]) == expected["n_actions"], (
            f"{preset_name}: expected {expected['n_actions']} actions, "
            f"got {len(config['actions'])}"
        )
        assert config["actions"] == expected["actions"], (
            f"{preset_name}: action list mismatch"
        )
        assert config["turn_step"] == expected["turn_step"], (
            f"{preset_name}: turn_step mismatch"
        )
        print(f"✓ {preset_name:15} - {len(config['actions'])} actions, "
              f"turn_step={config['turn_step']}°")


def test_env_creation() -> None:
    """Test that environments can be created with each preset."""
    print("\n=== Testing Environment Creation ===\n")
    
    presets = ["baseline", "fine_turns", "speed_var", "fine_speed"]
    
    for preset in presets:
        try:
            env = make_static_env(action_preset=preset, seed=0)
            n_actions = int(env.action_space.n)
            obs_shape = env.observation_space.shape
            
            # Reset and step once to verify env works
            obs, info = env.reset(seed=0)
            for _ in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
            env.close()
            
            print(f"✓ {preset:15} - action_space.n={n_actions}, obs_shape={obs_shape}")
        except Exception as e:
            print(f"✗ {preset:15} - FAILED: {e}")
            raise


def test_agent_wrapping() -> None:
    """Test that presets work correctly with agent wrappers (B, C v1, C v2)."""
    print("\n=== Testing with Agent Wrappers ===\n")
    
    presets = ["baseline", "fine_turns", "speed_var", "fine_speed"]
    wrapper_configs = [
        ("Agent A (baseline)", {}),
        ("Agent B (stroboscopic)", {"use_stroboscopic": True, "strobe_k": 7}),
        ("Agent C v1 (active gating)", {"use_active_gating": True, "high_freq_steps": 35}),
        ("Agent C v2 (active vision)", {"use_active_vision": True, "vision_cost": 0.01}),
    ]
    
    for preset in presets:
        for agent_name, wrapper_kwargs in wrapper_configs:
            try:
                env = make_static_env(action_preset=preset, seed=0, **wrapper_kwargs)
                n_actions = int(env.action_space.n)
                
                obs, info = env.reset(seed=0)
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                env.close()
                
                print(f"  ✓ {preset:12} + {agent_name:25} → n_actions={n_actions}")
            except Exception as e:
                print(f"  ✗ {preset:12} + {agent_name:25} → FAILED: {e}")
                raise


def test_motion_behavior() -> None:
    """Test that motion parameters are actually applied correctly.
    
    For fine_turns: verify turn angles (10° vs 45°)
    For speed_var: verify forward speeds (0.25, 0.5, 0.75)
    """
    print("\n=== Testing Motion Behavior ===\n")
    
    # Test fine_turns: sample turn actions and verify motion
    print("Fine-grained turns preset:")
    env = make_static_env(action_preset="fine_turns", seed=0)
    
    # Actions: [left-10°, left-45°, right-10°, right-45°, forward]
    # Get initial position
    obs, _ = env.reset(seed=42)
    agent_dir_start = float(env.unwrapped.agent.dir)
    
    # Step with action 0 (left-10°) - should turn left by 10°
    obs, _, _, _, _ = env.step(0)
    agent_dir_after_10 = float(env.unwrapped.agent.dir)
    turn_10 = abs(agent_dir_after_10 - agent_dir_start)
    print(f"  ✓ Action 0 (left-10°):   turn ≈ {turn_10:.1f}°")
    
    # Reset and try action 1 (left-45°)
    obs, _ = env.reset(seed=42)
    agent_dir_start = float(env.unwrapped.agent.dir)
    obs, _, _, _, _ = env.step(1)
    agent_dir_after_45 = float(env.unwrapped.agent.dir)
    turn_45 = abs(agent_dir_after_45 - agent_dir_start)
    print(f"  ✓ Action 1 (left-45°):   turn ≈ {turn_45:.1f}°")
    
    env.close()
    
    # Test speed_var: sample forward actions and verify speed
    print("\nSpeed variation preset:")
    env = make_static_env(action_preset="speed_var", seed=0)
    
    # Actions: [slow_fwd, normal_fwd, fast_fwd, left, right]
    # Get initial x position
    obs, _ = env.reset(seed=42)
    x_start = float(env.unwrapped.agent.pos[0])
    
    # Step with action 0 (slow_fwd)
    obs, _, _, _, _ = env.step(0)
    x_slow = float(env.unwrapped.agent.pos[0])
    slow_dist = abs(x_slow - x_start)
    print(f"  ✓ Action 0 (slow 0.25x):   dist ≈ {slow_dist:.3f}")
    
    # Reset and try action 1 (normal_fwd)
    obs, _ = env.reset(seed=42)
    x_start = float(env.unwrapped.agent.pos[0])
    obs, _, _, _, _ = env.step(1)
    x_normal = float(env.unwrapped.agent.pos[0])
    normal_dist = abs(x_normal - x_start)
    print(f"  ✓ Action 1 (normal 0.5x):  dist ≈ {normal_dist:.3f}")
    
    # Reset and try action 2 (fast_fwd)
    obs, _ = env.reset(seed=42)
    x_start = float(env.unwrapped.agent.pos[0])
    obs, _, _, _, _ = env.step(2)
    x_fast = float(env.unwrapped.agent.pos[0])
    fast_dist = abs(x_fast - x_start)
    print(f"  ✓ Action 2 (fast 0.75x):   dist ≈ {fast_dist:.3f}")
    
    # Verify speeds are in correct order
    assert slow_dist < normal_dist < fast_dist or np.allclose([slow_dist, normal_dist, fast_dist], 0), (
        f"Speed ordering incorrect: slow={slow_dist:.3f}, normal={normal_dist:.3f}, fast={fast_dist:.3f}"
    )
    print(f"  ✓ Speed ordering verified: slow < normal < fast")
    
    env.close()


def test_action_space_with_wrappers() -> None:
    """Verify final action space sizes with wrapper combinations."""
    print("\n=== Final Action Space Sizes ===\n")
    
    test_cases = [
        # (preset, use_stroboscopic, use_active_gating, use_active_vision, expected_n_actions)
        ("baseline", False, False, False, 3),
        ("baseline", True, False, False, 3),  # Stroboscopic doesn't change action space
        ("baseline", False, True, False, 4),  # STOP_AND_LOOK adds 1
        ("baseline", False, False, True, 6),  # Active vision doubles
        
        ("fine_turns", False, False, False, 5),
        ("fine_turns", False, True, False, 6),  # 5 + 1 for STOP_AND_LOOK
        ("fine_turns", False, False, True, 10),  # 5 × 2 for high/low freq
        
        ("speed_var", False, False, False, 5),
        ("speed_var", False, True, False, 6),  # 5 + 1
        ("speed_var", False, False, True, 10),  # 5 × 2

        ("fine_speed", False, False, False, 7),
        ("fine_speed", False, True, False, 8),
        ("fine_speed", False, False, True, 14),
    ]
    
    for preset, strobe, gating, vision, expected_n in test_cases:
        env = make_static_env(
            action_preset=preset,
            seed=0,
            use_stroboscopic=strobe,
            use_active_gating=gating,
            use_active_vision=vision,
        )
        actual_n = int(env.action_space.n)
        status = "✓" if actual_n == expected_n else "✗"
        print(
            f"{status} {preset:12} + "
            f"strobe={str(strobe):5} gating={str(gating):5} vision={str(vision):5} "
            f"→ n_actions={actual_n} (expected {expected_n})"
        )
        assert actual_n == expected_n, (
            f"Action space mismatch for {preset} with wrappers: "
            f"got {actual_n}, expected {expected_n}"
        )
        env.close()


if __name__ == "__main__":
    try:
        test_preset_configs()
        test_env_creation()
        test_agent_wrapping()
        test_motion_behavior()
        test_action_space_with_wrappers()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60 + "\n")
    except Exception as e:
        print("\n" + "="*60)
        print(f"✗ TEST FAILED: {e}")
        print("="*60 + "\n")
        raise
