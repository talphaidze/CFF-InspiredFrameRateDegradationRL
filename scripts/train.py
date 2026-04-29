"""Entrypoint for training an agent from a YAML config.

Usage:
    python scripts/train.py --config configs/agent_a_static.yaml
    python scripts/train.py --config configs/agent_a_static.yaml --track  # wandb on
    python scripts/train.py --config configs/agent_a_static.yaml --total-timesteps 10000  # smoke test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import yaml

from cff_rl.agents.ppo import PPOConfig, train as ppo_train_ff
from cff_rl.agents.ppo_lstm import train as ppo_train_lstm
from cff_rl.envs.static_maze import make_static_env
from cff_rl.envs.wrappers import VideoCompositeWrapper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--track", action="store_true", help="Enable wandb logging")
    p.add_argument("--total-timesteps", type=int, default=None, help="Override total_timesteps")
    p.add_argument("--seed", type=int, default=None, help="Override seed")
    p.add_argument("--exp-name", type=str, default=None, help="Override exp_name")
    p.add_argument("--record-video", action="store_true", help="Record videos on env 0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        raw = yaml.safe_load(f)

    env_id = raw.pop("env_id", "MiniWorld-OneRoom-v0")

    cfg = PPOConfig(**{k: v for k, v in raw.items() if k in PPOConfig.__dataclass_fields__})
    if args.track:
        cfg.track = True
    if args.total_timesteps is not None:
        cfg.total_timesteps = args.total_timesteps
    if args.seed is not None:
        cfg.seed = args.seed
    if args.exp_name is not None:
        cfg.exp_name = args.exp_name
    if args.record_video:
        cfg.record_video = True

    import time

    run_name = f"{cfg.exp_name}__{cfg.seed}__{int(time.time())}"
    video_dir = Path(cfg.log_dir) / run_name / "videos"
    every = max(1, cfg.video_every)

    def env_fn(seed: int, env_idx: int = 0):
        use_stroboscopic = cfg.exp_name.startswith("agent_b")
        common = dict(
            env_id=env_id,
            seed=seed,
            use_stroboscopic=use_stroboscopic,
            strobe_k=7,
            frame_stack=cfg.frame_stack,
            use_proprio=cfg.use_proprio,
            turn_step_deg=cfg.turn_step_deg,
        )
        if cfg.record_video and env_idx == 0:
            env = make_static_env(render_mode="rgb_array", **common)
            env = VideoCompositeWrapper(env, render_fps=4)
            video_dir.mkdir(parents=True, exist_ok=True)
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=str(video_dir),
                episode_trigger=lambda ep: ep % every == 0,
                disable_logger=True,
            )
            return env
        return make_static_env(**common)

    # Inject the pre-computed run_name so the PPO trainer uses the same dir as
    # the videos written above. PPOConfig has no run_name field; instead we
    # rely on cfg.exp_name + seed + the wall-clock timestamp baked into
    # run_name. Pass it through via an env var the trainer can pick up.
    import os as _os

    _os.environ["CFF_RUN_NAME"] = run_name

    train_fn = ppo_train_lstm if cfg.recurrent else ppo_train_ff
    print(f"agent: {'recurrent (LSTM)' if cfg.recurrent else 'feed-forward'}")
    train_fn(cfg, env_fn)


if __name__ == "__main__":
    main()
