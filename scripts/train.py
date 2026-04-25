"""Entrypoint for training an agent from a YAML config.

Usage:
    python scripts/train.py --config configs/agent_a_static.yaml
    python scripts/train.py --config configs/agent_a_static.yaml --track  # wandb on
    python scripts/train.py --config configs/agent_a_static.yaml --total-timesteps 10000  # smoke test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from cff_rl.agents.ppo import PPOConfig, train as ppo_train
from cff_rl.envs.static_maze import make_static_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--track", action="store_true", help="Enable wandb logging")
    p.add_argument("--total-timesteps", type=int, default=None, help="Override total_timesteps")
    p.add_argument("--seed", type=int, default=None, help="Override seed")
    p.add_argument("--exp-name", type=str, default=None, help="Override exp_name")
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

    def env_fn(seed: int):
        return make_static_env(env_id=env_id, seed=seed)

    ppo_train(cfg, env_fn)


if __name__ == "__main__":
    main()
