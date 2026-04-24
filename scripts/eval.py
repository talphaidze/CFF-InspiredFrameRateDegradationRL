"""Rollout a trained checkpoint and report Regime 1 metrics.

Primary metric (proposal § Method, Regime 1): steps to 90% success rate
across 100 episodes. This script reports success rate, mean return, mean
episode length, and mean direction-reversal count for a given checkpoint.

Usage:
    python scripts/eval.py --checkpoint runs/<run>/ckpt_000050.pt --episodes 100
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from cff_rl.agents.ppo import NatureCNN, _count_reversals
from cff_rl.envs.static_maze import make_static_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-id", type=str, default="MiniWorld-OneRoom-v0")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--deterministic", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_static_env(env_id=args.env_id, seed=args.seed)
    obs_shape = env.observation_space.shape
    n_actions = int(env.action_space.n)

    agent = NatureCNN(in_channels=obs_shape[0], n_actions=n_actions).to(device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    successes, returns, lengths, reversals = [], [], [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_ret, ep_len, ep_acts = 0.0, 0, []
        terminated = truncated = False
        while not (terminated or truncated):
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                if args.deterministic:
                    h = agent.encode(x)
                    action = int(agent.actor(h).argmax(dim=-1).item())
                else:
                    a, _, _, _ = agent.get_action_and_value(x)
                    action = int(a.item())
            obs, r, terminated, truncated, _ = env.step(action)
            ep_ret += float(r)
            ep_len += 1
            ep_acts.append(action)
        successes.append(1.0 if terminated else 0.0)
        returns.append(ep_ret)
        lengths.append(ep_len)
        reversals.append(_count_reversals(ep_acts))

    env.close()
    print(f"episodes: {args.episodes}")
    print(f"success_rate: {np.mean(successes):.3f}")
    print(f"mean_return:  {np.mean(returns):.3f}")
    print(f"mean_length:  {np.mean(lengths):.1f}")
    print(f"mean_reversals: {np.mean(reversals):.2f}")


if __name__ == "__main__":
    main()
