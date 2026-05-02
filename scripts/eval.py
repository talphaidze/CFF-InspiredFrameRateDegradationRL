"""Rollout a trained checkpoint and report Regime 1 metrics.

Primary metric (proposal § Method, Regime 1): steps to 90% success rate
across N episodes. This script reports success rate, mean return, mean
episode length, and mean direction-reversal count for a given checkpoint,
aggregated across one or more seeds.

Dispatches on `arch.recurrent` saved in the checkpoint to load either
the feed-forward `NatureCNN` (ppo.py) or the recurrent `RecurrentNatureCNN`
(ppo_lstm.py).

Usage:
    python scripts/eval.py --checkpoint runs/<run>/ckpt_000050.pt --episodes 50
    python scripts/eval.py --checkpoint runs/<run>/ckpt_000050.pt --seeds 1 2 3 4 5 --episodes 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from cff_rl.agents.ppo import NatureCNN, _count_reversals
from cff_rl.agents.ppo_lstm import RecurrentNatureCNN
from cff_rl.envs.static_maze import make_static_env
from cff_rl.envs.wrappers import VideoCompositeWrapper


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-id", type=str, default="MiniWorld-FourRooms-v0")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="One or more eval seeds. Defaults to [--seed].",
    )
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument(
        "--record-video",
        action="store_true",
        help="Record videos of eval episodes (uses first seed only).",
    )
    p.add_argument(
        "--video-dir",
        type=Path,
        default=None,
        help="Where to write videos. Defaults to <checkpoint_dir>/eval_videos.",
    )
    return p.parse_args()


def run_seed(
    agent,
    device: torch.device,
    env_id: str,
    seed: int,
    episodes: int,
    deterministic: bool,
    recurrent: bool,
    frame_stack: int = 4,
    use_proprio: bool = False,
    use_stroboscopic: bool = False,
    use_active_gating: bool = False,
    strobe_k: int = 7,
    high_freq_steps: int = 35,
    turn_step_deg: int = 90,
    video_dir: Path | None = None,
) -> dict:
    common = dict(
        env_id=env_id,
        seed=seed,
        frame_stack=frame_stack,
        use_proprio=use_proprio,
        use_stroboscopic=use_stroboscopic,
        use_active_gating=use_active_gating,
        strobe_k=strobe_k,
        high_freq_steps=high_freq_steps,
        turn_step_deg=turn_step_deg,
    )
    if video_dir is not None:
        env = make_static_env(render_mode="rgb_array", **common)
        env = VideoCompositeWrapper(env, render_fps=4)
        video_dir.mkdir(parents=True, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda ep: True,
            disable_logger=True,
            name_prefix=f"eval-seed{seed}",
        )
    else:
        env = make_static_env(**common)

    def _split(o):
        if use_proprio:
            img = torch.as_tensor(o["image"], dtype=torch.float32).unsqueeze(0).to(device)
            ext = torch.as_tensor(o["extras"], dtype=torch.float32).unsqueeze(0).to(device)
            return img, ext
        return torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).to(device), None

    successes, returns, lengths, reversals, sal_counts = [], [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_ret, ep_len, ep_acts, ep_sal = 0.0, 0, [], 0
        terminated = truncated = False
        if recurrent:
            lstm_state = agent.initial_state(1, device)
            done_t = torch.zeros(1, device=device)
        while not (terminated or truncated):
            x, ext = _split(obs)
            with torch.no_grad():
                if recurrent:
                    if deterministic:
                        h, lstm_state = agent.get_states(
                            x, lstm_state, done_t, extras=ext
                        )
                        action = int(agent.actor(h).argmax(dim=-1).item())
                    else:
                        a, _, _, _, lstm_state = agent.get_action_and_value(
                            x, lstm_state, done_t, extras=ext
                        )
                        action = int(a.item())
                else:
                    if deterministic:
                        h = agent.encode(x, ext)
                        action = int(agent.actor(h).argmax(dim=-1).item())
                    else:
                        a, _, _, _ = agent.get_action_and_value(x, ext)
                        action = int(a.item())
            obs, r, terminated, truncated, info = env.step(action)
            ep_ret += float(r)
            ep_len += 1
            ep_acts.append(action)
            if use_active_gating:
                ep_sal += int(info["stop_and_look"])
        successes.append(1.0 if terminated else 0.0)
        returns.append(ep_ret)
        lengths.append(ep_len)
        reversals.append(_count_reversals(ep_acts))
        sal_counts.append(ep_sal)
    env.close()
    return {
        "seed": seed,
        "episodes": episodes,
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "mean_reversals": float(np.mean(reversals)),
        "mean_sal_per_episode": float(np.mean(sal_counts)),
    }


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else [args.seed]
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    arch = ckpt.get("arch", {})
    # Backwards-compat: older LSTM checkpoints stored `use_lstm` instead.
    recurrent = arch.get("recurrent", arch.get("use_lstm", False))
    frame_stack = arch.get("frame_stack", 4)
    lstm_hidden_size = arch.get("lstm_hidden_size", 256)
    lstm_num_layers = arch.get("lstm_num_layers", 1)
    use_proprio = arch.get("use_proprio", False)
    use_stroboscopic = arch.get("use_stroboscopic", False)
    use_active_gating = arch.get("use_active_gating", False)
    strobe_k = arch.get("strobe_k", 7)
    high_freq_steps = arch.get("high_freq_steps", 35)
    n_extras = arch.get("n_extras", 0)
    turn_step_deg = arch.get("turn_step_deg", 90)

    probe = make_static_env(
        env_id=args.env_id,
        seed=seeds[0],
        frame_stack=frame_stack,
        use_proprio=use_proprio,
        use_stroboscopic=use_stroboscopic,
        use_active_gating=use_active_gating,
        strobe_k=strobe_k,
        high_freq_steps=high_freq_steps,
        turn_step_deg=turn_step_deg,
    )
    if use_proprio:
        obs_shape = probe.observation_space["image"].shape
    else:
        obs_shape = probe.observation_space.shape
    n_actions = int(probe.action_space.n)
    probe.close()

    if recurrent:
        agent = RecurrentNatureCNN(
            in_channels=obs_shape[0],
            n_actions=n_actions,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            n_extras=n_extras,
        ).to(device)
    else:
        agent = NatureCNN(
            in_channels=obs_shape[0], n_actions=n_actions, n_extras=n_extras
        ).to(device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()

    agent_type = (
       "active-gating (C)" if use_active_gating
        else "stroboscopic (B)" if use_stroboscopic
        else " normal (A)"
    )
    print(
        f"agent type: {agent_type}"
        f"agent: {'recurrent (LSTM)' if recurrent else 'feed-forward'}, "
        f"frame_stack={frame_stack}, use_proprio={use_proprio}, turn={turn_step_deg}°"
    )

    video_dir = None
    if args.record_video:
        video_dir = args.video_dir or (args.checkpoint.parent / "eval_videos")

    per_seed = [
        run_seed(
            agent,
            device,
            args.env_id,
            s,
            args.episodes,
            args.deterministic,
            recurrent=recurrent,
            frame_stack=frame_stack,
            use_proprio=use_proprio,
            use_stroboscopic=use_stroboscopic,
            use_active_gating=use_active_gating,
            strobe_k=strobe_k,
            high_freq_steps=high_freq_steps,
            turn_step_deg=turn_step_deg,
            video_dir=video_dir if i == 0 else None,
        )
        for i, s in enumerate(seeds)
    ]

    def agg(key: str) -> tuple[float, float]:
        vals = np.array([r[key] for r in per_seed], dtype=np.float64)
        return float(vals.mean()), float(vals.std(ddof=0))

    summary = {
        "checkpoint": str(args.checkpoint),
        "env_id": args.env_id,
        "seeds": seeds,
        "episodes_per_seed": args.episodes,
        "deterministic": args.deterministic,
        "per_seed": per_seed,
        "success_rate": agg("success_rate"),
        "mean_return": agg("mean_return"),
        "mean_length": agg("mean_length"),
        "mean_reversals": agg("mean_reversals"),
        "mean_sal_per_episode": agg("mean_sal_per_episode"),
    }

    out = args.checkpoint.parent / "eval_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"checkpoint: {args.checkpoint}")
    print(f"seeds: {seeds}  episodes/seed: {args.episodes}")
    print(f"{'metric':<22} {'mean':>10} {'std':>10}")
    metrics = ["success_rate", "mean_return", "mean_length", "mean_reversals"]
    if use_active_gating:
        metrics.append("mean_sal_per_episode")
    for k in metrics:
        m, s = summary[k]
        print(f"{k:<22} {m:>10.3f} {s:>10.3f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
