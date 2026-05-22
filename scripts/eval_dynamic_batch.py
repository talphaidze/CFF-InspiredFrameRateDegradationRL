"""Batch evaluator for all fourroomshard_dynamic agent runs.

Discovers runs/agent_{a,b,c2}_*fourroomshard_dynamic* under RUNS_DIR,
picks the latest checkpoint in each, and runs eval if eval_results.json
is missing or --force is passed.  Saves results to the run directory
(same convention as eval.py).

Usage:
    python scripts/eval_dynamic_batch.py --dry-run
    python scripts/eval_dynamic_batch.py
    python scripts/eval_dynamic_batch.py --force
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cff_rl.agents.ppo import NatureCNN
from cff_rl.agents.ppo_lstm import RecurrentNatureCNN
from cff_rl.envs.static_maze import make_static_env

RUNS_DIR = Path("runs")
ENV_ID = "MiniWorld-FourRoomsHardDynamic-v0"
SEEDS = [1, 2, 3, 4, 5]
EPISODES = 50
PREFIXES = ("agent_a_", "agent_b_", "agent_c2_")


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = sorted(run_dir.glob("ckpt_*.pt"))
    return ckpts[-1] if ckpts else None


def needs_eval(out_file: Path, force: bool) -> bool:
    if force or not out_file.exists():
        return True
    try:
        d = json.loads(out_file.read_text())
        return d.get("env_id") != ENV_ID
    except Exception:
        return True


def run_seed(agent, device, arch, seed: int) -> dict:
    from cff_rl.agents.ppo import _count_reversals

    recurrent = arch["recurrent"]
    frame_stack = arch["frame_stack"]
    use_proprio = arch["use_proprio"]
    use_stroboscopic = arch["use_stroboscopic"]
    use_active_gating = arch["use_active_gating"]
    use_active_vision = arch["use_active_vision"]
    vision_cost = arch["vision_cost"]
    strobe_k = arch["strobe_k"]
    hf_strobe_k = arch["hf_strobe_k"]
    high_freq_steps = arch["high_freq_steps"]
    turn_step_deg = arch["turn_step_deg"]
    use_fresh_gate = arch.get("use_fresh_gate", False)

    env = make_static_env(
        env_id=ENV_ID,
        seed=seed,
        frame_stack=frame_stack,
        use_proprio=use_proprio,
        use_stroboscopic=use_stroboscopic,
        use_active_gating=use_active_gating,
        use_active_vision=use_active_vision,
        vision_cost=vision_cost,
        strobe_k=strobe_k,
        hf_strobe_k=hf_strobe_k,
        high_freq_steps=high_freq_steps,
        turn_step_deg=turn_step_deg,
    )

    def _split(o):
        if use_proprio:
            img = torch.as_tensor(o["image"], dtype=torch.float32).unsqueeze(0).to(device)
            ext = torch.as_tensor(o["extras"], dtype=torch.float32).unsqueeze(0).to(device)
            return img, ext
        return torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).to(device), None

    inner = env.unwrapped
    MOVE_FORWARD_IDX = 2

    successes, returns, lengths, reversals, sal_counts, hf_counts, collisions_list = (
        [], [], [], [], [], [], []
    )
    for ep in range(EPISODES):
        obs, _ = env.reset(seed=seed + ep)
        ep_ret, ep_len, ep_acts, ep_sal, ep_hf, ep_col = 0.0, 0, [], 0, 0, 0
        terminated = truncated = False
        if recurrent:
            lstm_state = agent.initial_state(1, device)
            done_t = torch.zeros(1, device=device)
        # gate fully applied at eval; first obs treated as fresh
        is_fresh = torch.ones(1, 1, dtype=torch.float32, device=device) if use_fresh_gate else None
        while not (terminated or truncated):
            x, ext = _split(obs)
            with torch.no_grad():
                if recurrent:
                    a, _, _, _, lstm_state = agent.get_action_and_value(
                        x, lstm_state, done_t, extras=ext,
                        is_fresh=is_fresh, gate_alpha=1.0,
                    )
                    action = int(a.item())
                else:
                    a, _, _, _ = agent.get_action_and_value(x, ext)
                    action = int(a.item())
            pos_before = inner.agent.pos.copy()
            obs, r, terminated, truncated, info = env.step(action)
            pos_after = inner.agent.pos.copy()
            if action == MOVE_FORWARD_IDX and np.linalg.norm(pos_after - pos_before) < 1e-6:
                ep_col += 1
            ep_ret += float(r)
            ep_len += 1
            ep_acts.append(action)
            if use_active_gating:
                ep_sal += int(info["stop_and_look"])
            if use_active_vision:
                ep_hf += int(info["high_freq"])
                if use_fresh_gate:
                    is_fresh = torch.tensor(
                        [[float(info["high_freq"])]], dtype=torch.float32, device=device
                    )
        successes.append(1.0 if terminated else 0.0)
        returns.append(ep_ret)
        lengths.append(ep_len)
        reversals.append(_count_reversals(ep_acts))
        sal_counts.append(ep_sal)
        hf_counts.append(ep_hf)
        collisions_list.append(ep_col)
    env.close()
    return {
        "seed": seed,
        "episodes": EPISODES,
        "success_rate": float(np.mean(successes)),
        "mean_return": float(np.mean(returns)),
        "mean_length": float(np.mean(lengths)),
        "mean_reversals": float(np.mean(reversals)),
        "mean_collisions": float(np.mean(collisions_list)),
        "mean_sal_per_episode": float(np.mean(sal_counts)),
        "mean_highfreq_per_episode": float(np.mean(hf_counts)),
    }


def eval_run(run_dir: Path, dry_run: bool) -> None:
    ckpt_path = find_latest_checkpoint(run_dir)
    if ckpt_path is None:
        print(f"  [skip] no checkpoint in {run_dir.name}")
        return

    out_file = run_dir / "eval_results.json"
    if dry_run:
        print(f"  [would eval] {run_dir.name}  ckpt={ckpt_path.name}")
        return

    print(f"  [eval] {run_dir.name}  ckpt={ckpt_path.name}")
    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt_data.get("arch", {})
    arch = {
        "recurrent": raw.get("recurrent", raw.get("use_lstm", False)),
        "frame_stack": raw.get("frame_stack", 4),
        "lstm_hidden_size": raw.get("lstm_hidden_size", 256),
        "lstm_num_layers": raw.get("lstm_num_layers", 1),
        "use_proprio": raw.get("use_proprio", False),
        "use_stroboscopic": raw.get("use_stroboscopic", False),
        "use_active_gating": raw.get("use_active_gating", False),
        "use_active_vision": raw.get("use_active_vision", False),
        "vision_cost": raw.get("vision_cost", 0.01),
        "strobe_k": raw.get("strobe_k", 7),
        "hf_strobe_k": raw.get("hf_strobe_k", 1),
        "use_depth": raw.get("use_depth", False),
        "use_fresh_gate": raw.get("use_fresh_gate", False),
        "high_freq_steps": raw.get("high_freq_steps", 35),
        "n_extras": raw.get("n_extras", 0),
        "turn_step_deg": raw.get("turn_step_deg", 90),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe = make_static_env(
        env_id=ENV_ID,
        seed=1,
        frame_stack=arch["frame_stack"],
        use_proprio=arch["use_proprio"],
        use_stroboscopic=arch["use_stroboscopic"],
        use_active_gating=arch["use_active_gating"],
        use_active_vision=arch["use_active_vision"],
        vision_cost=arch["vision_cost"],
        strobe_k=arch["strobe_k"],
        hf_strobe_k=arch["hf_strobe_k"],
        high_freq_steps=arch["high_freq_steps"],
        use_depth=arch["use_depth"],
        turn_step_deg=arch["turn_step_deg"],
    )
    obs_shape = (
        probe.observation_space["image"].shape
        if arch["use_proprio"]
        else probe.observation_space.shape
    )
    n_actions = int(probe.action_space.n)
    probe.close()

    if arch["recurrent"]:
        agent = RecurrentNatureCNN(
            in_channels=obs_shape[0],
            n_actions=n_actions,
            lstm_hidden_size=arch["lstm_hidden_size"],
            lstm_num_layers=arch["lstm_num_layers"],
            n_extras=arch["n_extras"],
            use_fresh_gate=arch["use_fresh_gate"],
        ).to(device)
    else:
        agent = NatureCNN(
            in_channels=obs_shape[0], n_actions=n_actions, n_extras=arch["n_extras"]
        ).to(device)
    agent.load_state_dict(ckpt_data["agent"])
    agent.eval()

    per_seed = [run_seed(agent, device, arch, s) for s in SEEDS]

    def agg(key: str):
        vals = np.array([r[key] for r in per_seed], dtype=np.float64)
        return float(vals.mean()), float(vals.std(ddof=0))

    summary = {
        "checkpoint": str(ckpt_path),
        "env_id": ENV_ID,
        "seeds": SEEDS,
        "episodes_per_seed": EPISODES,
        "deterministic": False,
        "per_seed": per_seed,
        "success_rate": agg("success_rate"),
        "mean_return": agg("mean_return"),
        "mean_length": agg("mean_length"),
        "mean_reversals": agg("mean_reversals"),
        "mean_collisions": agg("mean_collisions"),
        "mean_sal_per_episode": agg("mean_sal_per_episode"),
        "mean_highfreq_per_episode": agg("mean_highfreq_per_episode"),
    }
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"    success_rate={summary['success_rate'][0]:.3f}  wrote {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-eval even if results exist")
    args = parser.parse_args()

    run_dirs = sorted(
        d
        for d in RUNS_DIR.iterdir()
        if d.is_dir()
        and "fourroomshard_dynamic" in d.name
        and any(d.name.startswith(p) for p in PREFIXES)
    )

    print(f"Found {len(run_dirs)} dynamic runs")
    to_eval = [d for d in run_dirs if needs_eval(d / "eval_results.json", args.force)]
    skip = len(run_dirs) - len(to_eval)
    if skip:
        print(f"Skipping {skip} already-evaluated runs (use --force to re-eval)")
    if not to_eval:
        print("Nothing to evaluate.")
        return

    for run_dir in to_eval:
        eval_run(run_dir, args.dry_run)


if __name__ == "__main__":
    main()
