"""Visualise agent trajectories on a top-down map with perception-mode heatmap.

Runs episodes identical to eval.py, but at every step records the agent's
(x, z) world position and whether it used high-freq vision (Agent C v2) or
STOP_AND_LOOK (Agent C v1).  After all episodes, renders:

    * top-down map as background
    * agent path (thin line per episode)
    * hot-coloured scatter points where the agent chose high-freq / SAL

Usage:
    python scripts/eval_heatmap.py --checkpoint runs/<run>/ckpt.pt \
        --env-id MiniWorld-FourRooms-v0 --episodes 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

from cff_rl.agents.ppo import NatureCNN
from cff_rl.agents.ppo_lstm import RecurrentNatureCNN
from cff_rl.envs.static_maze import make_static_env


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--env-id", type=str, required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path.  Defaults to <checkpoint_dir>/heatmap.png.",
    )
    # Overrides for older checkpoints missing perception flags.
    p.add_argument("--use-stroboscopic", action="store_true")
    p.add_argument("--use-active-gating", action="store_true")
    p.add_argument("--use-active-vision", action="store_true")
    return p.parse_args()


def collect_trajectories(
    agent,
    device: torch.device,
    env: gym.Env,
    seed: int,
    episodes: int,
    deterministic: bool,
    recurrent: bool,
    use_proprio: bool,
    use_active_gating: bool,
    use_active_vision: bool,
) -> list[dict]:
    """Run *episodes* and return per-episode trajectory data.

    Each element is a dict with keys:
        positions : (T, 2) float  – (x, z) world coords per step
        high_freq : (T,)  bool    – True where agent chose high-freq / SAL
        actions   : (T,)  int     – raw action index per step
        success   : bool
    """

    def _split(o):
        if use_proprio:
            img = torch.as_tensor(o["image"], dtype=torch.float32).unsqueeze(0).to(device)
            ext = torch.as_tensor(o["extras"], dtype=torch.float32).unsqueeze(0).to(device)
            return img, ext
        return torch.as_tensor(o, dtype=torch.float32).unsqueeze(0).to(device), None

    trajectories: list[dict] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)

        # Grab top-down snapshot right after reset so it matches this episode's layout.
        inner = env.unwrapped
        fb = inner.vis_fb
        topdown_img = inner.render_top_view(fb)

        # Compute viewport bounds (same logic as MiniWorld's render_top_view).
        td_min_x = float(inner.min_x) - 1
        td_max_x = float(inner.max_x) + 1
        td_min_z = float(inner.min_z) - 1
        td_max_z = float(inner.max_z) + 1
        w_width = td_max_x - td_min_x
        w_height = td_max_z - td_min_z
        aspect = w_width / w_height
        fb_aspect = fb.width / fb.height
        if aspect > fb_aspect:
            new_h = w_width / fb_aspect
            h_diff = new_h - w_height
            td_min_z -= h_diff / 2
            td_max_z += h_diff / 2
        elif aspect < fb_aspect:
            new_w = w_height * fb_aspect
            w_diff = new_w - w_width
            td_min_x -= w_diff / 2
            td_max_x += w_diff / 2

        positions, hf_flags, actions = [], [], []
        terminated = truncated = False

        if recurrent:
            lstm_state = agent.initial_state(1, device)
            done_t = torch.zeros(1, device=device)

        while not (terminated or truncated):
            # Record position BEFORE the step (where the agent decided).
            pos = env.unwrapped.agent.pos
            positions.append([float(pos[0]), float(pos[2])])

            x, ext = _split(obs)
            with torch.no_grad():
                if recurrent:
                    if deterministic:
                        h, lstm_state = agent.get_states(x, lstm_state, done_t, extras=ext)
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

            obs, _r, terminated, truncated, info = env.step(action)
            actions.append(action)

            # Record perception mode.
            if use_active_vision:
                hf_flags.append(bool(info["high_freq"]))
            elif use_active_gating:
                hf_flags.append(bool(info["stop_and_look"]))
            else:
                hf_flags.append(False)

        trajectories.append(
            {
                "positions": np.array(positions, dtype=np.float32),
                "high_freq": np.array(hf_flags, dtype=bool),
                "actions": np.array(actions, dtype=np.int64),
                "success": bool(terminated),
                "topdown": topdown_img,
                "bounds": (td_min_x, td_max_x, td_min_z, td_max_z),
            }
        )
    return trajectories


def get_topdown_image(env: gym.Env) -> tuple[np.ndarray, float, float, float, float]:
    """Reset the env and grab its top-down rendering + actual viewport bounds.

    Returns (image, min_x, max_x, min_z, max_z) matching the exact
    orthographic projection used by MiniWorld's render_top_view, which
    adds 1-unit padding and adjusts for framebuffer aspect ratio.
    """
    env.reset()
    inner = env.unwrapped
    fb = inner.vis_fb

    # Replicate the bounds logic from MiniWorld.render_top_view.
    min_x = float(inner.min_x) - 1
    max_x = float(inner.max_x) + 1
    min_z = float(inner.min_z) - 1
    max_z = float(inner.max_z) + 1

    width = max_x - min_x
    height = max_z - min_z
    aspect = width / height
    fb_aspect = fb.width / fb.height

    if aspect > fb_aspect:
        new_h = width / fb_aspect
        h_diff = new_h - height
        min_z -= h_diff / 2
        max_z += h_diff / 2
    elif aspect < fb_aspect:
        new_w = height * fb_aspect
        w_diff = new_w - width
        min_x -= w_diff / 2
        max_x += w_diff / 2

    img = inner.render_top_view(fb)  # (H, W, 3) uint8
    return img, min_x, max_x, min_z, max_z


def world_to_pixel(
    positions: np.ndarray,
    img_shape: tuple[int, int],
    min_x: float,
    max_x: float,
    min_z: float,
    max_z: float,
) -> np.ndarray:
    """Map (x, z) world coords → (col, row) pixel coords on the top-down image."""
    h, w = img_shape[:2]
    xs = positions[:, 0]
    zs = positions[:, 1]
    # MiniWorld top-down: x → column (left-right), z → row (top-bottom, flipped).
    cols = (xs - min_x) / (max_x - min_x) * w
    rows = (1.0 - (zs - min_z) / (max_z - min_z)) * h
    return np.stack([cols, rows], axis=-1)


def plot_heatmap(
    traj: dict,
    ep_index: int,
    use_active_gating: bool,
    use_active_vision: bool,
    n_base_actions: int,
    out_path: Path,
) -> None:
    """Plot a single episode's trajectory on its own top-down background."""
    from matplotlib.collections import LineCollection

    topdown = traj["topdown"]
    min_x, max_x, min_z, max_z = traj["bounds"]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(topdown, extent=[min_x, max_x, min_z, max_z], origin="lower")

    # Action indices (base actions): 0 = turn_left, 1 = turn_right, 2 = forward.
    TURN_LEFT = 0
    TURN_RIGHT = 1

    col_hf = np.array([1.0, 0.0, 0.0, 0.85])
    col_lf = np.array([0.2, 0.4, 1.0, 0.55])

    pos = traj["positions"]   # (T, 2) with (x, z)
    hf = traj["high_freq"]    # (T,) bool
    acts = traj["actions"]    # (T,) int

    if len(pos) >= 2:
        # Base action regardless of active-vision doubling.
        base_acts = acts % n_base_actions

        # --- forward-step segments (position actually changes) ---
        segments = np.stack([pos[:-1], pos[1:]], axis=1)  # (T-1, 2, 2)
        seg_colors = np.where(hf[:-1, None], col_hf, col_lf)
        lc = LineCollection(segments, colors=seg_colors, linewidths=10)
        ax.add_collection(lc)

        # --- turn markers (position stays the same) ---
        turn_mask = (base_acts == TURN_LEFT) | (base_acts == TURN_RIGHT)
        if turn_mask.any():
            turn_pos = pos[turn_mask]
            turn_hf = hf[turn_mask]
            turn_dir = base_acts[turn_mask]

            for tp, thf, td in zip(turn_pos, turn_hf, turn_dir):
                color = col_hf[:3] if thf else col_lf[:3]
                m = "<" if td == TURN_LEFT else ">"
                ax.plot(tp[0], tp[1], m, color=color, markersize=12,
                        alpha=0.7, zorder=4, markeredgecolor="none")

    # Start / end markers.
    ax.plot(pos[0, 0], pos[0, 1], "o", color="lime", markersize=6,
            zorder=6, label="start")
    marker = "*" if traj["success"] else "X"
    ax.plot(pos[-1, 0], pos[-1, 1], marker, color="white", markersize=8,
            markeredgecolor="black", zorder=6,
            label=f"end ({'ok' if traj['success'] else 'fail'})")

    # Dummy artists for the legend.
    ax.plot([], [], color=col_hf[:3], linewidth=4, label="high-freq")
    ax.plot([], [], color=col_lf[:3], linewidth=4, label="low-freq")
    ax.plot([], [], "<", color="grey", markersize=6, linestyle="none", label="turn left")
    ax.plot([], [], ">", color="grey", markersize=6, linestyle="none", label="turn right")

    if use_active_vision:
        mode_label = f"Ep {ep_index} — Active Vision (C v2) — red = high-freq"
    elif use_active_gating:
        mode_label = f"Ep {ep_index} — Active Gating (C v1) — red = STOP_AND_LOOK"
    else:
        mode_label = f"Ep {ep_index} — Baseline"
    ax.set_title(mode_label, fontsize=13)
    # Episode stats text box.
    n_steps = len(pos)
    n_hf = int(hf.sum())
    base_acts_all = acts % n_base_actions
    n_forward = int((base_acts_all == 2).sum())
    n_turns = int(((base_acts_all == 0) | (base_acts_all == 1)).sum())
    from cff_rl.agents.ppo import _count_reversals
    n_reversals = _count_reversals(acts.tolist() if hasattr(acts, 'tolist') else list(acts))
    stats_text = (
        f"steps: {n_steps}\n"
        f"result: {'success' if traj['success'] else 'timeout'}\n"
        f"forwards: {n_forward}  turns: {n_turns}\n"
        f"high-freq: {n_hf} ({100*n_hf/max(n_steps,1):.1f}%)\n"
        f"reversals: {n_reversals}"
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

    ax.set_xlabel("x (world)")
    ax.set_ylabel("z (world)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200)
    plt.close(fig)
    print(f"saved heatmap → {out_path}")


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- unpack arch from checkpoint ----
    arch = ckpt.get("arch", {})
    recurrent = arch.get("recurrent", arch.get("use_lstm", False))
    frame_stack = arch.get("frame_stack", 4)
    lstm_hidden_size = arch.get("lstm_hidden_size", 256)
    lstm_num_layers = arch.get("lstm_num_layers", 1)
    use_proprio = arch.get("use_proprio", False)
    use_stroboscopic = arch.get("use_stroboscopic", False) or args.use_stroboscopic
    use_active_gating = arch.get("use_active_gating", False) or args.use_active_gating
    use_active_vision = arch.get("use_active_vision", False) or args.use_active_vision
    strobe_k = arch.get("strobe_k", 7)
    high_freq_steps = arch.get("high_freq_steps", 35)
    n_extras = arch.get("n_extras", 0)
    turn_step_deg = arch.get("turn_step_deg", 90)
    vision_cost = arch.get("vision_cost", 0.01)

    # ---- build env (rgb_array for top-down snapshot) ----
    env_kwargs = dict(
        env_id=args.env_id,
        seed=args.seed,
        frame_stack=frame_stack,
        use_proprio=use_proprio,
        use_stroboscopic=use_stroboscopic,
        use_active_gating=use_active_gating,
        use_active_vision=use_active_vision,
        vision_cost=vision_cost,
        strobe_k=strobe_k,
        high_freq_steps=high_freq_steps,
        turn_step_deg=turn_step_deg,
        render_mode="rgb_array",
    )
    env = make_static_env(**env_kwargs)

    # ---- build agent ----
    if use_proprio:
        obs_shape = env.observation_space["image"].shape
    else:
        obs_shape = env.observation_space.shape
    n_actions = int(env.action_space.n)

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

    # ---- collect trajectories ----
    trajectories = collect_trajectories(
        agent=agent,
        device=device,
        env=env,
        seed=args.seed,
        episodes=args.episodes,
        deterministic=args.deterministic,
        recurrent=recurrent,
        use_proprio=use_proprio,
        use_active_gating=use_active_gating,
        use_active_vision=use_active_vision,
    )
    env.close()

    # ---- summary ----
    n_success = sum(t["success"] for t in trajectories)
    total_steps = sum(len(t["positions"]) for t in trajectories)
    total_hf = sum(t["high_freq"].sum() for t in trajectories)
    print(f"episodes: {args.episodes}  success: {n_success}/{args.episodes}")
    print(f"total steps: {total_steps}  high-freq steps: {int(total_hf)}")
    for i, traj in enumerate(trajectories):
        pos = traj["positions"]
        print(f"\n--- episode {i} ({'ok' if traj['success'] else 'fail'}) ---")
        for t, (x, z) in enumerate(pos):
            hf = traj["high_freq"][t]
            act = traj["actions"][t]
            print(f"  step {t:4d}: x={x:7.3f} z={z:7.3f}  action={act}  hf={hf}")

    # ---- plot one image per episode ----
    from cff_rl.envs.static_maze import STATIC_ACTIONS
    base_out = args.out or (args.checkpoint.parent / "heatmap.png")
    for i, traj in enumerate(trajectories):
        if args.episodes == 1:
            out_path = base_out
        else:
            out_path = base_out.with_stem(f"{base_out.stem}_ep{i}")
        plot_heatmap(
            traj=traj,
            ep_index=i,
            use_active_gating=use_active_gating,
            use_active_vision=use_active_vision,
            n_base_actions=len(STATIC_ACTIONS),
            out_path=out_path,
        )


if __name__ == "__main__":
    main()
