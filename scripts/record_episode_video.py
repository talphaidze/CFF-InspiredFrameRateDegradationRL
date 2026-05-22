"""Record a single episode as a video.

Left panel  : top-down schematic — maze, pillars, moving distractors, goal, agent
              arrow, trajectory coloured green (HF) / red (LF).
Right panel : agent's first-person 64×64 observation (upscaled).
Final frames: per-episode HF spatial heatmap held for 3 s.

Usage:
    uv run python scripts/record_episode_video.py
    uv run python scripts/record_episode_video.py --ckpt PATH --seed 7
    uv run python scripts/record_episode_video.py --find-seed-n 30 --out results/ep.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from cff_rl.agents.ppo_lstm import RecurrentNatureCNN
from cff_rl.envs.static_maze import make_static_env

ENV_ID = "MiniWorld-FourRoomsHardDynamic-v0"
DEFAULT_CKPT = (
    "runs/agent_c2_fourroomshard_dynamic_4d_lstm_proprio_turn10_6M_gate_"
    "vc000001__42__1779343926/ckpt_001464.pt"
)

MAZE_XLIM = (-7.8, 7.8)
MAZE_ZLIM = (-7.8, 7.8)
ROOM_BOUNDS = [(-7, -1, 1, 7), (1, 7, 1, 7), (1, 7, -7, -1), (-7, -1, -7, -1)]
PILLARS    = [(-4., 4.), (4., 4.), (4., -4.), (-4., -4.)]
DOORWAYS   = [(0., 4.), (4., 0.), (0., -4.), (-4., 0.)]
DISTRACTOR_COLORS = ["#f0a500", "#00b4d8", "#90e0ef", "#caf0f8"]

VIDEO_FPS       = 10
HEATMAP_HOLD_S  = 3
GRID_CELLS      = 40
MIN_VISITS      = 1   # single episode — relax threshold

FIG_W_IN = 13.0
FIG_H_IN =  6.5
DPI      = 100


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

def load_agent(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw  = ckpt.get("arch", {})
    arch = {
        "frame_stack":       raw.get("frame_stack", 1),
        "lstm_hidden_size":  raw.get("lstm_hidden_size", 256),
        "lstm_num_layers":   raw.get("lstm_num_layers", 1),
        "use_proprio":       raw.get("use_proprio", True),
        "use_active_vision": raw.get("use_active_vision", True),
        "vision_cost":       raw.get("vision_cost", 0.0001),
        "strobe_k":          raw.get("strobe_k", 7),
        "hf_strobe_k":       raw.get("hf_strobe_k", 1),
        "high_freq_steps":   raw.get("high_freq_steps", 35),
        "use_depth":         raw.get("use_depth", False),
        "use_fresh_gate":    raw.get("use_fresh_gate", False),
        "n_extras":          raw.get("n_extras", 5),
        "turn_step_deg":     raw.get("turn_step_deg", 10),
    }
    probe = make_static_env(
        env_id=ENV_ID, seed=1,
        frame_stack=arch["frame_stack"],
        use_proprio=arch["use_proprio"],
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
        if arch["use_proprio"] else probe.observation_space.shape
    )
    n_actions = int(probe.action_space.n)
    probe.close()

    agent = RecurrentNatureCNN(
        in_channels=obs_shape[0],
        n_actions=n_actions,
        lstm_hidden_size=arch["lstm_hidden_size"],
        lstm_num_layers=arch["lstm_num_layers"],
        n_extras=arch["n_extras"],
        use_fresh_gate=arch["use_fresh_gate"],
    ).to(device)
    agent.load_state_dict(ckpt["agent"])
    agent.eval()
    return agent, arch


# ---------------------------------------------------------------------------
# Seed selection
# ---------------------------------------------------------------------------

def find_far_seed(arch: dict, n: int = 30) -> tuple[int, float]:
    best_seed, best_dist = 1, 0.0
    for seed in range(1, n + 1):
        env = make_static_env(
            env_id=ENV_ID, seed=seed,
            frame_stack=arch["frame_stack"],
            use_proprio=arch["use_proprio"],
            use_active_vision=arch["use_active_vision"],
            vision_cost=arch["vision_cost"],
            strobe_k=arch["strobe_k"],
            hf_strobe_k=arch["hf_strobe_k"],
            high_freq_steps=arch["high_freq_steps"],
            use_depth=arch["use_depth"],
            turn_step_deg=arch["turn_step_deg"],
        )
        env.reset(seed=seed)
        inner = env.unwrapped
        d = float(np.linalg.norm(inner.agent.pos[[0, 2]] - inner.box.pos[[0, 2]]))
        env.close()
        print(f"  seed {seed:3d}  dist={d:.2f}")
        if d > best_dist:
            best_dist = d
            best_seed = seed
    return best_seed, best_dist


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def _draw_maze_base(ax):
    ax.set_facecolor("#111827")
    for (x0, x1, z0, z1) in ROOM_BOUNDS:
        ax.add_patch(mpatches.Rectangle(
            (x0, z0), x1 - x0, z1 - z0,
            linewidth=1.5, edgecolor="#d1d5db", facecolor="#1e3a5f", alpha=0.65,
        ))
    for px, pz in PILLARS:
        ax.add_patch(mpatches.Rectangle(
            (px - 0.5, pz - 0.5), 1.0, 1.0,
            linewidth=0, facecolor="#6b7280", alpha=0.95,
        ))
    for i, (dx, dz) in enumerate(DOORWAYS):
        ax.plot(dx, dz, "*", ms=8, color="cyan",
                markeredgecolor="#003355", markeredgewidth=0.5, zorder=3, alpha=0.7,
                label="doorway" if i == 0 else None)


def draw_episode_frame(
    ax_top, ax_fp,
    trajectory, hf_flags,
    agent_xz, agent_dir,
    goal_xz, distractor_xzs,
    obs_img, step,
):
    ax_top.cla()
    _draw_maze_base(ax_top)

    # Goal
    gx, gz = goal_xz
    ax_top.add_patch(mpatches.Rectangle(
        (gx - 0.45, gz - 0.45), 0.9, 0.9,
        linewidth=1.5, edgecolor="#ff8888", facecolor="#cc0000", alpha=0.95, zorder=4,
    ))

    # Moving distractors
    for i, (dx, dz) in enumerate(distractor_xzs):
        ax_top.add_patch(plt.Circle(
            (dx, dz), 0.35,
            color=DISTRACTOR_COLORS[i % len(DISTRACTOR_COLORS)], alpha=0.9, zorder=4,
        ))

    # Trajectory segments coloured by HF/LF
    for i in range(1, len(trajectory)):
        x0, z0 = trajectory[i - 1]
        x1, z1 = trajectory[i]
        col = "#22dd55" if hf_flags[i] else "#dd2222"
        ax_top.plot([x0, x1], [z0, z1], color=col, lw=2.2, alpha=0.85, zorder=5,
                    solid_capstyle="round")

    # Agent arrow
    ax, az = agent_xz
    fdx = 0.65 * float(np.sin(agent_dir))
    fdz = 0.65 * float(np.cos(agent_dir))
    ax_top.annotate(
        "", xy=(ax + fdx, az + fdz), xytext=(ax, az),
        arrowprops=dict(arrowstyle="-|>", color="white", lw=1.8, mutation_scale=16),
        zorder=7,
    )
    ax_top.plot(ax, az, "o", color="white", ms=5, zorder=8)

    ax_top.set_xlim(MAZE_XLIM)
    ax_top.set_ylim(MAZE_ZLIM)
    ax_top.set_aspect("equal")
    ax_top.axis("off")

    cur_hf = hf_flags[-1] if hf_flags else False
    mode_str = "HF" if cur_hf else "LF"
    mode_col = "#22dd55" if cur_hf else "#dd2222"
    ax_top.set_title(
        f"Step {step:4d}   mode: {mode_str}",
        fontsize=11, color=mode_col, pad=5,
    )

    # First-person view
    ax_fp.cla()
    frame = obs_img[-1] if obs_img.ndim == 3 else obs_img
    ax_fp.imshow(frame, cmap="gray", vmin=0, vmax=255, origin="upper",
                 aspect="equal", interpolation="nearest")
    ax_fp.axis("off")
    ax_fp.set_facecolor("#111827")
    ax_fp.set_title("Agent view (64×64)", fontsize=9, color="#9ca3af", pad=3)


def fig_to_bgr(fig) -> np.ndarray:
    fig.canvas.draw()
    buf  = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    rgba = buf.reshape(h, w, 4)
    return cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Heatmap (per-episode)
# ---------------------------------------------------------------------------

def render_heatmap(
    spatial_records, goal_xz, outcome_str, frame_wh: tuple[int, int],
    action_filter: str | None = None,
) -> np.ndarray:
    """action_filter: None = all steps, 'forward' = forward only, 'turn' = turns only."""
    # spatial_records: list of (x, z, hf, is_forward)
    if action_filter == "forward":
        recs = [s for s in spatial_records if s[3] == 1]
        filter_label = "forward actions only"
    elif action_filter == "turn":
        recs = [s for s in spatial_records if s[3] == 0]
        filter_label = "turn actions only"
    else:
        recs = spatial_records
        filter_label = "all actions"

    if not recs:
        recs = spatial_records  # fallback: no data after filter

    xs  = np.array([s[0] for s in recs])
    zs  = np.array([s[1] for s in recs])
    hfs = np.array([s[2] for s in recs])
    x_range = [MAZE_XLIM[0], MAZE_XLIM[1]]
    z_range = [MAZE_ZLIM[0], MAZE_ZLIM[1]]

    total, xe, ze = np.histogram2d(xs, zs, bins=GRID_CELLS, range=[x_range, z_range])
    hf_sum, _, _  = np.histogram2d(xs, zs, bins=GRID_CELLS, range=[x_range, z_range], weights=hfs)
    with np.errstate(invalid="ignore"):
        frac = np.where(total >= MIN_VISITS, hf_sum / total, np.nan)

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 7.0), dpi=DPI, facecolor="#111827")
    fig.subplots_adjust(left=0.1, right=0.88, top=0.85, bottom=0.1)
    ax.set_facecolor("#111827")

    extent = [xe[0], xe[-1], ze[0], ze[-1]]
    im = ax.imshow(frac.T, origin="lower", extent=extent,
                   cmap=plt.cm.RdYlGn, vmin=0.0, vmax=1.0, aspect="equal")

    # Maze overlay
    for (x0, x1, z0, z1) in ROOM_BOUNDS:
        ax.add_patch(mpatches.Rectangle(
            (x0, z0), x1 - x0, z1 - z0,
            linewidth=0.8, edgecolor="white", facecolor="none", alpha=0.5,
        ))
    for px, pz in PILLARS:
        ax.add_patch(mpatches.Rectangle(
            (px - 0.5, pz - 0.5), 1.0, 1.0,
            linewidth=0, facecolor="#888888", alpha=0.7,
        ))
    for i, (dx, dz) in enumerate(DOORWAYS):
        ax.plot(dx, dz, "*", ms=10, color="cyan",
                markeredgecolor="black", markeredgewidth=0.5, zorder=5,
                label="doorway" if i == 0 else None)
    gx, gz = goal_xz
    ax.add_patch(mpatches.Rectangle(
        (gx - 0.45, gz - 0.45), 0.9, 0.9,
        linewidth=1.5, edgecolor="#ff6666", facecolor="#cc0000", alpha=0.9, zorder=6,
    ))

    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("HF fraction", fontsize=11, color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlim(MAZE_XLIM)
    ax.set_ylim(MAZE_ZLIM)
    ax.set_xlabel("x (m)", fontsize=11, color="white")
    ax.set_ylabel("z (m)", fontsize=11, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    ax.set_title(
        f"Per-episode HF heatmap  —  {outcome_str}\n{filter_label}    ★ = doorway    red = goal",
        fontsize=12, color="white", pad=8,
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.5)

    bgr = fig_to_bgr(fig)
    plt.close(fig)
    bgr_r = cv2.resize(bgr, frame_wh, interpolation=cv2.INTER_AREA)
    return bgr_r


# ---------------------------------------------------------------------------
# Per-episode forward/turn split heatmap — static PNG
# ---------------------------------------------------------------------------

def save_episode_heatmap_plot(spatial_records, goal_xz, outcome_str, out_path: Path):
    """3-row PNG (all / forward / turns) matching plot_spatial_single style."""
    x_range = [MAZE_XLIM[0], MAZE_XLIM[1]]
    z_range = [MAZE_ZLIM[0], MAZE_ZLIM[1]]

    row_labels     = ["all actions", "forward only", "turns only"]
    action_filters = [None, "forward", "turn"]

    fig, axes = plt.subplots(3, 1, figsize=(5.5, 5.0 * 3), constrained_layout=True)

    im_ref = None
    for ax, row_lbl, af in zip(axes, row_labels, action_filters):
        if af == "forward":
            recs = [s for s in spatial_records if s[3] == 1]
        elif af == "turn":
            recs = [s for s in spatial_records if s[3] == 0]
        else:
            recs = spatial_records
        if not recs:
            recs = spatial_records

        xs  = np.array([s[0] for s in recs])
        zs  = np.array([s[1] for s in recs])
        hfs = np.array([s[2] for s in recs])
        total, xe, ze = np.histogram2d(xs, zs, bins=GRID_CELLS, range=[x_range, z_range])
        hf_sum, _, _  = np.histogram2d(xs, zs, bins=GRID_CELLS, range=[x_range, z_range], weights=hfs)
        with np.errstate(invalid="ignore"):
            frac = np.where(total >= MIN_VISITS, hf_sum / total, np.nan)

        extent = [xe[0], xe[-1], ze[0], ze[-1]]
        im = ax.imshow(frac.T, origin="lower", extent=extent,
                       cmap=plt.cm.RdYlGn, vmin=0.0, vmax=1.0, aspect="equal")
        im_ref = im

        # Maze overlay
        for (x0, x1, z0, z1) in ROOM_BOUNDS:
            ax.add_patch(mpatches.Rectangle(
                (x0, z0), x1 - x0, z1 - z0,
                linewidth=0.8, edgecolor="white", facecolor="none", alpha=0.5,
            ))
        for px, pz in PILLARS:
            ax.add_patch(mpatches.Rectangle(
                (px - 0.5, pz - 0.5), 1.0, 1.0,
                linewidth=0, facecolor="#888888", alpha=0.7,
            ))
        for i, (dx, dz) in enumerate(DOORWAYS):
            ax.plot(dx, dz, "*", ms=12, color="cyan",
                    markeredgecolor="black", markeredgewidth=0.5, zorder=5,
                    label="doorway" if i == 0 else None)
        gx, gz = goal_xz
        ax.add_patch(mpatches.Rectangle(
            (gx - 0.45, gz - 0.45), 0.9, 0.9,
            linewidth=1.5, edgecolor="#ff6666", facecolor="#cc0000", alpha=0.9, zorder=6,
        ))

        ax.set_xlim(x_range)
        ax.set_ylim(z_range)
        ax.set_xlabel("x (m)", fontsize=10)
        ax.set_ylabel("z (m)", fontsize=10)
        ax.set_title(f"{outcome_str} — {row_lbl}", fontsize=11)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.6)

    cb = fig.colorbar(im_ref, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("HF fraction per cell", fontsize=11)

    pillar_patch = mpatches.Patch(facecolor="#888888", label="pillar")
    fig.legend(handles=[pillar_patch], loc="lower center", ncol=2,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(
        f"Per-episode HF vision usage — {outcome_str}\n★ = doorway,  ■ = pillar,  red = goal",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--seed", type=int, default=None,
                   help="Episode seed (default: auto-select farthest)")
    p.add_argument("--find-seed-n", type=int, default=30,
                   help="Number of seeds to scan when auto-selecting")
    p.add_argument("--out", type=Path, default=Path("results/episode_video.mp4"))
    return p.parse_args()


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading agent ...")
    agent, arch = load_agent(args.ckpt, device)

    if args.seed is None:
        print(f"Scanning {args.find_seed_n} seeds for largest agent-goal distance ...")
        seed, dist = find_far_seed(arch, args.find_seed_n)
        print(f"  → chosen seed {seed}, distance {dist:.2f} m")
    else:
        seed = args.seed

    env = make_static_env(
        env_id=ENV_ID, seed=seed,
        frame_stack=arch["frame_stack"],
        use_proprio=arch["use_proprio"],
        use_active_vision=arch["use_active_vision"],
        vision_cost=arch["vision_cost"],
        strobe_k=arch["strobe_k"],
        hf_strobe_k=arch["hf_strobe_k"],
        high_freq_steps=arch["high_freq_steps"],
        use_depth=arch["use_depth"],
        turn_step_deg=arch["turn_step_deg"],
    )

    obs, _ = env.reset(seed=seed)
    inner  = env.unwrapped

    lstm_state = agent.initial_state(1, device)
    done_t     = torch.zeros(1, device=device)
    is_fresh   = (
        torch.ones(1, 1, dtype=torch.float32, device=device)
        if arch["use_fresh_gate"] else None
    )

    goal_xz   = (float(inner.box.pos[0]), float(inner.box.pos[2]))
    trajectory = [(float(inner.agent.pos[0]), float(inner.agent.pos[2]))]
    hf_flags   = [False]
    spatial    = []

    # Figure for episode frames
    fig, (ax_top, ax_fp) = plt.subplots(
        1, 2, figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI,
        facecolor="#111827",
        gridspec_kw={"width_ratios": [3, 1]},
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02, wspace=0.06)

    frame_w = int(FIG_W_IN * DPI)
    frame_h = int(FIG_H_IN * DPI)
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(str(args.out), fourcc, VIDEO_FPS, (frame_w, frame_h))

    step = 0
    terminated = truncated = False

    print("Recording episode ...")
    while not (terminated or truncated):
        pos         = inner.agent.pos.copy()
        agent_dir   = float(inner.agent.dir)
        distractors = [(float(b.pos[0]), float(b.pos[2])) for b in inner._distractors]
        img = obs["image"] if arch["use_proprio"] else obs

        draw_episode_frame(
            ax_top, ax_fp,
            trajectory, hf_flags,
            (float(pos[0]), float(pos[2])), agent_dir,
            goal_xz, distractors,
            img, step,
        )
        writer.write(fig_to_bgr(fig))

        x_t = torch.as_tensor(
            obs["image"] if arch["use_proprio"] else obs,
            dtype=torch.float32,
        ).unsqueeze(0).to(device)
        ext = (
            torch.as_tensor(obs["extras"], dtype=torch.float32).unsqueeze(0).to(device)
            if arch["use_proprio"] else None
        )

        with torch.no_grad():
            a, _, _, _, lstm_state = agent.get_action_and_value(
                x_t, lstm_state, done_t, extras=ext, is_fresh=is_fresh, gate_alpha=1.0,
            )
        action = int(a.item())
        obs, _, terminated, truncated, info = env.step(action)
        hf = bool(info.get("high_freq", False))
        if arch["use_fresh_gate"]:
            is_fresh = torch.tensor([[float(hf)]], dtype=torch.float32, device=device)

        new_pos    = inner.agent.pos.copy()
        is_forward = int(action % 3 == 2)  # base actions: 0=turn_left, 1=turn_right, 2=forward
        trajectory.append((float(new_pos[0]), float(new_pos[2])))
        hf_flags.append(hf)
        spatial.append((float(new_pos[0]), float(new_pos[2]), float(hf), is_forward))
        step += 1

    env.close()
    plt.close(fig)

    outcome = "SUCCESS" if terminated else "TIMEOUT"
    print(f"Episode done: {outcome} in {step} steps")

    # Heatmap frame (all actions only — forward/turn split saved as a separate PNG)
    hold_frames = HEATMAP_HOLD_S * VIDEO_FPS
    print("Rendering heatmap ...")
    heatmap_bgr = render_heatmap(spatial, goal_xz, outcome, (frame_w, frame_h))
    for _ in range(hold_frames):
        writer.write(heatmap_bgr)

    writer.release()
    total_frames = step + hold_frames
    print(f"Saved {args.out}  ({total_frames} frames @ {VIDEO_FPS} fps, ~{total_frames/VIDEO_FPS:.1f} s)")

    # Separate PNG: forward vs turn heatmap split
    plot_path = args.out.with_name(args.out.stem + "_heatmap.png")
    save_episode_heatmap_plot(spatial, goal_xz, outcome, plot_path)


if __name__ == "__main__":
    main()
