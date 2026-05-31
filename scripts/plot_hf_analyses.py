"""HF vision usage analyses for the vision-cost sweep.

Analysis — Spatial HF heatmap:
    For vc=2e-5 and vc=1e-4, grid the maze floor and compute the fraction
    of steps at each cell that used a HF action.  Doorways and pillars are
    overlaid as explicit markers so the reader can see whether HF concentrates
    at structurally important locations.

Usage:

    python scripts/plot_hf_analyses.py \
        --ckpt-vc2e5 runs/.../ckpt_vc2e-5.pt \
        --ckpt-vc1e4 runs/.../ckpt_vc1e-4.pt \
        --episodes 50 --out results/plots/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from gymnasium.envs.registration import register as _gym_register

from cff_rl.agents.ppo_lstm import RecurrentNatureCNN
from cff_rl.envs.static_maze import make_static_env
from miniworld.entity import Ball as _Ball
from cff_rl.envs.fourrooms_hard import FourRoomsHard


# ---------------------------------------------------------------------------
# Fixed-layout variant: goal and balls locked; only agent start varies
# ---------------------------------------------------------------------------

class FixedLayoutFourRoomsHard(FourRoomsHard):
    """FourRoomsHard where goal and distractor positions are fixed.

    The parent's _gen_world sets up rooms, pillars, and entities (all randomly
    placed), then we overwrite the xz coordinates of the goal and each
    distractor ball with canonical values.  The agent's starting position is
    left as-is so it still varies across episodes.

    Positions are chosen to be well clear of pillars (±4, ±4) and doorways
    (0, ±4) and (±4, 0).
    """
    _GOAL_XZ = (-6, -4.0)         # room2: x∈[1,7]  z∈[-7,-1]
    _BALL_XZ = [
        (-3.0,  6.0),               # room0: x∈[-7,-1] z∈[1,7]
        ( 6.0,  6.0),               # room1: x∈[1,7]   z∈[1,7]
        (-6.0, -6.0),               # room3: x∈[-7,-1] z∈[-7,-1]
        ( 6.0, -6.0),               # room2: x∈[1,7]   z∈[-7,-1]
    ]

    def _gen_world(self):
        super()._gen_world()        # randomises goal, balls, and agent
        # Snap goal to fixed xz; y (floor clearance) is preserved from parent
        self.box.pos[0] = self._GOAL_XZ[0]
        self.box.pos[2] = self._GOAL_XZ[1]
        # Collect distractor balls (all Ball entities that aren't the goal)
        distractors = [e for e in self.entities
                       if isinstance(e, _Ball) and e is not self.box]
        for ball, (bx, bz) in zip(distractors, self._BALL_XZ):
            ball.pos[0] = bx
            ball.pos[2] = bz


_gym_register(
    id="MiniWorld-FourRoomsHardFixed-v0",
    entry_point=FixedLayoutFourRoomsHard,
)

ENV_ID = "MiniWorld-FourRoomsHardFixed-v0"
SEEDS = [1, 2, 3, 4, 5]
GRID_CELLS = 60
MIN_VISITS = 3

# Blue (all-LF) → light grey (50/50) → orange (all-HF)
HF_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "hf_diverge",
    [(0.0, "#2166ac"), (0.5, "#d9d9d9"), (1.0, "#d6604d")],
)
HF_CMAP.set_bad(color="#1a1a1a")   # dark grey for unvisited cells

COLORS = {"vc=2e-5": "#2a7db5", "vc=1e-4": "#5ab55a"}

# ── Maze geometry (from fourrooms_hard.py _gen_world) ───────────────────────
# Rooms: room0 x∈[-7,-1] z∈[1,7], room1 x∈[1,7] z∈[1,7],
#        room2 x∈[1,7] z∈[-7,-1], room3 x∈[-7,-1] z∈[-7,-1]
# connect_rooms: room0↔room1 z∈[3,5]@x=0, room1↔room2 x∈[3,5]@z=0,
#                room2↔room3 z∈[-5,-3]@x=0, room3↔room0 x∈[-5,-3]@z=0
DOORWAYS = [
    (0.0,  4.0),   # room0 ↔ room1  (top wall,   z midpoint of [3,5])
    (4.0,  0.0),   # room1 ↔ room2  (right wall,  x midpoint of [3,5])
    (0.0, -4.0),   # room2 ↔ room3  (bottom wall, z midpoint of [-5,-3])
    (-4.0, 0.0),   # room3 ↔ room0  (left wall,   x midpoint of [-5,-3])
]
PILLARS = [(-4.0, 4.0), (4.0, 4.0), (4.0, -4.0), (-4.0, -4.0)]

# Wall segments in (x,z) space for reference outline
ROOM_BOUNDS = [
    (-7, -1, 1, 7),   # room0: (min_x, max_x, min_z, max_z)
    ( 1,  7, 1, 7),   # room1
    ( 1,  7,-7,-1),   # room2
    (-7, -1,-7,-1),   # room3
]


# ---------------------------------------------------------------------------
# Agent loading
# ---------------------------------------------------------------------------

def load_agent(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt.get("arch", {})
    arch = {
        "frame_stack":      raw.get("frame_stack", 1),
        "lstm_hidden_size": raw.get("lstm_hidden_size", 256),
        "lstm_num_layers":  raw.get("lstm_num_layers", 1),
        "use_proprio":      raw.get("use_proprio", True),
        "use_active_vision":raw.get("use_active_vision", True),
        "vision_cost":      raw.get("vision_cost", 0.0001),
        "strobe_k":         raw.get("strobe_k", 7),
        "hf_strobe_k":      raw.get("hf_strobe_k", 1),
        "high_freq_steps":  raw.get("high_freq_steps", 35),
        "use_depth":        raw.get("use_depth", False),
        "use_fresh_gate":   raw.get("use_fresh_gate", False),
        "n_extras":         raw.get("n_extras", 5),
        "turn_step_deg":    raw.get("turn_step_deg", 10),
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
# Rollout collection
# ---------------------------------------------------------------------------

def collect_rollouts(agent, arch, device, seeds, episodes_per_seed):
    """Returns per-step spatial records."""
    # spatial: list of (x, z, hf_float, success, is_forward)
    spatial: list[tuple] = []

    for seed in seeds:
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
        inner = env.unwrapped

        for ep in range(episodes_per_seed):
            obs, _ = env.reset(seed=seed + ep)
            lstm_state = agent.initial_state(1, device)
            done_t = torch.zeros(1, device=device)
            is_fresh = (
                torch.ones(1, 1, dtype=torch.float32, device=device)
                if arch["use_fresh_gate"] else None
            )
            ep_steps: list[tuple] = []   # (x, z, hf, is_forward)
            terminated = truncated = False

            while not (terminated or truncated):
                pos = inner.agent.pos.copy()
                if arch["use_proprio"]:
                    x_t = torch.as_tensor(obs["image"], dtype=torch.float32).unsqueeze(0).to(device)
                    ext = torch.as_tensor(obs["extras"], dtype=torch.float32).unsqueeze(0).to(device)
                else:
                    x_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                    ext = None

                with torch.no_grad():
                    a, _, _, _, lstm_state = agent.get_action_and_value(
                        x_t, lstm_state, done_t,
                        extras=ext, is_fresh=is_fresh, gate_alpha=1.0,
                    )
                action = int(a.item())
                obs, _, terminated, truncated, info = env.step(action)
                hf = float(info["high_freq"])
                is_forward = int(action % 3 == 2)
                ep_steps.append((float(pos[0]), float(pos[2]), hf, is_forward))
                if arch["use_fresh_gate"]:
                    is_fresh = torch.tensor([[hf]], dtype=torch.float32, device=device)

            success = float(terminated)
            for x, z, hf, is_fwd in ep_steps:
                spatial.append((x, z, hf, success, is_fwd))

        env.close()
        print(f"  seed {seed} done  ({episodes_per_seed} eps)")

    return {"spatial": spatial}


# ---------------------------------------------------------------------------
# Plot: Spatial HF heatmap with maze overlay
# ---------------------------------------------------------------------------

def build_hf_grid(spatial, x_range, z_range, n_cells, action_filter=None):
    """action_filter: None=all, 'forward'=forward only, 'turn'=turns only."""
    if action_filter == "forward":
        spatial = [s for s in spatial if s[4] == 1]
    elif action_filter == "turn":
        spatial = [s for s in spatial if s[4] == 0]
    if not spatial:
        dummy = np.full((n_cells, n_cells), np.nan)
        xe = np.linspace(x_range[0], x_range[1], n_cells + 1)
        ze = np.linspace(z_range[0], z_range[1], n_cells + 1)
        return dummy, xe, ze, np.zeros((n_cells, n_cells))
    xs  = np.array([s[0] for s in spatial])
    zs  = np.array([s[1] for s in spatial])
    hfs = np.array([s[2] for s in spatial])
    total, xe, ze = np.histogram2d(xs, zs, bins=n_cells, range=[x_range, z_range])
    hf_sum, _, _  = np.histogram2d(xs, zs, bins=n_cells, range=[x_range, z_range], weights=hfs)
    with np.errstate(invalid="ignore"):
        frac = np.where(total >= MIN_VISITS, hf_sum / total, np.nan)
    return frac, xe, ze, total


def draw_maze_overlay(ax):
    """Draw room outlines, pillars, doorway markers, goal, and distractor balls."""
    # Room outlines — bright white, clearly visible
    for (x0, x1, z0, z1) in ROOM_BOUNDS:
        rect = mpatches.Rectangle(
            (x0, z0), x1 - x0, z1 - z0,
            linewidth=1.8, edgecolor="white", facecolor="none", alpha=0.85,
            zorder=3,
        )
        ax.add_patch(rect)

    # Pillars — solid dark fill with a crisp white border
    for px, pz in PILLARS:
        sq = mpatches.FancyBboxPatch(
            (px - 0.45, pz - 0.45), 0.9, 0.9,
            boxstyle="square,pad=0",
            linewidth=1.2, edgecolor="white", facecolor="#444444", alpha=0.95,
            zorder=4,
        )
        ax.add_patch(sq)

    # Goal (yellow box, 0.9×0.9 to match MiniWorld Box size)
    gx, gz = FixedLayoutFourRoomsHard._GOAL_XZ
    _BOX_SIZE = 0.9
    ax.add_patch(mpatches.Rectangle(
        (gx - _BOX_SIZE / 2, gz - _BOX_SIZE / 2), _BOX_SIZE, _BOX_SIZE,
        linewidth=1.5, edgecolor="black", facecolor="#FFD700", alpha=0.95, zorder=6,
    ))

    # Distractor balls (red circles)
    for bx, bz in FixedLayoutFourRoomsHard._BALL_XZ[:1]:
        ax.plot(bx, bz, marker="o", markersize=9, color="#e84040",
                markeredgecolor="black", markeredgewidth=0.9, zorder=6)


def _single_heatmap(ax, spatial, x_range, z_range, action_filter, title):
    frac, xe, ze, _ = build_hf_grid(spatial, x_range, z_range, GRID_CELLS, action_filter)
    extent = [xe[0], xe[-1], ze[0], ze[-1]]

    # Dark background for unvisited cells
    ax.set_facecolor("#1a1a1a")

    im = ax.imshow(
        frac.T, origin="lower", extent=extent,
        cmap=HF_CMAP, vmin=0.0, vmax=1.0, aspect="equal",
    )
    draw_maze_overlay(ax)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("z (m)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(x_range)
    ax.set_ylim(z_range)
    # Clean up spines
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
        spine.set_linewidth(0.8)
    return im


def plot_spatial(data_by_label, heatmap_labels, out_path):
    all_x = [s[0] for lbl in data_by_label for s in data_by_label[lbl]["spatial"]]
    all_z = [s[1] for lbl in data_by_label for s in data_by_label[lbl]["spatial"]]
    pad = 0.5
    x_range = [min(all_x) - pad, max(all_x) + pad]
    z_range = [min(all_z) - pad, max(all_z) + pad]

    # 3 rows (all / forward / turns) × 2 cols (vc=2e-5 / vc=1e-4)
    row_labels    = ["all actions", "forward only", "turns only"]
    action_filters = [None, "forward", "turn"]

    fig, axes = plt.subplots(
        3, len(heatmap_labels),
        figsize=(5.5 * len(heatmap_labels), 5.0 * 3),
        constrained_layout=True,
    )

    im_ref = None
    for row, (row_lbl, af) in enumerate(zip(row_labels, action_filters)):
        for col, cost_label in enumerate(heatmap_labels):
            ax = axes[row, col]
            title = f"{cost_label} — {row_lbl}"
            im = _single_heatmap(
                ax,
                data_by_label[cost_label]["spatial"],
                x_range, z_range, af, title,
            )
            im_ref = im

    cb = fig.colorbar(im_ref, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("HF fraction per cell  (blue = LF, orange = HF)", fontsize=10)

    pillar_patch = mpatches.Patch(facecolor="#888888", label="pillar")
    goal_patch = mpatches.Patch(facecolor="#FFD700", edgecolor="black",  label="Goal")
    obstacle_handle = plt.Line2D(
        [0], [0], marker="o", color="none", markerfacecolor="#e84040",
        markeredgecolor="black", markeredgewidth=0.9, markersize=9, label="Obstacle",
    )
    doorway_handle  = plt.Line2D(
        [0], [0], marker="D", color="none", markerfacecolor="white",
        markeredgecolor="#333333", markeredgewidth=0.8, markersize=8, label="Doorway",
    )
    fig.legend(
        handles=[pillar_patch, goal_patch, obstacle_handle, doorway_handle],
        loc="lower center", ncol=4, fontsize=9,
        bbox_to_anchor=(0.5, -0.01), 
        handlelength=1.0,
    )
    fig.suptitle(
        "Spatial HF vision usage — FourRoomsHard",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=1000, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_spatial_single(label, data, out_path):
    """Heatmap for a single run: 3 rows (all / forward / turns), 1 column."""
    spatial = data["spatial"]
    all_x = [s[0] for s in spatial]
    all_z = [s[1] for s in spatial]
    pad = 0.5
    x_range = [min(all_x) - pad, max(all_x) + pad]
    z_range = [min(all_z) - pad, max(all_z) + pad]

    row_labels    = ["all actions", "forward only", "turns only"]
    action_filters = [None, "forward", "turn"]

    fig, axes = plt.subplots(3, 1, figsize=(5.5, 5.0 * 3), constrained_layout=True)

    im_ref = None
    for ax, row_lbl, af in zip(axes, row_labels, action_filters):
        im = _single_heatmap(ax, spatial, x_range, z_range, af, f"{label} — {row_lbl}")
        im_ref = im

    cb = fig.colorbar(im_ref, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("HF fraction per cell  (blue = LF, orange = HF)", fontsize=10)

    pillar_patch = mpatches.Patch(facecolor="#888888", label="pillar")
    goal_patch = mpatches.Patch(facecolor="#FFD700", edgecolor="black",  label="Goal")
    obstacle_handle = plt.Line2D(
        [0], [0], marker="o", color="none", markerfacecolor="#e84040",
        markeredgecolor="black", markeredgewidth=0.9, markersize=9, label="Obstacle",
    )
    doorway_handle  = plt.Line2D(
        [0], [0], marker="D", color="none", markerfacecolor="white",
        markeredgecolor="#333333", markeredgewidth=0.8, markersize=8, label="Doorway",
    )
    fig.legend(
        handles=[pillar_patch, goal_patch, obstacle_handle, doorway_handle],
        loc="lower center", ncol=4, fontsize=9,
        bbox_to_anchor=(0.5, -0.01), 
        handlelength=1.0,
    )
    fig.suptitle(
        f"Spatial HF vision usage — {label}",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=1000, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot spatial HF heatmaps for vc=2e-5 and vc=1e-4 agents."
    )
    p.add_argument(
        "--ckpt-vc2e5", required=True, metavar="PATH",
        help="Checkpoint path for the vc=2e-5 agent.",
    )
    p.add_argument(
        "--ckpt-vc1e4", required=True, metavar="PATH",
        help="Checkpoint path for the vc=1e-4 agent.",
    )
    p.add_argument("--episodes", type=int, default=30,
                   help="Episodes per seed (default: 30).")
    p.add_argument("--out", type=Path, default=Path("results/plots"),
                   help="Output directory for saved figures.")
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs: dict[str, str] = {
        "vc=2e-5": args.ckpt_vc2e5,
        "vc=1e-4": args.ckpt_vc1e4,
    }

    data_by_label: dict[str, dict] = {}
    for label, ckpt_path in runs.items():
        print(f"\n--- Collecting rollouts: {label} ---")
        agent, arch = load_agent(ckpt_path, device)
        data_by_label[label] = collect_rollouts(
            agent, arch, device, SEEDS, args.episodes
        )

    plot_spatial(
        data_by_label,
        heatmap_labels=["vc=2e-5", "vc=1e-4"],
        out_path=args.out / "hf_spatial_heatmap.png",
    )
    for label, data in data_by_label.items():
        safe = label.replace("=", "").replace("-", "n")
        plot_spatial_single(label, data, args.out / f"hf_spatial_{safe}.png")


if __name__ == "__main__":
    main()