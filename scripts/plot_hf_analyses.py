"""HF vision usage analyses for the vision-cost sweep.

Analysis 1 — Spatial HF heatmap:
    For vc=0.00002 and vc=0.0001, grid the maze floor and compute the fraction
    of steps at each cell that used a HF action.  Doorways and pillars are
    overlaid as explicit markers so the reader can see whether HF concentrates
    at structurally important locations.

Analysis 2 — Within-episode HF timing, split by outcome:
    For all three cost conditions, bin steps by normalised episode time t/T and
    plot mean HF fraction separately for successful and failed episodes.
    Reveals whether the agent's HF allocation is outcome-predictive even when
    the aggregate curve looks flat.

Usage:
    python scripts/plot_hf_analyses.py
    python scripts/plot_hf_analyses.py --episodes 50 --out results/plots/
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

from cff_rl.agents.ppo_lstm import RecurrentNatureCNN
from cff_rl.envs.static_maze import make_static_env

ENV_ID = "MiniWorld-FourRoomsHardDynamic-v0"
SEEDS = [1, 2, 3, 4, 5]
N_TIME_BINS = 20
GRID_CELLS = 60
MIN_VISITS = 3

RUNS: dict[str, str] = {
    "vc=1e-5": "runs/agent_c2_fourroomshard_dynamic_4d_lstm_proprio_turn10_6M_gate_vc000001__42__1779343926/ckpt_001464.pt",
    "vc=2e-5": "runs/agent_c2_fourroomshard_dynamic_4d_lstm_proprio_turn10_6M_gate_vc000002__42__1779355584/ckpt_001400.pt",
    "vc=1e-4": "runs/agent_c2_fourroomshard_dynamic_4d_lstm_proprio_turn10_6M_gate_v2__42__1779311643/ckpt_001464.pt",
}

COLORS = {"vc=1e-5": "#e07b39", "vc=2e-5": "#2a7db5", "vc=1e-4": "#5ab55a"}

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
    """Returns per-step records with outcome label."""
    # spatial: list of (x, z, hf_float, success)
    spatial: list[tuple] = []
    # temporal: list of (t_norm, hf_float, success)
    temporal: list[tuple] = []

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
            ep_steps: list[tuple] = []   # (x, z, hf)
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
                # is_forward: action % n_base_actions == 2 (move_forward index)
                is_forward = int(action % 3 == 2)
                ep_steps.append((float(pos[0]), float(pos[2]), hf, is_forward))
                if arch["use_fresh_gate"]:
                    is_fresh = torch.tensor([[hf]], dtype=torch.float32, device=device)

            success = float(terminated)
            T = len(ep_steps)
            for t, (x, z, hf, is_fwd) in enumerate(ep_steps):
                spatial.append((x, z, hf, success, is_fwd))
                temporal.append((t / max(T - 1, 1), hf, success))

        env.close()
        print(f"  seed {seed} done  ({episodes_per_seed} eps)")

    return {"spatial": spatial, "temporal": temporal}


# ---------------------------------------------------------------------------
# Plot 1: Spatial HF heatmap with maze overlay
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
    """Draw room outlines, pillars, and doorway markers."""
    # Room outlines (thin white lines)
    for (x0, x1, z0, z1) in ROOM_BOUNDS:
        rect = mpatches.Rectangle(
            (x0, z0), x1 - x0, z1 - z0,
            linewidth=0.8, edgecolor="white", facecolor="none", alpha=0.5,
        )
        ax.add_patch(rect)

    # Pillars (grey squares)
    for px, pz in PILLARS:
        sq = mpatches.Rectangle(
            (px - 0.5, pz - 0.5), 1.0, 1.0,
            linewidth=0, facecolor="#888888", alpha=0.7,
        )
        ax.add_patch(sq)

    # Doorways (cyan stars with label on first only)
    for i, (dx, dz) in enumerate(DOORWAYS):
        ax.plot(dx, dz, marker="*", markersize=12, color="cyan",
                markeredgecolor="black", markeredgewidth=0.5, zorder=5,
                label="doorway" if i == 0 else None)


def _single_heatmap(ax, spatial, x_range, z_range, action_filter, title):
    frac, xe, ze, _ = build_hf_grid(spatial, x_range, z_range, GRID_CELLS, action_filter)
    extent = [xe[0], xe[-1], ze[0], ze[-1]]
    im = ax.imshow(
        frac.T, origin="lower", extent=extent,
        cmap=plt.cm.RdYlGn, vmin=0.0, vmax=1.0, aspect="equal",
    )
    draw_maze_overlay(ax)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (m)", fontsize=10)
    ax.set_ylabel("z (m)", fontsize=10)
    ax.set_xlim(x_range)
    ax.set_ylim(z_range)
    return im


def plot_spatial(data_by_label, heatmap_labels, out_path):
    all_x = [s[0] for lbl in data_by_label for s in data_by_label[lbl]["spatial"]]
    all_z = [s[1] for lbl in data_by_label for s in data_by_label[lbl]["spatial"]]
    pad = 0.5
    x_range = [min(all_x) - pad, max(all_x) + pad]
    z_range = [min(all_z) - pad, max(all_z) + pad]

    # 3 rows (all / forward / turns) × 2 cols (vc=2e-5 / vc=1e-4)
    row_labels   = ["all actions", "forward only", "turns only"]
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
            if col == 0:
                ax.legend(loc="lower right", fontsize=8, framealpha=0.6)

    cb = fig.colorbar(im_ref, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("HF fraction per cell", fontsize=11)

    pillar_patch = mpatches.Patch(facecolor="#888888", label="pillar")
    fig.legend(
        handles=[pillar_patch],
        loc="lower center", ncol=2, fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Spatial HF vision usage — FourRoomsHardDynamic\n★ = doorway,  ■ = pillar",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
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
        ax.legend(loc="lower right", fontsize=8, framealpha=0.6)

    cb = fig.colorbar(im_ref, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("HF fraction per cell", fontsize=11)

    pillar_patch = mpatches.Patch(facecolor="#888888", label="pillar")
    fig.legend(
        handles=[pillar_patch],
        loc="lower center", ncol=2, fontsize=9,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        f"Spatial HF vision usage — {label}\n★ = doorway,  ■ = pillar",
        fontsize=14,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Plot 2: Within-episode HF timing — success vs failure
# ---------------------------------------------------------------------------

def bin_temporal(temporal, success_flag, n_bins=N_TIME_BINS):
    bins = np.linspace(0, 1, n_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    subset = [(t, hf) for t, hf, s in temporal if s == success_flag]
    if not subset:
        return centers, np.full(n_bins, np.nan), np.full(n_bins, np.nan)
    t_arr  = np.array([r[0] for r in subset])
    hf_arr = np.array([r[1] for r in subset])
    means, stds = [], []
    for b in range(n_bins):
        mask = (t_arr >= bins[b]) & (t_arr < bins[b + 1])
        vals = hf_arr[mask]
        means.append(vals.mean() if len(vals) > 0 else np.nan)
        stds.append(vals.std()  if len(vals) > 0 else np.nan)
    return centers, np.array(means), np.array(stds)


def plot_timing(data_by_label, out_path):
    bins = np.linspace(0, 1, N_TIME_BINS + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    fig, axes = plt.subplots(
        1, len(data_by_label),
        figsize=(5 * len(data_by_label), 4),
        sharey=True, constrained_layout=True,
    )
    if len(data_by_label) == 1:
        axes = [axes]

    for ax, (label, data) in zip(axes, data_by_label.items()):
        color = COLORS[label]
        temporal = data["temporal"]

        # Count success/failure for subtitle
        n_success = sum(1 for _, _, s in temporal if s == 1.0)
        n_fail    = sum(1 for _, _, s in temporal if s == 0.0)
        # Approx episode counts (each episode contributes many steps)
        ep_success = sum(1 for ep in _iter_episodes(temporal) if ep[0][2] == 1.0)
        ep_fail    = sum(1 for ep in _iter_episodes(temporal) if ep[0][2] == 0.0)

        for success_flag, linestyle, outcome_label in [
            (1.0, "-",  f"success (n={ep_success})"),
            (0.0, "--", f"failure (n={ep_fail})"),
        ]:
            c, means, stds = bin_temporal(temporal, success_flag)
            if np.all(np.isnan(means)):
                continue
            ax.plot(c, means, color=color, linewidth=2,
                    linestyle=linestyle, label=outcome_label)
            ax.fill_between(
                c,
                np.clip(means - stds, 0, 1),
                np.clip(means + stds, 0, 1),
                color=color, alpha=0.12,
            )

        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_title(label, fontsize=13, fontweight="bold", color=color)
        ax.set_xlabel("Normalised episode time  t / T", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Mean HF fraction", fontsize=11)
    fig.suptitle(
        "Within-episode HF timing by outcome — FourRoomsHardDynamic",
        fontsize=13,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def _iter_episodes(temporal):
    """Group temporal records into episodes by detecting t_norm resets."""
    if not temporal:
        return
    ep = [temporal[0]]
    for rec in temporal[1:]:
        if rec[0] < ep[-1][0]:   # t_norm decreased → new episode
            yield ep
            ep = [rec]
        else:
            ep.append(rec)
    yield ep


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--out", type=Path, default=Path("results/plots"))
    return p.parse_args()


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_by_label: dict[str, dict] = {}
    for label, ckpt_path in RUNS.items():
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
    plot_timing(
        data_by_label,
        out_path=args.out / "hf_timing.png",
    )


if __name__ == "__main__":
    main()
