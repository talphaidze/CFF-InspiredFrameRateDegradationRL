"""3D analysis plot: vision cost × HF Hz rate × success rate.

Reads eval_results.json from every runs/*fourroomshard_dynamic* directory,
loads the corresponding checkpoint arch to determine agent type, vision_cost,
and hf_strobe_k, then produces:

  - A 3D scatter: X=vision_cost (log), Y=HF Hz rate (35/hf_strobe_k),
    Z=success rate, color=HF usage fraction.
  - Three 2D panels: cost vs. success, HF Hz vs. success,
    HF fraction vs. success.

Agent A (always 35 Hz, no cost) and Agent B (always 5 Hz, no cost) are
annotated as reference baselines throughout.

Output saved to results/plots/dynamic_3d_cost_hf_perf.png.

Usage:
    python scripts/plot_dynamic_3d.py
    python scripts/plot_dynamic_3d.py --out results/plots/my_fig.png
    python scripts/plot_dynamic_3d.py --no-show
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

RUNS_DIR = Path("runs")
OUT_DEFAULT = Path("results/plots/dynamic_3d_cost_hf_perf.png")
PREFIXES = ("agent_a_", "agent_b_", "agent_c2_")
PHYSICS_HZ = 35.0  # MiniWorld physics tick rate


# ── data loading ─────────────────────────────────────────────────────────────

def load_arch(run_dir: Path) -> dict | None:
    ckpts = sorted(run_dir.glob("ckpt_*.pt"))
    if not ckpts:
        return None
    raw = torch.load(ckpts[-1], map_location="cpu", weights_only=False).get("arch", {})
    return {
        "use_active_vision": raw.get("use_active_vision", False),
        "use_stroboscopic": raw.get("use_stroboscopic", False),
        "use_active_gating": raw.get("use_active_gating", False),
        "vision_cost": float(raw.get("vision_cost", 0.01)),
        "hf_strobe_k": int(raw.get("hf_strobe_k", 1)),
        "turn_step_deg": raw.get("turn_step_deg", 90),
    }


def collect_runs() -> list[dict]:
    rows = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        if not any(run_dir.name.startswith(p) for p in PREFIXES):
            continue
        if "fourroomshard_dynamic" not in run_dir.name:
            continue

        result_file = run_dir / "eval_results.json"
        if not result_file.exists():
            print(f"  [skip] {run_dir.name}: no eval_results.json")
            continue
        result = json.loads(result_file.read_text())
        if result.get("env_id") != "MiniWorld-FourRoomsHardDynamic-v0":
            print(f"  [skip] {run_dir.name}: wrong env_id ({result.get('env_id')})")
            continue

        arch = load_arch(run_dir)
        if arch is None:
            print(f"  [skip] {run_dir.name}: no checkpoint")
            continue

        success_mean, success_std = result["success_rate"]
        mean_hf = result["mean_highfreq_per_episode"][0]
        mean_len = result["mean_length"][0]
        hf_fraction = mean_hf / mean_len if mean_len > 0 else 0.0

        if arch["use_active_vision"]:
            agent_type = "C2"
            vision_cost = arch["vision_cost"]
            hf_hz = PHYSICS_HZ / arch["hf_strobe_k"]
        elif arch["use_stroboscopic"]:
            agent_type = "B"
            vision_cost = None
            hf_hz = None
            hf_fraction = 0.0
        else:
            agent_type = "A"
            vision_cost = None
            hf_hz = None
            hf_fraction = 1.0

        rows.append(
            {
                "run": run_dir.name,
                "agent_type": agent_type,
                "vision_cost": vision_cost,
                "hf_hz": hf_hz,
                "hf_strobe_k": arch["hf_strobe_k"],
                "hf_fraction": hf_fraction,
                "success_rate": success_mean,
                "success_std": success_std,
                "turn_step_deg": arch["turn_step_deg"],
            }
        )
    return rows


# ── plot helpers ──────────────────────────────────────────────────────────────

# Fixed colors per HF Hz rate for consistent styling across panels.
HZ_PALETTE = {
    35.0: "#e63946",   # red — highest rate
    17.5: "#f4a261",   # orange
    11.666666666666666: "#2a9d8f",  # teal
    5.0: "#457b9d",    # blue — lowest (same as LF)
}

def _hz_color(hz: float) -> str:
    for key, col in HZ_PALETTE.items():
        if abs(hz - key) < 0.1:
            return col
    return "#888888"


def _jitter(n: int, scale: float = 0.015, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, n)


def make_figure(rows: list[dict], out: Path, show: bool) -> None:
    a_rows = [r for r in rows if r["agent_type"] == "A"]
    b_rows = [r for r in rows if r["agent_type"] == "B"]
    c2_rows = [r for r in rows if r["agent_type"] == "C2"]

    if not c2_rows:
        print("No C2 runs found — cannot produce cost-sweep plot.")
        return

    sr_a = float(np.mean([r["success_rate"] for r in a_rows])) if a_rows else None
    sr_b = float(np.mean([r["success_rate"] for r in b_rows])) if b_rows else None

    vc = np.array([r["vision_cost"] for r in c2_rows], dtype=float)
    hf_hz = np.array([r["hf_hz"] for r in c2_rows], dtype=float)
    hf_frac = np.array([r["hf_fraction"] for r in c2_rows], dtype=float)
    sr = np.array([r["success_rate"] for r in c2_rows], dtype=float)
    log_vc = np.log10(vc)

    # Color by HF fraction (0=LF, 1=always HF).
    colors = [_hz_color(h) for h in hf_hz]
    unique_hz = sorted(set(hf_hz), reverse=True)

    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        "When is HF vision worth paying for?\n"
        "Agent C2 (active vision) on FourRoomsHardDynamic",
        fontsize=13, fontweight="bold",
    )

    # ── 3D scatter ────────────────────────────────────────────────────────────
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")

    for hz in unique_hz:
        mask = np.abs(hf_hz - hz) < 0.1
        label = f"HF={hz:.1f} Hz"
        ax3d.scatter(
            log_vc[mask] + _jitter(mask.sum(), 0.05),
            hf_hz[mask] + _jitter(mask.sum(), 0.4),
            sr[mask],
            c=_hz_color(hz),
            s=80,
            edgecolors="k",
            linewidths=0.5,
            zorder=5,
            label=label,
        )

    # Reference baselines
    vc_range = np.array([log_vc.min() - 0.5, log_vc.max() + 0.5])
    hz_vals = np.array(unique_hz)
    if sr_a is not None:
        ax3d.plot(
            [vc_range.mean()] * 2,
            [hz_vals.max() * 1.1] * 2,
            [sr_a, sr_a],
            color="steelblue", linewidth=2, linestyle="--",
        )
        ax3d.text(
            vc_range.mean(), hz_vals.max() * 1.15, sr_a,
            f"A  sr={sr_a:.2f}", color="steelblue", fontsize=8,
        )
    if sr_b is not None:
        ax3d.text(
            vc_range.mean(), hz_vals.min() * 0.85, sr_b,
            f"B  sr={sr_b:.2f}", color="coral", fontsize=8,
        )

    ax3d.set_xlabel("log₁₀(vision cost)", fontsize=9, labelpad=8)
    ax3d.set_ylabel("HF Hz rate", fontsize=9, labelpad=8)
    ax3d.set_zlabel("Success rate", fontsize=9, labelpad=8)
    ax3d.set_title("3D: cost × HF rate × performance", fontsize=10)
    ax3d.set_zlim(0, 1.05)
    ax3d.legend(fontsize=7, loc="upper left")

    # ── Panel 1: vision_cost vs. success rate ─────────────────────────────────
    ax1 = fig.add_subplot(2, 2, 2)
    for hz in unique_hz:
        mask = np.abs(hf_hz - hz) < 0.1
        idx = np.argsort(log_vc[mask])
        ax1.plot(
            log_vc[mask][idx], sr[mask][idx],
            color=_hz_color(hz), marker="o", markersize=6,
            linewidth=1.2, label=f"HF={hz:.1f} Hz",
        )
    if sr_a is not None:
        ax1.axhline(sr_a, color="steelblue", linestyle="--", linewidth=1.2,
                    label=f"Agent A ({sr_a:.2f})")
    if sr_b is not None:
        ax1.axhline(sr_b, color="coral", linestyle="--", linewidth=1.2,
                    label=f"Agent B ({sr_b:.2f})")
    ax1.set_xlabel("log₁₀(vision cost)")
    ax1.set_ylabel("Success rate")
    ax1.set_title("Cost vs. Performance  (by HF rate)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    _annotate_vc_ticks(ax1, vc, log_vc)

    # ── Panel 2: HF Hz rate vs. success rate ─────────────────────────────────
    ax2 = fig.add_subplot(2, 2, 3)
    # Group by vision_cost for connected lines
    unique_vc = sorted(set(vc))
    vc_cmap = plt.cm.get_cmap("plasma", len(unique_vc))
    for i, v in enumerate(unique_vc):
        mask = np.abs(vc - v) < v * 0.01 + 1e-10
        if mask.sum() == 0:
            continue
        idx = np.argsort(hf_hz[mask])
        ax2.plot(
            hf_hz[mask][idx], sr[mask][idx],
            color=vc_cmap(i), marker="o", markersize=6,
            linewidth=1.2, label=f"vc={v:.1e}",
        )
    if sr_a is not None:
        ax2.scatter([PHYSICS_HZ], [sr_a], marker="*", s=200, color="steelblue",
                    zorder=6, label=f"Agent A ({sr_a:.2f})")
    if sr_b is not None:
        ax2.scatter([PHYSICS_HZ / 7], [sr_b], marker="*", s=200, color="coral",
                    zorder=6, label=f"Agent B ({sr_b:.2f})")
    ax2.set_xlabel("HF Hz rate")
    ax2.set_ylabel("Success rate")
    ax2.set_title("HF Rate vs. Performance  (by cost)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: HF fraction vs. success rate ─────────────────────────────────
    ax3 = fig.add_subplot(2, 2, 4)
    for hz in unique_hz:
        mask = np.abs(hf_hz - hz) < 0.1
        ax3.scatter(
            hf_frac[mask] + _jitter(mask.sum(), 0.01),
            sr[mask],
            c=_hz_color(hz), s=60, zorder=5, label=f"HF={hz:.1f} Hz",
        )
    if sr_a is not None:
        ax3.axvline(1.0, color="steelblue", linestyle="--", linewidth=1.2,
                    label=f"Agent A ({sr_a:.2f})")
        ax3.scatter([1.0], [sr_a], marker="*", s=200, color="steelblue", zorder=6)
    if sr_b is not None:
        ax3.axvline(0.0, color="coral", linestyle="--", linewidth=1.2,
                    label=f"Agent B ({sr_b:.2f})")
        ax3.scatter([0.0], [sr_b], marker="*", s=200, color="coral", zorder=6)
    ax3.set_xlabel("HF usage fraction (steps)")
    ax3.set_ylabel("Success rate")
    ax3.set_title("HF Usage vs. Performance")
    ax3.set_xlim(-0.05, 1.1)
    ax3.set_ylim(-0.05, 1.05)
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    if show:
        plt.show()
    plt.close()


def _annotate_vc_ticks(ax, vc: np.ndarray, log_vc: np.ndarray) -> None:
    unique_log = np.unique(np.round(log_vc, 8))
    unique_vc = 10 ** unique_log
    ax.set_xticks(unique_log)
    ax.set_xticklabels([f"{v:.0e}" for v in unique_vc], rotation=30, fontsize=7)


def print_table(rows: list[dict]) -> None:
    print(
        f"\n{'run':<65} {'type':<4} {'vision_cost':<12} "
        f"{'hf_hz':<8} {'hf_frac':<9} {'success':<8}"
    )
    print("-" * 112)
    for r in sorted(rows, key=lambda x: (x["agent_type"], x.get("hf_hz") or 0, x.get("vision_cost") or 0)):
        vc = f"{r['vision_cost']:.2e}" if r["vision_cost"] is not None else "N/A"
        hz = f"{r['hf_hz']:.1f}" if r["hf_hz"] is not None else "N/A"
        print(
            f"{r['run']:<65} {r['agent_type']:<4} {vc:<12} "
            f"{hz:<8} {r['hf_fraction']:<9.4f} {r['success_rate']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    print("Loading runs...")
    rows = collect_runs()
    if not rows:
        print("No data found.")
        return

    print_table(rows)
    make_figure(rows, args.out, show=not args.no_show)


if __name__ == "__main__":
    main()
