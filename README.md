# CFF-Inspired Frame Rate Degradation in RL

EPFL CS-503 Visual Intelligence project. Investigates whether degrading the
frame rate of visual observations helps or hurts RL, and whether an agent can
learn *when* to spend high-frequency perception — inspired by biological
**Critical Flicker Fusion (CFF)**.

We compare three architecturally identical agents (Nature-CNN + LSTM + PPO)
that differ **only in how they observe**, on the **MiniWorld FourRooms** suite
across three difficulties: Easy Static, Hard Static, and Hard Dynamic.

- **Agent A — 35 Hz baseline.** Fresh 64×64 grayscale frame every step.
- **Agent B — 5 Hz stroboscopic.** Each frame held for `k=7` steps (`StroboscopicWrapper`).
- **Agent C v1 — active gating.** Adds a `STOP_AND_LOOK` action (no reward penalty)
  over a 5 Hz default. *Discarded* after weak Hard-Static performance.
- **Agent C v2 — active perception.** Action space doubled to 3 moves × 2 frequency
  modes; high-frequency perception costs a small per-step penalty `c_vision`, so the
  agent learns when to use fresh frames vs. coast on LSTM memory.

📄 **Project website:** https://manuelcurnis.github.io/cff-rl-webpage (source:
[manuelcurnis/cff-rl-webpage](https://github.com/manuelcurnis/cff-rl-webpage))

## Team

- Tamar Alphaidze
- Sajal Chaurasia
- Manuel Curnis
- Alessandro Di Maria

## Repo layout

```
src/cff_rl/
  envs/static_maze.py             MiniWorld-FourRooms-v0 (Easy Static)
  envs/fourrooms_hard.py          MiniWorld-FourRoomsHard-v0 (fixed pillars + distractor)
  envs/fourrooms_hard_dynamic.py  MiniWorld-FourRoomsHardDynamic-v0 (moving obstacles)
  envs/wrappers.py                Grayscale64, Stroboscopic (Agent B), ActiveGating (C v1),
                                  ActiveVision (C v2), proprioceptive obs
  agents/ppo.py                   CleanRL-style single-file PPO (Nature-CNN encoder)
  agents/ppo_lstm.py              LSTM PPO backbone shared by all agents
configs/                          YAML configs (see Agent Configs below)
scripts/train.py                  Config-driven training entrypoint
scripts/eval.py                   Rollout + evaluation metrics
scripts/eval_dynamic_batch.py     Batched Hard-Dynamic evaluation
scripts/eval_heatmap.py           Spatial HF-usage heatmaps
scripts/plot_hf_analyses*.py      HF-usage / training-curve plots
scripts/plot_dynamic_3d.py        3D trajectory plots
scripts/record_episode_video.py   Episode video rendering
setup_env_izar.sh                 One-time Izar (SCITAS) env setup
submit_job.sh / submit_job_*.sh   SLURM submission scripts
```

## Agent Configs

### Easy Static
| Agent      | Config File |
|------------|-------------|
| Agent A    | `agent_a_fourrooms_lstm_turn10.yaml` |
| Agent B    | `agent_b_fourrooms_lstm_turn10.yaml` |
| Agent C v1 | `agent_c_fourrooms_lstm_turn10_hf1.yaml` |
| Agent C v2 | `agent_c2_fourrooms_lstm_turn10.yaml` |

### Hard Static
| Agent                        | Config File |
|------------------------------|-------------|
| Agent A                      | `agent_a_fourroomshard_v2_lstm_proprio_turn10_3M.yaml` |
| Agent B                      | `agent_b_fourroomshard_v2_lstm_proprio_turn10.yaml` |
| Agent C v2 (static trained)  | `agent_c2_fourroomshard_4d_lstm_proprio_turn10_6M_gate_vc000001.yaml` |
| Agent C v2 (dynamic trained) | `agent_c2_fourroomshard_dynamic_4d_lstm_proprio_turn10_6M_gate_vc000002.yaml` |

### Hard Dynamic
| Agent      | Config File |
|------------|-------------|
| Agent A    | `agent_a_fourroomshard_dynamic_lstm_proprio_turn10.yaml` |
| Agent B    | `agent_b_fourroomshard_dynamic_4d_lstm_proprio_turn10.yaml` |
| Agent C v2 | `agent_c2_fourroomshard_dynamic_4d_lstm_proprio_turn10_6M_gate_vc000002.yaml` |

## Key Hyperparameters Default Values

| Hyperparameter | Value |
|---|---|
| Total timesteps | 6 M |
| Parallel envs / rollout steps | 16 / 256 |
| Learning rate | 2.5 × 10⁻⁴ (linear → 0) |
| Discount γ / GAE λ | 0.99 / 0.95 |
| Clip coef / minibatches / epochs | 0.2 / 4 / 4 |
| LSTM hidden size | 256 |
| Turn step | 10° |
| Strobe `k` (Agent B / C v2 low-freq) | 7 (≈5 Hz) |
| Vision cost `c_vision` (Agent C v2) | 1 × 10⁻⁴ |
| Entropy coef | 0.005 (C v2 gate curriculum: 0.02 → 0.001) |

## Local development (macOS / Linux with uv)

```bash
# Install uv if needed: https://docs.astral.sh/uv/
uv sync

# Env smoke test
uv run python -m cff_rl.envs.static_maze

# Short training run
uv run python scripts/train.py --config configs/agent_a_static.yaml --total-timesteps 10000
```

MiniWorld uses pyglet for rendering. On a headless machine it will pick up
EGL/OSMesa automatically; no display server is required.

## Training on Izar (SCITAS)

One-time setup on a login node:

```bash
ssh <username>@izar.epfl.ch
cd CFF-InspiredFrameRateDegradationRL
bash setup_env_izar.sh
```

Submit a training job:

```bash
sbatch submit_job.sh "$WANDB_API_KEY"
```

Monitor:

```bash
squeue -u "$USER"
tail -f cff_agent_a_<jobid>.out
```

For interactive debugging on a GPU node, see `scitas_tutorial.md`.
