# CFF-Inspired Frame Rate Degradation in RL

EPFL CS-503 Visual Intelligence project. Investigates whether degrading the
frame rate of visual observations improves RL sample efficiency, inspired by
Critical Flicker Fusion.

Three agents, identical except for observation frame rate:

- **Group A** — 35 Hz continuous (this deliverable)
- **Group B** — 5 Hz stroboscopic (to be added)
- **Group C** — Active Gating with `STOP_AND_LOOK` (to be added)

## Team

- Tamar Alphaidze
- Sajal Chaurasia
- Manuel Curnis
- Alessandro Di Maria

## Repo layout

```
src/cff_rl/
  envs/static_maze.py    Regime 1 MiniWorld env (OneRoom, 3 discrete actions, 64x64 grayscale, 4-frame stack)
  envs/wrappers.py       Grayscale64, FrameStack4, ActionFilter. Stroboscopic/STOP_AND_LOOK will live here.
  agents/ppo.py          CleanRL-style single-file PPO with Nature-CNN encoder
configs/agent_a_static.yaml
scripts/train.py         Config-driven training entrypoint
scripts/eval.py          Rollout + Regime 1 metrics
setup_env_izar.sh        One-time Izar env setup (miniforge + pip install -e .)
submit_job.sh            SLURM submission script for Agent A
```

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

Submit an Agent A training job:

```bash
sbatch submit_job.sh "$WANDB_API_KEY"
```

Monitor:

```bash
squeue -u "$USER"
tail -f cff_agent_a_<jobid>.out
```

For interactive debugging on a GPU node, see `scitas_tutorial.md`.
