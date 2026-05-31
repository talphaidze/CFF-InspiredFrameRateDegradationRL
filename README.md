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
scripts/eval_heatmap.py           Another type of Spatial HF-usage heatmaps
scripts/plot_hf_analyses.py       HeatMap (present in Website)
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
| Agent C v2                   | `agent_c2_fourroomshard_4d_lstm_proprio_turn10_6M_gate_vc000001.yaml` |
| Agent C v2 (dynamic trained) | `agent_c2_fourroomshard_dynamic_4d_lstm_proprio_turn10_6M_gate_vc000002.yaml` |

### Hard Dynamic
| Agent      | Config File |
|------------|-------------|
| Agent A    | `agent_a_fourroomshard_dynamic_lstm_proprio_turn10.yaml` |
| Agent B    | `agent_b_fourroomshard_dynamic_4d_lstm_proprio_turn10.yaml` |
| Agent C v2 | `agent_c2_fourroomshard_dynamic_4d_lstm_proprio_turn10_6M_gate_vc000002.yaml` |

## Setup on Izar (SCITAS)

One-time setup on a login node:

```bash
ssh <username>@izar.epfl.ch
cd CFF-InspiredFrameRateDegradationRL
bash setup_env_izar.sh
```

### Training

Submit a training job:

```bash
sbatch submit_job.sh configs/<config_file> "$WANDB_API_KEY"
```
Please replace `<config_file>` with the config file of the agent you want to train (see the `Agent Configs` table for the config file name corresponding to each agent). `WANDB_API_KEY` is your personal wandb's access api key.

### Evaluation

Submit an evaluation job:

```bash
sbatch submit_job_eval.sh <run_name> <env_id>
```
Please replace `<run_name>` with the folder name which is created after training an agent (it contains the trained agent's checkpoints and config), and replace the `<env_id>` with the name of the env you want to test your agent in (MiniWorld-FourRooms-v0/ MiniWorld-FourRoomsHard-v0/ MiniWorld-FourRoomsHardDynamic-v0)

The results (`eval_results.json`) will get saved in the same `<run_name>` folder.

For example:
```bash
sbatch submit_job_eval.sh agent_a_fourroomshard_dynamic_4d_lstm_proprio_turn10__42__1780100225 MiniWorld-FourRoomsHardDynamic-v0
```

### Heatmap

```bash
sbatch submit_job_heatmap.sh <vc2e5ckpt> <vc1e4ckpt>
```
where `<vc2e5ckpt>` is the Agent C v2 checkpoint when trained with `agent_c2_fourroomshard_dynamic_4d_lstm_proprio_turn10_6M_gate_vc000002.yaml` file and `<vc1e4ckpt>` is the Agent C v2 checkpoint when trained with `agent_c2_fourroomshard_dynamic_4d_lstm_proprio_turn10_6M_gate_vc00001.yaml` file

### Episode Video
```bash
sbatch submit_job_epvideo.sh <vc1e4ckpt>
```
where `<vc1e4ckpt>` is the Agent C v2 checkpoint when trained with `agent_c2_fourroomshard_dynamic_4d_lstm_proprio_turn10_6M_gate_vc00001.yaml` file

For interactive debugging on a GPU node, see `scitas_tutorial.md`.
