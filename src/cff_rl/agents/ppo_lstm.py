"""Recurrent PPO (CleanRL `ppo_atari_lstm.py` style).

CNN encoder → FC(512) → LSTM(hidden) → shared actor / critic heads.
Per-env hidden state is carried across rollout steps and zeroed on
episode boundaries. PPO minibatching shuffles ENVIRONMENTS only — each
minibatch is unrolled for the full num_steps with its saved initial
LSTM state so BPTT and resets are preserved.

Frame stacking is still available (`cfg.frame_stack`); the LSTM and
stack k are independent knobs.

Shares `PPOConfig`, `layer_init`, and `_count_reversals` with `ppo.py`.
"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

from cff_rl.agents.ppo import PPOConfig, _count_reversals, layer_init


class RecurrentNatureCNN(nn.Module):
    """CNN + LSTM recurrent agent over a (C, 64, 64) uint8 image stack."""

    def __init__(
        self,
        in_channels: int,
        n_actions: int,
        lstm_hidden_size: int = 256,
        lstm_num_layers: int = 1,
        n_extras: int = 0,
    ):
        super().__init__()
        self.n_extras = int(n_extras)
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 64, 64)
            feat_dim = self.network(dummy).shape[1]
        self.fc = nn.Sequential(
            layer_init(nn.Linear(feat_dim, 512)),
            nn.ReLU(),
        )
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        lstm_input_dim = 512 + self.n_extras
        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_size, num_layers=lstm_num_layers)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0.0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.actor = layer_init(nn.Linear(lstm_hidden_size, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(lstm_hidden_size, 1), std=1.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.network(x / 255.0))

    def initial_state(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(
            self.lstm_num_layers, batch_size, self.lstm_hidden_size, device=device
        )
        return (z, z.clone())

    def get_states(
        self,
        x: torch.Tensor,
        lstm_state: tuple[torch.Tensor, torch.Tensor],
        done: torch.Tensor,
        extras: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        feats = self.encode(x)
        if self.n_extras > 0:
            assert extras is not None, "extras required when n_extras > 0"
            feats = torch.cat([feats, extras], dim=1)
        n_envs = lstm_state[0].shape[1]
        feats = feats.reshape((-1, n_envs, feats.shape[1]))  # (T, N, 512+n_extras)
        done = done.reshape((-1, n_envs))  # (T, N)
        outputs: list[torch.Tensor] = []
        for h_t, d_t in zip(feats, done):
            mask = (1.0 - d_t).view(1, -1, 1)
            lstm_state = (mask * lstm_state[0], mask * lstm_state[1])
            out, lstm_state = self.lstm(h_t.unsqueeze(0), lstm_state)
            outputs.append(out)
        flat = torch.flatten(torch.cat(outputs), 0, 1)
        return flat, lstm_state

    def get_value(
        self,
        x: torch.Tensor,
        lstm_state: tuple[torch.Tensor, torch.Tensor],
        done: torch.Tensor,
        extras: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h, _ = self.get_states(x, lstm_state, done, extras)
        return self.critic(h)

    def get_action_and_value(
        self,
        x: torch.Tensor,
        lstm_state: tuple[torch.Tensor, torch.Tensor],
        done: torch.Tensor,
        action: torch.Tensor | None = None,
        extras: torch.Tensor | None = None,
    ):
        h, lstm_state = self.get_states(x, lstm_state, done, extras)
        logits = self.actor(h)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return (
            action,
            dist.log_prob(action),
            dist.entropy(),
            self.critic(h),
            lstm_state,
        )


def train(
    cfg: PPOConfig,
    env_fn: Callable[..., gym.Env],
) -> None:
    """Train recurrent (LSTM) PPO."""
    cfg.finalize()

    run_name = os.environ.get(
        "CFF_RUN_NAME", f"{cfg.exp_name}__{cfg.seed}__{int(time.time())}"
    )
    run_dir = Path(cfg.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if cfg.track:
        import wandb

        wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            name=run_name,
            config=vars(cfg),
            sync_tensorboard=True,
            save_code=True,
        )
    writer = SummaryWriter(str(run_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n"
        + "\n".join(f"|{k}|{v}|" for k, v in vars(cfg).items()),
    )

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.torch_deterministic

    device = torch.device(
        "cuda" if cfg.cuda and torch.cuda.is_available() else "cpu"
    )

    import inspect

    _env_fn_arity = len(inspect.signature(env_fn).parameters)

    def _make(i: int):
        if _env_fn_arity >= 2:
            return env_fn(cfg.seed + i, i)
        return env_fn(cfg.seed + i)

    envs = gym.vector.SyncVectorEnv(
        [lambda i=i: _make(i) for i in range(cfg.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)
    n_actions = int(envs.single_action_space.n)

    if cfg.use_proprio:
        assert isinstance(envs.single_observation_space, gym.spaces.Dict), (
            "use_proprio=True requires Dict obs from ProprioWrapper"
        )
        obs_shape = envs.single_observation_space["image"].shape
        n_extras = int(envs.single_observation_space["extras"].shape[0])
    else:
        obs_shape = envs.single_observation_space.shape
        n_extras = 0

    agent = RecurrentNatureCNN(
        in_channels=obs_shape[0],
        n_actions=n_actions,
        lstm_hidden_size=cfg.lstm_hidden_size,
        lstm_num_layers=cfg.lstm_num_layers,
        n_extras=n_extras,
    ).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    obs = torch.zeros((cfg.num_steps, cfg.num_envs) + obs_shape, dtype=torch.uint8).to(device)
    extras_buf = (
        torch.zeros((cfg.num_steps, cfg.num_envs, n_extras), dtype=torch.float32).to(device)
        if n_extras > 0
        else None
    )
    actions = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.long).to(device)
    logprobs = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    rewards = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    dones = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    values = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)

    ep_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    ep_lengths = np.zeros(cfg.num_envs, dtype=np.int64)
    ep_action_hist: list[list[int]] = [[] for _ in range(cfg.num_envs)]
    # Agent C v1: count STOP_AND_LOOK uses within each episode.
    ep_sal_count = np.zeros(cfg.num_envs, dtype=np.int64)
    # Agent C v2: count high-freq action choices within each episode.
    ep_hf_count = np.zeros(cfg.num_envs, dtype=np.int64)

    def _split_obs(o):
        if n_extras > 0:
            img = torch.as_tensor(o["image"], dtype=torch.uint8).to(device)
            ext = torch.as_tensor(o["extras"], dtype=torch.float32).to(device)
            return img, ext
        return torch.as_tensor(o, dtype=torch.uint8).to(device), None

    global_step = 0
    start_time = time.time()
    next_obs_np, _ = envs.reset(seed=cfg.seed)
    next_obs, next_extras = _split_obs(next_obs_np)
    next_done = torch.zeros(cfg.num_envs).to(device)
    next_lstm_state = agent.initial_state(cfg.num_envs, device)

    for iteration in range(1, cfg.num_iterations + 1):
        initial_lstm_state = (
            next_lstm_state[0].clone(),
            next_lstm_state[1].clone(),
        )
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        for step in range(cfg.num_steps):
            global_step += cfg.num_envs
            obs[step] = next_obs
            if extras_buf is not None:
                extras_buf[step] = next_extras
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = (
                    agent.get_action_and_value(
                        next_obs.float(),
                        next_lstm_state,
                        next_done,
                        extras=next_extras,
                    )
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32).to(device)
            next_obs, next_extras = _split_obs(next_obs_np)
            next_done = torch.as_tensor(done, dtype=torch.float32).to(device)

            a_np = action.cpu().numpy()
            # Extract per-env STOP_AND_LOOK flag from vectorised infos (Agent C v1).
            sal_step = (
                np.asarray(
                    infos["stop_and_look"], dtype=bool
                )
                if cfg.use_active_gating else None
            )
            # Extract per-env high-freq flag from vectorised infos (Agent C v2).
            hf_step = (
                np.asarray(
                    infos["high_freq"], dtype=bool
                )
                if cfg.use_active_vision else None
            )
            for i in range(cfg.num_envs):
                ep_returns[i] += float(reward[i])
                ep_lengths[i] += 1
                ep_action_hist[i].append(int(a_np[i]))
                if cfg.use_active_gating:
                    ep_sal_count[i] += int(sal_step[i])
                if cfg.use_active_vision:
                    ep_hf_count[i] += int(hf_step[i])
                if done[i]:
                    success = 1.0 if bool(terminated[i]) else 0.0
                    writer.add_scalar("charts/episodic_return", ep_returns[i], global_step)
                    writer.add_scalar("charts/episodic_length", ep_lengths[i], global_step)
                    writer.add_scalar("charts/success", success, global_step)
                    writer.add_scalar(
                        "charts/reversals",
                        _count_reversals(ep_action_hist[i]),
                        global_step,
                    )
                    if cfg.use_active_gating:
                        writer.add_scalar(
                            "charts/sal_per_episode", ep_sal_count[i], global_step
                        )
                        writer.add_scalar(
                            "charts/sal_rate",
                            ep_sal_count[i] / ep_lengths[i],
                            global_step,
                        )
                        ep_sal_count[i] = 0
                    if cfg.use_active_vision:
                        writer.add_scalar(
                            "charts/highfreq_per_episode", ep_hf_count[i], global_step
                        )
                        writer.add_scalar(
                            "charts/highfreq_rate",
                            ep_hf_count[i] / ep_lengths[i],
                            global_step,
                        )
                        ep_hf_count[i] = 0
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0
                    ep_action_hist[i] = []

        with torch.no_grad():
            next_value = agent.get_value(
                next_obs.float(), next_lstm_state, next_done, extras=next_extras
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t]
                    + cfg.gamma * nextvalues * nextnonterminal
                    - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # Recurrent minibatching: shuffle ENV indices only; each minibatch
        # is unrolled with its own initial LSTM state so BPTT/per-env
        # resets are preserved.
        env_inds = np.arange(cfg.num_envs)
        clipfracs = []
        for _ in range(cfg.update_epochs):
            np.random.shuffle(env_inds)
            for start in range(0, cfg.num_envs, cfg.envs_per_minibatch):
                mb_envs = env_inds[start : start + cfg.envs_per_minibatch]
                mb_envs_t = torch.as_tensor(mb_envs, dtype=torch.long, device=device)

                mb_obs = obs[:, mb_envs_t]
                mb_dones = dones[:, mb_envs_t]
                mb_logprobs_old = logprobs[:, mb_envs_t].reshape(-1)
                mb_actions = actions[:, mb_envs_t].reshape(-1)
                mb_advantages = advantages[:, mb_envs_t].reshape(-1)
                mb_returns = returns[:, mb_envs_t].reshape(-1)
                mb_values_old = values[:, mb_envs_t].reshape(-1)
                mb_initial_state = (
                    initial_lstm_state[0][:, mb_envs_t],
                    initial_lstm_state[1][:, mb_envs_t],
                )

                flat_obs = mb_obs.reshape((-1,) + obs_shape).float()
                flat_dones = mb_dones.reshape(-1)
                flat_extras = (
                    extras_buf[:, mb_envs_t].reshape(-1, n_extras)
                    if extras_buf is not None
                    else None
                )
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    flat_obs,
                    mb_initial_state,
                    flat_dones,
                    mb_actions,
                    extras=flat_extras,
                )

                logratio = newlogprob - mb_logprobs_old
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    )

                norm_advantages = mb_advantages
                if cfg.norm_adv:
                    norm_advantages = (norm_advantages - norm_advantages.mean()) / (
                        norm_advantages.std() + 1e-8
                    )

                pg_loss1 = -norm_advantages * ratio
                pg_loss2 = -norm_advantages * torch.clamp(
                    ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - mb_returns) ** 2
                    v_clipped = mb_values_old + torch.clamp(
                        newvalue - mb_values_old,
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - mb_returns) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - mb_returns) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        y_pred = values.reshape(-1).cpu().numpy()
        y_true = returns.reshape(-1).cpu().numpy()
        var_y = np.var(y_true)
        explained_var = (
            float("nan") if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        )

        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / (time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)
        writer.add_scalar(
            "charts/learning_rate",
            optimizer.param_groups[0]["lr"],
            global_step,
        )
        print(
            f"iter {iteration}/{cfg.num_iterations}  step={global_step}  SPS={sps}"
        )

        if iteration % cfg.checkpoint_every == 0 or iteration == cfg.num_iterations:
            ckpt = run_dir / f"ckpt_{iteration:06d}.pt"
            torch.save(
                {
                    "agent": agent.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iteration": iteration,
                    "global_step": global_step,
                    "config": vars(cfg),
                    "arch": {
                        "in_channels": obs_shape[0],
                        "n_actions": n_actions,
                        "recurrent": True,
                        "lstm_hidden_size": cfg.lstm_hidden_size,
                        "lstm_num_layers": cfg.lstm_num_layers,
                        "frame_stack": cfg.frame_stack,
                        "use_proprio": cfg.use_proprio,
                        "use_stroboscopic": cfg.use_stroboscopic,
                        "use_active_gating": cfg.use_active_gating,
                        "use_active_vision": cfg.use_active_vision,
                        "vision_cost": cfg.vision_cost,
                        "strobe_k": cfg.strobe_k,
                        "high_freq_steps": cfg.high_freq_steps,
                        "n_extras": n_extras,
                        "turn_step_deg": cfg.turn_step_deg,
                    },
                },
                ckpt,
            )

    envs.close()
    writer.close()
    if cfg.track:
        import wandb

        wandb.finish()
