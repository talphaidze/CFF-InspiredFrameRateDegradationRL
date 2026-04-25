"""PPO for image-observation RL, adapted from CleanRL's ppo_atari.py.

Single-file style (CleanRL convention) so later we can fork/modify it for
Groups B and C with minimal diff. References:
- https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py
- PPO paper: Schulman et al., 2017.
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym


@dataclass
class PPOConfig:
    exp_name: str = "agent_a_static"
    seed: int = 42
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project: str = "cff-rl"
    wandb_entity: str | None = None

    total_timesteps: int = 1_000_000
    learning_rate: float = 2.5e-4
    num_envs: int = 8
    num_steps: int = 128
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.1
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = None

    log_dir: str = "runs"
    checkpoint_every: int = 50  # updates

    # Derived at runtime
    batch_size: int = field(init=False, default=0)
    minibatch_size: int = field(init=False, default=0)
    num_iterations: int = field(init=False, default=0)

    def finalize(self) -> None:
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.num_iterations = self.total_timesteps // self.batch_size


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    """CNN encoder over a (C, 64, 64) uint8 image stack."""

    def __init__(self, in_channels: int, n_actions: int):
        super().__init__()
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
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.network(x / 255.0))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.critic(self.encode(x))

    def get_action_and_value(
        self, x: torch.Tensor, action: torch.Tensor | None = None
    ):
        h = self.encode(x)
        logits = self.actor(h)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(h)


def _count_reversals(actions: list[int], left: int = 0, right: int = 1) -> int:
    """Count left↔right turn-switch events in an episode's action sequence.

    Used as a perceptual-confusion proxy for the static-task analysis
    (proposal § Method, Regime 1 metrics).
    """
    n = 0
    prev_turn: int | None = None
    for a in actions:
        if a == left or a == right:
            if prev_turn is not None and a != prev_turn:
                n += 1
            prev_turn = a
    return n


def train(
    cfg: PPOConfig,
    env_fn: Callable[[int], gym.Env],
) -> None:
    cfg.finalize()

    run_name = f"{cfg.exp_name}__{cfg.seed}__{int(time.time())}"
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

    envs = gym.vector.SyncVectorEnv(
        [lambda i=i: env_fn(cfg.seed + i) for i in range(cfg.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)
    obs_shape = envs.single_observation_space.shape  # (C, H, W)
    n_actions = int(envs.single_action_space.n)

    agent = NatureCNN(in_channels=obs_shape[0], n_actions=n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # Rollout buffers
    obs = torch.zeros((cfg.num_steps, cfg.num_envs) + obs_shape, dtype=torch.uint8).to(device)
    actions = torch.zeros((cfg.num_steps, cfg.num_envs), dtype=torch.long).to(device)
    logprobs = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    rewards = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    dones = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)
    values = torch.zeros((cfg.num_steps, cfg.num_envs)).to(device)

    # Per-env episode tracking for custom metrics
    ep_returns = np.zeros(cfg.num_envs, dtype=np.float32)
    ep_lengths = np.zeros(cfg.num_envs, dtype=np.int64)
    ep_action_hist: list[list[int]] = [[] for _ in range(cfg.num_envs)]

    global_step = 0
    start_time = time.time()
    next_obs_np, _ = envs.reset(seed=cfg.seed)
    next_obs = torch.as_tensor(next_obs_np, dtype=torch.uint8).to(device)
    next_done = torch.zeros(cfg.num_envs).to(device)

    for iteration in range(1, cfg.num_iterations + 1):
        if cfg.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
            optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

        for step in range(cfg.num_steps):
            global_step += cfg.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs.float()
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs_np, reward, terminated, truncated, infos = envs.step(
                action.cpu().numpy()
            )
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32).to(device)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.uint8).to(device)
            next_done = torch.as_tensor(done, dtype=torch.float32).to(device)

            a_np = action.cpu().numpy()
            for i in range(cfg.num_envs):
                ep_returns[i] += float(reward[i])
                ep_lengths[i] += 1
                ep_action_hist[i].append(int(a_np[i]))
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
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0
                    ep_action_hist[i] = []

        # GAE
        with torch.no_grad():
            next_value = agent.get_value(next_obs.float()).reshape(1, -1)
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

        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(cfg.batch_size)
        clipfracs = []
        for _ in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds].float(), b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    )

                mb_advantages = b_advantages[mb_inds]
                if cfg.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if cfg.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -cfg.clip_coef,
                        cfg.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
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
                },
                ckpt,
            )

    envs.close()
    writer.close()
    if cfg.track:
        import wandb

        wandb.finish()
