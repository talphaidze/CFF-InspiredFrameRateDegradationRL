"""Feed-forward PPO for image-observation RL (CleanRL `ppo_atari.py` style).

Single-file. The `NatureCNN` agent here has no recurrence — temporal
context comes only from the k-frame stack in the env wrappers
(`PPOConfig.frame_stack`). For the recurrent variant see `ppo_lstm.py`.

References:
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

    record_video: bool = False
    video_every: int = 50  # episodes (env 0 only)

    # Architecture
    # `recurrent` selects between this module's feed-forward train() and
    # the LSTM-aware train() in ppo_lstm.py — train.py dispatches on it.
    recurrent: bool = False
    frame_stack: int = 4
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 1

    # Proprioceptive extras (NavA3C-style): prev_action one-hot + prev_reward
    # + heading sin/cos. Toggles a Dict obs space and concat-after-FC.
    use_proprio: bool = False
    # Agent B: hold each visual frame for strobe_k steps (~5 Hz).
    use_stroboscopic: bool = False
    strobe_k: int = 7
    # Agent C: stroboscopic by default; STOP_AND_LOOK triggers 35 Hz for
    # high_freq_steps consecutive steps (proposal: 35 steps = 1 s).
    # Mutually exclusive with use_stroboscopic.
    use_active_gating: bool = False
    high_freq_steps: int = 35
    # FourRooms uses 90° turns by default; lower values are a learning-side
    # ablation, not a CFF claim.
    turn_step_deg: int = 90

    # Derived at runtime
    batch_size: int = field(init=False, default=0)
    minibatch_size: int = field(init=False, default=0)
    num_iterations: int = field(init=False, default=0)
    envs_per_minibatch: int = field(init=False, default=0)

    def finalize(self) -> None:
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.num_iterations = self.total_timesteps // self.batch_size
        if self.recurrent:
            assert (
                self.num_envs % self.num_minibatches == 0
            ), "num_envs must be divisible by num_minibatches for recurrent PPO"
            self.envs_per_minibatch = self.num_envs // self.num_minibatches


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    """Feed-forward CNN policy over a (C, 64, 64) uint8 image stack.

    Temporal context comes from frame stacking only. Optional `n_extras`
    proprioceptive scalars are concatenated after the FC(512) layer; when
    `n_extras == 0` the network is identical to the original.
    """

    def __init__(self, in_channels: int, n_actions: int, n_extras: int = 0):
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
        head_dim = 512 + self.n_extras
        self.actor = layer_init(nn.Linear(head_dim, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(head_dim, 1), std=1.0)

    def encode(
        self, x: torch.Tensor, extras: torch.Tensor | None = None
    ) -> torch.Tensor:
        h = self.fc(self.network(x / 255.0))
        if self.n_extras > 0:
            assert extras is not None, "extras required when n_extras > 0"
            h = torch.cat([h, extras], dim=1)
        return h

    def get_value(
        self, x: torch.Tensor, extras: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.critic(self.encode(x, extras))

    def get_action_and_value(
        self,
        x: torch.Tensor,
        extras: torch.Tensor | None = None,
        action: torch.Tensor | None = None,
    ):
        h = self.encode(x, extras)
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
    env_fn: Callable[..., gym.Env],
) -> None:
    """Train feed-forward PPO. env_fn is called as env_fn(seed, env_idx)
    when it accepts two args, else env_fn(seed)."""
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
        obs_shape = envs.single_observation_space["image"].shape  # (C, H, W)
        n_extras = int(envs.single_observation_space["extras"].shape[0])
    else:
        obs_shape = envs.single_observation_space.shape  # (C, H, W)
        n_extras = 0

    agent = NatureCNN(
        in_channels=obs_shape[0], n_actions=n_actions, n_extras=n_extras
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
    # Agent C only: count STOP_AND_LOOK uses within each episode.
    ep_sal_count = np.zeros(cfg.num_envs, dtype=np.int64)

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

    for iteration in range(1, cfg.num_iterations + 1):
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
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs.float(), next_extras
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
            # Extract per-env STOP_AND_LOOK flag from vectorised infos (Agent C).
            sal_step = (
                np.asarray(
                    infos["stop_and_look"], dtype=bool
                )
                if cfg.use_active_gating else None
            )
            for i in range(cfg.num_envs):
                ep_returns[i] += float(reward[i])
                ep_lengths[i] += 1
                ep_action_hist[i].append(int(a_np[i]))
                if cfg.use_active_gating:
                    ep_sal_count[i] += int(sal_step[i])
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
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0
                    ep_action_hist[i] = []

        with torch.no_grad():
            next_value = agent.get_value(next_obs.float(), next_extras).reshape(1, -1)
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
        b_extras = (
            extras_buf.reshape(-1, n_extras) if extras_buf is not None else None
        )
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

                mb_extras = b_extras[mb_inds] if b_extras is not None else None
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds].float(), mb_extras, b_actions[mb_inds]
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
                    "arch": {
                        "in_channels": obs_shape[0],
                        "n_actions": n_actions,
                        "recurrent": False,
                        "frame_stack": cfg.frame_stack,
                        "use_proprio": cfg.use_proprio,
                        "use_stroboscopic": cfg.use_stroboscopic,
                        "use_active_gating": cfg.use_active_gating,
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
