# Adapted from vendor/cleanrl_ppo_reference.py (CleanRL ppo.py, commit
# fe8d8a03c41a7ef5b523e2e354bd01c363e786bb). The vendored file is kept
# byte-identical for upstream diffability; the changes for HexFallEnv are:
#   - Dict observation encoder (flatten + concat all keys except action_mask,
#     then MLP 256/256/tanh) — see encode_obs and Agent.
#   - Action-mask logits masking via masked_fill(-inf) inside the policy
#     forward pass — see Agent._masked_dist. Applied both at rollout sampling
#     and at PPO update time.
#   - num_envs = 1 (vectorization deferred per PROJECT_STATE.md).
#   - 30-minute wall-clock checkpointing with --resume (network + optimizer +
#     global_step + Python/NumPy/Torch RNG states).
# The PPO update math (GAE, ratio clip, value loss, entropy bonus, KL,
# advantage normalization, gradient clipping) is unchanged from the template.
import argparse
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from hexfall.env import HexFallEnv


DEFAULT_CHECKPOINT_INTERVAL_SECONDS = 30 * 60


def make_env(level_path: str, seed: int):
    def thunk():
        env = HexFallEnv(level_path, seed=seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def encode_obs(obs_dict, device):
    """Flatten + concat every observation key except action_mask.

    Returns:
        enc:   (num_envs, obs_dim) float32 tensor — input to the encoder MLP.
        mask:  (num_envs, n_actions) bool tensor — True on legal cells.
    """
    flat_keys = []
    mask = None
    for k in sorted(obs_dict.keys()):
        v = np.asarray(obs_dict[k])
        t = torch.as_tensor(v, dtype=torch.float32, device=device)
        if k == "action_mask":
            mask = t.bool()
        else:
            flat_keys.append(t.reshape(t.shape[0], -1))
    enc = torch.cat(flat_keys, dim=1)
    return enc, mask


class Agent(nn.Module):
    """Two-tower MLP. Mask is applied to actor logits before constructing the
    Categorical distribution. Critic does not see the mask.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, n_actions), std=0.01),
        )

    def _masked_dist(self, x_enc: torch.Tensor, mask: torch.Tensor) -> Categorical:
        logits = self.actor(x_enc)
        logits = logits.masked_fill(~mask, float("-inf"))
        if not torch.isfinite(logits).any(dim=1).all():
            raise RuntimeError(
                "All-illegal action mask hit in policy forward — env should "
                "have terminated before this state was observed."
            )
        return Categorical(logits=logits)

    def get_value(self, x_enc):
        return self.critic(x_enc)

    def get_action_and_value(self, x_enc, mask, action=None):
        dist = self._masked_dist(x_enc, mask)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x_enc)


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--level-path", default="levels/tiny_solvable.json")
    p.add_argument("--exp-name", default="train_ppo")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cuda", action="store_true", default=True)
    p.add_argument("--total-timesteps", type=int, default=100_000)
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--anneal-lr", action="store_true", default=True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--norm-adv", action="store_true", default=True)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--clip-vloss", action="store_true", default=True)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument(
        "--checkpoint-interval-seconds",
        type=float,
        default=DEFAULT_CHECKPOINT_INTERVAL_SECONDS,
    )
    p.add_argument("--run-name", default=None)
    return p.parse_args(argv)


def save_checkpoint(path: Path, agent, optimizer, global_step: int, run_name: str):
    torch.save(
        {
            "agent_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": int(global_step),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
            "run_name": run_name,
        },
        path,
    )


def load_checkpoint(path: str, agent, optimizer, device):
    blob = torch.load(path, map_location=device, weights_only=False)
    agent.load_state_dict(blob["agent_state_dict"])
    optimizer.load_state_dict(blob["optimizer_state_dict"])
    rng = blob["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"].cpu())
    if rng["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["torch_cuda"]])
    return int(blob["global_step"])


def _log_episode_infos(infos: dict, writer: SummaryWriter, global_step: int):
    """Gymnasium 1.x SyncVectorEnv exposes RecordEpisodeStatistics output via
    infos['episode'] (dict of arrays) gated by infos['_episode'] (bool mask).
    """
    ep = infos.get("episode")
    mask = infos.get("_episode")
    if not isinstance(ep, dict) or mask is None:
        return
    mask = np.asarray(mask)
    rewards = np.asarray(ep.get("r", []))
    lengths = np.asarray(ep.get("l", []))
    for i in range(len(mask)):
        if mask[i] and i < len(rewards) and i < len(lengths):
            writer.add_scalar("charts/episodic_return", float(rewards[i]), global_step)
            writer.add_scalar("charts/episodic_length", float(lengths[i]), global_step)


def main(argv=None):
    args = parse_args(argv)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # At least one iteration even when total_timesteps < batch_size (sanity runs).
    args.num_iterations = max(1, args.total_timesteps // args.batch_size)

    run_name = args.run_name or (
        f"{Path(args.level_path).stem}__{args.exp_name}__{args.seed}__{int(time.time())}"
    )
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(run_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # SAME_STEP autoreset: prevents the all-zero action_mask of terminal HexFall states from
    # reaching the policy and tripping the all-illegal bug guard. Default NEXT_STEP returns the
    # terminal obs once before reset; HexFall terminal states have no legal actions, which would
    # trigger the guard on non-bug behavior. See issue #8 worker session 2 report.
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.level_path, args.seed + i) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), (
        "only discrete action space is supported"
    )
    n_actions = int(envs.single_action_space.n)

    probe_obs, _ = envs.reset(seed=args.seed)
    probe_enc, probe_mask = encode_obs(probe_obs, device)
    obs_dim = int(probe_enc.shape[1])

    agent = Agent(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    global_step = 0
    if args.resume is not None:
        global_step = load_checkpoint(args.resume, agent, optimizer, device)
        print(f"Resumed from {args.resume} at global_step={global_step}")
        # Reset envs to fresh state after resume; the restored RNG controls
        # subsequent stochastic sampling.
        next_obs, _ = envs.reset(seed=args.seed + 10_000 + global_step)
        next_enc, next_mask = encode_obs(next_obs, device)
    else:
        next_enc, next_mask = probe_enc, probe_mask

    obs_buf = torch.zeros((args.num_steps, args.num_envs, obs_dim), device=device)
    mask_buf = torch.zeros(
        (args.num_steps, args.num_envs, n_actions), dtype=torch.bool, device=device
    )
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    start_time = time.time()
    last_checkpoint_time = time.monotonic()
    next_done = torch.zeros(args.num_envs, device=device)

    while global_step < args.total_timesteps:
        if args.anneal_lr:
            frac = 1.0 - min(1.0, global_step / max(1, args.total_timesteps))
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_enc
            mask_buf[step] = next_mask
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_enc, next_mask)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terms, truncs, infos = envs.step(action.cpu().numpy())
            next_done_np = np.logical_or(terms, truncs).astype(np.float32)
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device).view(-1)
            next_enc, next_mask = encode_obs(next_obs, device)
            next_done = torch.as_tensor(next_done_np, device=device)
            _log_episode_infos(infos, writer, global_step)

        with torch.no_grad():
            next_value = agent.get_value(next_enc).reshape(1, -1)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs_buf.reshape(-1, obs_dim)
        b_masks = mask_buf.reshape(-1, n_actions)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_masks[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("losses/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", float(np.mean(clipfracs)), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        sps = int(global_step / max(1e-6, time.time() - start_time))
        writer.add_scalar("charts/SPS", sps, global_step)

        now = time.monotonic()
        saved_this_iter = 0
        if now - last_checkpoint_time >= args.checkpoint_interval_seconds:
            ckpt_path = run_dir / f"checkpoint_{int(time.time())}.pt"
            save_checkpoint(ckpt_path, agent, optimizer, global_step, run_name)
            last_checkpoint_time = now
            saved_this_iter = 1
            print(f"Checkpoint -> {ckpt_path} @ global_step={global_step}")
        writer.add_scalar("charts/checkpoint_saved", saved_this_iter, global_step)

    save_checkpoint(run_dir / "checkpoint_final.pt", agent, optimizer, global_step, run_name)
    writer.add_scalar("charts/checkpoint_saved", 1, global_step)
    print(f"Done. global_step={global_step}, run_dir={run_dir}")

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
