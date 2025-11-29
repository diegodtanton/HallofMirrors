# ============================================
# File: shared/agent.py
# ============================================

import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Callable
import os

from .config import PLOT_INTERVAL, CHECKPOINT_INTERVAL
from .model import ActorCriticNet


@dataclass
class PPOConfig:
    total_steps: int = 200_000
    update_steps: int = 2048
    minibatch_size: int = 256
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PPOAgent:
    def __init__(self, env, hidden_size=64, n_hidden_layers=1, config=PPOConfig()):
        self.env = env
        self.config = config
        self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers

        self.device = config.device
        self.net = ActorCriticNet(
            env.observation_space.shape, env.action_space.n, hidden_size, n_hidden_layers
        ).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=config.lr)
        self.total_env_steps = 0

    # --------------------------
    # I/O Helpers
    # --------------------------
    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.total_env_steps = ckpt.get("total_env_steps", 0)
        print(f"Loaded checkpoint from {path}")

    def save_checkpoint(self, path):
        torch.save(
            {
                "model_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "total_env_steps": self.total_env_steps,
                "config": self.config,
                "hidden_size": self.hidden_size,
                "n_hidden_layers": self.n_hidden_layers,
            },
            path,
        )

    def get_latent(self, obs_np, prev_reward_val=0.0):
        # Helper for analysis
        obs_t = torch.from_numpy(obs_np).float().to(self.device)
        pr_t = torch.tensor([[prev_reward_val]], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            _, _, z = self.net(obs_t, pr_t)
        return z.cpu().numpy()

    def select_action(self, obs, prev_reward):
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        pr_t = torch.tensor([[prev_reward]], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits, val, _ = self.net(obs_t, pr_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
        return int(action.item()), float(logprob.item()), float(val.item())

    # --------------------------
    # PPO Training Loop
    # --------------------------
    def train(
        self,
        verbose: bool = True,
        stop_at_return: Optional[float] = None,
        checkpoint_dir: Optional[str] = None,
        plot_callback: Optional[Callable] = None,
    ):
        cfg = self.config
        step = self.total_env_steps

        # Initial Reset
        obs, _ = self.env.reset()
        obs = obs.astype(np.float32)
        prev_reward = 0.0

        # Buffers now include prev_rewards (pr_b)
        obs_b, pr_b, act_b, lp_b, rew_b, done_b, val_b = [], [], [], [], [], [], []

        return_hist = deque(maxlen=20)
        training_stats = []

        next_plot = (step // PLOT_INTERVAL + 1) * PLOT_INTERVAL
        next_ckpt = (step // CHECKPOINT_INTERVAL + 1) * CHECKPOINT_INTERVAL

        while step < cfg.total_steps:
            # 1. ROLLOUT
            for _ in range(cfg.update_steps):
                action, lp, val = self.select_action(obs, prev_reward)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                obs_b.append(obs)
                pr_b.append(prev_reward) # Store input PR
                act_b.append(action)
                lp_b.append(lp)
                rew_b.append(reward)
                done_b.append(done)
                val_b.append(val)

                obs = next_obs.astype(np.float32)
                prev_reward = float(reward)
                step += 1

                if done:
                    obs, _ = self.env.reset()
                    obs = obs.astype(np.float32)
                    prev_reward = 0.0

                if step >= cfg.total_steps:
                    break

            # 2. CALCULATION
            with torch.no_grad():
                # Value of NEXT state
                obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                pr_t = torch.tensor([[prev_reward]], dtype=torch.float32).to(self.device)
                _, last_val, _ = self.net(obs_t, pr_t)
                last_val = float(last_val.item())

            T_rew = torch.tensor(rew_b, dtype=torch.float32, device=self.device)
            T_val = torch.tensor(val_b, dtype=torch.float32, device=self.device)
            T_done = torch.tensor(done_b, dtype=torch.float32, device=self.device)

            advs = []
            gae = 0.0
            for i in reversed(range(len(rew_b))):
                mask = 1.0 - T_done[i].item()
                next_val = last_val if i == len(rew_b) - 1 else T_val[i + 1].item()
                delta = (
                    T_rew[i].item()
                    + cfg.gamma * next_val * mask
                    - T_val[i].item()
                )
                gae = delta + cfg.gamma * cfg.gae_lambda * mask * gae
                advs.insert(0, gae)

            T_obs = torch.tensor(np.array(obs_b), dtype=torch.float32, device=self.device)
            T_pr = torch.tensor(np.array(pr_b), dtype=torch.float32, device=self.device).unsqueeze(1)
            T_act = torch.tensor(act_b, dtype=torch.long, device=self.device)
            T_lp = torch.tensor(lp_b, dtype=torch.float32, device=self.device)
            T_adv = torch.tensor(advs, dtype=torch.float32, device=self.device)
            T_ret = T_adv + T_val

            T_adv = (T_adv - T_adv.mean()) / (T_adv.std() + 1e-8)

            dataset_size = len(obs_b)
            idxs = np.arange(dataset_size)

            # 3. OPTIMIZATION
            for _ in range(cfg.ppo_epochs):
                np.random.shuffle(idxs)
                for start in range(0, dataset_size, cfg.minibatch_size):
                    end = start + cfg.minibatch_size
                    b_idx = idxs[start:end]
                    
                    # Forward pass with OBS and PREV_REWARD
                    logits, v, _ = self.net(T_obs[b_idx], T_pr[b_idx])
                    
                    dist = torch.distributions.Categorical(logits=logits)
                    new_lp = dist.log_prob(T_act[b_idx])
                    entropy = dist.entropy().mean()

                    ratio = (new_lp - T_lp[b_idx]).exp()
                    surr1 = ratio * T_adv[b_idx]
                    surr2 = torch.clamp(
                        ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps
                    ) * T_adv[b_idx]

                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = (T_ret[b_idx] - v.squeeze(-1)).pow(2).mean()
                    loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), cfg.max_grad_norm)
                    self.optimizer.step()

            # Clear buffers
            obs_b.clear(); pr_b.clear(); act_b.clear(); lp_b.clear()
            rew_b.clear(); done_b.clear(); val_b.clear()

            self.total_env_steps = step

            # 4. MAINTENANCE
            if step >= next_plot or step >= cfg.total_steps:
                avg_ret = evaluate_agent(self, self.env, n_episodes=10)
                
                # CRITICAL: Reset env after eval to sync state
                obs, _ = self.env.reset()
                obs = obs.astype(np.float32)
                prev_reward = 0.0

                return_hist.append(avg_ret)
                rolling = float(np.mean(return_hist))

                stats_entry = {
                    "step": step,
                    "instant_return": avg_ret,
                    "rolling_return": rolling,
                }
                training_stats.append(stats_entry)

                if verbose:
                    print(
                        f"  Step {step} | Instant: {avg_ret:.2f} | Rolling: {rolling:.2f}"
                    )

                if plot_callback:
                    plot_callback(training_stats)

                if (
                    stop_at_return is not None
                    and len(return_hist) >= 10
                    and rolling >= stop_at_return
                ):
                    print(
                        f"  -> Solved! Reached {stop_at_return:.2f} (Rolling: {rolling:.2f})"
                    )
                    if checkpoint_dir:
                        self.save_checkpoint(
                            os.path.join(checkpoint_dir, "final_solved.pt")
                        )
                    break

                while next_plot <= step:
                    next_plot += PLOT_INTERVAL

            if checkpoint_dir and step >= next_ckpt:
                filename = f"ckpt_step_{step}.pt"
                self.save_checkpoint(os.path.join(checkpoint_dir, filename))
                while next_ckpt <= step:
                    next_ckpt += CHECKPOINT_INTERVAL

        return training_stats


def evaluate_agent(agent: PPOAgent, env, n_episodes=20) -> float:
    total = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs = obs.astype(np.float32)
        prev_reward = 0.0
        done = False
        ep_ret = 0.0
        while not done:
            action, _, _ = agent.select_action(obs, prev_reward)
            obs, r, term, trunc, _ = env.step(action)
            obs = obs.astype(np.float32)
            prev_reward = float(r)
            ep_ret += r
            done = term or trunc
        total += ep_ret
    return total / n_episodes