# a2c.py  -- drop-in replacement
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from snd_w2 import build_move_cost, snd_w2_per_agent  # 新增：W2-SND

Tensor = torch.Tensor

# --------------------------- Model ---------------------------

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )
        self.actor_move = nn.Linear(hidden, 5)  # L,R,U,D,Stay
        self.actor_msg  = nn.Linear(hidden, 2)  # 1-bit message
        self.critic     = nn.Linear(hidden, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.constant_(m.bias, 0.)

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.base(obs)
        move_logits = self.actor_move(x)  # [B,5]
        msg_logits  = self.actor_msg(x)   # [B,2]
        value       = self.critic(x).squeeze(-1)  # [B]
        return move_logits, msg_logits, value

    def act(self, obs: Tensor):
        move_logits, msg_logits, value = self.forward(obs)
        dist_m = Categorical(logits=move_logits)
        dist_c = Categorical(logits=msg_logits)
        a_m = dist_m.sample()
        a_c = dist_c.sample()
        logp = dist_m.log_prob(a_m) + dist_c.log_prob(a_c)
        entropy = dist_m.entropy() + dist_c.entropy()
        return a_m, a_c, logp, value, (move_logits, msg_logits, entropy)

    def evaluate(self, obs: Tensor, a_m: Tensor, a_c: Tensor):
        move_logits, msg_logits, value = self.forward(obs)
        dist_m = Categorical(logits=move_logits)
        dist_c = Categorical(logits=msg_logits)
        logp = dist_m.log_prob(a_m) + dist_c.log_prob(a_c)
        entropy = dist_m.entropy() + dist_c.entropy()
        return logp, entropy, value, move_logits, msg_logits

# --------------------------- Storage ---------------------------

@dataclass
class RolloutStorage:
    num_steps: int
    num_envs: int
    obs_dim: int
    device: torch.device

    def __post_init__(self):
        T, N, D = self.num_steps, self.num_envs, self.obs_dim
        self.obs      = torch.zeros(T + 1, N, D, device=self.device)
        self.rewards  = torch.zeros(T, N, device=self.device)
        self.masks    = torch.ones(T + 1, N, device=self.device)
        self.a_move   = torch.zeros(T, N, dtype=torch.long, device=self.device)
        self.a_msg    = torch.zeros(T, N, dtype=torch.long, device=self.device)
        self.logp     = torch.zeros(T, N, device=self.device)
        self.values   = torch.zeros(T + 1, N, device=self.device)
        self.returns  = torch.zeros(T + 1, N, device=self.device)
        self.step = 0

    def insert(self, obs, a_m, a_c, logp, value, reward, mask):
        t = self.step
        self.obs[t+1].copy_(obs)
        self.a_move[t].copy_(a_m)
        self.a_msg[t].copy_(a_c)
        self.logp[t].copy_(logp)
        self.values[t].copy_(value)
        self.rewards[t].copy_(reward)
        self.masks[t+1].copy_(mask)
        self.step = t + 1

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.values[-1].zero_()
        self.step = 0

    def compute_returns(self, gamma: float, gae_lambda: float):
        T = self.num_steps
        gae = torch.zeros(self.num_envs, device=self.device)
        self.returns[-1].copy_(self.values[-1])  # bootstrap
        for t in reversed(range(T)):
            delta = self.rewards[t] + gamma * self.values[t+1] * self.masks[t+1] - self.values[t]
            gae = delta + gamma * gae_lambda * self.masks[t+1] * gae
            self.returns[t] = gae + self.values[t]

# --------------------------- Agent ---------------------------

class A2C:
    def __init__(self, obs_dim: int, num_envs: int, cfg: dict, device: torch.device):
        self.device = device
        self.gamma = float(cfg.get("gamma", 0.99))
        self.gae_lambda = float(cfg.get("gae_lambda", 0.95))
        self.entropy_coef = float(cfg.get("entropy_coef", 0.01))
        self.value_coef = float(cfg.get("value_coef", 0.5))
        self.lr = float(cfg.get("lr", 3e-4))
        self.max_grad_norm = float(cfg.get("max_grad_norm", 0.5))
        self.amp = bool(cfg.get("amp", True))

        self.snd_metric = cfg.get("snd_metric", "w2")  # 'w2' or 'js'
        self.snd_coef = float(cfg.get("snd_coef", 0.10))
        self.snd_alpha_msg = float(cfg.get("snd_alpha_msg", 0.25))
        self.snd_warmup_updates = int(cfg.get("snd_warmup_updates", 2000))

        self.seac_coef = float(cfg.get("seac_coef", 1.0))
        self.seac_clip = float(cfg.get("seac_clip", 5.0))
        self.seac_warmup_updates = int(cfg.get("seac_warmup_updates", 2000))

        self.entropy_anneal_to = float(cfg.get("entropy_anneal_to", 0.001))
        self.entropy_anneal_steps = int(cfg.get("entropy_anneal_steps", 20000))
        self.update_count = 0

        self.model = ActorCritic(obs_dim).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        self.storage = RolloutStorage(
            num_steps=int(cfg.get("num_steps", 5)),
            num_envs=num_envs, obs_dim=obs_dim, device=self.device
        )

        # W2 代价矩阵（移动头）
        self.C_move = build_move_cost(self.device)

    # ---- 动作 ----
    @torch.no_grad()
    def act(self, obs: Tensor):
        return self.model.act(obs)

    # ---- 训练一步 ----
    def update(self, all_agents: List["A2C"]) -> Dict[str, float]:
        self.storage.compute_returns(self.gamma, self.gae_lambda)

        T, N = self.storage.num_steps, self.storage.num_envs
        obs_flat = self.storage.obs[:-1].reshape(T*N, -1)
        a_m_flat = self.storage.a_move.reshape(T*N)
        a_c_flat = self.storage.a_msg.reshape(T*N)
        ret_flat = self.storage.returns[:-1].reshape(T*N)
        val_flat = self.storage.values[:-1].reshape(T*N)
        adv = ret_flat - val_flat
        # 优势归一化（每次 update）
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # 主策略项
        with torch.cuda.amp.autocast(enabled=self.amp):
            logp, entropy, value_pred, move_logits, msg_logits = self.model.evaluate(obs_flat, a_m_flat, a_c_flat)
            policy_loss = -(adv * logp).mean()
            value_loss = F.mse_loss(value_pred, ret_flat)
            entropy_loss = - entropy.mean()  # 惩罚项写成 loss 相加

        # ---- SEAC ----
        # 用本智能体的观测 obs_flat，对其他智能体的 (a_m,a_c) 计算对数似然，
        # 权重使用本智能体的 adv；外加温和裁剪与退火。
        seac_loss = torch.tensor(0., device=self.device)
        if self.seac_coef > 0 and len(all_agents) > 1:
            with torch.cuda.amp.autocast(enabled=self.amp):
                for ag in all_agents:
                    if ag is self:  # 跳过自己
                        continue
                    a_m_j = ag.storage.a_move.reshape(T*N).detach()
                    a_c_j = ag.storage.a_msg.reshape(T*N).detach()
                    logp_j, _, _, _, _ = self.model.evaluate(obs_flat, a_m_j, a_c_j)
                    # 比率裁剪（用 exp(logp_j - self.logp_i) 近似），防止偏移项过大
                    ratio = torch.exp(logp_j - logp.clamp(-20, 20)).clamp(0., self.seac_clip)
                    seac_term = -(ratio * adv.detach() * logp_j).mean()
                    seac_loss = seac_loss + seac_term
            seac_loss = seac_loss / max(1, len(all_agents)-1)
            # 退火：前 warmup 逐步从 0 → seac_coef
            warm = min(1.0, self.update_count / max(1, self.seac_warmup_updates))
            seac_loss = self.seac_coef * warm * seac_loss

        # ---- SND（W2 或 JS）----
        snd_loss = torch.tensor(0., device=self.device)
        if self.snd_coef > 0 and len(all_agents) > 1:
            with torch.no_grad():
                # 准备各 agent 的 softmax 概率（基于各自的 obs_flat）
                # 这里“对齐时间步”的做法与 SEAC 一致：同一批次 (T*N)。
                mv_probs, ms_probs = [], []
                for ag in all_agents:
                    mv_lg, ms_lg, _ = ag.model.forward(ag.storage.obs[:-1].reshape(T*N, -1))
                    mv_probs.append(F.softmax(mv_lg, dim=-1))
                    ms_probs.append(F.softmax(ms_lg, dim=-1))
            # 只给当前 agent 传梯度：其自身概率用 requires_grad 的前向再算一次
            mv_lg_i, ms_lg_i, _ = self.model.forward(self.storage.obs[:-1].reshape(T*N, -1))
            mv_probs[self._idx_in(all_agents)] = F.softmax(mv_lg_i, dim=-1)  # 替换本体
            ms_probs[self._idx_in(all_agents)] = F.softmax(ms_lg_i, dim=-1)

            # 计算 per-agent 的 SND 贡献
            snd_per_agent = snd_w2_per_agent(
                move_probs_list=mv_probs,
                msg_probs_list=ms_probs,
                C_move=self.C_move,
                alpha_msg=self.snd_alpha_msg,
                detach_others=True
            )
            # 我只取自己的那份
            snd_i = snd_per_agent[self._idx_in(all_agents)]
            # 退火：前 warmup 逐步从 0 → snd_coef，后期可保持或缓降（需可配）
            warm = min(1.0, self.update_count / max(1, self.snd_warmup_updates))
            snd_loss = -(self.snd_coef * warm) * snd_i  # maximize SND → minimize (−SND)

        # ---- 熵系数退火 ----
        if self.entropy_anneal_steps > 0:
            frac = min(1.0, self.update_count / self.entropy_anneal_steps)
            ent_coef = (1-frac) * self.entropy_coef + frac * self.entropy_anneal_to
        else:
            ent_coef = self.entropy_coef

        total_loss = policy_loss + self.value_coef * value_loss + ent_coef * entropy_loss + seac_loss + snd_loss

        self.optim.zero_grad(set_to_none=True)
        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optim)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optim)
        self.scaler.update()

        self.update_count += 1

        return {
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy":    float((-entropy_loss).item()),
            "seac_loss":  float(seac_loss.item()),
            "snd_loss":   float(snd_loss.item()),
            "ent_coef":   float(ent_coef),
        }

    def _idx_in(self, all_agents: List["A2C"]) -> int:
        for k, ag in enumerate(all_agents):
            if ag is self: return k
        return 0
