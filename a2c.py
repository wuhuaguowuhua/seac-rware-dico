#!/usr/bin/env python
# a2c.py -- minimal A2C scaffold that works with DiCoSNDPolicy.
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.distributions import Categorical
from sacred import Ingredient

algorithm = Ingredient("algorithm")

@algorithm.config
def _algo_cfg():
    num_processes = 1
    num_steps = 5
    gamma = 0.99
    tau = 1.00          # 未使用，保留占位
    lr = 7e-4
    eps = 1e-5
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5
    device = "cpu"


class RolloutStorage:
    def __init__(self, num_steps: int, num_envs: int, obs_dim: int, device: str = "cpu"):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.device = torch.device(device)

        self.obs = torch.zeros(num_steps + 1, num_envs, obs_dim, dtype=torch.float32, device=self.device)
        self.recurrent_hidden_states = torch.zeros(num_steps + 1, num_envs, 1, dtype=torch.float32, device=self.device)
        self.actions = torch.zeros(num_steps, num_envs, 1, dtype=torch.long, device=self.device)
        self.action_log_probs = torch.zeros(num_steps, num_envs, 1, dtype=torch.float32, device=self.device)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 1, dtype=torch.float32, device=self.device)
        self.returns = torch.zeros(num_steps + 1, num_envs, 1, dtype=torch.float32, device=self.device)
        self.rewards = torch.zeros(num_steps, num_envs, 1, dtype=torch.float32, device=self.device)
        self.masks = torch.ones(num_steps + 1, num_envs, 1, dtype=torch.float32, device=self.device)
        self.bad_masks = torch.ones(num_steps + 1, num_envs, 1, dtype=torch.float32, device=self.device)
        self.step = 0

    def to(self, device: str):
        dev = torch.device(device)
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                setattr(self, k, v.to(dev))
        self.device = dev

    def insert(
        self,
        obs: Tensor,
        recurrent_hidden_states: Tensor,
        actions: Tensor,
        action_log_probs: Tensor,
        value_preds: Tensor,
        rewards: Tensor,
        masks: Tensor,
        bad_masks: Tensor,
    ):
        s = self.step
        self.obs[s + 1].copy_(obs)
        self.recurrent_hidden_states[s + 1].copy_(recurrent_hidden_states)
        self.actions[s].copy_(actions)
        self.action_log_probs[s].copy_(action_log_probs)
        self.value_preds[s].copy_(value_preds)
        self.rewards[s].copy_(rewards)
        self.masks[s + 1].copy_(masks)
        self.bad_masks[s + 1].copy_(bad_masks)
        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.recurrent_hidden_states[0].copy_(self.recurrent_hidden_states[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.step = 0

    def compute_returns(self, gamma: float = 0.99):
        # A2C 版纯折扣回报（不做 GAE）
        self.returns[-1] = self.value_preds[-1]  # bootstrap 可写成 0；保持兼容性
        for s in reversed(range(self.num_steps)):
            self.returns[s] = (
                self.rewards[s]
                + gamma * self.returns[s + 1] * self.masks[s + 1] * self.bad_masks[s + 1]
            )


class A2C:
    def __init__(self, agent_id: int, obs_space, act_space, num_envs=1, num_steps=5, device="cpu", **_):
        self.agent_id = agent_id
        self.obs_dim = int(np.prod(obs_space.shape))
        self.act_dim = int(getattr(act_space, "n", 1))
        self.device = torch.device(device)

        # 超参在 attach_model 时会用
        self._hparams = {
            "num_envs": num_envs,
            "num_steps": num_steps,
            "gamma": 0.99,
            "lr": 7e-4,
            "eps": 1e-5,
            "value_loss_coef": 0.5,
            "entropy_coef": 0.01,
            "max_grad_norm": 0.5,
        }

        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.storage = RolloutStorage(num_steps, num_envs, self.obs_dim, device=device)

    # 显式挂接外部策略并初始化优化器
    def attach_model(self, model: nn.Module, lr: float = None, eps: float = None):
        self.model = model
        self.model.train()
        lr = self._hparams["lr"] if lr is None else lr
        eps = self._hparams["eps"] if eps is None else eps
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=eps)

    # 兼容旧用法：直接赋值 .model 后手动调用
    def reset_optimizer(self):
        assert self.model is not None
        self.attach_model(self.model)

    def compute_returns(self, gamma: float = 0.99):
        self.storage.compute_returns(gamma=gamma)

    def update(self, _all_storages: List[RolloutStorage]):
        """
        这里忽略其它智能体的 storage，做单体 A2C 更新。
        关键点：**重新前向**，不要用 rollout 里无梯度的 logp/value。
        """
        assert self.model is not None and self.optimizer is not None
        self.model.train()

        # ---- 准备 batch ----
        num_steps = self.storage.num_steps
        num_envs = self.storage.num_envs

        obs_batch = self.storage.obs[:-1].reshape(num_steps * num_envs, self.obs_dim)
        actions_batch = self.storage.actions.reshape(num_steps * num_envs, 1)
        returns_batch = self.storage.returns[:-1].reshape(num_steps * num_envs, 1)

        # ---- 重新前向，构建计算图 ----
        logits, values = self.model.forward(obs_batch)  # 需要 grad
        dist = Categorical(logits=logits)
        action_log_probs = dist.log_prob(actions_batch.squeeze(-1)).unsqueeze(-1)
        entropy = dist.entropy().mean()

        with torch.no_grad():
            advantages = returns_batch - values  # baseline 不反传

        policy_loss = -(advantages * action_log_probs).mean()
        value_loss = F.mse_loss(values, returns_batch)

        loss = (
            policy_loss
            + self._hparams["value_loss_coef"] * value_loss
            - self._hparams["entropy_coef"] * entropy
        )

        # ---- 反向 & step ----
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._hparams["max_grad_norm"])
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
        }

    # 兼容外部调用
    def save(self, path: str):
        assert self.model is not None
        torch.save({"model": self.model.state_dict()}, path)

    def load(self, path: str):
        assert self.model is not None
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model"])

    # 让外部可以无害调用
    def step_scheduler(self):
        sched = getattr(self.model, "scheduler", None)
        if sched is not None:
            try:
                sched.step()
            except Exception:
                pass
