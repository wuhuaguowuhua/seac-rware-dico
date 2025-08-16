# a2c.py — A2C / SEAC (with SND) 针对旧式 RolloutStorage(obs_space, action_space, rhs, num_steps, num_processes) 适配版
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from sacred import Ingredient

from storage import RolloutStorage  # 你的项目里已有

# ============== Sacred ingredient: algorithm 超参 ==============
algorithm = Ingredient("algorithm")

@algorithm.config
def _algo_cfg():
    # 采样与设备
    num_processes = 1
    num_steps = 5
    device = "cpu"

    # 学习率与优化
    lr = 3e-4
    eps = 1e-5
    max_grad_norm = 0.5

    # A2C 损失系数
    gamma = 0.99
    gae_lambda = 0.95
    entropy_coef = 0.01
    value_loss_coef = 0.5

    # SEAC / SND
    seac_coef = 1.0   # 共享经验项权重
    snd_coef  = 0.10  # SND 正则权重（可调）

# ========================= A2C 封装 ============================
class A2C:
    def __init__(
        self,
        agent_id: int,
        obs_space,
        act_space,
        num_envs: int,
        num_steps: int,
        device: str = "cpu",
        lr: float = 3e-4,
        eps: float = 1e-5,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
        max_grad_norm: float = 0.5,
    ):
        self.agent_id = agent_id
        self.device = torch.device(device)

        # 保存 Gym space 本体，便于构造旧式 RolloutStorage
        self._obs_space = obs_space
        self._act_space = act_space

        self.obs_dim = int(np.prod(obs_space.shape))
        self.act_dim = int(getattr(act_space, "n", 1))

        # 损失/优化超参
        self.lr = lr
        self.eps = eps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm

        # rollout / 模型 / 优化器
        self.storage = None  # type: ignore
        self.model = None    # attach_model 后可用
        self.optimizer = None

        self._num_envs = int(num_envs)
        self._num_steps = int(num_steps)

    # -------- 在创建好策略网络后调用 ----------
    def attach_model(self, model: torch.nn.Module):
        """挂载策略/价值网络，并创建 optimizer 与 RolloutStorage（按你工程的旧式签名构造）。"""
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, eps=self.eps)

        rhss = int(getattr(self.model, "recurrent_hidden_state_size", 1))

        # 你的 RolloutStorage 签名：
        # __init__(obs_space, action_space, recurrent_hidden_state_size, num_steps, num_processes)
        self.storage = RolloutStorage(
            self._obs_space,
            self._act_space,
            rhss,
            self._num_steps,
            self._num_envs,
        )

    # --------- GAE/Returns（用当前模型估值末状态） ----------
    @torch.no_grad()
    def compute_returns(self):
        assert self.model is not None and self.storage is not None
        N = self._num_envs
        obs_last = self.storage.obs[-1].view(-1, self.obs_dim)  # (n_envs, obs_dim)
        _, value_last = self.model.forward(obs_last)  # type: ignore
        value_last = value_last.view(N, 1)

        # 你的 RolloutStorage 版本需要: (next_value, use_gae, gamma, gae_lambda)
        # 我们优先尝试这个签名；若失败，再逐级回退，最大化兼容不同老版本。
        try:
            # 首选：位置参数形式
            self.storage.compute_returns(value_last, True, self.gamma, self.gae_lambda)
            return
        except TypeError:
            pass

        try:
            # 其次：关键字形式
            self.storage.compute_returns(
                next_value=value_last, use_gae=True, gamma=self.gamma, gae_lambda=self.gae_lambda
            )
            return
        except TypeError:
            pass

        try:
            # 再次退回：老式 (next_value, gamma, gae_lambda)
            self.storage.compute_returns(value_last, self.gamma, self.gae_lambda)
            return
        except TypeError:
            pass

        try:
            # 最老的实现：只接受 next_value
            self.storage.compute_returns(value_last)
            return
        except TypeError:
            # 如果还不行，就明确报错，提醒贴出 compute_returns 的源码
            raise TypeError(
                "Unsupported RolloutStorage.compute_returns signature. "
                "Please share its definition so we can adapt."
            )

    # --------- 核心：一次参数更新（支持 SEAC + SND） ----------
    def update(
        self,
        storages_all_agents,
        snd_logits_batch=None,
        snd_coef: float = 0.0,
        seac_coef: float = 1.0,
    ):
        """
        storages_all_agents: list[RolloutStorage]，包含所有智能体的 rollout
        snd_logits_batch:    list[Tensor]，每个元素是“某个智能体在自己的 rollout 上 forward 得到的 logits”，
                             只作为 SND 参照；**请在外部对非本体 logits 调用 .detach()**。
        """
        assert self.model is not None and self.optimizer is not None and self.storage is not None
        device = next(self.model.parameters()).device

        # ---- 本体 on-policy 数据 ----
        T, N = self._num_steps, self._num_envs
        obs      = self.storage.obs[:-1].reshape(T * N, self.obs_dim).to(device)
        actions  = self.storage.actions.reshape(T * N, 1).to(device).long()
        returns  = self.storage.returns[:-1].reshape(T * N, 1).to(device)

        logits, values = self.model.forward(obs)  # (TN, act_dim), (TN, 1)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)  # (TN,1)
        entropy = dist.entropy().mean()

        advantages = (returns - values).detach()
        policy_loss = -(log_prob * advantages).mean()
        value_loss  = F.mse_loss(values, returns)

        # ---- SEAC: 用“他人”rollout 帮我更新（重要性采样校正）----
        seac_loss = torch.zeros((), device=device)
        others = [s for s in storages_all_agents if s is not self.storage]
        for other in others:
            o_obs     = other.obs[:-1].reshape(T * N, self.obs_dim).to(device)
            o_actions = other.actions.reshape(T * N, 1).to(device).long()
            o_returns = other.returns[:-1].reshape(T * N, 1).to(device)

            # 当前 agent 在“他人”的数据上评估
            o_logits, o_values = self.model.forward(o_obs)  # (TN,a), (TN,1)
            o_dist   = Categorical(logits=o_logits)
            log_pi_i = o_dist.log_prob(o_actions.squeeze(-1)).unsqueeze(-1)
            # 他人在采样时的 log pi（缓冲区已存）
            log_pi_j = other.action_log_probs.reshape(T * N, 1).to(device).detach()
            rho = torch.exp(log_pi_i - log_pi_j)  # importance ratio（只对本体开梯度）

            o_adv = (o_returns - o_values).detach()
            seac_loss = seac_loss - (rho * log_pi_i * o_adv).mean()

        if len(others) > 0:
            seac_loss = seac_loss / len(others)

        # ---- SND 正则（真正加入 loss）----
        snd_loss = torch.zeros((), device=device)
        if snd_logits_batch is not None:
            snd_loss = self.model.snd_regularisation(snd_logits_batch)  # type: ignore

        # ---- 总损失 & 更新 ----
        loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy
            + seac_coef * seac_loss
            + snd_coef  * snd_loss
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss":        float(loss.detach().cpu().item()),
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "value_loss":  float(value_loss.detach().cpu().item()),
            "entropy":     float(entropy.detach().cpu().item()),
            "seac_loss":   float(seac_loss.detach().cpu().item()),
            "snd_loss":    float(snd_loss.detach().cpu().item()),
        }

    # --------- 便捷的存取接口（与 train_new.py 配合） ----------
    def save(self, file_path: str):
        assert self.model is not None and self.optimizer is not None
        torch.save(
            {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict()},
            file_path,
        )

    def load(self, file_path: str):
        assert self.model is not None
        state = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(state["model"], strict=False)
        if "optimizer" in state and self.optimizer is not None:
            self.optimizer.load_state_dict(state["optimizer"])
