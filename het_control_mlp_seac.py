# ===================================================================================
# het_control_mlp_seac.py  —— 将 HetControlMlpEmpirical 改造为 RWARE + SEAC 可用版本
# ===================================================================================

from __future__ import annotations
from typing import Sequence, Optional, Type

import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import NormalParamExtractor
from torchrl.modules import MultiAgentMLP

from benchmarl.models.common import Model, ModelConfig
from het_control.snd import compute_behavioral_distance
from het_control.utils import overflowing_logits_norm
from .utils import squash

# -----------------------------------------------------------------------------------
# 1. 改造后的策略网络（脱离 BenchmarL 原有 pipeline，直接面向 RWARE + SEAC）
# -----------------------------------------------------------------------------------
class HetControlMlpForSEAC(Model):
    """
    改造自 HetControlMlpEmpirical（:contentReference[oaicite:1]{index=1}），用于：
      1) RWARE 环境（离散动作）
      2) SEAC 算法（所有 agent 共享 & 执行 SND 缩放）
    """

    def __init__(
        self,
        activation_class: Type[nn.Module],
        num_cells: Sequence[int],
        desired_snd: float,
        tau: float,
        bootstrap_from_desired_snd: bool,
        process_shared: bool,
        n_agents: int,
        obs_dim: int,
        n_actions: int,
        **kwargs,
    ):
        """
        Args:
            activation_class       : 激活函数类型（如 nn.ReLU, nn.Tanh 等）。
            num_cells              : 隐藏层单元数列表。例如 [128,128] 或单个 int。
            desired_snd            : 目标多样性 SND_des。
            tau                    : 软更新因子 τ ∈ [0,1]。
            bootstrap_from_desired_snd : 首次迭代时，是否用 desired_snd 来初始化 \widehat{SND}。
            process_shared         : 是否对 shared 部分输出做 squash（本例中一般不需要）。
            n_agents               : 智能体数量。
            obs_dim                : 每个智能体的观测维度（RWARE 下的展平成一维）。
            n_actions              : 离散动作空间大小（RWARE 下每个 agent 的动作个数）。
            **kwargs               : 其他预留参数（兼容原 BenchmarL 构造）。
        """
        super().__init__(**kwargs)

        # ——————————————————————————————————————————
        # 1) 保存配置，注册 buffer
        # ——————————————————————————————————————————
        self.n_agents = n_agents
        self.activation_class = activation_class
        self.num_cells = num_cells
        self.process_shared = process_shared

        # desired_snd 和 estimated_snd 都注册为 buffer，以便入 GPU/CPU
        self.register_buffer(
            "desired_snd",
            torch.tensor([desired_snd], dtype=torch.float, device=self.device),
        )
        # \widehat{SND} 一开始如果用 desired_snd，设定为 nan 或直接赋值 desired_snd
        init_snd = desired_snd if bootstrap_from_desired_snd else float("nan")
        self.register_buffer(
            "estimated_snd",
            torch.tensor([init_snd], dtype=torch.float, device=self.device),
        )
        self.tau = tau
        self.bootstrap_from_desired_snd = bootstrap_from_desired_snd

        # 输出维度： 离散动作只需 logits，agent_out_features = n_actions
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        agent_output_features = n_actions  # 对应每个 agent 的“偏差logits”输出

        # ——————————————————————————————————————————
        # 2) 构建共享网络 & 偏差网络（MultiAgentMLP）
        # ——————————————————————————————————————————
        # Shared MLP: 输出 n_agents * n_actions 的“共同 logits”（没有分 agent）
        # 这里 share_params=True，表示所有 agent 用同一组权重 -> π_h 部分
        self.shared_mlp = MultiAgentMLP(
            n_agent_inputs=self.obs_dim,
            n_agent_outputs=self.n_actions,
            n_agents=self.n_agents,
            centralised=False,
            share_params=True,
            device=self.device,
            activation_class=self.activation_class,
            num_cells=self.num_cells,
        )
        # Agent-specific MLP: 输出 n_agents * agent_output_features 的“偏差 logits” -> π_{h,i} 部分
        self.agent_mlps = MultiAgentMLP(
            n_agent_inputs=self.obs_dim,
            n_agent_outputs=agent_output_features,
            n_agents=self.n_agents,
            centralised=False,
            share_params=False,  # 每个 agent 自己的一套权重 -> π_{h,i}
            device=self.device,
            activation_class=self.activation_class,
            num_cells=self.num_cells,
        )

    def _perform_checks(self):
        super()._perform_checks()
        # 本例不需要额外检查，假设输入 tensordict 上的形状总是 [B, n_agents, obs_dim]

    def forward(
        self,
        tensordict: TensorDictBase,
        agent_index: int | None = None,
        update_estimate: bool = True,
        compute_estimate: bool = True,
    ) -> TensorDictBase:
        """
        取出观测，将它送进 shared & agent-specific MLP，负责：
          1) 计算 \widehat{SND}（在训练阶段）
          2) 计算 scaling_ratio = desired_snd / \widehat{SND}
          3) 最终输出每个 agent 的 logits（多样性已缩放）
        tensordict 中预计至少包含：
          tensordict["obs"] 的形状: [B, n_agents, obs_dim]
        返回时，tensordict 会写入：
          - (agent_group, "logits"): [B, n_agents, n_actions]
          - (agent_group, "estimated_snd"): [B, n_agents, 1]（每个 agent 相同，占位）
          - (agent_group, "scaling_ratio"): [B, n_agents, 1]
        """
        # ——————————————————————————————————————————
        # 1) 拿到 obs，形状 [B, n_agents, obs_dim]
        # ——————————————————————————————————————————
        obs = tensordict.get(self.in_key)  # 比如 in_key="obs" ，保证 in_key 已在父类 ModelConfig 中设定
        # 确保维度正确
        # (假设 tensordict 的 obs 已经过 flatten、无其他维度：[B, n_agents, obs_dim])
        B, n, d_dim = obs.shape
        assert (
            n == self.n_agents and d_dim == self.obs_dim
        ), f"Obs shape mismatch: (got {obs.shape}, expected [B, {self.n_agents}, {self.obs_dim}])"

        # ——————————————————————————————————————————
        # 2) shared 部分 + agent-specific 部分
        # ——————————————————————————————————————————
        # shared_out: [B, n_agents, n_actions]
        shared_out = self.shared_mlp(obs)
        # agent_out: [B, n_agents, n_actions] （直接理解为“偏差 logits”）
        agent_out = self.agent_mlps(obs)

        # 如果需要对 shared 输出做 squash（离散动作一般不需要，但保留接口）
        shared_out = self.process_shared_out(shared_out)

        # ——————————————————————————————————————————
        # 3) 先计算 \widehat{SND}（只在训练阶段，且 desired_snd>0 且 agent 数 >1 时）
        # ——————————————————————————————————————————
        if (
            self.desired_snd.item() > 0
            and torch.is_grad_enabled()
            and compute_estimate
            and self.n_agents > 1
        ):
            # compute_behavioral_distance 会对每个 agent 用当前 agent_mlps 输出的“偏差 logits” 
            # 进行 Wasserstein 距离计算（just_mean=True 表示仅用均值）
            # 注意：compute_behavioral_distance 要求输入是一个 list，内部对每个元素都执行 .mean() 得到单个值
            dev_logits_list = []
            # 逐 agent 拿出“偏差”，形状 [B, n_actions]
            for i, agent_net in enumerate(self.agent_mlps.agent_networks):
                dev_logits_list.append(agent_net(obs) )  # [B, n_actions]
            # distance: 单个 scalar（batch 内先 .mean()，然后再 .unsqueeze(-1)）
            distance = (
                compute_behavioral_distance(agent_actions=dev_logits_list, just_mean=True)
                .mean()
                .unsqueeze(-1)
            )  # [B, 1] 的 tensor
            # 如果是第一次更新，通过 bootstrap 决定是用 desired_snd 还是当前测得的 distance
            if self.estimated_snd.isnan().any():
                distance = self.desired_snd if self.bootstrap_from_desired_snd else distance

            if update_estimate:
                # 软更新 \widehat{SND} ← (1−τ) * \widehat{SND} + τ * distance
                new_hat = (1 - self.tau) * self.estimated_snd + self.tau * distance
                self.estimated_snd[:] = new_hat.detach()
        else:
            # 如果不更新，就直接用上一次保存的 \widehat{SND}
            distance = self.estimated_snd

        # ——————————————————————————————————————————
        # 4) 根据 distance 计算 scaling_ratio
        # ——————————————————————————————————————————
        if self.desired_snd.item() == 0:
            scaling_ratio = torch.zeros_like(distance)  # 全部树立为0，偏差直接被抹去
        elif (
            self.desired_snd.item() < 0
            or distance.isnan().any()
            or self.n_agents == 1
        ):
            scaling_ratio = torch.ones_like(distance)  # 不做约束（按原始共享+偏差输出）
        else:
            # 正常 DiCo： scaling_ratio = desired_snd / distance
            scaling_ratio = self.desired_snd / distance  # [B,1]
        
        # 把 scaling_ratio 扩展到 [B, n_agents, 1]
        scaling_ratio_expanded = scaling_ratio.unsqueeze(1).expand(-1, self.n_agents, 1)  # [B, n, 1]

        # ——————————————————————————————————————————
        # 5) 生成最终 logits： shared_out + scaling_ratio * agent_out
        #    然后存到 tensordict 中
        # ——————————————————————————————————————————
        # shared_out, agent_out 都是 [B, n_agents, n_actions]
        # scaling_ratio_expanded: [B, n_agents, 1] -> 自动广播到 [B, n_agents, n_actions]
        final_logits = shared_out + scaling_ratio_expanded * agent_out  # [B, n_agents, n_actions]
        # 对 logits 做一次 overflow 归一化/剪切，仅用于监控，不影响最终 softmax
        final_loc_norm = overflowing_logits_norm(final_logits, 
                                                 self.action_spec[self.agent_group, "action"])

        # 写入 tensordict
        tensordict.set(
            (self.agent_group, "logits"), 
            final_logits
        )  # [B, n_agents, n_actions]
        # 将 \widehat{SND} 写入每个 agent 的 MdP, 形状 [B, n_agents, 1]
        tensordict.set(
            (self.agent_group, "estimated_snd"), 
            self.estimated_snd.unsqueeze(0).unsqueeze(0).expand(B, self.n_agents, 1)
        )
        # 将 scaling_ratio 写入，形状同上
        tensordict.set(
            (self.agent_group, "scaling_ratio"), 
            scaling_ratio_expanded
        )
        # 记录一个监控值：norm 后的 logits
        tensordict.set(
            (self.agent_group, "out_loc_norm"), 
            final_loc_norm
        )

        # 最后返回给 SEAC 使用
        return tensordict

    def process_shared_out(self, logits: torch.Tensor) -> torch.Tensor:
        """
        如果 process_shared=True，可以对 shared_out 先做一个 squash，
        将输出约束到 RWARE 离散动作对应的近似实值范围。本示例中一般设 process_shared=False。
        """
        if self.process_shared:
            return squash(
                logits, 
                action_spec=self.action_spec[self.agent_group, "action"], 
                clamp=False,
            )
        else:
            return logits


@dataclass
class HetControlMlpForSEACConfig(ModelConfig):
    activation_class: Type[nn.Module] = MISSING
    num_cells: Sequence[int] = MISSING
    desired_snd: float = MISSING
    tau: float = MISSING
    bootstrap_from_desired_snd: bool = MISSING
    process_shared: bool = MISSING
    n_agents: int = MISSING
    obs_dim: int = MISSING
    n_actions: int = MISSING

    @staticmethod
    def associated_class():
        return HetControlMlpForSEAC
