# snd_w2.py  --  Wasserstein-2 SND for discrete actions (RWARE)
from __future__ import annotations
import torch
import torch.nn.functional as F

@torch.no_grad()
def build_move_cost(device: torch.device) -> torch.Tensor:
    """5 移动动作的 2D 坐标并生成平方欧氏距离矩阵 C∈R^{5x5}."""
    # L,R,U,D,Stay  →  (-1,0),(1,0),(0,1),(0,-1),(0,0)
    xy = torch.tensor([[-1.,0.],[1.,0.],[0.,1.],[0.,-1.],[0.,0.]], device=device)
    # C_ab = ||x_a - x_b||^2
    diff = xy[:,None,:] - xy[None,:,:]
    C = (diff**2).sum(-1)  # [5,5]
    return C  # 不需要归一化，保持“米”尺度更贴近几何

def _sinkhorn_batch_cost(p: torch.Tensor, q: torch.Tensor, C: torch.Tensor,
                         eps: float=0.08, iters: int=20) -> torch.Tensor:
    """
    熵正则 Sinkhorn 近似 W2^2(p,q) 的批量实现。
    p,q: [B, K], C: [K, K] (非负代价矩阵).
    返回: [B] 对应每一批样本的近似 W2^2.
    """
    # 保障数值稳定
    p = (p + 1e-8); p = p / p.sum(-1, keepdim=True)
    q = (q + 1e-8); q = q / q.sum(-1, keepdim=True)

    B, K = p.shape
    # exp(-C/eps)
    Kmat = torch.exp(-C / eps).to(p.dtype)  # [K,K]
    u = torch.ones(B, K, device=p.device, dtype=p.dtype)
    v = torch.ones(B, K, device=p.device, dtype=p.dtype)

    # 迭代
    for _ in range(iters):
        Kv = torch.clamp_min(Kmat @ v.transpose(0,1), 1e-12).transpose(0,1)  # [B,K]
        u = p / Kv
        Ku = torch.clamp_min((Kmat.transpose(0,1) @ u.transpose(0,1)), 1e-12).transpose(0,1)
        v = q / Ku

    # 传输 T = diag(u) K diag(v)
    T = (u.unsqueeze(2) * Kmat) * v.unsqueeze(1)  # [B,K,K]
    cost = (T * C).sum(dim=(1,2))
    return cost

def bernoulli_w2_sq(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    二元分布在 0-1 线上，带平方欧氏距离的 W2^2 等于概率差的绝对值。
    p,q: [B,2] 概率分布；返回 [B].
    """
    p1 = p[..., 1]
    q1 = q[..., 1]
    return (p1 - q1).abs()

def snd_w2_per_agent(move_probs_list, msg_probs_list, C_move,
                     alpha_msg: float = 0.25,
                     detach_others: bool=True) -> list[torch.Tensor]:
    """
    对于 N 个智能体，给出每个智能体 i 的 SND 贡献（仅对自身求梯度）。
    move_probs_list: 长度 N，每个 [B,5]
    msg_probs_list:  长度 N，每个 [B,2]
    C_move: [5,5] 代价矩阵
    返回: 长度 N 的张量，每个为 [ ] 标量（已对 batch 求均值）
    """
    N = len(move_probs_list)
    outs = []
    for i in range(N):
        pi_m = move_probs_list[i]
        pi_c = msg_probs_list[i]
        acc = []
        for j in range(N):
            if j == i: continue
            pj_m = move_probs_list[j].detach() if detach_others else move_probs_list[j]
            pj_c = msg_probs_list[j].detach()  if detach_others else msg_probs_list[j]
            w2_move = _sinkhorn_batch_cost(pi_m, pj_m, C_move)  # [B]
            w2_msg  = bernoulli_w2_sq(pi_c, pj_c)               # [B]
            acc.append(w2_move + alpha_msg * w2_msg)
        if len(acc) == 0:
            outs.append(torch.zeros([], device=pi_m.device, dtype=pi_m.dtype))
        else:
            snd_i = torch.stack(acc, dim=0).mean(0).mean()  # pair均值 + batch均值
            outs.append(snd_i)
    return outs
