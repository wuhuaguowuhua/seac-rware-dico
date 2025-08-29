# train_new.py  -- SEAC + SND(W2/JS) 训练脚本（Sacred + 可选 W&B）
from __future__ import annotations
import os, random, time
from typing import List, Dict

import numpy as np
import torch
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds

from envs import make_vec_envs
from a2c import A2C

# ---- 可选 W&B（未安装也不影响运行）----
try:
    import wandb
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False

ex = Experiment("seac-rware")
ex.captured_out_filter = apply_backspaces_and_linefeeds

# -------------------- 默认配置 --------------------
@ex.config
def cfg():
    env_name = "rware-tiny-4ag-easy-v1"
    seed = 0
    dummy_vecenv = True
    log_interval = 500
    save_interval = 10000
    eval_interval = 0
    results_dir = "./results"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 这里加入 num_env_steps，避免 Sacred 报“新增未使用”
    num_env_steps = 400000

    # W&B（默认关闭；按需在命令行打开 use_wandb=True）
    use_wandb = False
    wandb_project = "seac-rware"
    wandb_run_name = None

    algorithm = dict(
        num_processes = 1,
        num_steps = 5,
        gamma = 0.99,
        gae_lambda = 0.95,
        lr = 3e-4,
        value_coef = 0.5,
        entropy_coef = 0.01,
        entropy_anneal_to = 0.001,
        entropy_anneal_steps = 20000,
        max_grad_norm = 0.5,
        amp = True,

        # SND
        snd_metric = "w2",    # {'w2','js'}
        snd_coef = 0.10,
        snd_alpha_msg = 0.25,
        snd_warmup_updates = 2000,

        # SEAC
        seac_coef = 1.0,
        seac_clip = 5.0,
        seac_warmup_updates = 2000,

        # （可选）若你坚持传 algorithm.device=cuda，也支持
        # device = None,
    )

# -------------------- 小工具 --------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# -------------------- 主函数 --------------------
@ex.automain
def main(_config):
    cfg = _config
    set_seed(int(cfg["seed"]))

    # 允许 algorithm.device 覆盖顶层 device（两种写法都生效）
    alg_dev = cfg["algorithm"].get("device", None)
    dev_name = alg_dev if alg_dev is not None else cfg.get("device", "cpu")
    device = torch.device(dev_name)

    # ----- W&B 可选启用 -----
    use_wandb = bool(cfg.get("use_wandb", False))
    if use_wandb and not _WANDB_AVAILABLE:
        print("[warn] wandb 未安装，已自动关闭（pip install wandb 即可启用）。")
        use_wandb = False
    if use_wandb:
        run_name = cfg.get("wandb_run_name") or f"{cfg['env_name']}-{time.strftime('%m%d-%H%M')}"
        wandb.init(project=cfg.get("wandb_project","seac-rware"),
                   name=run_name, config=cfg)

    # 1) 环境
    envs = make_vec_envs(
    cfg["env_name"],                          # env_name (str)
    int(cfg["seed"]),                         # seed
    bool(cfg.get("dummy_vecenv", True)),      # dummy_vecenv
    int(cfg["algorithm"]["num_processes"]),   # num_processes
    None,                                     # video_dir/time_limit（你这版是 None）
    (),                                       # frame_stack / wrappers
    "cpu",                                    # device（底层 env 用 CPU，模型 device=cfg['device']）
    )
    obs = envs.reset()  # 约定: list[n_agents] of Tensor(n_envs, feat)
    # 直接从 obs 推断智能体数量，避免访问 env 属性
    n_agents = len(obs) if isinstance(obs, (list, tuple)) else 1
    n_envs   = envs.num_envs
    feat_dim = obs[0].shape[-1]

    # 2) 智能体
    agents: List[A2C] = []
    for i in range(n_agents):
        ag = A2C(obs_dim=feat_dim, num_envs=n_envs, cfg=cfg["algorithm"], device=device)
        ag.storage.obs[0].copy_(obs[i].to(device))
        agents.append(ag)

    # 3) 训练循环
    steps_per_update = n_envs * int(cfg["algorithm"]["num_steps"])
    total_env_steps  = int(cfg.get("num_env_steps", 400000))
    num_updates = max(1, total_env_steps // steps_per_update)

    t0 = time.time()
    for j in range(1, num_updates + 1):
        # --- 收集 rollout ---
        for step in range(cfg["algorithm"]["num_steps"]):
            actions = []
            for i, ag in enumerate(agents):
                with torch.no_grad():
                    a_m, a_c, logp, value, _ = ag.act(ag.storage.obs[step].to(device))
                a_np = torch.stack([a_m, a_c], dim=1).cpu().numpy().astype(np.int64)
                actions.append(a_np)

            next_obs, reward, done, info = envs.step(actions)

            # 规整奖励/掩码
            if isinstance(reward, list):
                rew = torch.as_tensor(np.asarray(reward), dtype=torch.float32)
            else:
                rew = torch.as_tensor(reward, dtype=torch.float32)
            if isinstance(done, (list, tuple, np.ndarray)):
                m = 1.0 - torch.as_tensor(np.asarray(done), dtype=torch.float32)
            else:
                m = 1.0 - torch.as_tensor(done, dtype=torch.float32)

            for i, ag in enumerate(agents):
                ag.storage.insert(
                    obs=next_obs[i].to(device),
                    a_m=torch.as_tensor(actions[i][:,0], device=device),
                    a_c=torch.as_tensor(actions[i][:,1], device=device),
                    logp=torch.zeros(n_envs, device=device),
                    value=torch.zeros(n_envs, device=device),
                    reward=rew[:, i] if rew.ndim == 2 else rew,  # 兼容 list/ndarray
                    mask=m,
                )

        # --- 更新 ---
        per_agent_logs: List[Dict[str, float]] = []
        for ag in agents:
            logs = ag.update(agents)
            per_agent_logs.append(logs)
            ag.storage.after_update()

        # --- 日志 ---
        steps_done = j * steps_per_update
        dt = time.time() - t0
        fps = steps_done / max(1e-6, dt)

        meanR = 0.0
        if isinstance(info, dict) and "episode_reward" in info:
            ep = info["episode_reward"]
            if isinstance(ep, (list, np.ndarray)):
                meanR = float(np.asarray(ep).mean())
            elif torch.is_tensor(ep):
                meanR = float(ep.float().mean().item())
            else:
                meanR = float(ep)

        if j % int(cfg["log_interval"]) == 0:
            pol = np.mean([l["policy_loss"] for l in per_agent_logs])
            val = np.mean([l["value_loss"]  for l in per_agent_logs])
            ent = np.mean([l["entropy"]     for l in per_agent_logs])
            seac= np.mean([l["seac_loss"]   for l in per_agent_logs])
            snd = np.mean([l["snd_loss"]    for l in per_agent_logs])
            print(f"(Upd {j:>5d}) steps {steps_done:>7d} | {fps:4.0f} FPS | meanR {meanR:6.3f} "
                  f"| pol {pol:+.4f} | val {val:+.4f} | ent {ent:.3f} | seac {seac:+.4f} | snd {snd:+.4f}")

            if use_wandb:
                metrics = {
                    "global_step": steps_done,
                    "fps": fps,
                    "mean_reward": meanR,
                    "policy_loss": pol,
                    "value_loss":  val,
                    "entropy":     ent,
                    "seac_loss":   seac,
                    "snd_loss":    snd,
                }
                # 逐 agent
                for idx, l in enumerate(per_agent_logs):
                    for k, v in l.items():
                        metrics[f"agent{idx}/{k}"] = v
                wandb.log(metrics)

    envs.close()
    if use_wandb:
        wandb.finish()
