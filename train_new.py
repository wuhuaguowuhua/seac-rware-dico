# train_new.py — SEAC + SND + W&B(TensorBoard sync)
import glob
import logging
import os
import shutil
import time
from collections import deque
from os import path
from pathlib import Path

import numpy as np
import torch
import gym
from gym import spaces
from sacred import Experiment
from sacred.observers import FileStorageObserver
from torch.utils.tensorboard import SummaryWriter

import utils
from a2c import A2C, algorithm
from envs import make_vec_envs
from wrappers import RecordEpisodeStatistics, SquashDones
from het_control.het_control_mlp_empirical import DiCoSNDPolicy

import rware      # noqa: 注册环境
import lbforaging # noqa: 注册环境

# ---------------------- Sacred & logging -------------------------
ex = Experiment("seac-rware", ingredients=[algorithm], save_git_info=False)
ex.captured_out_filter = lambda captured_output: "Output capturing turned off."
ex.observers.append(FileStorageObserver("./results/sacred"))

logging.basicConfig(
    level=logging.INFO,
    format="(%(process)d) [%(levelname).1s] - (%(asctime)s) - %(name)s >> %(message)s",
    datefmt="%m/%d %H:%M:%S",
)

@ex.config
def _config():
    env_name = None
    time_limit = None
    wrappers = (RecordEpisodeStatistics, SquashDones)
    dummy_vecenv = False

    num_env_steps = 100e6  # 可在命令行覆盖

    eval_dir = "./results/video/{id}"
    loss_dir = "./results/loss/{id}"
    save_dir = "./results/trained_models/{id}"

    log_interval = 2000
    save_interval = int(1e6)
    eval_interval = int(1e6)
    episodes_per_eval = 8

    # ====== 新增：W&B 开关与项目名 ======
    use_wandb = True
    wandb_project = "seac-rware-dico"

# 载入额外 YAML 配置（如存在）
for conf in glob.glob("configs/*.yaml"):
    ex.add_named_config(Path(conf).stem, conf)

def _squash_info(info):
    info = [i for i in info if i]
    keys = {k for d in info for k in d.keys() if k != "TimeLimit.truncated"}
    return {k: np.mean([np.array(d[k]).sum() for d in info if k in d]) for k in keys}

# ------------------------ 评估（可被 eval_interval 触发） ------------------------
@ex.capture
def evaluate(
    agents,
    monitor_dir,
    episodes_per_eval,
    env_name,
    seed,
    wrappers,
    dummy_vecenv,
    time_limit,
    algorithm,
    _log,
):
    device = algorithm["device"]

    eval_envs = make_vec_envs(
        env_name,
        seed,
        dummy_vecenv,
        episodes_per_eval,
        time_limit,
        wrappers,
        device,
        monitor_dir=monitor_dir,
    )

    n_obs = eval_envs.reset()  # list[n_agents] of (episodes_per_eval, feat)
    n_recurrent_hidden_states = [
        torch.zeros(episodes_per_eval, a.model.recurrent_hidden_state_size, device=device)
        for a in agents
    ]
    masks = torch.zeros(episodes_per_eval, 1, device=device)

    all_infos = []
    while len(all_infos) < episodes_per_eval:
        with torch.no_grad():
            _, n_action, _, n_recurrent_hidden_states = zip(
                *[
                    a.model.act(n_obs[a.agent_id], h, masks)
                    for a, h in zip(agents, n_recurrent_hidden_states)
                ]
            )
        n_obs, _, done, infos = eval_envs.step(n_action)
        all_infos.extend([i for i in infos if i])

    eval_envs.close()
    s = _squash_info(all_infos)
    if s:
        _log.info(f"Eval mean reward {s.get('episode_reward', 0.0):.4f}")

# ------------------------------- main -------------------------------
@ex.automain
def main(
    _run,
    _log,
    num_env_steps,
    env_name,
    seed,
    algorithm,
    dummy_vecenv,
    time_limit,
    wrappers,
    save_dir,
    eval_dir,
    loss_dir,
    log_interval,
    save_interval,
    eval_interval,
    use_wandb,
    wandb_project,
):
    # -------- 先解析路径（W&B 需要 TB 的目录） --------
    loss_dir_resolved = path.expanduser(loss_dir.format(id=_run._id)) if loss_dir else None
    eval_dir = path.expanduser(eval_dir.format(id=_run._id))
    save_dir = path.expanduser(save_dir.format(id=_run._id))
    utils.cleanup_log_dir(eval_dir)
    utils.cleanup_log_dir(save_dir)

    # -------- [新增] W&B 初始化（必须在创建 SummaryWriter 之前） --------
    if use_wandb:
        try:
            import wandb  # 懒加载，避免没装包时报错
            cfg = dict(algorithm)
            cfg.update({
                "env_name": env_name,
                "seed": seed,
                "num_env_steps": num_env_steps,
                "log_interval": log_interval,
                "save_interval": save_interval,
                "eval_interval": eval_interval,
            })
            wandb.init(
                project=wandb_project,
                name=f"{env_name}-seed{seed}-id{_run._id}",
                config=cfg,
                sync_tensorboard=True,  # 同步 TB 到 W&B
            )
            if loss_dir_resolved:  # 可选：确保目录存在
                os.makedirs(loss_dir_resolved, exist_ok=True)
        except Exception as e:
            _log.warning(f"W&B 初始化失败（将继续本地 TensorBoard）：{e}")
            use_wandb = False

    # -------- 创建 TensorBoard writer（W&B 若开启会自动同步） --------
    writer = SummaryWriter(loss_dir_resolved) if loss_dir_resolved else None

    torch.set_num_threads(1)

    # ---------- 创建向量环境 ----------
    envs = make_vec_envs(
        env_name,
        seed,
        dummy_vecenv,
        algorithm["num_processes"],
        time_limit,
        wrappers,
        algorithm["device"],
    )

    # 观测形状：(n_agents, feat_dim)
    obs_shape = envs.observation_space.shape
    n_agents  = int(obs_shape[0])
    feat_dim  = int(obs_shape[1])

    # 按智能体的 obs/act 空间
    obs_spaces = [
        spaces.Box(low=-np.inf, high=np.inf, shape=(feat_dim,), dtype=np.float32)
        for _ in range(n_agents)
    ]
    act_spaces = envs.action_space
    assert isinstance(act_spaces, (list, tuple)) and len(act_spaces) == n_agents, \
        "env.action_space 必须是长度为 n_agents 的列表"

    act_dims = np.asarray([int(getattr(a, "n", 1)) for a in act_spaces], dtype=np.int64)

    # ---------- 构造 A2C + DiCoSNDPolicy ----------
    agents = []
    for i in range(n_agents):
        a = A2C(
            i, obs_spaces[i], act_spaces[i],
            num_envs=algorithm["num_processes"],
            num_steps=algorithm["num_steps"],
            device=algorithm["device"],
            lr=algorithm["lr"],
            eps=algorithm["eps"],
            gamma=algorithm["gamma"],
            gae_lambda=algorithm["gae_lambda"],
            entropy_coef=algorithm["entropy_coef"],
            value_loss_coef=algorithm["value_loss_coef"],
            max_grad_norm=algorithm["max_grad_norm"],
        )
        m = DiCoSNDPolicy(
            obs_dim=feat_dim,
            act_dim=int(act_dims[i]),
            num_agents=n_agents,
            hidden_dim=128,
            desired_snd=1.0,
            snd_warmup_steps=5_000,
        ).to(algorithm["device"])
        a.attach_model(m)
        agents.append(a)

    # ---------- 初始化 rollout ----------
    obs = envs.reset()  # list[n_agents] of (n_envs, feat)
    for i in range(n_agents):
        agents[i].storage.obs[0].copy_(obs[i])   # 放入起始 obs
        agents[i].storage.to(algorithm["device"])

    start = time.time()
    num_updates = int(num_env_steps) // algorithm["num_steps"] // algorithm["num_processes"]
    all_infos = deque(maxlen=10)

    # ============================ 训练主循环 ============================
    for j in range(1, num_updates + 1):
        # ------- 采样 num_steps -------
        for step in range(algorithm["num_steps"]):
            with torch.no_grad():
                outs = [
                    a.model.act(  # type: ignore
                        a.storage.obs[step],
                        a.storage.recurrent_hidden_states[step],
                        a.storage.masks[step],
                    ) for a in agents
                ]
                n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(*outs)

            next_obs, reward, done, infos = envs.step(n_action)
            masks = torch.tensor([[0.0] if d else [1.0] for d in done], dtype=torch.float32)
            bad_masks = torch.tensor(
                [[0.0] if info.get("TimeLimit.truncated", False) else [1.0] for info in infos],
                dtype=torch.float32,
            )

            for a in agents:
                i = a.agent_id
                a.storage.insert(
                    next_obs[i],
                    n_recurrent_hidden_states[i],
                    n_action[i],
                    n_action_log_prob[i],
                    n_value[i],
                    reward[:, i].unsqueeze(1),  # (n_envs,1)
                    masks,
                    bad_masks,
                )
            all_infos.extend([i for i in infos if i])

        # ------- 计算 returns -------
        for a in agents:
            a.compute_returns()

        # ------- 更新每个智能体（SEAC + SND）-------
        for a in agents:
            # 组装 SND 的参照 logits（他人的 logits 不反传）
            logits_batch = [
                other.model.forward(other.storage.obs[:-1].view(-1, other.model.obs_dim))[0].detach()  # type: ignore
                for other in agents
            ]
            loss_dict = a.update(
                [ag.storage for ag in agents],
                snd_logits_batch=logits_batch,
                snd_coef=float(algorithm.get("snd_coef", 0.10)),
                seac_coef=float(algorithm.get("seac_coef", 1.0)),
            )

            if writer:
                for k, v in loss_dict.items():
                    writer.add_scalar(f"agent{a.agent_id}/{k}", v, j)

            # 可选的内部调度（若模型内定义了 SND warmup）
            a.model.step_scheduler()  # type: ignore
            a.storage.after_update()

        # ------- 日志 / 保存 / 评估 -------
        if j % log_interval == 0 and all_infos:
            squashed = _squash_info(all_infos)
            total_steps = (j + 1) * algorithm["num_processes"] * algorithm["num_steps"]
            fps = int(total_steps / (time.time() - start))
            logging.getLogger("main").info(
                f"Upd {j} | steps {total_steps} | FPS {fps} | meanR {squashed.get('episode_reward', 0.0):.3f}"
            )
            # Sacred 记录
            for k, v in squashed.items():
                _run.log_scalar(k, v, j)
            # ====== 新增：把聚合指标也写到 TB（W&B 将自动同步这些标量）======
            if writer:
                for k, v in squashed.items():
                    writer.add_scalar(k, v, j)
            all_infos.clear()

        if save_interval and (j % save_interval == 0 or j == num_updates):
            cur_dir = path.join(save_dir, f"u{j}")
            os.makedirs(cur_dir, exist_ok=True)
            for a in agents:
                a.save(path.join(cur_dir, f"agent{a.agent_id}"))
            shutil.make_archive(cur_dir, "xztar", save_dir, f"u{j}")
            shutil.rmtree(cur_dir)

        if eval_interval and (j % eval_interval == 0 or j == num_updates):
            evaluate(agents, os.path.join(eval_dir, f"u{j}"))

    envs.close()
    if writer:
        writer.close()
    # 优雅结束 W&B（若开启）
    if use_wandb:
        try:
            import wandb
            wandb.finish()
        except Exception:
            pass
