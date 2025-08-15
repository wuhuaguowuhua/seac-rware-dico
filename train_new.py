#!/usr/bin/env python
"""Training SEAC/A2C on RWARE with DiCoSNDPolicy. Final aligned version."""
from __future__ import annotations
import glob, logging, os, shutil, time
from collections import deque
from copy import deepcopy
from os import path
from pathlib import Path

import gym
import numpy as np
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver  # noqa
from torch.utils.tensorboard import SummaryWriter
from gym import spaces

import utils
from a2c import A2C, algorithm
from envs import make_vec_envs
from wrappers import RecordEpisodeStatistics, SquashDones
from het_control.het_control_mlp_empirical import DiCoSNDPolicy

import rware  # noqa: env registration
import lbforaging  # noqa: env registration

# ------------------------ Sacred & logging ------------------------
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
    num_env_steps = 100e6
    eval_dir = "./results/video/{id}"
    loss_dir = "./results/loss/{id}"
    save_dir = "./results/trained_models/{id}"
    log_interval = 2000
    save_interval = int(1e6)
    eval_interval = int(1e6)
    episodes_per_eval = 8

# load extra YAML configs
for conf in glob.glob("configs/*.yaml"):
    ex.add_named_config(Path(conf).stem, conf)

def _squash_info(info):
    info = [i for i in info if i]
    keys = {k for d in info for k in d.keys() if k != "TimeLimit.truncated"}
    return {k: np.mean([np.array(d[k]).sum() for d in info if k in d]) for k in keys}

# ------------------------ evaluation helper -----------------------
@ex.capture
def evaluate(
    agents, monitor_dir, episodes_per_eval, env_name, seed, wrappers,
    dummy_vecenv, time_limit, algorithm, _log,
):
    device = algorithm["device"]
    eval_envs = make_vec_envs(
        env_name, seed, dummy_vecenv, episodes_per_eval,
        time_limit, wrappers, device, monitor_dir=monitor_dir,
    )
    n_obs = eval_envs.reset()  # list[n_agents] of (episodes_per_eval, feat)
    # RNN 兼容：没有 hidden_size 就设为 1
    hid_sizes = [int(getattr(a.model, "recurrent_hidden_state_size", 1)) for a in agents]
    n_recurrent_hidden_states = [torch.zeros(episodes_per_eval, h, device=device) for h in hid_sizes]
    masks = torch.zeros(episodes_per_eval, 1, device=device)

    all_infos = []
    while len(all_infos) < episodes_per_eval:
        with torch.no_grad():
            outs = []
            for a, h in zip(agents, n_recurrent_hidden_states):
                ret = a.model.act(n_obs[a.agent_id], h, masks)
                if isinstance(ret, (list, tuple)) and len(ret) == 4:
                    v, act, logp, h_new = ret
                else:
                    v, act, logp = ret
                    h_new = h
                outs.append((v, act, logp, h_new))
            _, n_action, _, n_recurrent_hidden_states = zip(*outs)
        n_obs, _, done, infos = eval_envs.step(n_action)
        all_infos.extend([i for i in infos if i])

    eval_envs.close()
    if all_infos:
        _log.info(f"Eval mean reward { _squash_info(all_infos).get('episode_reward', 0.0):.4f}")

# ------------------------------------------------------------------
# main training -----------------------------------------------------
# ------------------------------------------------------------------
@ex.automain
def main(
    _run, _log, num_env_steps, env_name, seed, algorithm, dummy_vecenv,
    time_limit, wrappers, save_dir, eval_dir, loss_dir, log_interval,
    save_interval, eval_interval,
):
    writer = SummaryWriter(path.expanduser(loss_dir.format(id=_run._id))) if loss_dir else None
    eval_dir = path.expanduser(eval_dir.format(id=_run._id))
    save_dir = path.expanduser(save_dir.format(id=_run._id))
    utils.cleanup_log_dir(eval_dir); utils.cleanup_log_dir(save_dir)

    torch.set_num_threads(1)
    envs = make_vec_envs(
        env_name, seed, dummy_vecenv, algorithm["num_processes"],
        time_limit, wrappers, algorithm["device"],
    )

    # —— 从 observation_space 的形状得到 n_agents 与 feat_dim —— 
    obs_shape = envs.observation_space.shape  # (n_agents, feat_dim)
    n_agents  = int(obs_shape[0]); feat_dim = int(obs_shape[1])

    # —— 构造“按智能体”的 obs/action space 列表 —— 
    obs_spaces = [spaces.Box(low=-np.inf, high=np.inf, shape=(feat_dim,), dtype=np.float32)
                  for _ in range(n_agents)]
    act_spaces = envs.action_space
    assert isinstance(act_spaces, (list, tuple)) and len(act_spaces) == n_agents
    act_dims = np.asarray([int(getattr(a, "n", 1)) for a in act_spaces], dtype=np.int64)

    assert algorithm["num_processes"] == 1, "先用单进程跑通：请把 algorithm.num_processes 设为 1"

    # -------- 构造多智能体 A2C + DiCoSNDPolicy ----------
    agents = []
    for i in range(n_agents):
        osp, asp = obs_spaces[i], act_spaces[i]
        agent = A2C(
            i, osp, asp,
            num_envs=algorithm["num_processes"],
            num_steps=algorithm["num_steps"],
            device=algorithm["device"],
        )
        model = DiCoSNDPolicy(
            obs_dim=int(np.prod(osp.shape)),
            act_dim=int(act_dims[i]),
            num_agents=n_agents,
            hidden_dim=128,
            desired_snd=1.0,
            snd_warmup_steps=5_000,
        ).to(algorithm["device"])
        agent.attach_model(model)     # ← 初始化优化器在这里
        agents.append(agent)

    # -------- 初始化 rollout storage ----------
    obs = envs.reset()  # list[n_agents] of (n_envs, feat)
    for i in range(n_agents):
        agents[i].storage.obs[0].copy_(obs[i].to(algorithm["device"]))
        agents[i].storage.to(algorithm["device"])

    # ----------------------- training loop ------------------------
    start = time.time()
    num_updates = int(num_env_steps) // algorithm["num_steps"] // algorithm["num_processes"]
    all_infos = deque(maxlen=64)

    for j in range(1, num_updates + 1):
        # ---- collect rollout ------------------------------------
        for step in range(algorithm["num_steps"]):
            with torch.no_grad():
                outs = []
                for a in agents:
                    ret = a.model.act(
                        a.storage.obs[step],
                        a.storage.recurrent_hidden_states[step],
                        a.storage.masks[step],
                    )
                    if isinstance(ret, (list, tuple)) and len(ret) == 4:
                        v, act, logp, h_new = ret
                    else:
                        v, act, logp = ret
                        h_new = a.storage.recurrent_hidden_states[step]
                    outs.append((v, act, logp, h_new))
                n_value, n_action, n_action_log_prob, n_recurrent_hidden_states = zip(*outs)

            next_obs, reward, done, infos = envs.step(n_action)

            masks = torch.tensor([[0.0] if d else [1.0] for d in done],
                                 dtype=torch.float32, device=algorithm["device"])
            bad_masks = torch.tensor(
                [[0.0] if info.get("TimeLimit.truncated", False) else [1.0] for info in infos],
                dtype=torch.float32, device=algorithm["device"]
            )

            for a in agents:
                i = a.agent_id
                agents[i].storage.insert(
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

        # ---- update each agent ---------------------------------
        for a in agents:
            a.compute_returns(gamma=algorithm["gamma"])

        for a in agents:
            # 可选：SND 只做监控，不入图
            try:
                logits_batch = [
                    other.model.forward(other.storage.obs[:-1].view(-1, other.model.obs_dim))[0]
                    for other in agents
                ]
                snd_loss_val = float(a.model.snd_regularisation(logits_batch).item())
            except Exception:
                snd_loss_val = 0.0

            loss_dict = a.update([ag.storage for ag in agents])  # ← 这里内部会重新前向，带梯度
            loss_dict["snd_loss"] = snd_loss_val

            if writer:
                for k, v in loss_dict.items():
                    writer.add_scalar(f"agent{a.agent_id}/{k}", v, j)

            a.step_scheduler()
            a.storage.after_update()

        # ---- logging/checkpoint/eval ----------------------------
        if j % log_interval == 0 and all_infos:
            squashed = _squash_info(all_infos)
            total_steps = (j + 1) * algorithm["num_processes"] * algorithm["num_steps"]
            fps = int(total_steps / (time.time() - start))
            _log.info(
                f"Upd {j} | steps {total_steps} | FPS {fps} | "
                f"meanR {squashed.get('episode_reward', 0.0):.3f}"
            )
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
