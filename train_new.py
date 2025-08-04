"""Modified training script that plugs **DiCoSNDPolicy** into the
existing A2C / SEAC training loop.  Only the *agent construction*
section changed – everything else is identical to the original file.
"""

import glob
import logging
import os
import shutil
import time
from collections import deque
from os import path
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import gym
from sacred import Experiment
from sacred.observers import FileStorageObserver  # noqa
from torch.utils.tensorboard import SummaryWriter

import utils
from a2c import A2C, algorithm
from envs import make_vec_envs
from wrappers import RecordEpisodeStatistics, SquashDones
from het_control.het_control_mlp_empirical import DiCoSNDPolicy  # NEW

import rware  # noqa – env registration
import lbforaging  # noqa – env registration

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

    n_obs = eval_envs.reset()
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
    _log.info(f"Eval mean reward { _squash_info(all_infos)['episode_reward']:.4f}")


# ------------------------------------------------------------------
# main training -----------------------------------------------------
# ------------------------------------------------------------------
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
):
    writer = SummaryWriter(path.expanduser(loss_dir.format(id=_run._id))) if loss_dir else None
    eval_dir = path.expanduser(eval_dir.format(id=_run._id))
    save_dir = path.expanduser(save_dir.format(id=_run._id))
    utils.cleanup_log_dir(eval_dir)
    utils.cleanup_log_dir(save_dir)

    torch.set_num_threads(1)
    envs = make_vec_envs(
        env_name,
        seed,
        dummy_vecenv,
        algorithm["num_processes"],
        time_limit,
        wrappers,
        algorithm["device"],
    )

    # --------------------- swap in DiCoSNDPolicy ------------------
    obs_space = envs.observation_space   # FlattenMAObs 已经把 obs 变成 (n_agents, feat)
    act_space = envs.action_space

    # -------- 解析多智能体观测空间 ----------
    if isinstance(obs_space, gym.spaces.Box):
        n_agents = obs_space.shape[0]
        obs_spaces = [
            gym.spaces.Box(low=-np.inf, high=np.inf,
                           shape=(int(np.prod(obs_space.shape[1:])),),
                           dtype=obs_space.dtype)
            for _ in range(n_agents)
        ]
    else:  # Tuple(Box, …)
        n_agents = len(obs_space)
        obs_spaces = [deepcopy(sp) for sp in obs_space]

    # -------- 解析动作空间 ----------
    act_spaces, act_dims = [], []

    # 单一 MultiDiscrete 或 Tuple( … )
    if isinstance(act_space, gym.spaces.MultiDiscrete):
        # env 已经把所有智能体拼到一个 MultiDiscrete 里
        for dim in act_space.nvec.astype(int):
            act_spaces.append(gym.spaces.Discrete(int(dim)))
            act_dims.append(int(dim))

    elif isinstance(act_space, gym.spaces.Tuple):
        # Tuple 每个元素对应 1 个智能体
        for sp in act_space:
            if isinstance(sp, gym.spaces.Discrete):
                act_spaces.append(sp)
                act_dims.append(int(sp.n))

            elif isinstance(sp, gym.spaces.MultiDiscrete):
                # 把该智能体所有离散通道最大值作为统一动作维度
                dim = int(sp.nvec.max())
                act_spaces.append(gym.spaces.Discrete(dim))
                act_dims.append(dim)
            else:
                raise TypeError(f"Unsupported sub-action space: {sp}")
    else:
        raise TypeError(f"Unsupported action space: {act_space}")

    act_dims = np.asarray(act_dims, dtype=int)
    n_agents = len(act_spaces)

    # -------- 构造多智能体 A2C + DiCoSNDPolicy ----------
    agents = []
    for i in range(n_agents):
        osp, asp = obs_spaces[i], act_spaces[i]

        agent = A2C(i, osp, asp)  # vanilla scaffolding
        agent.model = DiCoSNDPolicy(
            obs_dim=int(np.prod(osp.shape)),
            act_dim=int(act_dims[i]),
            num_agents=n_agents,
            hidden_dim=128,
            desired_snd=1.0,
            snd_warmup_steps=5_000,
        ).to(algorithm["device"])
        agents.append(agent)

    # -------- 初始化 rollout storage ----------
    obs = envs.reset()  # ndarray (n_agents, feat_dim)
    for i in range(n_agents):
        agents[i].storage.obs[0].copy_(obs[i])
        agents[i].storage.to(algorithm["device"])

    # ----------------------- training loop ------------------------
    start = time.time()
    num_updates = int(num_env_steps) // algorithm["num_steps"] // algorithm["num_processes"]
    all_infos = deque(maxlen=10)

    for j in range(1, num_updates + 1):
        # ---- collect rollout ------------------------------------
        for step in range(algorithm["num_steps"]):
            with torch.no_grad():
                outs = [
                    a.model.act(
                        a.storage.obs[step],
                        a.storage.recurrent_hidden_states[step],
                        a.storage.masks[step],
                    )
                    for a in agents
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
                    reward[:, i].unsqueeze(1),
                    masks,
                    bad_masks,
                )
            all_infos.extend([i for i in infos if i])

        # ---- update each agent ---------------------------------
        for a in agents:
            a.compute_returns()

        for a in agents:
            logits_batch = [
                other.model.forward(other.storage.obs[:-1].view(-1, other.model.obs_dim))[0]
                for other in agents
            ]
            snd_loss = a.model.snd_regularisation(logits_batch)
            loss_dict = a.update([ag.storage for ag in agents])
            loss_dict["snd_loss"] = snd_loss.item()
            if writer:
                for k, v in loss_dict.items():
                    writer.add_scalar(f"agent{a.agent_id}/{k}", v, j)
            a.model.step_scheduler()
            a.storage.after_update()

        # ---- logging / checkpoint / eval -----------------------
        if j % log_interval == 0 and all_infos:
            squashed = _squash_info(all_infos)
            total_steps = (j + 1) * algorithm["num_processes"] * algorithm["num_steps"]
            fps = int(total_steps / (time.time() - start))
            _log.info(
                f"Upd {j} | steps {total_steps} | FPS {fps} | meanR {squashed['episode_reward']:.3f}"
            )
            for k, v in squashed.items():
                _run.log_scalar(k, v, j)
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
