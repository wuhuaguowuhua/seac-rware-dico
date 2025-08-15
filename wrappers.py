#!/usr/bin/env python
"""wrappers.py – robust helpers for evaluating multi-agent RWARE models."""
from __future__ import annotations
from collections import deque
from time import perf_counter
from typing import Any

import gym
import numpy as np
from gym.wrappers.record_video import RecordVideo as GymMonitor  # type: ignore
from gym.wrappers.time_limit import TimeLimit as GymTimeLimit    # type: ignore
from gym import spaces

def _safe_unwrap_reset(out):
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, {}

def _to_list(x: Any, length: int) -> list:
    if isinstance(x, (bool, int, np.bool_)):
        return [bool(x)] * length
    return list(x)

# ---- 替换 wrappers.py 中的 RecordEpisodeStatistics 整个类 ----
class RecordEpisodeStatistics(gym.Wrapper):
    """在 info 中累计单局 reward / length / time（多智能体兼容）。"""

    def __init__(self, env: gym.Env, deque_size: int = 100):
        super().__init__(env)
        # 1) 尽量从底层环境拿真实的智能体数量
        try:
            n = int(getattr(env.unwrapped, "num_agents", 0))
        except Exception:
            n = 0
        if n <= 0:
            # 退化兜底：先按 1 起步，见到 reward 再自适应调整
            n = 1
        self.n_agents = n

        # 2) 累计器与队列
        self._ep_rew = np.zeros(self.n_agents, np.float64)
        self._ep_len = 0
        self._start = perf_counter()
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def _ensure_rew_buffer(self, reward):
        """根据 reward 的形状自适应调整内部累计向量长度。"""
        r = np.asarray(reward, dtype=np.float64)
        if r.ndim == 0:
            # 标量：将内部缓冲调整为长度 1
            if self._ep_rew.shape[0] != 1:
                self._ep_rew = np.zeros(1, np.float64)
                self.n_agents = 1
        else:
            # 按智能体向量：将内部缓冲调整为对应长度
            if self._ep_rew.shape[0] != r.shape[0]:
                self._ep_rew = np.zeros(r.shape[0], np.float64)
                self.n_agents = r.shape[0]
        return r

    # -- reset ------------------------------------------------------
    def reset(self, **kw):                              # type: ignore[override]
        res = self.env.reset(**kw)
        obs = res[0] if isinstance(res, tuple) else res  # 丢弃 info 以适配 SB3 1.x
        self._ep_rew.fill(0.0)
        self._ep_len = 0
        self._start = perf_counter()
        return obs

    # -- step -------------------------------------------------------
    def step(self, action):                             # type: ignore[override]
        res = self.env.step(action)
        if len(res) == 4:                               # Gym ≤0.25
            obs, reward, done, info = res              # done：可能是 bool 或 list[bool]
        else:                                           # Gymnasium / Gym ≥0.26
            obs, reward, terminated, truncated, info = res
            # 合并为 env 级别或 per-agent：
            term  = np.asarray(terminated)
            trunc = np.asarray(truncated)
            # 这里不压成标量，仍让 done 保持原来的结构（标量或 per-agent）
            done  = (term | trunc) if term.shape == trunc.shape else (term if term.ndim else bool(term or trunc))

        # —— 关键：按 reward 的形状自适应内部缓存维度 ----
        r = self._ensure_rew_buffer(reward)
        if r.ndim == 0:
            self._ep_rew[0] += float(r)
        else:
            self._ep_rew += r

        self._ep_len += 1

        # 归一化 done：把标量/列表统一成 list[bool] 再判断是否全部结束
        done_list = _to_list(done, self.n_agents)
        if all(done_list):
            info["episode_reward"] = self._ep_rew.copy()
            info["episode_length"] = self._ep_len
            info["episode_time"]   = perf_counter() - self._start
            self.return_queue.append(self._ep_rew.copy())
            self.length_queue.append(self._ep_len)
            self._ep_rew.fill(0.0)
            self._ep_len = 0
            self._start = perf_counter()

        return (obs, reward, done, info) if len(res) == 4 else (
            obs, reward, terminated, truncated, info
        )


class TimeLimit(GymTimeLimit):
    def step(self, action):
        res = self.env.step(action)
        if len(res) == 4:
            obs, reward, done, info = res
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = res
            done = [t or tr for t, tr in zip(_to_list(terminated, len(action)), _to_list(truncated, len(action)))]
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not all(done)
            terminated = done = [True] * len(_to_list(done, len(action))); truncated = True
        return (obs, reward, done, info) if len(res) == 4 else (obs, reward, terminated, truncated, info)

class Monitor(GymMonitor): pass
SafeMonitor = Monitor

class SquashDones(gym.Wrapper):
    def step(self, action):
        res = self.env.step(action)
        if len(res) == 4:
            return res
        else:
            obs, reward, terminated, truncated, info = res
            done = terminated or truncated
            return obs, reward, done, info

# --------- 扁平化工具 ----------
def _flatten_value(v) -> np.ndarray:
    if isinstance(v, dict):
        parts = [_flatten_value(v[k]) for k in sorted(v.keys(), key=lambda x: str(x))]
        return np.concatenate(parts, dtype=np.float32) if parts else np.asarray([], dtype=np.float32)
    if isinstance(v, (list, tuple)):
        parts = [_flatten_value(x) for x in v]
        return np.concatenate(parts, dtype=np.float32) if parts else np.asarray([], dtype=np.float32)
    return np.asarray(v, dtype=np.float32).ravel().astype(np.float32, copy=False)

def _flat_agent(obs_single) -> np.ndarray:
    return _flatten_value(obs_single)

def _agents_seq_from_obs(obs, expected_n: int | None = None):
    # A) list/tuple already per-agent
    if isinstance(obs, (list, tuple)):
        return list(obs)
    # B) dict with 'agents'
    if isinstance(obs, dict):
        if "agents" in obs and isinstance(obs["agents"], (list, tuple)):
            return list(obs["agents"])
        # dict-of-fields (each field length is agent count)
        vals = list(obs.values()); lens = []
        for v in vals:
            try:
                lens.append(len(v))
            except Exception:
                lens.append(None)
        candidate_n = None
        if any(isinstance(x, int) for x in lens):
            counts = {}
            for x in lens:
                if isinstance(x, int): counts[x] = counts.get(x, 0) + 1
            if counts: candidate_n = max(counts, key=counts.get)
        n = expected_n or candidate_n
        if n and all((isinstance(l, int) and l >= n) or (l is None) for l in lens):
            seq = []
            for i in range(n):
                item = {}
                for k, v in obs.items():
                    try: vv = v[i]
                    except Exception: vv = v
                    item[k] = vv
                seq.append(item)
            return seq
        # dict: {agent_id: obs}
        try:
            keys = sorted(obs.keys(), key=lambda k: str(k))
        except Exception:
            keys = list(obs.keys())
        return [obs[k] for k in keys]
    raise AssertionError(f"Unsupported obs type for multi-agent flattening: {type(obs)}")

class FlattenMAObs(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        try:
            expected_n = int(getattr(env.unwrapped, "num_agents", 0)) or None
        except Exception:
            expected_n = None
        res = env.reset()
        sample = res[0] if (isinstance(res, tuple) and len(res) == 2) else res
        sample_seq = _agents_seq_from_obs(sample, expected_n)
        flats = [_flat_agent(a) for a in sample_seq]
        self._n_agents = len(flats)
        self._flat_dim = int(max(x.shape[0] for x in flats))
        # 防御性：零填齐
        if any(x.shape[0] != self._flat_dim for x in flats):
            flats = [np.pad(x, (0, self._flat_dim - x.shape[0]), mode="constant") for x in flats]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._n_agents, self._flat_dim), dtype=np.float32
        )

    def _flat(self, obs):
        seq = _agents_seq_from_obs(obs, expected_n=self._n_agents)
        flats = [_flat_agent(a) for a in seq]
        out = np.zeros((self._n_agents, self._flat_dim), dtype=np.float32)
        for i, x in enumerate(flats):
            x = np.asarray(x, dtype=np.float32).ravel()
            if x.shape[0] >= self._flat_dim:
                out[i, :] = x[: self._flat_dim]
            else:
                out[i, : x.shape[0]] = x
        return out

    def reset(self, **kw):
        res = self.env.reset(**kw)
        obs = res[0] if (isinstance(res, tuple) and len(res) == 2) else res
        return self._flat(obs)

    def step(self, action):
        res = self.env.step(action)
        if len(res) == 4:
            obs, rew, done, info = res
            done_arr = np.asarray(done)
            done_bool = bool(done_arr.any()) if done_arr.ndim > 0 else bool(done)
        else:
            obs, rew, term, trunc, info = res
            term = np.asarray(term); trunc = np.asarray(trunc)
            done_bool = bool((term | trunc).any()) if term.ndim > 0 else bool(term or trunc)

        rew_arr = np.asarray(rew, dtype=np.float32)
        rew_float = float(rew_arr.sum()) if rew_arr.ndim > 0 else float(rew_arr)
        if isinstance(info, dict):
            info.setdefault("agent_rewards", rew_arr.tolist() if rew_arr.ndim > 0 else [float(rew_arr)])
        return self._flat(obs), rew_float, done_bool, info
