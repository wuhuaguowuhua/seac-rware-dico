#!/usr/bin/env python
"""wrappers.py – robust helpers for evaluating multi-agent RWARE models.

Exports
~~~~~~~
* RecordEpisodeStatistics – per-agent reward/length tracker supporting both
  Gym 0.26+ (5-tuple) 和旧 Gym 0.21- (4-tuple) API。
* TimeLimit               – 多智能体感知的安全 TimeLimit。
* Monitor / SafeMonitor   – 兼容两种录像包装器。
* FlattenMAObs            – **新实现**，把 RWARE 的 dict-list 观测压平为
  (n_agents, flat_dim) ndarray，保证各子进程输出形状恒定。
* SafeReset / SquashDones – 若干 API 兼容辅助。
"""
from __future__ import annotations

from collections import deque
from time import perf_counter
from typing import Any, Tuple

import gym
import numpy as np
from gym.wrappers.record_video import RecordVideo as GymMonitor      # type: ignore
from gym.wrappers.time_limit import TimeLimit as GymTimeLimit        # type: ignore
from gym import spaces

# ------------------------------------------------------------------
# 通用小工具
# ------------------------------------------------------------------
def _safe_unwrap_reset(out):
    """确保 reset() 的返回值永远是 (obs, info)。"""
    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, {}

def _to_list(x: Any, length: int) -> list:
    """把标量 bool/int 转成长度为 *length* 的 list[bool]。"""
    if isinstance(x, (bool, int, np.bool_)):
        return [bool(x)] * length
    return list(x)

# ------------------------------------------------------------------
# RecordEpisodeStatistics
# ------------------------------------------------------------------
class RecordEpisodeStatistics(gym.Wrapper):
    """在 info 中累计单局 reward / length / time（多智能体兼容）。"""

    def __init__(self, env: gym.Env, deque_size: int = 100):
        super().__init__(env)
        self.n_agents = len(env.action_space)           # type: ignore[arg-type]
        self._ep_rew = np.zeros(self.n_agents, np.float64)
        self._ep_len = 0
        self._start = perf_counter()
        self.return_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    # -- reset ------------------------------------------------------
    def reset(self, **kw):                              # type: ignore[override]
        obs, info = _safe_unwrap_reset(self.env.reset(**kw))
        self._ep_rew.fill(0.0)
        self._ep_len = 0
        self._start = perf_counter()
        return obs, info

    # -- step -------------------------------------------------------
    def step(self, action):                             # type: ignore[override]
        res = self.env.step(action)
        if len(res) == 4:                               # Gym 0.21
            obs, reward, done, info = res
        else:                                           # Gym-nasium / 0.26
            obs, reward, terminated, truncated, info = res
            done = [t or tr
                    for t, tr in zip(_to_list(terminated, self.n_agents),
                                     _to_list(truncated,  self.n_agents))]

        self._ep_rew += np.asarray(reward, np.float64)
        self._ep_len += 1

        if all(_to_list(done, self.n_agents)):
            info["episode_reward"] = self._ep_rew.copy()
            info["episode_length"] = self._ep_len
            info["episode_time"]   = perf_counter() - self._start
            self.return_queue.append(self._ep_rew.copy())
            self.length_queue.append(self._ep_len)

        return (obs, reward, done, info) if len(res) == 4 else (
            obs, reward, terminated, truncated, info
        )

# ------------------------------------------------------------------
# TimeLimit
# ------------------------------------------------------------------
class TimeLimit(GymTimeLimit):
    """兼容 4/5-tuple 的 multi-agent TimeLimit。"""

    def step(self, action):                             # type: ignore[override]
        res = self.env.step(action)
        if len(res) == 4:
            obs, reward, done, info = res
            terminated, truncated = done, False
        else:
            obs, reward, terminated, truncated, info = res
            done = [t or tr
                    for t, tr in zip(_to_list(terminated, len(action)),
                                     _to_list(truncated,  len(action)))]

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not all(done)
            terminated = done = [True] * len(_to_list(done, len(action)))
            truncated = True

        return (obs, reward, done, info) if len(res) == 4 else (
            obs, reward, terminated, truncated, info
        )

# ------------------------------------------------------------------
# Video wrapper alias
# ------------------------------------------------------------------
class Monitor(GymMonitor):
    """Just an alias; choose whichever RecordVideo/Monitor Gym provides."""
    pass

SafeMonitor = Monitor

# ------------------------------------------------------------------
# 简单小 Wrapper 们
# ------------------------------------------------------------------
class SquashDones(gym.Wrapper):
    """把 Gymnasium (terminated, truncated) 压成 Gym 风格的 done。"""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        return obs, reward, done, info

class SafeReset(gym.Wrapper):
    """确保 reset() 始终返回 (obs, info)。"""
    def reset(self, **kw):                              # type: ignore[override]
        return _safe_unwrap_reset(self.env.reset(**kw))

# ------------------------------------------------------------------
# ========== 全新实现：FlattenMAObs ==========
#   针对 RWARE dict-obs，固定键顺序展开，保证长度一致
# ------------------------------------------------------------------
_RWARE_KEYS_ORDER = (
    "vec",          # global robot position one-hot
    "goal_vec",     # goal position
    "box_vec",      # carried box indicator
    "local_message" # 1-bit local comm
)

def _flat_rware_agent(obs: Any) -> np.ndarray:
    """将单个 agent 的 RWARE 观测展平成 1-D float32 向量。"""

    # 旧版 RWARE 返回 dict 观测，键顺序为 ``_RWARE_KEYS_ORDER``；
    # 新版/``fast_obs=True`` 时已经是 ndarray，只需展平即可。
    if isinstance(obs, dict):
        parts: list[np.ndarray] = []
        for k in _RWARE_KEYS_ORDER:
            v = obs[k]
            parts.append(np.asarray(v, dtype=np.float32).ravel())
        return np.concatenate(parts, dtype=np.float32)

    # fallback: treat as array-like
    return np.asarray(obs, dtype=np.float32).ravel()

class FlattenMAObs(gym.Wrapper):
    """
    list/tuple 结构的多智能体观测  →  (n_agents, flat_dim) ndarray
    （专门针对 RWARE；键顺序固定，dim 恒定）
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        sample: Tuple[Any, ...] = env.observation_space.sample()
        assert isinstance(sample, (list, tuple)), "期望 MA obs 是 list/tuple"
        flats = [_flat_rware_agent(a) for a in sample]
        self._n_agents  = len(flats)
        self._flat_dim  = flats[0].shape[0]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._n_agents, self._flat_dim),
            dtype=np.float32,
        )

    # -- helpers ----------------------------------------------------
    def _flat(self, obs):
        return np.stack([_flat_rware_agent(a) for a in obs], axis=0)

    # -- gym API ----------------------------------------------------
    def reset(self, **kw):
        obs, info = _safe_unwrap_reset(self.env.reset(**kw))
        return self._flat(obs), info

    def step(self, action):
        res = self.env.step(action)
        if len(res) == 4:  # old Gym API
            obs, rew, done, info = res
            term, trunc = done, False
        else:
            obs, rew, term, trunc, info = res
        return self._flat(obs), rew, term, trunc, info
# ========== 实现结束 ==========
