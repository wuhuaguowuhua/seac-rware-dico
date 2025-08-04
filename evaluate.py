#!/usr/bin/env python
"""Evaluate pretrained SEAC agents on RWARE-small-4ag (Gymnasium-friendly)."""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List

import gymnasium as gym
import numpy as np
import torch
import rware                    # noqa: F401  # registers envs

from a2c import A2C
from wrappers import RecordEpisodeStatistics, TimeLimit, SafeMonitor


# --------------------------------------------------------------------------- #
parser = argparse.ArgumentParser("Evaluate RWARE pretrained agents")
parser.add_argument("--env",        default="rware-small-4ag-v2")
parser.add_argument("--ckpt_dir",   default="pretrained/rware-small-4ag")
parser.add_argument("--episodes",   type=int, default=5)
parser.add_argument("--time_limit", type=int, default=500)
parser.add_argument("--video",      action="store_true")
args = parser.parse_args()

# --------------------------------------------------------------------------- #
# build agents (dummy env only to read spaces)
_dummy = gym.make(args.env, disable_env_checker=True)
agents: List[A2C] = []
for i, (obs_sp, act_sp) in enumerate(zip(_dummy.observation_space,
                                         _dummy.action_space)):
    ag = A2C(i, obs_sp, act_sp, device="cpu")
    ag.restore(Path(args.ckpt_dir) / f"agent{i}")
    agents.append(ag)
num_agents = len(agents)
_dummy.close()

# --------------------------------------------------------------------------- #
for ep in range(args.episodes):
    env = gym.make(args.env, disable_env_checker=True, render_mode="human")
    if args.video:
        env = SafeMonitor(env, video_folder="videos",
                          episode_trigger=lambda _: True)
    if args.time_limit:
        env = TimeLimit(env, args.time_limit)
    env = RecordEpisodeStatistics(env)

    obs, _ = env.reset()
    done = [False] * num_agents

    while not all(done):
        tensor_obs = [torch.as_tensor(o, dtype=torch.float32) for o in obs]
        _, actions, _, _ = zip(*[ag.model.act(t, None, None)
                                for ag, t in zip(agents, tensor_obs)])

        res = env.step([a.item() for a in actions])

        if len(res) == 5:                          # Gymnasium ≥0.26
            obs, _, terminated, truncated, info = res  # type: ignore
            # ── 广播 scalar → list ───────────────────────────────────────────
            if isinstance(terminated, (bool, np.bool_)):
                terminated = [terminated] * num_agents      # type: ignore[list-item]
            if isinstance(truncated, (bool, np.bool_)):
                truncated  = [truncated]  * num_agents      # type: ignore[list-item]
            done = [t or tr for t, tr in zip(terminated, truncated)]
        else:                                      # Old Gym (tuple len == 4)
            obs, _, done, info = res               # type: ignore

    total_r = float(np.sum(info["episode_reward"]))
    steps   = int(info["episode_length"])
    print(f"Ep {ep+1}/{args.episodes} | reward={total_r:.1f} | steps={steps}")
    env.close()

print("Evaluation finished.")
