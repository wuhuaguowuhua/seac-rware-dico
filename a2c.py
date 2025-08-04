"""Patched a2c.py for evaluation only (updated loader for varied checkpoints)."""
from __future__ import annotations

import collections
from pathlib import Path
from typing import Any, Dict

import gym
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Optional Sacred stub (unchanged)
# ---------------------------------------------------------------------------
try:
    from sacred import Ingredient  # type: ignore
except ModuleNotFoundError:

    class Ingredient:  # pylint: disable=too-few-public-methods
        def __init__(self, *_: Any, **__: Any):
            pass

        def config(self, fn=None):  # type: ignore
            return fn

        def capture(self, fn=None):  # type: ignore
            return fn if fn is not None else (lambda f: f)

algorithm = Ingredient("algorithm", save_git_info=False)
@algorithm.config
def _default_algo_cfg():
    num_processes = 4
    num_steps = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
try:
    from gymnasium.spaces.utils import flatdim as _flatdim
except ImportError:
    from gym.spaces.utils import flatdim as _flatdim  # type: ignore


def _space_dim(space: gym.Space) -> int:
    try:
        return int(_flatdim(space))
    except NotImplementedError:
        if isinstance(space, gym.spaces.Box):
            return int(np.prod(space.shape))
        if isinstance(space, gym.spaces.Discrete):
            return int(space.n)
        if isinstance(space, gym.spaces.MultiBinary):
            return int(space.n)
        if isinstance(space, gym.spaces.MultiDiscrete):
            return int(np.prod(space.nvec))
        raise


class Policy(torch.nn.Module):
    def __init__(self, obs_space: gym.Space, action_space: gym.Space):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(_space_dim(obs_space), 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, _space_dim(action_space)),
        )

    def act(self, obs: torch.Tensor, *_):
        logits = self.net(obs)
        action = torch.argmax(logits)
        return None, action, None, None


class A2C:  # evaluation‑only skeleton
    def __init__(self, agent_id: int, obs_space: gym.Space, action_space: gym.Space, *, device: str = "cpu", **_: Any):
        self.agent_id = agent_id
        self.device = torch.device(device)
        self.model: Policy | torch.nn.Module = Policy(obs_space, action_space).to(self.device)

    # ---------------------------------------------------------------------
    # Robust restore that understands several checkpoint layouts
    # ---------------------------------------------------------------------
    def restore(self, ckpt_dir: str | Path) -> None:
        ckpt_file = Path(ckpt_dir) / "models.pt"
        if not ckpt_file.exists():
            raise FileNotFoundError(ckpt_file)

        obj = torch.load(ckpt_file, weights_only=False, map_location=self.device)

        # Case A: full Module
        if isinstance(obj, torch.nn.Module):
            self.model = obj.to(self.device)
            return

        # Case B: dict variants
        if isinstance(obj, dict):
            # common keys holding either Module or state_dict
            for key in [
                "model",
                "policy",
                "actor_critic",
                "ac",
                "network",
                "model_state_dict",
            ]:
                if key in obj:
                    cand = obj[key]
                    if isinstance(cand, torch.nn.Module):
                        self.model = cand.to(self.device)
                        return
                    if isinstance(cand, (dict, collections.OrderedDict)):
                        self.model.load_state_dict(cand)
                        return
            # maybe entire dict is state_dict
            try:
                self.model.load_state_dict(obj)  # type: ignore[arg-type]
                return
            except Exception:  # pragma: no cover
                pass

        raise RuntimeError("Unrecognized checkpoint format")
