import gym
import numpy as np
import torch
from gym import spaces

# 触发 RWARE 的注册
import rware

from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper
from wrappers import TimeLimit, Monitor, FlattenMAObs

def make_env(env_id, seed, rank, time_limit, wrappers, monitor_dir):
    def _thunk(rank=0):
        import rware  # 再保一手注册
        env = gym.make(env_id, fast_obs=False, msg_bits=1, disable_env_checker=True)

        # 某些版本把多离散拼成 MultiDiscrete：拆成每 agent 一格 Discrete
        if isinstance(env.action_space, spaces.MultiDiscrete):
            nvec = env.action_space.nvec
            env.action_space = [spaces.Discrete(int(n)) for n in nvec]

        env.seed(seed + rank)

        if time_limit:
            env = TimeLimit(env, time_limit)
        for wrapper in wrappers:
            env = wrapper(env)

        env = FlattenMAObs(env)

        # —— 设定“按智能体的动作空间列表”（只暴露移动动作；通信位由 VecPyTorch 补 0）
        try:
            from rware.warehouse import Action as RwareAction
            n_move = int(len(RwareAction))
        except Exception:
            n_move = 5
        try:
            n_agents = int(env.observation_space.shape[0])
        except Exception:
            n_agents = int(getattr(env.unwrapped, "num_agents", 1))
        env.action_space = [spaces.Discrete(n_move) for _ in range(n_agents)]

        if monitor_dir:
            env = Monitor(env, monitor_dir, episode_trigger=lambda ep: ep == 0, name_prefix=f"env{rank}")
        return env
    return _thunk

def make_vec_envs(env_name, seed, dummy_vecenv, num_processes,
                  time_limit, wrappers, device="cpu", monitor_dir=None):
    env_fns = [make_env(env_name, seed, i, time_limit, wrappers, monitor_dir) for i in range(num_processes)]
    if num_processes == 1 or dummy_vecenv:
        venv = DummyVecEnv(env_fns)
    else:
        venv = SubprocVecEnv(env_fns, start_method="fork")
    return VecPyTorch(venv, device)

class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        super().__init__(venv)
        self.device = device
        # 推断需要的列数：移动(1) + msg_bits
        self.required_cols = 2
        try:
            if hasattr(venv, "envs") and len(venv.envs) > 0:
                mb = int(getattr(venv.envs[0].unwrapped, "msg_bits", 1))
                self.required_cols = 1 + max(1, mb)
        except Exception:
            pass

    def reset(self):
        obs = self.venv.reset()
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        return [obs_t[:, i] for i in range(obs_t.shape[1])]

    def step_async(self, actions):
        n_envs = self.num_envs
        per_agent = []
        for a in actions:
            if isinstance(a, torch.Tensor):
                arr = a.detach().cpu().numpy()
            else:
                arr = np.asarray(a)
            arr = arr.astype(np.int64, copy=False)

            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            elif arr.ndim == 1:
                arr = arr.reshape(n_envs, 1) if arr.shape[0] == n_envs else arr.reshape(1, -1)
            elif arr.ndim >= 2:
                if arr.shape[0] != n_envs and arr.ndim == 2 and arr.shape[1] == n_envs:
                    arr = arr.T
                elif arr.shape[0] != n_envs and n_envs == 1:
                    arr = arr.reshape(1, -1)

            # 自动补通信位列
            if arr.shape[1] < self.required_cols:
                pad = np.zeros((arr.shape[0], self.required_cols - arr.shape[1]), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=1)
            per_agent.append(arr)

        per_env_actions = [tuple(arr[e] for arr in per_agent) for e in range(n_envs)]
        return self.venv.step_async(per_env_actions)

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device)
        n_envs, n_agents = obs_t.shape[0], obs_t.shape[1]
        obs_list = [obs_t[:, i] for i in range(n_agents)]
        r = torch.as_tensor(rew, dtype=torch.float32, device=self.device).view(n_envs, 1).expand(n_envs, n_agents)
        done_np = np.asarray(done, dtype=bool)
        return (obs_list, r, done_np, info)
