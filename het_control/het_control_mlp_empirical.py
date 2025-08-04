"""A very small, self‑contained re‑implementation of
`DiCoSNDPolicy` from *Controlling Behavioural Diversity in MARL*.
The interface is purposely kept identical to the `Policy` class used
in the original A2C/SEAC codebase so that the new policy can be
*drop‑in* swapped without touching the downstream algorithm.
"""

from __future__ import annotations
import math
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Local helper comes from the first module
from .snd import compute_behavioral_distance

class DiCoSNDPolicy(nn.Module):
    """Multi‑layer perceptron actor‑critic with *diversity control*.

    Parameters
    ----------
    obs_dim: int
        Flattened observation size per‑agent.
    act_dim: int
        Number of discrete actions.
    num_agents: int
        Total agents in the environment (needed for the diversity loss).
    hidden_dim: int, default 128
    discrete_actions: bool, default True
        Set *False* if you are using continuous actions (not the case in RWARE).
    desired_snd: float, default 1.0
        Target average pair‑wise behavioural distance.
    snd_warmup_steps: int, default 5_000
        Over how many global updates to linearly ramp‑up the diversity loss.
    """

    recurrent_hidden_state_size: int = 1  # keep A2C happy (non‑recurrent)

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        num_agents: int,
        *,
        hidden_dim: int = 128,
        discrete_actions: bool = True,
        desired_snd: float = 1.0,
        snd_warmup_steps: int = 5_000,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents
        self.discrete = discrete_actions
        self.desired_snd = desired_snd
        self.snd_warmup_steps = snd_warmup_steps
        self.register_buffer("_update", torch.zeros(1, dtype=torch.long))

        # Simple two‑layer MLP shared by actor & critic
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_head = nn.Linear(hidden_dim, act_dim)
        self.critic_head = nn.Linear(hidden_dim, 1)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------
    def _body(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return F.relu(self.fc2(x))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (logits, value) for given observations."""
        h = self._body(x)
        return self.actor_head(h), self.critic_head(h)

    # ------------------------------------------------------------------
    # A2C / SEAC required API
    # ------------------------------------------------------------------
    def act(
        self,
        obs: torch.Tensor,
        recurrent_hidden_states: torch.Tensor | None,
        masks: torch.Tensor | None,
    ):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return (
            value,
            action.unsqueeze(-1),  # envs expect shape (n_proc, 1)
            action_log_prob.unsqueeze(-1),
            recurrent_hidden_states,  # untouched (non‑recurrent model)
        )

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        recurrent_hidden_states: torch.Tensor | None,
        masks: torch.Tensor | None,
    ):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action_log_probs = dist.log_prob(action.squeeze(-1)).unsqueeze(-1)
        dist_entropy = dist.entropy().mean()

        # Diversity regulariser (only computed when called by *update*)
        snd_loss = torch.zeros_like(dist_entropy)
        if self.training:
            # Expect a list[Tensor] later; here we pass single batch, compute 0 loss
            pass
        return value, action_log_probs, dist_entropy, snd_loss

    # ------------------------------------------------------------------
    # Diversity loss helper (called externally once per *global* update)
    # ------------------------------------------------------------------
    def snd_regularisation(
        self, agent_logits: List[torch.Tensor]
    ) -> torch.Tensor:
        """Return the scalar SND regulariser for **this** agent.

        The loss pushes the running mean pair‑wise distance towards
        ``self.desired_snd``.
        """
        with torch.no_grad():
            distances = compute_behavioral_distance(agent_logits, just_mean=True)
            current = distances.mean()
        coeff = (
            torch.clamp_min(self._update.float(), 0) / float(self.snd_warmup_steps)
        ).clamp_max(1.0)
        return coeff * (current - self.desired_snd) ** 2

    # Call by the trainer *once* every global update after .update += 1
    def step_scheduler(self):
        self._update += 1
