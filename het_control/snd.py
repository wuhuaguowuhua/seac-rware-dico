"""Helper utilities to compute statistical (Wasserstein‑2) distance
between agents' action distributions — used by DiCoSNDPolicy to
measure behavioural diversity at training time.

This is the *exact* file you uploaded earlier (commit preserved).
"""

from __future__ import annotations
from typing import List
import torch

def compute_behavioral_distance(
    agent_actions: List[torch.Tensor],
    just_mean: bool,
):
    """Return pair‑wise distances for a list of action logits tensors.

    Each tensor must be shaped ``[*batch, action_features]``.  If the logits
    only contain the mean of a categorical distribution (`just_mean=True`)
    the closed‑form Wasserstein‑2 reduces to the L2 distance between the
    means; otherwise the full mean+variance term is used.
    """
    n_agents = len(agent_actions)
    pair_results = []
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            pair_results.append(
                compute_statistical_distance(
                    agent_actions[i], agent_actions[j], just_mean=just_mean
                )
            )
    result = torch.stack(pair_results, dim=-1)
    n_pairs = n_agents * (n_agents - 1) // 2
    return result.view((*agent_actions[0].shape[:-1], n_pairs))

def compute_statistical_distance(logits_i, logits_j, just_mean: bool):
    if just_mean:
        loc_i, loc_j = logits_i, logits_j
        var_i = var_j = None
    else:
        loc_i, scale_i = logits_i.chunk(2, dim=-1)
        loc_j, scale_j = logits_j.chunk(2, dim=-1)
        var_i, var_j = scale_i.pow(2), scale_j.pow(2)

    return _wasserstein_2(
        loc_i,
        var_i,
        loc_j,
        var_j,
        just_mean=just_mean,
    ).view(logits_i.shape[:-1])

def _wasserstein_2(mean1, var1, mean2, var2, *, just_mean: bool):
    mean_term = torch.linalg.norm(mean1 - mean2, dim=-1)
    if just_mean:
        return mean_term
    # variance term (Frobenius norm between covariance square‑roots)
    s1 = torch.linalg.cholesky(torch.diag_embed(var1))
    s2 = torch.linalg.cholesky(torch.diag_embed(var2))
    return torch.sqrt(mean_term ** 2 + torch.linalg.norm(s1 - s2, dim=(-1, -2)) ** 2)