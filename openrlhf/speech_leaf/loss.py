from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GRPOConfig:
    clip_eps: float = 0.2
    beta_kl: float = 0.02


def leaf_policy_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    mask: torch.Tensor,
    per_token_advantage: torch.Tensor,
    cfg: GRPOConfig,
    logp_ref: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    ratio = torch.exp(logp_new - logp_old)

    unclipped = ratio * per_token_advantage
    clipped = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * per_token_advantage
    obj = torch.minimum(unclipped, clipped)
    pg_loss = -(obj * mask).sum() / mask.sum().clamp_min(1.0)

    if cfg.beta_kl > 0 and logp_ref is not None:
        log_ratio = logp_ref - logp_new
        kl = (log_ratio.exp() - log_ratio - 1.0) * mask
        kl_loss = kl.sum() / mask.sum().clamp_min(1.0)
        return pg_loss + cfg.beta_kl * kl_loss

    return pg_loss
