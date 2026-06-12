"""L_dir : 관계 방향성 제약 손실 (본 연구 기여 ①).

f(h, r, t) 와 f(t, r, h) 를 명시적으로 구분하도록 학습.
폐지(A, B) ≠ 폐지(B, A) 같은 비대칭/방향성 관계를 벡터 공간에서 구분.

수식 :  L_dir = max(0, γ_dir + f(h,r,t) − f(t,r,h))
       단, 대칭 관계 (owl:SymmetricProperty) 는 적용에서 제외.
"""
from __future__ import annotations

from typing import Set

import torch
import torch.nn as nn


class DirectionLoss(nn.Module):
    """관계 방향성 제약 손실 (기여 ①)."""

    def __init__(
        self,
        margin_dir: float = 1.0,
        skip_symmetric: bool = True,
    ) -> None:
        super().__init__()
        self.margin_dir = margin_dir
        self.skip_symmetric = skip_symmetric

    def forward(
        self,
        scoring_fn,                # 모델의 score(h, r, t) callable
        h: torch.Tensor,           # (B,)
        r: torch.Tensor,           # (B,)
        t: torch.Tensor,           # (B,)
        symmetric_relation_ids: Set[int] | None = None,
    ) -> torch.Tensor:
        """대칭 관계는 손실에서 제외하고 나머지에 대해 방향성 패널티 적용."""
        # 정방향 점수와 역방향 점수
        f_forward = scoring_fn(h, r, t)   # (B,)
        f_backward = scoring_fn(t, r, h)  # (B,)

        # 정방향이 역방향보다 margin_dir 이상 낮아야 함 (즉 점수가 작아야 좋음)
        loss = torch.clamp(self.margin_dir + f_forward - f_backward, min=0)

        if self.skip_symmetric and symmetric_relation_ids:
            # 대칭 관계 마스킹
            sym_tensor = torch.tensor(
                list(symmetric_relation_ids), device=r.device, dtype=r.dtype
            )
            mask = ~torch.isin(r, sym_tensor)
            loss = loss * mask.float()
            denom = mask.float().sum().clamp(min=1.0)
            return loss.sum() / denom

        return loss.mean()
