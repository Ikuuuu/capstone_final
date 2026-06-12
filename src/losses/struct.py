"""L_struct : 구조 손실 (TransE margin-based ranking loss).

f(h,r,t) 가 양성 트리플은 작게, 부정 샘플은 크게 되도록 학습.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class StructLoss(nn.Module):
    """Margin-based pairwise ranking loss."""

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        pos_scores: torch.Tensor,  # (B,)
        neg_scores: torch.Tensor,  # (B, K)
    ) -> torch.Tensor:
        # 점수 함수 f(h,r,t) 는 "거리" 의미 (작을수록 좋음).
        # max(0, γ + f(pos) − f(neg)) 의 평균
        diff = self.margin + pos_scores.unsqueeze(1) - neg_scores
        return torch.clamp(diff, min=0).mean()
