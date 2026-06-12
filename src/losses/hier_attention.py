"""L_hier_attn : Attention 기반 적응형 계층 가중치 손실 (본 연구 기여 ②).

기존 attention 기반 KGE 는 엔티티·관계 임베딩 자체에 attention 을 적용했지만,
본 연구는 온톨로지 클래스 계층 손실 가중치 자체에 attention 을 적용하여
도메인 별 가중치를 손실 함수 수준에서 자동 학습.

수식 :  L_hier_attn = Σ_c [ softmax(W · c_emb)_c · L_hier_c ]
       (c : 클래스, W : 학습 가능 attention 가중치)
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class HierarchyAttentionLoss(nn.Module):
    """Attention 기반 적응형 계층 가중치 손실 (기여 ②)."""

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        attention_hidden_dim: int = 64,
        initial_uniform_epochs: int = 50,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.initial_uniform_epochs = initial_uniform_epochs
        # Attention 모듈 : 클래스 임베딩 → scalar score
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, 1),
        )

    def compute_class_weights(
        self,
        class_embeddings: torch.Tensor,    # (C, d)
        current_epoch: int = 0,
    ) -> torch.Tensor:
        """클래스별 attention 가중치 계산."""
        if current_epoch < self.initial_uniform_epochs:
            # 학습 초기는 균등 가중치로 안정화
            return torch.full(
                (self.num_classes,),
                1.0 / self.num_classes,
                device=class_embeddings.device,
            )
        scores = self.attention(class_embeddings).squeeze(-1)  # (C,)
        return F.softmax(scores, dim=0)

    def forward(
        self,
        per_class_hier_losses: torch.Tensor,    # (C,) 클래스별 L_hier_c
        class_embeddings: torch.Tensor,         # (C, d)
        current_epoch: int = 0,
    ) -> torch.Tensor:
        weights = self.compute_class_weights(class_embeddings, current_epoch)
        return torch.sum(weights * per_class_hier_losses)
