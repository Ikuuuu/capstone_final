"""KGE 모델의 공통 베이스 클래스."""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseKGEModel(nn.Module, ABC):
    """모든 KGE 모델이 상속할 공통 인터페이스."""

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.entity_emb = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    @abstractmethod
    def score(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """점수 함수 f(h, r, t).  값이 작을수록 양성 가능성 높음."""

    def init_with_lm(self, entity_init_vectors: torch.Tensor) -> None:
        """한국어 LM 임베딩으로 엔티티 초기화 (보강 장치).

        Args:
            entity_init_vectors: shape (num_entities, embedding_dim)
        """
        assert entity_init_vectors.shape == self.entity_emb.weight.shape
        with torch.no_grad():
            self.entity_emb.weight.copy_(entity_init_vectors)
