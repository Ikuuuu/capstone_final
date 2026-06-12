"""TransE 베이스라인 모델."""
from __future__ import annotations

import torch

from .base import BaseKGEModel


class TransE(BaseKGEModel):
    """f(h, r, t) = || h + r − t ||_p."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 100,
        norm: int = 1,
    ) -> None:
        super().__init__(num_entities, num_relations, embedding_dim)
        self.norm = norm

    def score(
        self,
        h: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        h_e = self.entity_emb(h)
        r_e = self.relation_emb(r)
        t_e = self.entity_emb(t)
        return torch.norm(h_e + r_e - t_e, p=self.norm, dim=-1)
