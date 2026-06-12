"""본 연구 모델 : Ontology-Integrated KGE.

기본 구조 : TransE 기반 점수 함수
+ LM 기반 엔티티 초기화 (보강 장치)
+ 4 종 손실 함수 (Trainer 에서 결합 적용)
  - L_struct
  - L_type, L_attr
  - L_hier_attn  (기여 ②)
  - L_dir         (기여 ①)
"""
from __future__ import annotations

import torch

from .transe import TransE


class OursOntologyKGE(TransE):
    """본 연구의 메인 모델.

    점수 함수는 TransE 와 동일하지만, Trainer 가 4종 손실의 가중합을 최적화한다.
    """

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        num_classes: int,
        embedding_dim: int = 100,
        norm: int = 1,
    ) -> None:
        super().__init__(num_entities, num_relations, embedding_dim, norm)
        # 온톨로지 클래스 임베딩 (계층 가중치 attention 입력으로 사용)
        self.class_emb = torch.nn.Embedding(num_classes, embedding_dim)
        torch.nn.init.xavier_uniform_(self.class_emb.weight)
        self.num_classes = num_classes

    def get_class_embeddings(self) -> torch.Tensor:
        return self.class_emb.weight
