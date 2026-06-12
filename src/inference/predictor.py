"""학습된 KGE 모델로부터 의미 검색 / 관계 예측 추론을 수행하는 Predictor."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch

from ..models import build_model
from ..utils.checkpoint import load_checkpoint


class Predictor:
    """체크포인트로부터 모델 로드 → 추론 인터페이스 제공."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        entity2id: Dict[str, int],
        relation2id: Dict[str, int],
        model_name: str = "ours",
        num_classes: int = 10,
        embedding_dim: int = 100,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.id2entity = {v: k for k, v in entity2id.items()}
        self.id2relation = {v: k for k, v in relation2id.items()}
        kwargs = dict(
            num_entities=len(entity2id),
            num_relations=len(relation2id),
            embedding_dim=embedding_dim,
        )
        if model_name == "ours":
            kwargs["num_classes"] = num_classes
        self.model = build_model(model_name, **kwargs).to(device)
        load_checkpoint(checkpoint_path, self.model, map_location=device)
        self.model.eval()

    @torch.no_grad()
    def find_similar_entities(self, query_entity: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """질의 엔티티와 의미적으로 가까운 엔티티 Top-K 반환."""
        if query_entity not in self.entity2id:
            raise KeyError(f"Unknown entity: {query_entity}")
        q_id = torch.tensor([self.entity2id[query_entity]], device=self.device)
        q_emb = self.model.entity_emb(q_id)              # (1, d)
        all_emb = self.model.entity_emb.weight           # (E, d)
        dists = torch.norm(all_emb - q_emb, p=2, dim=-1).cpu()
        ranked = torch.argsort(dists)
        results: List[Tuple[str, float]] = []
        for idx in ranked[: top_k + 1].tolist():
            if idx == self.entity2id[query_entity]:
                continue
            results.append((self.id2entity[idx], float(dists[idx])))
            if len(results) >= top_k:
                break
        return results

    @torch.no_grad()
    def predict_tail(self, head: str, relation: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """주어진 (head, relation, ?) 에 대한 tail 예측 Top-K."""
        if head not in self.entity2id or relation not in self.relation2id:
            raise KeyError(f"Unknown head/relation: {head}/{relation}")
        h_id = torch.tensor([self.entity2id[head]], device=self.device)
        r_id = torch.tensor([self.relation2id[relation]], device=self.device)
        num_e = len(self.entity2id)
        all_t = torch.arange(num_e, device=self.device)
        h_t = h_id.expand(num_e)
        r_t = r_id.expand(num_e)
        scores = self.model.score(h_t, r_t, all_t).cpu()
        ranked = torch.argsort(scores)
        return [(self.id2entity[i.item()], float(scores[i])) for i in ranked[:top_k]]
