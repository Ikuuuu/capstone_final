"""의미 유사 검색 정성 평가 (동의어·약어·인용 관계)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch


@torch.no_grad()
def evaluate_semantic_search(
    model: torch.nn.Module,
    eval_path: str | Path,
    entity2id: Dict[str, int],
    top_k: List[int] = [1, 3, 5, 10],
    device: str = "cpu",
) -> Dict[str, float]:
    """평가셋(JSON) 로드 → 각 질의에 대해 가장 가까운 엔티티 Top-k 정확도 측정.

    평가셋 형식 (data/eval/semantic_eval.json):
    [
      {"query": "지자체", "gold": "지방자치단체", "type": "synonym"},
      {"query": "행안부", "gold": "행정안전부", "type": "abbreviation"},
      ...
    ]
    """
    eval_path = Path(eval_path)
    if not eval_path.exists():
        return {"semantic_no_data": 0.0}

    with eval_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    hits_at = {k: 0 for k in top_k}
    n = 0
    entity_names = sorted(entity2id, key=lambda e: entity2id[e])

    for item in items:
        if item["query"] not in entity2id or item["gold"] not in entity2id:
            continue
        q_id = entity2id[item["query"]]
        g_id = entity2id[item["gold"]]

        q_emb = model.entity_emb(torch.tensor([q_id], device=device))  # (1, d)
        all_emb = model.entity_emb.weight                                # (E, d)
        dists = torch.norm(all_emb - q_emb, p=2, dim=-1)                 # (E,)
        ranked = torch.argsort(dists)
        gold_rank = (ranked == g_id).nonzero(as_tuple=True)[0].item() + 1
        for k in top_k:
            if gold_rank <= k:
                hits_at[k] += 1
        n += 1

    if n == 0:
        return {"semantic_no_match": 0.0}
    return {f"semantic_hits@{k}": hits_at[k] / n for k in top_k}
