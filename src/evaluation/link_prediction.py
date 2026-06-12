"""링크 예측 평가 (MRR, Hits@K)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from tqdm import tqdm


@torch.no_grad()
def evaluate_link_prediction(
    model: torch.nn.Module,
    eval_triples: List[Tuple[int, int, int]],
    num_entities: int,
    filter_set: set | None = None,
    device: str = "cpu",
    batch_size: int = 64,
) -> Dict[str, float]:
    """링크 예측 평가 : MRR, Hits@1/3/10 (Raw + Filter)."""
    model.eval()
    all_entities = torch.arange(num_entities, device=device)
    ranks_raw: List[int] = []
    ranks_filt: List[int] = []

    for h, r, t in tqdm(eval_triples, desc="Link prediction"):
        # tail prediction
        h_t = torch.tensor([h], device=device).expand(num_entities)
        r_t = torch.tensor([r], device=device).expand(num_entities)
        scores = model.score(h_t, r_t, all_entities)  # (E,) — 작을수록 좋음
        target_score = scores[t].item()

        rank_raw = (scores < target_score).sum().item() + 1
        ranks_raw.append(rank_raw)

        if filter_set is not None:
            # 양성 트리플 (현재 (h, r, *)) 의 점수는 제외하여 filtered rank 계산
            mask = torch.zeros(num_entities, dtype=torch.bool, device=device)
            for tt in range(num_entities):
                if (h, r, tt) in filter_set and tt != t:
                    mask[tt] = True
            scores_filt = scores.clone()
            scores_filt[mask] = float("inf")
            rank_filt = (scores_filt < target_score).sum().item() + 1
            ranks_filt.append(rank_filt)

    def _metrics(ranks: List[int]) -> Dict[str, float]:
        if not ranks:
            return {"mrr": 0.0, "hits@1": 0.0, "hits@3": 0.0, "hits@10": 0.0}
        import numpy as np
        ranks_arr = np.array(ranks)
        return {
            "mrr": float((1.0 / ranks_arr).mean()),
            "hits@1": float((ranks_arr <= 1).mean()),
            "hits@3": float((ranks_arr <= 3).mean()),
            "hits@10": float((ranks_arr <= 10).mean()),
        }

    out = {f"raw_{k}": v for k, v in _metrics(ranks_raw).items()}
    if ranks_filt:
        out.update({f"filt_{k}": v for k, v in _metrics(ranks_filt).items()})
    return out
