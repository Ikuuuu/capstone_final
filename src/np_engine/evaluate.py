"""링크 예측 평가 : Filtered MRR / Hits@1/3/10.

head & tail 양방향 예측의 평균(표준 KGE 프로토콜).
전체·관계별·관계그룹별(방향성/계층) 분해 및 트리플별 reciprocal rank 배열
(bootstrap 입력) 을 반환한다.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np

from .data import KG


def _build_filter(all_triples):
    tails = defaultdict(set)   # (h,r) -> {t}
    heads = defaultdict(set)   # (r,t) -> {h}
    for h, r, t in all_triples:
        tails[(h, r)].add(t)
        heads[(r, t)].add(h)
    return tails, heads


def evaluate(model, P, kg: KG, eval_triples: np.ndarray, batch_ent: int = 2048) -> Dict:
    if hasattr(model, "prepare_eval"):
        model.prepare_eval(P)
    tails, heads = _build_filter(kg.all_triples)
    ne = kg.num_ent
    all_ent = np.arange(ne)

    rr = np.zeros(len(eval_triples))       # 트리플별 평균 reciprocal rank (head+tail)
    hits1 = np.zeros(len(eval_triples))
    hits3 = np.zeros(len(eval_triples))
    hits10 = np.zeros(len(eval_triples))
    rel_of = eval_triples[:, 1].copy()

    for i, (h, r, t) in enumerate(eval_triples.tolist()):
        ranks = []
        # ─ tail prediction : (h, r, ?) ─
        e = model.energy(P, np.full(ne, h), np.full(ne, r), all_ent)   # 낮을수록 정답
        filt = tails[(h, r)] - {t}
        if filt:
            e[np.fromiter(filt, dtype=int)] = np.inf
        ranks.append(int((e < e[t]).sum()) + 1)
        # ─ head prediction : (?, r, t) ─
        e = model.energy(P, all_ent, np.full(ne, r), np.full(ne, t))
        filt = heads[(r, t)] - {h}
        if filt:
            e[np.fromiter(filt, dtype=int)] = np.inf
        ranks.append(int((e < e[h]).sum()) + 1)

        rk = np.array(ranks, dtype=float)
        rr[i] = np.mean(1.0 / rk)
        hits1[i] = np.mean(rk <= 1)
        hits3[i] = np.mean(rk <= 3)
        hits10[i] = np.mean(rk <= 10)

    def agg(mask=None):
        idx = slice(None) if mask is None else mask
        return {
            "mrr": float(rr[idx].mean()) if np.size(rr[idx]) else 0.0,
            "hits@1": float(hits1[idx].mean()) if np.size(hits1[idx]) else 0.0,
            "hits@3": float(hits3[idx].mean()) if np.size(hits3[idx]) else 0.0,
            "hits@10": float(hits10[idx].mean()) if np.size(hits10[idx]) else 0.0,
            "n": int(np.size(rr[idx])),
        }

    out = {"overall": agg()}

    # 관계별
    per_rel = {}
    for rid in np.unique(rel_of):
        per_rel[kg.id2rel[int(rid)]] = agg(rel_of == rid)
    out["per_relation"] = per_rel

    # 관계 그룹별 (가설 검증 자산)
    dir_mask = np.isin(rel_of, list(kg.directional_rel_ids))
    hier_mask = np.isin(rel_of, list(kg.hierarchy_rel_ids))
    out["groups"] = {
        "directional": agg(dir_mask),   # H1
        "hierarchy": agg(hier_mask),    # H2
        "other": agg(~dir_mask & ~hier_mask),
    }

    # bootstrap 입력용 원자료
    out["_rr"] = rr
    out["_rel_of"] = rel_of
    out["_dir_mask"] = dir_mask
    out["_hier_mask"] = hier_mask
    return out
