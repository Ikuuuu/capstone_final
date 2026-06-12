"""통계 검정 : paired bootstrap + 가설 H1/H2/H3 자동 판정.

paired bootstrap (가이드라인 §3.5, 1,000 samples) :
  동일 테스트 트리플에 대한 두 모델의 reciprocal rank 를 짝지어
  (Ours − baseline) 평균차의 분포를 재표집해 p-value 와 신뢰구간 산출.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def paired_bootstrap(rr_a: np.ndarray, rr_b: np.ndarray, n_boot: int = 1000,
                     seed: int = 0) -> Dict:
    """H0: mean(rr_a) == mean(rr_b).  반환: a−b 평균차, p_value(양측), 95% CI."""
    rng = np.random.RandomState(seed)
    n = len(rr_a)
    if n == 0:
        return {"delta": 0.0, "p_value": 1.0, "ci95": [0.0, 0.0], "n": 0}
    diff = rr_a - rr_b
    obs = float(diff.mean())
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        boot[i] = diff[idx].mean()
    # 양측 p : 부호 반대(또는 0 포함) 비율 기반
    if obs >= 0:
        p = 2.0 * float((boot <= 0).mean())
    else:
        p = 2.0 * float((boot >= 0).mean())
    p = min(1.0, p)
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    return {"delta": obs, "p_value": p, "ci95": ci, "n": int(n)}


def test_H1(ours_eval, transo_eval, threshold_pp=5.0, n_boot=1000) -> Dict:
    """H1 방향성 : 방향성 관계에서 Ours Hits@1 ≥ TransO +5%p & bootstrap p<0.05."""
    dm = ours_eval["_dir_mask"]
    rr_o = ours_eval["_rr"][dm]
    rr_t = transo_eval["_rr"][dm]
    bs = paired_bootstrap(rr_o, rr_t, n_boot=n_boot)
    h1_o = ours_eval["groups"]["directional"]["hits@1"]
    h1_t = transo_eval["groups"]["directional"]["hits@1"]
    delta_pp = (h1_o - h1_t) * 100
    mrr_delta_pp = (ours_eval["groups"]["directional"]["mrr"]
                    - transo_eval["groups"]["directional"]["mrr"]) * 100
    passed = (delta_pp >= threshold_pp) and (bs["p_value"] < 0.05)
    return {
        "hypothesis": "H1_directionality",
        "criterion": f"방향성관계 Hits@1 ≥ TransO +{threshold_pp}%p & paired-bootstrap p<0.05",
        "ours_hits@1": h1_o, "transo_hits@1": h1_t,
        "delta_hits@1_pp": delta_pp, "delta_mrr_pp": mrr_delta_pp,
        "bootstrap_mrr": bs, "passed": bool(passed),
    }


def test_H2(full_eval, ablation_eval, threshold_pp=3.0) -> Dict:
    """H2 계층 적응 : L_hier_attn 제거 시 (계층관계) MRR 이 ≥3%p 저하."""
    mrr_full = full_eval["groups"]["hierarchy"]["mrr"]
    mrr_abl = ablation_eval["groups"]["hierarchy"]["mrr"]
    drop_pp = (mrr_full - mrr_abl) * 100
    passed = drop_pp >= threshold_pp
    return {
        "hypothesis": "H2_hierarchy_adaptation",
        "criterion": f"L_hier_attn 제거 시 계층관계 MRR ≥ {threshold_pp}%p 저하",
        "mrr_full": mrr_full, "mrr_ablation": mrr_abl,
        "drop_pp": drop_pp, "passed": bool(passed),
    }


def test_H3(deltas_ours: Dict[float, float], deltas_transe: Dict[float, float],
            factor=0.7) -> Dict:
    """H3 강건성 : 데이터 50% 축소 시 Ours 의 ΔMRR < TransE 의 ΔMRR × 0.7."""
    d_o = deltas_ours.get(0.5, None)
    d_t = deltas_transe.get(0.5, None)
    passed = (d_o is not None and d_t is not None and d_o < d_t * factor)
    return {
        "hypothesis": "H3_robustness",
        "criterion": f"50% 축소 시 ΔMRR(Ours) < ΔMRR(TransE) × {factor}",
        "delta_mrr_ours_50pct": d_o, "delta_mrr_transe_50pct": d_t,
        "threshold": (d_t * factor) if d_t is not None else None,
        "passed": bool(passed),
    }
