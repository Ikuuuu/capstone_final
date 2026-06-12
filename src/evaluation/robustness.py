"""강건성 평가 : 학습 데이터 25 / 50 / 75 / 100 % 축소 시 성능 저하 폭."""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple


def stratified_subsample(
    triples: List[Tuple[int, int, int]],
    fraction: float,
    seed: int = 42,
) -> List[Tuple[int, int, int]]:
    """관계별 stratified 로 일정 비율 sampling (KG 구조 보존)."""
    rng = random.Random(seed)
    by_rel: Dict[int, list] = {}
    for h, r, t in triples:
        by_rel.setdefault(r, []).append((h, r, t))
    sampled: List[Tuple[int, int, int]] = []
    for rel, ts in by_rel.items():
        rng.shuffle(ts)
        n = max(1, int(len(ts) * fraction))
        sampled.extend(ts[:n])
    return sampled


def evaluate_robustness(
    train_fn: Callable,                # config 받아서 학습 → metrics 반환
    base_config: Dict,
    fractions: List[float] = [0.25, 0.5, 0.75, 1.0],
    n_runs: int = 3,
) -> Dict[float, Dict[str, float]]:
    """각 fraction 에 대해 n_runs 회 학습 → 평균 ± 표준편차."""
    import numpy as np
    results: Dict[float, Dict[str, float]] = {}
    for frac in fractions:
        runs: List[Dict[str, float]] = []
        for seed in range(n_runs):
            cfg = dict(base_config)
            cfg["_data_fraction"] = frac
            cfg["_seed"] = seed
            runs.append(train_fn(cfg))
        keys = runs[0].keys()
        results[frac] = {
            f"{k}_mean": float(np.mean([r[k] for r in runs])) for k in keys
        }
        results[frac].update({
            f"{k}_std": float(np.std([r[k] for r in runs])) for k in keys
        })
    return results
