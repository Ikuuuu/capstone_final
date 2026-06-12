"""통계적 유의성 검정 (paired bootstrap)."""
from __future__ import annotations

import random
from typing import List


def paired_bootstrap_test(
    a_scores: List[float],
    b_scores: List[float],
    n_samples: int = 1000,
    seed: int = 42,
) -> float:
    """a vs b 의 paired bootstrap p-value 반환.

    Args:
        a_scores, b_scores: 각 모델의 sample-level 점수 (예: 트리플별 reciprocal rank)
        n_samples: bootstrap iteration 수
    """
    assert len(a_scores) == len(b_scores)
    rng = random.Random(seed)
    n = len(a_scores)
    if n == 0:
        return 1.0
    observed_diff = (sum(a_scores) - sum(b_scores)) / n
    count_geq = 0
    for _ in range(n_samples):
        diffs = [
            a_scores[rng.randrange(n)] - b_scores[rng.randrange(n)]
            for _ in range(n)
        ]
        mean_diff = sum(diffs) / n
        if mean_diff >= observed_diff:
            count_geq += 1
    return count_geq / n_samples
