"""트리플 데이터셋의 train/valid/test 분할 (stratified)."""
from __future__ import annotations

import random
from collections import Counter
from typing import List, Tuple


def stratified_split(
    triples: List[Tuple[str, str, str]],
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    stratify_by: str = "entity_frequency",
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """엔티티 빈도 stratified 또는 관계 stratified 로 분할.

    KG 구조 특성(허브 엔티티의 분포 등)을 train/valid/test 에 균형 있게 분배.

    Args:
        triples: (head, relation, tail) 문자열 트리플 리스트
        stratify_by: "entity_frequency" | "relation" | "random"
    """
    assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6

    rng = random.Random(seed)

    if stratify_by == "random":
        shuffled = list(triples)
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)
        return (
            shuffled[:n_train],
            shuffled[n_train : n_train + n_valid],
            shuffled[n_train + n_valid :],
        )

    if stratify_by == "relation":
        # 관계별로 균등 분할
        by_rel: dict[str, list] = {}
        for tr in triples:
            by_rel.setdefault(tr[1], []).append(tr)
        train, valid, test = [], [], []
        for rel, ts in by_rel.items():
            rng.shuffle(ts)
            n = len(ts)
            n_tr = int(n * train_ratio)
            n_va = int(n * valid_ratio)
            train.extend(ts[:n_tr])
            valid.extend(ts[n_tr : n_tr + n_va])
            test.extend(ts[n_tr + n_va :])
        rng.shuffle(train)
        rng.shuffle(valid)
        rng.shuffle(test)
        return train, valid, test

    # entity_frequency : 엔티티 빈도 4분위로 stratified
    ent_freq = Counter()
    for h, _, t in triples:
        ent_freq[h] += 1
        ent_freq[t] += 1
    # 빈도 4분위 경계
    freqs = sorted(ent_freq.values())
    n = len(freqs)
    if n == 0:
        return [], [], []
    q1, q2, q3 = freqs[n // 4], freqs[n // 2], freqs[3 * n // 4]

    def bucket(triple) -> int:
        h, _, t = triple
        avg_freq = (ent_freq[h] + ent_freq[t]) / 2
        if avg_freq <= q1:
            return 0
        if avg_freq <= q2:
            return 1
        if avg_freq <= q3:
            return 2
        return 3

    by_bucket: dict[int, list] = {0: [], 1: [], 2: [], 3: []}
    for tr in triples:
        by_bucket[bucket(tr)].append(tr)

    train, valid, test = [], [], []
    for b in by_bucket.values():
        rng.shuffle(b)
        n_b = len(b)
        n_tr = int(n_b * train_ratio)
        n_va = int(n_b * valid_ratio)
        train.extend(b[:n_tr])
        valid.extend(b[n_tr : n_tr + n_va])
        test.extend(b[n_tr + n_va :])

    rng.shuffle(train)
    rng.shuffle(valid)
    rng.shuffle(test)
    return train, valid, test
