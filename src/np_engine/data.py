"""KG 로드 + 관계 카테고리 정의 (가설별 검증 자산).

졸업논문 가이드라인 §3.1 / §3.2 에 정의된 관계 그룹을 코드로 고정한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

Triple = Tuple[int, int, int]


# ── 관계 이름 (relation2id.tsv 와 일치해야 함) ──
# 가설별 관계 그룹 (이름 기준; id 는 로드시 매핑)
DIRECTIONAL_RELATION_NAMES = [  # H1 : 방향성/비대칭 관계
    "인용", "외부인용", "위임", "위임받음", "소속", "포함",
]
HIERARCHY_RELATION_NAMES = ["상위법령", "하위법령"]  # H2 : 계층 관계
# inverse pair (양방향 관계 쌍) — L_dir / L_attr 검증 자산
INVERSE_PAIR_NAMES = [
    ("소속", "포함"),
    ("위임", "위임받음"),
    ("상위법령", "하위법령"),
]
# 대칭 관계 (방향성 제약에서 제외) — 본 도메인엔 없음
SYMMETRIC_RELATION_NAMES: List[str] = []


@dataclass
class KG:
    train: np.ndarray            # (N_train, 3) int
    valid: np.ndarray
    test: np.ndarray
    ent2id: Dict[str, int]
    rel2id: Dict[str, int]
    id2rel: Dict[int, str] = field(default_factory=dict)
    entity_type: np.ndarray = None      # (num_ent,) int, type 클래스 엔티티 id (없으면 -1)
    num_ent: int = 0
    num_rel: int = 0
    all_triples: Set[Triple] = field(default_factory=set)

    # 관계 그룹 (id 집합)
    directional_rel_ids: Set[int] = field(default_factory=set)
    hierarchy_rel_ids: Set[int] = field(default_factory=set)
    symmetric_rel_ids: Set[int] = field(default_factory=set)
    inverse_pairs: Dict[int, int] = field(default_factory=dict)  # r -> r_inv (양방향)


def _read_triples(p: Path) -> np.ndarray:
    rows = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            a = line.strip().split("\t")
            if len(a) >= 3:
                rows.append((int(a[0]), int(a[1]), int(a[2])))
    return np.asarray(rows, dtype=np.int64)


def _read_map(p: Path) -> Dict[str, int]:
    m: Dict[str, int] = {}
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            a = line.strip().split("\t")
            if len(a) >= 2:
                m[a[0]] = int(a[1])
    return m


def load_kg(processed_dir: str | Path) -> KG:
    d = Path(processed_dir)
    train = _read_triples(d / "train.tsv")
    valid = _read_triples(d / "valid.tsv")
    test = _read_triples(d / "test.tsv")
    ent2id = _read_map(d / "entity2id.tsv")
    rel2id = _read_map(d / "relation2id.tsv")
    id2rel = {v: k for k, v in rel2id.items()}
    num_ent = len(ent2id)
    num_rel = len(rel2id)

    all_triples = set(map(tuple, train.tolist()))
    all_triples |= set(map(tuple, valid.tolist()))
    all_triples |= set(map(tuple, test.tolist()))

    # 관계 그룹 매핑
    directional = {rel2id[n] for n in DIRECTIONAL_RELATION_NAMES if n in rel2id}
    hierarchy = {rel2id[n] for n in HIERARCHY_RELATION_NAMES if n in rel2id}
    symmetric = {rel2id[n] for n in SYMMETRIC_RELATION_NAMES if n in rel2id}
    inverse_pairs: Dict[int, int] = {}
    for a, b in INVERSE_PAIR_NAMES:
        if a in rel2id and b in rel2id:
            inverse_pairs[rel2id[a]] = rel2id[b]
            inverse_pairs[rel2id[b]] = rel2id[a]

    # 엔티티 타입 라벨 추출 : (e, type, class) 트리플의 tail 이 클래스 엔티티
    entity_type = np.full(num_ent, -1, dtype=np.int64)
    type_rel = rel2id.get("type", None)
    if type_rel is not None:
        for h, r, t in train.tolist():
            if r == type_rel:
                entity_type[h] = t

    return KG(
        train=train, valid=valid, test=test,
        ent2id=ent2id, rel2id=rel2id, id2rel=id2rel,
        entity_type=entity_type, num_ent=num_ent, num_rel=num_rel,
        all_triples=all_triples,
        directional_rel_ids=directional, hierarchy_rel_ids=hierarchy,
        symmetric_rel_ids=symmetric, inverse_pairs=inverse_pairs,
    )


def subsample_train(train: np.ndarray, fraction: float, seed: int = 0) -> np.ndarray:
    """H3 강건성 : train 트리플을 fraction 비율로 stratified(관계별) 축소."""
    if fraction >= 1.0:
        return train
    rng = np.random.RandomState(seed)
    keep_idx = []
    rels = train[:, 1]
    for r in np.unique(rels):
        idx = np.where(rels == r)[0]
        n_keep = max(1, int(round(len(idx) * fraction)))
        keep_idx.append(rng.choice(idx, size=n_keep, replace=False))
    keep = np.concatenate(keep_idx)
    rng.shuffle(keep)
    return train[keep]
