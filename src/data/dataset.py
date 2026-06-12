"""KG 트리플 데이터셋 (PyTorch Dataset)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class KGData:
    """KG 데이터의 메모리 표현."""

    train_triples: List[Tuple[int, int, int]]   # (h, r, t)
    valid_triples: List[Tuple[int, int, int]]
    test_triples: List[Tuple[int, int, int]]
    entity2id: Dict[str, int]
    relation2id: Dict[str, int]

    @property
    def num_entities(self) -> int:
        return len(self.entity2id)

    @property
    def num_relations(self) -> int:
        return len(self.relation2id)

    @property
    def all_triples(self) -> List[Tuple[int, int, int]]:
        return self.train_triples + self.valid_triples + self.test_triples


def load_kg(processed_dir: str | Path) -> KGData:
    """data/processed/ 에서 KG 로드.

    기대 파일:
      - train.tsv, valid.tsv, test.tsv : "h\\tr\\tt" 형식 (id)
      - entity2id.tsv, relation2id.tsv : "name\\tid"
    """
    processed_dir = Path(processed_dir)

    def _read_triples(path: Path) -> List[Tuple[int, int, int]]:
        triples: List[Tuple[int, int, int]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    triples.append((int(parts[0]), int(parts[1]), int(parts[2])))
        return triples

    def _read_mapping(path: Path) -> Dict[str, int]:
        mapping: Dict[str, int] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    mapping[parts[0]] = int(parts[1])
        return mapping

    return KGData(
        train_triples=_read_triples(processed_dir / "train.tsv"),
        valid_triples=_read_triples(processed_dir / "valid.tsv"),
        test_triples=_read_triples(processed_dir / "test.tsv"),
        entity2id=_read_mapping(processed_dir / "entity2id.tsv"),
        relation2id=_read_mapping(processed_dir / "relation2id.tsv"),
    )


class TripleDataset(Dataset):
    """학습용 트리플 데이터셋 (negative sampling 포함)."""

    def __init__(
        self,
        triples: List[Tuple[int, int, int]],
        num_entities: int,
        num_negatives: int = 10,
        filter_triples: set | None = None,
    ) -> None:
        self.triples = triples
        self.num_entities = num_entities
        self.num_negatives = num_negatives
        # 부정 샘플링 시 제외할 양성 트리플 집합 (filtered eval 시 사용)
        self.filter_set = filter_triples or set(self.triples)

    def __len__(self) -> int:
        return len(self.triples)

    def __getitem__(self, idx: int):
        h, r, t = self.triples[idx]
        return {
            "h": torch.tensor(h, dtype=torch.long),
            "r": torch.tensor(r, dtype=torch.long),
            "t": torch.tensor(t, dtype=torch.long),
        }

    def sample_negatives(self, h: int, r: int, t: int) -> List[Tuple[int, int, int]]:
        """주어진 양성 트리플에 대해 num_negatives 개의 부정 샘플 생성."""
        import random
        negatives: List[Tuple[int, int, int]] = []
        while len(negatives) < self.num_negatives:
            corrupt_head = random.random() < 0.5
            if corrupt_head:
                h_neg = random.randrange(self.num_entities)
                candidate = (h_neg, r, t)
            else:
                t_neg = random.randrange(self.num_entities)
                candidate = (h, r, t_neg)
            if candidate not in self.filter_set:
                negatives.append(candidate)
        return negatives
