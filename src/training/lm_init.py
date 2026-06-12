"""한국어 사전학습 LM 으로 엔티티 임베딩 초기화 (보강 장치).

엔티티 명칭·정의문을 LM 에 입력 → [CLS] 표현을 d 차원으로 투영하여 e_init 으로 사용.
관계 r 은 정규분포로 초기화 (변경 없음).
"""
from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger("kge.training.lm_init")


def build_lm_entity_embeddings(
    entity2id: Dict[str, int],
    embedding_dim: int,
    lm_name: str = "klue/roberta-base",
    device: str = "cpu",
) -> torch.Tensor:
    """엔티티 명칭을 LM 에 통과시켜 (|E|, d) 임베딩 행렬 생성.

    실제 구현 시 :  엔티티 정의문(설명) 까지 함께 입력하면 더 좋음.
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        logger.warning("transformers 미설치 - random init 으로 fallback")
        return torch.randn(len(entity2id), embedding_dim)

    logger.info(f"Loading LM: {lm_name}")
    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    model = AutoModel.from_pretrained(lm_name).to(device).eval()

    lm_hidden = model.config.hidden_size
    projection = nn.Linear(lm_hidden, embedding_dim).to(device)

    embeddings = torch.zeros(len(entity2id), embedding_dim)

    # 엔티티 명을 ID 순으로 배치
    entities_sorted = sorted(entity2id.items(), key=lambda x: x[1])

    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(entities_sorted), batch_size):
            batch = entities_sorted[i : i + batch_size]
            texts = [name for name, _ in batch]
            ids = [eid for _, eid in batch]
            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            ).to(device)
            outputs = model(**enc)
            cls = outputs.last_hidden_state[:, 0, :]  # [CLS]
            projected = projection(cls).cpu()
            for j, eid in enumerate(ids):
                embeddings[eid] = projected[j]

    return embeddings
