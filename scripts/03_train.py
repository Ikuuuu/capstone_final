"""단계 3 : KGE 모델 학습."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

import torch

from src.data.dataset import load_kg
from src.data.ontology import OntologySchema, load_from_owl
from src.models import build_model
from src.training.lm_init import build_lm_entity_embeddings
from src.training.trainer import Trainer
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ours.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["project"]["seed"])
    logger = setup_logger("kge.train", cfg["paths"]["logs"])

    logger.info("=" * 60)
    logger.info(f"단계 3 : 학습 시작 — {cfg['experiment']['name']}")
    logger.info("=" * 60)

    # KG · 온톨로지 로드
    kg = load_kg(cfg["paths"]["data_processed"])
    schema_path = Path(cfg["paths"]["ontology"])
    schema = load_from_owl(schema_path) if schema_path.exists() else OntologySchema()
    logger.info(f"KG : |E|={kg.num_entities}, |R|={kg.num_relations}, "
                f"train={len(kg.train_triples)}, valid={len(kg.valid_triples)}, "
                f"test={len(kg.test_triples)}")

    # 모델 생성
    model_kwargs = dict(
        num_entities=kg.num_entities,
        num_relations=kg.num_relations,
        embedding_dim=cfg["model"]["embedding_dim"],
    )
    if cfg["model"]["name"] == "ours":
        # 본 연구 : 클래스 수 = (스키마에 있으면 거기서, 없으면 기본 10)
        num_classes = max(10, len(set(schema.entity_types.values())))
        model_kwargs["num_classes"] = num_classes
    model = build_model(cfg["model"]["name"], **model_kwargs)

    # LM 초기화 (보강 장치)
    device = cfg["project"]["device"] if torch.cuda.is_available() else "cpu"
    if cfg["model"].get("lm_init", False):
        logger.info("LM 임베딩 초기화 적용 (보강 장치)")
        lm_embs = build_lm_entity_embeddings(
            entity2id=kg.entity2id,
            embedding_dim=cfg["model"]["embedding_dim"],
            lm_name=cfg["model"].get("lm_name", "klue/roberta-base"),
            device=device,
        )
        model.init_with_lm(lm_embs)

    # 학습
    trainer = Trainer(model=model, kg=kg, schema=schema, cfg=cfg, device=device)
    result = trainer.train()
    logger.info(f"완료 : {result}")


if __name__ == "__main__":
    main()
