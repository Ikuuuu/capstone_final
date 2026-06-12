"""단계 4 : 학습된 모델 평가 (링크 예측 + 의미 검색 + 강건성)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
from pathlib import Path

import torch

from src.data.dataset import load_kg
from src.evaluation.link_prediction import evaluate_link_prediction
from src.evaluation.semantic_search import evaluate_semantic_search
from src.models import build_model
from src.utils.checkpoint import load_checkpoint
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ours.yaml")
    ap.add_argument("--checkpoint", default=None,
                    help="기본 : checkpoints/<experiment_name>/best.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["project"]["seed"])
    logger = setup_logger("kge.evaluate", cfg["paths"]["logs"])

    logger.info("=" * 60)
    logger.info(f"단계 4 : 평가 시작 — {cfg['experiment']['name']}")
    logger.info("=" * 60)

    kg = load_kg(cfg["paths"]["data_processed"])
    device = cfg["project"]["device"] if torch.cuda.is_available() else "cpu"

    model_kwargs = dict(
        num_entities=kg.num_entities,
        num_relations=kg.num_relations,
        embedding_dim=cfg["model"]["embedding_dim"],
    )
    if cfg["model"]["name"] == "ours":
        model_kwargs["num_classes"] = 10
    model = build_model(cfg["model"]["name"], **model_kwargs).to(device)

    ckpt = args.checkpoint or str(
        Path(cfg["paths"]["checkpoints"]) / cfg["experiment"]["name"] / "best.pt"
    )
    load_checkpoint(ckpt, model, map_location=device)
    logger.info(f"체크포인트 로드 : {ckpt}")

    # 1. 링크 예측
    lp_metrics = evaluate_link_prediction(
        model=model,
        eval_triples=kg.test_triples,
        num_entities=kg.num_entities,
        filter_set=set(kg.all_triples),
        device=device,
    )
    logger.info(f"링크 예측 : {lp_metrics}")

    # 2. 의미 유사 검색
    eval_path = Path(cfg["paths"]["data_eval"]) / "semantic_eval.json"
    if eval_path.exists():
        sem = evaluate_semantic_search(
            model=model,
            eval_path=eval_path,
            entity2id=kg.entity2id,
            top_k=cfg["evaluation"]["semantic_search"]["top_k"],
            device=device,
        )
        logger.info(f"의미 검색 : {sem}")
    else:
        sem = {}
        logger.warning(f"의미 평가셋 없음 ({eval_path}) — 정성 평가 생략")

    # 결과 저장
    exp_dir = Path(cfg["paths"]["experiments"]) / cfg["experiment"]["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    with (exp_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"link_prediction": lp_metrics, "semantic": sem}, f, ensure_ascii=False, indent=2)
    logger.info(f"결과 저장 : {exp_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
