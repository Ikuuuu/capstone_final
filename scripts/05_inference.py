"""단계 5 : 학습된 모델로 실제 질의 추론 (의미 검색 / 관계 예측)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from pathlib import Path

from src.data.dataset import load_kg
from src.inference.predictor import Predictor
from src.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ours.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--query", default=None, help="유사 엔티티를 찾을 질의 엔티티")
    ap.add_argument("--head", default=None, help="(head, relation, ?) 예측의 head")
    ap.add_argument("--relation", default=None, help="(head, relation, ?) 예측의 relation")
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    cfg = load_config(args.config)
    kg = load_kg(cfg["paths"]["data_processed"])
    ckpt = args.checkpoint or str(
        Path(cfg["paths"]["checkpoints"]) / cfg["experiment"]["name"] / "best.pt"
    )

    predictor = Predictor(
        checkpoint_path=ckpt,
        entity2id=kg.entity2id,
        relation2id=kg.relation2id,
        model_name=cfg["model"]["name"],
        embedding_dim=cfg["model"]["embedding_dim"],
    )

    if args.query:
        print(f"\n=== 질의 : '{args.query}' 와 의미적으로 가까운 엔티티 Top-{args.top_k} ===")
        for ent, dist in predictor.find_similar_entities(args.query, top_k=args.top_k):
            print(f"  {ent:40s}  (distance={dist:.4f})")

    if args.head and args.relation:
        print(f"\n=== ({args.head}, {args.relation}, ?) Top-{args.top_k} 예측 ===")
        for ent, score in predictor.predict_tail(args.head, args.relation, top_k=args.top_k):
            print(f"  {ent:40s}  (score={score:.4f})")


if __name__ == "__main__":
    main()
