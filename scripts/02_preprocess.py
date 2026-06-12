"""단계 2 : 트리플 추출 + 분할 (최고 품질 옵션 활성화)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse, json
from src.data.ontology import OntologySchema, load_from_owl
from src.data.preprocess import (
    build_id_mappings, extract_triples_from_documents, load_raw_documents,
    save_processed_kg, validate_triples,
)
from src.data.splitter import stratified_split
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["project"]["seed"])
    logger = setup_logger("kge.preprocess", cfg["paths"]["logs"])

    logger.info("=" * 60)
    logger.info("단계 2 : 전처리 + 트리플 추출 + 분할 시작 (최고 품질 옵션)")
    logger.info("=" * 60)

    schema_path = Path(cfg["paths"]["ontology"])
    schema = load_from_owl(schema_path) if schema_path.exists() else OntologySchema()

    documents = load_raw_documents(cfg["paths"]["data_raw"])
    logger.info(f"raw 문서 : {len(documents)} 건")

    ext_cfg = cfg["preprocess"]["triple_extraction"]
    triples = extract_triples_from_documents(
        documents,
        include_articles=ext_cfg["include_articles"],
        include_addenda=ext_cfg["include_addenda"],
        include_jurisdiction=ext_cfg["include_jurisdiction"],
        include_external_citations=ext_cfg.get("include_external_citations", True),
        article_level_internal_citation=ext_cfg.get("article_level_internal_citation", True),
    )
    logger.info(f"추출 트리플 (중복 제거 후) : {len(triples)}")

    valid_triples, violations = validate_triples(triples, schema)
    logger.info(f"검증 통과 : {len(valid_triples)}, 위반 : {len(violations)}")

    interim_dir = Path(cfg["paths"]["data_interim"])
    interim_dir.mkdir(parents=True, exist_ok=True)
    with (interim_dir / "triples_all.json").open("w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False, indent=2)

    entity2id, relation2id = build_id_mappings(valid_triples)
    split = stratified_split(
        valid_triples,
        train_ratio=cfg["preprocess"]["split"]["train"],
        valid_ratio=cfg["preprocess"]["split"]["valid"],
        test_ratio=cfg["preprocess"]["split"]["test"],
        seed=cfg["project"]["seed"],
        stratify_by=cfg["preprocess"]["split"]["stratified_by"],
    )
    save_processed_kg(split, entity2id, relation2id, cfg["paths"]["data_processed"])

    from collections import Counter
    rel_dist = Counter(r for _, r, _ in valid_triples)
    logger.info("관계별 트리플 분포 :")
    for r, c in rel_dist.most_common():
        logger.info(f"  {r:18s} = {c}")
    logger.info("전처리 완료")


if __name__ == "__main__":
    main()
