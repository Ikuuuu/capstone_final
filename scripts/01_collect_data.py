"""단계 1 : 원본 문서 수집 → data/raw/"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from src.data.collect import collect_law_documents
from src.utils.config import load_config
from src.utils.logging import setup_logger
from src.utils.seed import seed_everything


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg["project"]["seed"])
    logger = setup_logger("kge.collect", cfg["paths"]["logs"])

    logger.info("=" * 60)
    logger.info(f"단계 1 : 데이터 수집 시작 (source={cfg['collect']['source']})")
    logger.info("=" * 60)

    n = collect_law_documents(
        source=cfg["collect"]["source"],
        pdf_dir=cfg["collect"].get("pdf_dir"),
        api_key=cfg["collect"].get("api_key", ""),
        domains=cfg["collect"].get("domains", []),
        max_documents=cfg["collect"].get("max_documents", 100),
        output_dir=cfg["paths"]["data_raw"],
    )
    logger.info(f"수집 완료 : {n} 문서 → {cfg['paths']['data_raw']}")


if __name__ == "__main__":
    main()
