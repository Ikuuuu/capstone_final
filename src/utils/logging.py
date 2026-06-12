"""로깅 설정 유틸."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "kge",
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """파일 + 콘솔 동시 로깅 설정."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger
