"""단계 1 : 외부 자료원에서 원본 문서 수집 → data/raw/

지원 모드 :
  - pdf_dir   : 로컬 PDF 디렉토리 (국가법령정보센터에서 직접 다운로드한 파일)
  - api       : OpenAPI 호출 (구현 예정)
  - demo      : 합성 데모 데이터
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from .pdf_parser import parse_directory

logger = logging.getLogger("kge.data.collect")


def _resolve_env(value: str) -> str:
    """${oc.env:VAR,DEFAULT} 패턴을 환경변수로 해석."""
    if not isinstance(value, str):
        return value
    m = re.match(r"\$\{oc\.env:([A-Z_]+)(?:,([^}]+))?\}", value)
    if not m:
        return value
    var, default = m.group(1), m.group(2) or ""
    return os.environ.get(var, default).strip()


def _doc_to_dict(doc) -> Dict[str, Any]:
    """LawDocument dataclass → JSON 직렬화 가능한 dict (raw_text 제외)."""
    d = asdict(doc)
    d.pop("raw_text", None)
    return d


def collect_from_pdf_directory(
    pdf_dir: str | Path,
    output_dir: str | Path,
) -> int:
    """로컬 PDF 디렉토리에서 법령 파싱 → data/raw/ 에 JSON 으로 저장."""
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF 디렉토리 없음: {pdf_dir}")

    logger.info(f"PDF 디렉토리 파싱: {pdf_dir}")
    documents = parse_directory(pdf_dir)

    for doc in documents:
        out_path = output_dir / f"{doc.law_id}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(_doc_to_dict(doc), f, ensure_ascii=False, indent=2)
    logger.info(f"저장 완료: {len(documents)} 개 → {output_dir}")
    return len(documents)


def collect_law_documents(
    source: str = "demo",
    pdf_dir: str | Path | None = None,
    api_key: str = "",
    domains: List[str] | None = None,
    max_documents: int = 100,
    output_dir: str | Path = "data/raw",
) -> int:
    """통합 진입점.

    Args:
        source: "pdf_dir" | "api" | "demo"
        pdf_dir: source=pdf_dir 일 때의 PDF 디렉토리 경로
        api_key: source=api 일 때의 OpenAPI 키
    """
    if source == "pdf_dir":
        if not pdf_dir:
            raise ValueError("source=pdf_dir 인 경우 pdf_dir 인자 필요")
        return collect_from_pdf_directory(pdf_dir, output_dir)

    if source == "demo" or _resolve_env(api_key) in ("", "DEMO_KEY"):
        return _collect_demo(output_dir)

    # TODO : 실제 OpenAPI 호출 구현
    raise NotImplementedError(
        "source=api 모드는 Phase 1 본격 구현 시 채워 넣을 것 "
        "(https://www.law.go.kr/DRF/lawService.do)"
    )


def _collect_demo(output_dir: str | Path) -> int:
    """합성 데모 데이터."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_docs = [
        {"law_id": "demo_001", "title": "지방자치법", "doc_type": "Law",
         "doc_type_raw": "법률", "promulgation_no": None, "promulgation_date": None,
         "effective_date": None, "revision_type": None, "short_title": None,
         "jurisdiction": ["행정안전부"],
         "articles": [{"no": "15", "title": "조례", "text": "지방자치단체는 …"}],
         "addenda": []},
    ]
    for doc in demo_docs:
        with (output_dir / f"{doc['law_id']}.json").open("w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
    return len(demo_docs)
