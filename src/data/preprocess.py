"""단계 2 : 원본 → 트리플 추출 → 검증 → 분할 (최고 품질판).

품질 향상 사항:
  - 외부 법령 인용을 placeholder 노드로 모두 KG 에 포함 (B-1 옵션)
  - 다중 부모 지원 (A-3 옵션) - 한 자식이 여러 부모를 가질 수 있음
  - 조문 단위 인용 (양 끝이 corpus 내일 때)
  - 동의어 / 약칭 canonical 매핑으로 노드 중복 방지
  - inverse pair (포함 ↔ 소속, 상위법령 ↔ 하위법령, 위임 ↔ 위임받음)
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from .ontology import OntologySchema
from .pdf_parser import (
    canonical_name,
    extract_citations_from_article,
    extract_delegation_relations,
    infer_hierarchy,
    LawDocument,
)
from .splitter import stratified_split

logger = logging.getLogger("kge.data.preprocess")
Triple = Tuple[str, str, str]


def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    return " ".join(s.split())


def _dict_to_lawdoc(d: Dict[str, Any]) -> LawDocument:
    return LawDocument(
        law_id=d["law_id"],
        title=d["title"],
        short_title=d.get("short_title"),
        doc_type=d["doc_type"],
        doc_type_raw=d.get("doc_type_raw", ""),
        issuing_org=d.get("issuing_org"),
        promulgation_no=d.get("promulgation_no"),
        promulgation_date=d.get("promulgation_date"),
        effective_date=d.get("effective_date"),
        revision_type=d.get("revision_type"),
        jurisdiction=d.get("jurisdiction", []),
        articles=d.get("articles", []),
        addenda=d.get("addenda", []),
        raw_text="",
    )


def load_raw_documents(raw_dir: str | Path) -> List[LawDocument]:
    raw_dir = Path(raw_dir)
    docs: List[LawDocument] = []
    for p in sorted(raw_dir.glob("*.json")):
        with p.open("r", encoding="utf-8") as f:
            docs.append(_dict_to_lawdoc(json.load(f)))
    return docs


def _article_node(law_title: str, article_no: str) -> str:
    return f"{law_title} 제{article_no}조"


def extract_triples_from_documents(
    documents: List[LawDocument],
    include_articles: bool = True,
    include_addenda: bool = False,
    include_jurisdiction: bool = True,
    include_external_citations: bool = True,   # ★ 외부 법령 placeholder
    article_level_internal_citation: bool = True,
) -> List[Triple]:
    triples: List[Triple] = []
    titles_in_corpus = {d.title for d in documents}
    # 외부 법령 placeholder 누적 집합 (canonical 명칭)
    external_laws: Set[str] = set()

    # 1. rdf:type — 법령 단위
    for d in documents:
        triples.append((d.title, "type", d.doc_type))

    # 2. 소관 기관
    if include_jurisdiction:
        for d in documents:
            for org in d.jurisdiction:
                triples.append((d.title, "소관기관", org))
                triples.append((org, "관할법령", d.title))   # inverse pair
                triples.append((org, "type", "Organization"))
            if d.issuing_org and d.issuing_org not in d.jurisdiction:
                triples.append((d.title, "발령기관", d.issuing_org))
                triples.append((d.issuing_org, "type", "Organization"))

    # 3. 상위법령 + 하위법령 (다중 부모 지원)
    for edge in infer_hierarchy(documents):
        triples.append((edge["child"], "상위법령", edge["parent"]))
        triples.append((edge["parent"], "하위법령", edge["child"]))

    # 4. 위임 관계 + inverse
    for edge in extract_delegation_relations(documents):
        triples.append((edge["delegator"], "위임", edge["delegatee"]))
        triples.append((edge["delegatee"], "위임받음", edge["delegator"]))

    # 5. 조문 노드 + 소속 + 인용 관계
    if include_articles:
        for d in documents:
            for art in d.articles:
                art_node = _article_node(d.title, art["no"])
                triples.append((art_node, "type", "Article"))
                triples.append((art_node, "소속", d.title))
                triples.append((d.title, "포함", art_node))
                # 인용 관계
                for c in extract_citations_from_article(art["text"], d.title):
                    target_raw = c["target"]
                    target = canonical_name(target_raw)  # 동의어 → canonical
                    if target == d.title:
                        continue
                    if c["article_no"] and article_level_internal_citation and target in titles_in_corpus:
                        # 조문-조문 인용
                        target_node = _article_node(target, c["article_no"])
                        triples.append((art_node, "인용", target_node))
                    elif target in titles_in_corpus:
                        # 조문 → 법령 (corpus 내)
                        triples.append((art_node, "인용", target))
                    else:
                        # 외부 법령
                        if include_external_citations:
                            triples.append((art_node, "외부인용", target))
                            external_laws.add(target)

    # 6. 외부 법령 placeholder 노드 type 등록
    if include_external_citations:
        for ext in external_laws:
            triples.append((ext, "type", "ExternalLegalDocument"))

    # 7. 부칙
    if include_addenda:
        for d in documents:
            for art in d.addenda:
                art_node = f"{d.title} 부칙 제{art['no']}조"
                triples.append((art_node, "type", "AddendumArticle"))
                triples.append((art_node, "소속부칙", d.title))

    # 정규화 (공백·줄바꿈 처리)
    triples = [(_normalize_text(h), _normalize_text(r), _normalize_text(t)) for (h, r, t) in triples]
    triples = [(h, r, t) for (h, r, t) in triples if h and r and t]
    # 중복 제거
    triples = list(dict.fromkeys(triples))
    return triples


def validate_triples(
    triples: List[Triple],
    schema: OntologySchema,
) -> Tuple[List[Triple], List[Dict]]:
    """타입 일치 검증 + 경고만 반환 (현재는 모두 통과)."""
    entity_types: Dict[str, str] = {}
    for h, r, t in triples:
        if r == "type":
            entity_types[h] = t
    valid: List[Triple] = []
    violations: List[Dict] = []
    for h, r, t in triples:
        valid.append((h, r, t))
    return valid, violations


def build_id_mappings(triples: List[Triple]) -> Tuple[Dict[str, int], Dict[str, int]]:
    ent_counter = Counter()
    rel_counter = Counter()
    for h, r, t in triples:
        ent_counter[h] += 1
        ent_counter[t] += 1
        rel_counter[r] += 1
    entity2id = {e: i for i, (e, _) in enumerate(ent_counter.most_common())}
    relation2id = {r: i for i, (r, _) in enumerate(rel_counter.most_common())}
    return entity2id, relation2id


def save_processed_kg(
    triples_split: Tuple[List, List, List],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, valid, test = triples_split

    def _write_triples(name: str, ts: List[Triple]) -> None:
        with (output_dir / name).open("w", encoding="utf-8") as f:
            for h, r, t in ts:
                if h in entity2id and t in entity2id and r in relation2id:
                    f.write(f"{entity2id[h]}\t{relation2id[r]}\t{entity2id[t]}\n")

    def _write_mapping(name: str, m: Dict[str, int]) -> None:
        with (output_dir / name).open("w", encoding="utf-8") as f:
            for k, v in m.items():
                f.write(f"{k}\t{v}\n")

    _write_triples("train.tsv", train)
    _write_triples("valid.tsv", valid)
    _write_triples("test.tsv", test)
    _write_mapping("entity2id.tsv", entity2id)
    _write_mapping("relation2id.tsv", relation2id)
    logger.info(
        f"KG 저장 — train={len(train)}, valid={len(valid)}, test={len(test)}, "
        f"|E|={len(entity2id)}, |R|={len(relation2id)}"
    )
