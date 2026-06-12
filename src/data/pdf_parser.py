"""국가법령정보센터 PDF 파일 파싱 모듈 (최고 품질판).

기능:
  - 행정규칙·훈령·시행세칙 클래스 지원
  - 다중 부모 추론 (제1조에 여러 모법령 명시 시 모두 부모로)
  - 위임 키워드 확장 + 첫 5조까지 스캔
  - 시행세칙 → 시행규칙 자동 계층
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("kge.data.pdf_parser")


DOC_TYPE_MAP = {
    "법률":             "Law",
    "대통령령":          "EnforcementOrdinance",
    "총리령":            "EnforcementRule",
    "국방부령":          "EnforcementRule",
    "부령":              "EnforcementRule",
    "특례규정":          "SpecialRegulation",
    "국방부훈령":         "AdministrativeOrder",
    "방위사업청훈령":      "AdministrativeOrder",
    "훈령":              "AdministrativeOrder",
    "공동훈령":          "JointAdministrativeOrder",
    "예규":              "AdministrativeRule",
    "고시":              "AdministrativeNotice",
}

DELEGATE_KEYWORDS = [
    "위임된 사항", "위임받은 사항", "위임에 따라",
    "위임에 의하여", "위임에 의한", "에서 위임한", "에 따라",
    "에 근거하여", "규정에 의하여", "에서 위임된",
]


@dataclass
class LawDocument:
    law_id: str
    title: str
    short_title: Optional[str]
    doc_type: str
    doc_type_raw: str
    issuing_org: Optional[str]
    promulgation_no: Optional[str]
    promulgation_date: Optional[str]
    effective_date: Optional[str]
    revision_type: Optional[str]
    jurisdiction: List[str] = field(default_factory=list)
    articles: List[Dict[str, Any]] = field(default_factory=list)
    addenda: List[Dict[str, Any]] = field(default_factory=list)
    raw_text: str = ""


FILENAME_RE = re.compile(
    r"^(?P<title>.+?)\((?P<dtype>[^)]+)\)\((?P<pno>제[0-9]+호)\)\((?P<edate>[0-9]{8})\)\.pdf$"
)


def parse_filename(filename: str) -> Dict[str, Optional[str]]:
    m = FILENAME_RE.match(filename)
    if not m:
        return {"title": Path(filename).stem, "doc_type_raw": None,
                "promulgation_no": None, "effective_date": None, "issuing_org": None}
    dtype_raw = m.group("dtype").strip()
    issuing_org = None
    for org in ["국방부", "방위사업청", "산업통상자원부", "총리실"]:
        if dtype_raw.startswith(org):
            issuing_org = org
            break
    if dtype_raw == "공동훈령":
        issuing_org = "공동"
    return {
        "title":           m.group("title").strip(),
        "doc_type_raw":    dtype_raw,
        "promulgation_no": m.group("pno"),
        "effective_date":  m.group("edate"),
        "issuing_org":     issuing_org,
    }


def normalize_doc_type(dtype_raw: str) -> str:
    if not dtype_raw:
        return "LegalDocument"
    keys_sorted = sorted(DOC_TYPE_MAP.keys(), key=len, reverse=True)
    for k in keys_sorted:
        if dtype_raw == k or dtype_raw.endswith(k):
            return DOC_TYPE_MAP[k]
    return "LegalDocument"


META_RE = re.compile(
    r"\[\s*시행\s+([0-9.\s]+?)\s*\.\s*\]\s*"
    r"\[\s*(?P<dtype>[^\s\d]+)\s+(?P<pno>제[0-9]+호)\s*,\s*"
    r"([0-9.\s]+?)\s*\.\s*,\s*(?P<rtype>[^]]+)\]"
)
SHORT_TITLE_RE = re.compile(r"약칭\s*:\s*([^)\n]+)")
JURISDICTION_RE = re.compile(
    r"^(?P<org>[가-힣]+(?:부|청|위원회|원|처|실|총리실|위원장))\s*\("
)


def _normalize_date(s: str) -> Optional[str]:
    if not s:
        return None
    parts = re.findall(r"\d+", s)
    if len(parts) >= 3:
        y, m, d = parts[0], parts[1].zfill(2), parts[2].zfill(2)
        return f"{y}{m}{d}"
    return None


def parse_metadata_block(text: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "short_title": None, "effective_date_meta": None,
        "promulgation_date": None, "doc_type_meta": None,
        "promulgation_no_meta": None, "revision_type": None,
        "jurisdiction": [],
    }
    head = text[:3000]
    m = SHORT_TITLE_RE.search(head)
    if m:
        meta["short_title"] = m.group(1).strip()
    m = META_RE.search(head)
    if m:
        meta["effective_date_meta"] = _normalize_date(m.group(1))
        meta["doc_type_meta"] = m.group("dtype")
        meta["promulgation_no_meta"] = m.group("pno")
        meta["promulgation_date"] = _normalize_date(m.group(3))
        meta["revision_type"] = m.group("rtype").strip()
    seen = set()
    for line in head.splitlines():
        line = line.strip()
        m = JURISDICTION_RE.match(line)
        if m:
            org = m.group("org").strip()
            if org and org not in seen and len(org) <= 12:
                seen.add(org)
                meta["jurisdiction"].append(org)
    return meta


ADDENDUM_HEADER_RE = re.compile(r"^\s*부\s*칙\s*(?:\(.*?\))?\s*$", re.MULTILINE)


def split_main_and_addenda(text: str) -> tuple[str, str]:
    matches = list(ADDENDUM_HEADER_RE.finditer(text))
    if not matches:
        return text, ""
    first = matches[0]
    return text[: first.start()], text[first.start():]


def parse_articles(text: str) -> List[Dict[str, Any]]:
    headers = list(re.finditer(
        r"^제(\d+(?:의\d+)?)\s*조(?:\s*\(([^)]*)\))?",
        text, re.MULTILINE,
    ))
    articles: List[Dict[str, Any]] = []
    seen_no = set()
    for i, h in enumerate(headers):
        no = h.group(1)
        if no in seen_no:
            continue
        seen_no.add(no)
        start = h.start()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[start:end].strip()
        body_lines = body.split("\n", 1)
        full_body = body_lines[1].strip() if len(body_lines) > 1 else ""
        articles.append({
            "no": no,
            "title": (h.group(2) or "").strip(),
            "text": full_body,
        })
    return articles


CITATION_RE = re.compile(r"「\s*([^」]+?)\s*」(?:\s*제(\d+(?:의\d+)?)\s*조)?")


def extract_citations_from_article(article_text: str, self_title: str) -> List[Dict[str, str]]:
    citations: List[Dict[str, str]] = []
    seen = set()
    for m in CITATION_RE.finditer(article_text):
        target = m.group(1).strip()
        art_no = m.group(2) or ""
        if target == self_title or target.endswith(self_title):
            continue
        key = (target, art_no)
        if key in seen:
            continue
        seen.add(key)
        citations.append({"target": target, "article_no": art_no, "kind": "citation"})
    return citations


def extract_parents_from_purpose(doc: "LawDocument") -> List[str]:
    """다중 부모 추출 : 제1~5조 안에서 위임 키워드 + 「」 인용을 모두 부모 후보로.

    반환 : 부모 후보 리스트 (중복 제거)
    """
    parents: List[str] = []
    seen: Set[str] = set()
    for art in doc.articles[:5]:
        body = art.get("text", "") + " " + (art.get("title") or "")
        # 위임 키워드가 포함된 경우 우선 처리
        has_delegate = any(kw in body for kw in DELEGATE_KEYWORDS)
        for m in CITATION_RE.finditer(body):
            target = m.group(1).strip()
            if target == doc.title or target in seen:
                continue
            # 위임 키워드가 본문에 있거나 art_no=1 (목적) 인 경우 부모로 인정
            if has_delegate or art.get("no") == "1":
                seen.add(target)
                parents.append(target)
    return parents


def extract_delegation_relations(documents: List["LawDocument"]) -> List[Dict[str, str]]:
    """위임 관계 추출 (corpus 내 매칭만)."""
    titles = {d.title: d for d in documents}
    edges: List[Dict[str, str]] = []
    seen = set()
    for d in documents:
        candidates = extract_parents_from_purpose(d)
        for parent in candidates:
            if parent not in titles or parent == d.title:
                continue
            key = (parent, d.title)
            if key in seen:
                continue
            seen.add(key)
            edges.append({"delegator": parent, "delegatee": d.title, "kind": "위임"})
    return edges


# ───── 동의어/약칭 canonical 매핑 ─────
SYNONYM_MAP: Dict[str, str] = {
    # 약칭 → 정식 명칭
    "국가계약법":        "국가를 당사자로 하는 계약에 관한 법률",
    "지방계약법":        "지방자치단체를 당사자로 하는 계약에 관한 법률",
    "대외무역법":        "대외무역법",  # 자기 자신
    "공직자윤리법":       "공직자윤리법",
    "방위사업법":        "방위사업법",
    # 약칭 + 시행령/시행규칙
    "국가계약법 시행령":   "국가를 당사자로 하는 계약에 관한 법률 시행령",
    "국가계약법 시행규칙": "국가를 당사자로 하는 계약에 관한 법률 시행규칙",
}


def canonical_name(raw_name: str) -> str:
    """동의어/약칭 → 정식 명칭으로 정규화."""
    raw_name = raw_name.strip()
    return SYNONYM_MAP.get(raw_name, raw_name)


def infer_hierarchy(documents: List["LawDocument"]) -> List[Dict[str, str]]:
    """4단 계층 추론 + 다중 부모 + 시행세칙 → 시행규칙 규칙."""
    titles = {d.title: d for d in documents}
    edges: List[Dict[str, str]] = []
    seen = set()

    def _add(child: str, parent: str):
        key = (child, parent)
        if parent in titles and parent != child and key not in seen:
            seen.add(key)
            edges.append({"child": child, "parent": parent, "kind": "상위법령"})

    # 1단계 : 파일명 suffix 규칙 (시행령/시행규칙/시행세칙)
    for d in documents:
        added = False
        for suffix in [" 시행세칙", " 시행규칙", " 시행령"]:
            if d.title.endswith(suffix):
                parent = d.title[: -len(suffix)]
                if parent in titles:
                    _add(d.title, parent)
                    added = True
                    break
        # 2단계 : 시행세칙 추가 매핑 ─ corpus 내 시행규칙 후보 탐색
        if not added and "시행세칙" in d.title:
            # "방산원가대상물자의 원가계산에 관한 시행세칙" → "방산원가대상물자의 원가계산에 관한 규칙"
            for cand in titles:
                if cand == d.title:
                    continue
                # 단어 prefix 일치 (앞 단어 50% 이상)
                tok_d = d.title.replace("시행세칙", "").strip().split()
                tok_c = cand.split()
                if len(tok_d) >= 2 and len(tok_c) >= 2 and tok_d[:2] == tok_c[:2]:
                    if cand.endswith("규칙") or cand.endswith("시행규칙"):
                        _add(d.title, cand)
                        added = True
                        break
        if added:
            continue
        # 3단계 : 행정규칙 → 본문 제1조 다중 부모 추론
        if d.doc_type in {"AdministrativeOrder", "AdministrativeRule",
                          "AdministrativeNotice", "JointAdministrativeOrder",
                          "SpecialRegulation"}:
            for parent in extract_parents_from_purpose(d):
                if parent in titles:
                    _add(d.title, parent)
    return edges


def read_pdf_text(pdf_path: str | Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ImportError("pypdf 설치 필요 : pip install pypdf") from e
    reader = PdfReader(str(pdf_path))
    chunks: List[str] = []
    for p in reader.pages:
        try:
            chunks.append(p.extract_text() or "")
        except Exception as e:
            logger.warning(f"page 추출 실패 ({Path(pdf_path).name}): {e}")
    return "\n".join(chunks)


def parse_pdf(pdf_path: str | Path) -> LawDocument:
    pdf_path = Path(pdf_path)
    fn_meta = parse_filename(pdf_path.name)
    raw_text = read_pdf_text(pdf_path)
    body_meta = parse_metadata_block(raw_text)
    doc_type_raw = fn_meta.get("doc_type_raw") or body_meta.get("doc_type_meta") or ""
    doc_type = normalize_doc_type(doc_type_raw)
    title = fn_meta.get("title") or pdf_path.stem
    main_text, addendum_text = split_main_and_addenda(raw_text)
    return LawDocument(
        law_id=pdf_path.stem,
        title=title,
        short_title=body_meta.get("short_title"),
        doc_type=doc_type,
        doc_type_raw=doc_type_raw,
        issuing_org=fn_meta.get("issuing_org"),
        promulgation_no=fn_meta.get("promulgation_no") or body_meta.get("promulgation_no_meta"),
        promulgation_date=body_meta.get("promulgation_date"),
        effective_date=fn_meta.get("effective_date") or body_meta.get("effective_date_meta"),
        revision_type=body_meta.get("revision_type"),
        jurisdiction=body_meta.get("jurisdiction", []),
        articles=parse_articles(main_text),
        addenda=parse_articles(addendum_text) if addendum_text else [],
        raw_text=raw_text,
    )


def parse_directory(pdf_dir: str | Path) -> List[LawDocument]:
    pdf_dir = Path(pdf_dir)
    documents: List[LawDocument] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        try:
            doc = parse_pdf(pdf_path)
            documents.append(doc)
        except Exception as e:
            logger.error(f"  ✗ {pdf_path.name} 실패 : {e}")
    return documents
