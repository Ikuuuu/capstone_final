"""OWL Lite 수준 온톨로지 스키마 정의 및 로드 유틸.

본 연구는 다음 OWL Lite 구성요소를 활용:
- rdf:type, rdfs:subClassOf, rdfs:domain, rdfs:range
- owl:inverseOf, owl:TransitiveProperty, owl:SymmetricProperty, owl:FunctionalProperty
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set


@dataclass
class OntologySchema:
    """OWL Lite 수준 온톨로지 스키마 (메모리 표현)."""

    # 엔티티 → 타입 매핑 (rdf:type)
    entity_types: Dict[str, str] = field(default_factory=dict)
    # 타입 계층 (자식 → 부모, rdfs:subClassOf)
    type_hierarchy: Dict[str, str] = field(default_factory=dict)
    # 관계별 domain/range (rdfs:domain, rdfs:range)
    relation_domain: Dict[str, str] = field(default_factory=dict)
    relation_range: Dict[str, str] = field(default_factory=dict)
    # 관계 속성
    symmetric_relations: Set[str] = field(default_factory=set)
    transitive_relations: Set[str] = field(default_factory=set)
    functional_relations: Set[str] = field(default_factory=set)
    inverse_pairs: Dict[str, str] = field(default_factory=dict)  # r → r_inv

    def is_symmetric(self, relation: str) -> bool:
        return relation in self.symmetric_relations

    def get_inverse(self, relation: str) -> str | None:
        return self.inverse_pairs.get(relation)

    def get_type(self, entity: str) -> str | None:
        return self.entity_types.get(entity)

    def get_parent_type(self, type_name: str) -> str | None:
        return self.type_hierarchy.get(type_name)

    def get_ancestors(self, type_name: str) -> List[str]:
        """주어진 타입의 모든 상위 타입을 반환 (자기 자신 제외)."""
        ancestors: List[str] = []
        cur = self.type_hierarchy.get(type_name)
        while cur is not None:
            if cur in ancestors:
                break
            ancestors.append(cur)
            cur = self.type_hierarchy.get(cur)
        return ancestors


def load_from_owl(path: str | Path) -> OntologySchema:
    """OWL 파일을 파싱하여 OntologySchema 로 변환.

    TODO : rdflib 활용한 본 파싱 구현
    현재는 JSON 형식의 간이 스키마도 받을 수 있도록 지원.
    """
    import json

    path = Path(path)
    schema = OntologySchema()

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        schema.entity_types = data.get("entity_types", {})
        schema.type_hierarchy = data.get("type_hierarchy", {})
        schema.relation_domain = data.get("relation_domain", {})
        schema.relation_range = data.get("relation_range", {})
        schema.symmetric_relations = set(data.get("symmetric_relations", []))
        schema.transitive_relations = set(data.get("transitive_relations", []))
        schema.functional_relations = set(data.get("functional_relations", []))
        schema.inverse_pairs = data.get("inverse_pairs", {})
        return schema

    # OWL 파싱 (rdflib)
    try:
        import rdflib
        from rdflib.namespace import OWL, RDF, RDFS
    except ImportError as e:
        raise ImportError("rdflib 설치 필요 : pip install rdflib") from e

    g = rdflib.Graph()
    g.parse(str(path))

    # rdf:type
    for s, _, o in g.triples((None, RDF.type, None)):
        if isinstance(s, rdflib.URIRef) and isinstance(o, rdflib.URIRef):
            schema.entity_types[str(s)] = str(o)
    # rdfs:subClassOf
    for s, _, o in g.triples((None, RDFS.subClassOf, None)):
        schema.type_hierarchy[str(s)] = str(o)
    # rdfs:domain, range
    for s, _, o in g.triples((None, RDFS.domain, None)):
        schema.relation_domain[str(s)] = str(o)
    for s, _, o in g.triples((None, RDFS.range, None)):
        schema.relation_range[str(s)] = str(o)
    # owl 속성
    for s in g.subjects(RDF.type, OWL.SymmetricProperty):
        schema.symmetric_relations.add(str(s))
    for s in g.subjects(RDF.type, OWL.TransitiveProperty):
        schema.transitive_relations.add(str(s))
    for s in g.subjects(RDF.type, OWL.FunctionalProperty):
        schema.functional_relations.add(str(s))
    for s, _, o in g.triples((None, OWL.inverseOf, None)):
        schema.inverse_pairs[str(s)] = str(o)

    return schema
