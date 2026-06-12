from .base import BaseKGEModel
from .transe import TransE
from .ours import OursOntologyKGE

__all__ = ["BaseKGEModel", "TransE", "OursOntologyKGE"]


def build_model(name: str, **kwargs):
    """이름으로 모델 생성 (config 에서 사용)."""
    name = name.lower()
    if name == "transe":
        return TransE(**kwargs)
    if name == "ours":
        return OursOntologyKGE(**kwargs)
    raise ValueError(f"Unknown model: {name}. (transo/transc/rotate/rgcn/compgcn 은 Phase 2 에서 추가 구현)")
