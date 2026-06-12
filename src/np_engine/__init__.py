"""NumPy 기반 경량 KGE 실험 엔진.

PyTorch 설치가 차단된 폐쇄망/샌드박스 환경에서도 졸업논문 흐름 가이드라인
(§3 실험 설계) 의 전체 실험을 **실제로 실행**할 수 있도록 만든 자립형 엔진.

- 모델 : TransE, RotatE, DistMult, ComplEx, TransO, Ours (+ ablation)
- 손실 : L_struct, L_dir(기여①), L_hier_attn(기여②), L_attr, L_type
- 평가 : Filtered MRR / Hits@1/3/10 (전체·관계별·방향성관계별)
- 통계 : paired bootstrap (H1/H2/H3 가설 자동 판정)

참조용 PyTorch 구현은 src/ 의 나머지 모듈에 있으며, 본 엔진은 그와
동일한 손실 정의를 NumPy 로 재현한 것이다 (수식 일치, 구현만 경량화).
"""
from __future__ import annotations

__all__ = ["data", "models", "train", "evaluate", "stats"]
