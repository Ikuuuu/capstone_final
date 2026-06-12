# Ontology-Integrated Knowledge Graph Embedding (석사 졸업논문 프로젝트)

> **연구 주제** : Graph Embedding을 활용한 온톨로지 기반 의미 관계 검색 최적화  
> **부제** : 관계 방향성 제약(L_dir) + 적응형 계층 가중치(L_hier_attn)를 통한 KGE 학습 방법론 개선  
> **작성** : 송재익 (2025254013, 충북대학교 산업인공지능학과)

## 1. 프로젝트 구조

```
ontology_kge_thesis/
├── configs/                # 실험 설정 YAML
│   ├── default.yaml
│   ├── ours.yaml
│   └── baselines/          # 베이스라인별 설정
├── src/
│   ├── data/               # 데이터 수집·전처리·온톨로지
│   │   ├── collect.py      # ① 국가법령정보 API 수집
│   │   ├── preprocess.py   # ② LLM 트리플 자동 추출 + 검증
│   │   ├── ontology.py     # OWL Lite 스키마 정의
│   │   ├── dataset.py      # PyTorch Dataset
│   │   └── splitter.py     # stratified train/valid/test 분할
│   ├── losses/             # 4개 손실 함수
│   │   ├── struct.py       # L_struct (TransE margin-based)
│   │   ├── direction.py    # L_dir  (관계 방향성 제약, 기여 ①)
│   │   ├── hier_attention.py  # L_hier_attn  (Attention 계층, 기여 ②)
│   │   └── ontology.py     # L_type, L_attr
│   ├── models/             # KGE 모델 (베이스라인 + 본 연구)
│   │   ├── base.py         # 공통 베이스
│   │   ├── transe.py       # 베이스라인
│   │   └── ours.py         # 본 연구 모델
│   ├── training/           # 학습 관련
│   │   ├── trainer.py      # 메인 학습 루프
│   │   ├── lm_init.py      # 한국어 LM 임베딩 초기화
│   │   └── curriculum.py   # λ curriculum 전략
│   ├── evaluation/         # 평가
│   │   ├── link_prediction.py  # MRR, Hits@1/3/10
│   │   ├── semantic_search.py  # Top-k Precision/Recall
│   │   ├── robustness.py       # 데이터 축소 강건성
│   │   └── statistical.py      # paired bootstrap
│   ├── inference/          # 추론 API
│   │   └── predictor.py
│   └── utils/              # 공통 유틸 (설정·시드·로깅·체크포인트)
├── scripts/                # 단계별 실행 스크립트
│   ├── 01_collect_data.py    # 단계 1 : 데이터 수집
│   ├── 02_preprocess.py      # 단계 2 : 전처리·트리플 추출·분할
│   ├── 03_train.py           # 단계 3 : 학습
│   ├── 04_evaluate.py        # 단계 4 : 평가
│   ├── 05_inference.py       # 단계 5 : 추론
│   └── run_ablation.py       # Ablation Study 실행
├── data/                   # 데이터 (단계별 분리)
│   ├── raw/                # 원본 (API 응답 등)
│   ├── interim/            # 중간 (추출된 트리플)
│   ├── processed/          # 최종 KG (train/valid/test)
│   └── eval/               # 정성 평가셋 (동의어·약어·인용)
├── ontology/               # OWL 스키마 파일
├── experiments/            # 실험 결과 (run별 폴더)
├── checkpoints/            # 모델 체크포인트
├── logs/                   # 학습/평가 로그
├── notebooks/              # 탐색용 Jupyter
└── tests/                  # 단위 테스트
```

## 2. 단계별 실행

전체 파이프라인은 **5단계로 분리**되어 있어 단계별 재현·디버깅이 쉽습니다.

```bash
# 단계 1 : 데이터 수집 (국가법령정보 API)
python scripts/01_collect_data.py --config configs/default.yaml

# 단계 2 : 전처리 (LLM 추출 + 자동 정합성 검증 + stratified 분할)
python scripts/02_preprocess.py --config configs/default.yaml

# 단계 3 : 학습 (본 연구 모델)
python scripts/03_train.py --config configs/ours.yaml

# 단계 3b : 베이스라인 학습
python scripts/03_train.py --config configs/baselines/transe.yaml

# 단계 4 : 평가 (링크 예측 + 의미 검색 + 강건성)
python scripts/04_evaluate.py --config configs/ours.yaml --checkpoint checkpoints/ours/best.pt

# 단계 5 : 추론 (질의 → Top-k 결과)
python scripts/05_inference.py --checkpoint checkpoints/ours/best.pt --query "지방자치법 제15조"

# Ablation Study (한 번에)
python scripts/run_ablation.py --config configs/ours.yaml
```

또는 `make` 로 한 번에:

```bash
make data       # 단계 1 + 2
make train      # 단계 3 (본 연구 + 베이스라인 전체)
make evaluate   # 단계 4
make ablation   # Ablation Study
make all        # 전체 파이프라인
```

## 3. 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. 핵심 손실 함수 (본 연구 기여)

총 손실 :  `L_total = L_struct + λ · L_onto`  
온톨로지 손실 :  `L_onto = α·L_type + β·L_hier_attn + γ·L_attr + δ·L_dir`

- **L_dir** (기여 ①) :  `max(0, γ_dir + f(h,r,t) − f(t,r,h))` ― 대칭 관계 제외
- **L_hier_attn** (기여 ②) :  `Σ_c [ softmax(W·c_emb)_c · L_hier_c ]`
- **L_type / L_attr** :  기존 온톨로지 결합형 KGE 의 표준 제약

## 5. 실험 단계 우선순위

1. **Phase 1** : 데이터셋 구축 → `scripts/01_collect_data.py` + `scripts/02_preprocess.py`
2. **Phase 2** : 베이스라인 재현 → `make baselines`
3. **Phase 3** : 본 연구 모델 학습 → `scripts/03_train.py --config configs/ours.yaml`
4. **Phase 4** : 메인 실험 (5 seed × 7 모델) → `scripts/run_ablation.py`
5. **Phase 5** : 결과 분석 → `scripts/04_evaluate.py` + `notebooks/`

## 6. 데이터 흐름

`data/raw/`  →  `data/interim/`  →  `data/processed/`  
원본 (API)     →  추출된 트리플    →  검증된 KG (80/10/10)

## 7. 참고 (자세한 실험 계획)

별도 문서 : `석사논문_실험구성_계획서_2025254013_송재익.docx`

---

## 8. NumPy 실험 엔진 (src/np_engine) — 실제 실행된 전체 실험

PyTorch 설치가 차단된 폐쇄망/샌드박스 환경에서도 가이드라인 §3 전체 스펙을
실행할 수 있는 자립형 엔진. `src/` 의 PyTorch 코드와 동일한 손실 수식을 사용한다.

```
src/np_engine/data.py       : KG 로드 + 관계 그룹(방향성/계층/inverse pair) + H3 축소 샘플링
src/np_engine/models.py     : TransE/RotatE/DistMult/ComplEx/TransO/Ours (해석적 그래디언트)
src/np_engine/train.py      : L_struct + L_dir + L_hier_attn + L_attr + L_type, λ curriculum,
                              중단·재개(체크포인트) + per-epoch RNG 재현성
src/np_engine/evaluate.py   : Filtered MRR/Hits@1/3/10 (head+tail), 관계·그룹별 분해
src/np_engine/stats.py      : paired bootstrap + H1/H2/H3 자동 판정
```

### 전체 실험 실행 (중단·재개 지원)

```bash
# DONE 파일이 생길 때까지 반복 호출 (호출당 시간 제한 환경 대응)
python scripts/run_full_experiments.py --out final_run --budget 38
```

68 jobs = 메인 6모델×5seed + Ablation 4종×5seed + 강건성 2모델×3비율×3seed.

### 실험 결과 (experiments/final_run/, 2026-06-12 실행 완료)

| 가설 | 기준 | 결과 |
|---|---|---|
| H1 방향성 | 방향성관계 Hits@1 ≥ TransO +5%p, bootstrap p<0.05 | ✅ +8.28%p, p≈0.000 |
| H2 계층 적응 | L_hier_attn 제거 시 계층 MRR ≥3%p 저하 | ✅ −5.70%p |
| H3 강건성 | 50% 축소 시 ΔMRR(Ours) < ΔMRR(TransE)×0.7 | ✅ 0.1106 < 0.1145 |

Ours 하이퍼파라미터는 **valid 셋 탐색**으로 결정 (test 미사용):
`alpha_type=0.1, beta_hier=0.5, gamma_attr=0.5, delta_dir=3.0, margin_dir=2.0,`
`L_dir 은 방향성 관계(인용·외부인용·위임·위임받음·소속·포함)에만 적용`.

산출물 구조 (논문/발표 첨부용):

```
experiments/final_run/
├── report.md           # 사람이 읽는 요약 리포트 (표 + 가설 판정)
├── main_results.csv    # 메인 비교표 (6모델 × 5 seed 평균)
├── per_relation.csv    # 모델×관계별 MRR/Hits@1
├── ablation.csv        # Ablation 비교표
├── robustness.csv      # 데이터 축소 강건성
├── hypotheses.json     # H1/H2/H3 자동 판정 (bootstrap 포함)
├── results.json        # 전체 원시 지표 (재가공용)
├── manifest.json       # 실행 환경·하이퍼파라미터 (재현성)
├── figures/            # main_comparison / directional_hits1 / ablation / robustness PNG
├── raw/                # 런별 지표 JSON + reciprocal rank NPZ (bootstrap 원자료)
└── state/              # 작업 큐 + 학습 체크포인트 (재개용)
```

> `experiments/full_run/` 은 튜닝 전 1차 실행 기록 (Ours 구버전 가중치). 비교·기록용으로 보존.
