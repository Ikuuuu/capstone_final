# === 졸업논문 실험 파이프라인 ===
PYTHON ?= python
CFG    ?= configs/default.yaml
OURS   ?= configs/ours.yaml
RUN    ?= experiments/final_run2

.PHONY: help data collect preprocess train baselines ablation evaluate inference figures-core4 all clean

help:
	@echo "사용 가능 타겟:"
	@echo "  make collect    - 단계 1: 데이터 수집"
	@echo "  make preprocess - 단계 2: 전처리 + 트리플 추출 + 분할"
	@echo "  make data       - 단계 1 + 2"
	@echo "  make train      - 단계 3: 본 연구 모델 학습"
	@echo "  make baselines  - 단계 3b: 모든 베이스라인 학습"
	@echo "  make evaluate   - 단계 4: 평가"
	@echo "  make ablation   - Ablation Study 실행"
	@echo "  make inference  - 단계 5: 추론 데모"
	@echo "  make figures-core4 - 핵심 4개 모델 비교 그림 생성(RUN=실험폴더)"
	@echo "  make all        - 전체 파이프라인 실행"
	@echo "  make clean      - 임시 파일 정리"

collect:
	$(PYTHON) scripts/01_collect_data.py --config $(CFG)

preprocess:
	$(PYTHON) scripts/02_preprocess.py --config $(CFG)

data: collect preprocess

train:
	$(PYTHON) scripts/03_train.py --config $(OURS)

baselines:
	@for cfg in configs/baselines/*.yaml; do \
		echo "===> Training: $$cfg"; \
		$(PYTHON) scripts/03_train.py --config $$cfg; \
	done

evaluate:
	$(PYTHON) scripts/04_evaluate.py --config $(OURS)

ablation:
	$(PYTHON) scripts/run_ablation.py --config $(OURS)

inference:
	$(PYTHON) scripts/05_inference.py --checkpoint checkpoints/ours/best.pt

figures-core4:
	$(PYTHON) scripts/plot_core4.py --run-dir $(RUN)

all: data train baselines evaluate ablation

clean:
	rm -rf experiments/tmp_* logs/*.log __pycache__ src/__pycache__ src/*/__pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
