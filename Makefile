.PHONY: all test data smoke clean e1 e2 e3 e4 e5

all:            ## 전 실험 실행 → results/RESULTS.md (~65초, numpy 만 필요)
	python experiments/run_all.py

test:           ## 설계 무결성 테스트 (pytest 없이도 동작)
	python tests/run_tests.py

e1 e2 e3 e4 e5: ## 실험 하나만
	@python $(wildcard experiments/$@_*.py)

data:           ## 합성 세계 → 한국어 텍스트 SFT 데이터 (finetune/data/*.jsonl)
	python finetune/build_sft_data.py

smoke:          ## LoRA 학습·생성·채점 전 경로 검증 (CPU, 다운로드 없음, ~90초)
	python finetune/train_qlora.py --smoke --epochs 8 --bs 8 --grad-accum 1 --lr 8e-3 --limit 256
	python finetune/evaluate.py --smoke --limit 64

clean:
	rm -rf results/*.md results/*.json finetune/runs __pycache__ */__pycache__
