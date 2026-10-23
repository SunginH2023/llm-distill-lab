.PHONY: all test data smoke clean e1 e2 e3 e4 e5

all:            ## run all experiments -> results/RESULTS.md (numpy only; about 17 s on an Apple M1 laptop)
	python experiments/run_all.py

test:           ## design-integrity tests (works without pytest)
	python tests/run_tests.py

e1 e2 e3 e4 e5: ## run a single experiment
	@python $(wildcard experiments/$@_*.py)

data:           ## synthetic world -> Korean-text SFT data (finetune/data/*.jsonl)
	PYTHONHASHSEED=0 python finetune/build_sft_data.py

smoke:          ## end-to-end check of LoRA training, generation and scoring (CPU, no downloads; train + eval about 26 s on an Apple M1 laptop)
	python finetune/train_qlora.py --smoke --epochs 8 --bs 8 --grad-accum 1 --lr 8e-3 --limit 256
	python finetune/evaluate.py --smoke --limit 64

clean:
	rm -rf results/*.md results/*.json finetune/runs __pycache__ */__pycache__
