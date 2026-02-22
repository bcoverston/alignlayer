PYTHON      := model/.venv/bin/python3
CHECKPOINT  ?= model/checkpoints/best.pt
CORPUS      ?= data/synthetic/scores-cache.jsonl
SCENARIOS   ?= data/test_scenarios.json
PAIRS_OUT   ?= data/synthetic/pairs-v2.jsonl
MAX_PAIRS   ?= 2000000
BOUNDARY_PAIRS ?= 50000
K           ?= 5

.PHONY: eval eval-report adversarial pairs train help

## Run scenario benchmark against current checkpoint
eval:
	$(PYTHON) -u model/eval.py \
		--checkpoint $(CHECKPOINT) \
		--corpus $(CORPUS) \
		--scenarios $(SCENARIOS) \
		--k $(K)

## Run adversarial regression suite
adversarial:
	$(PYTHON) -u model/eval.py \
		--checkpoint $(CHECKPOINT) \
		--corpus $(CORPUS) \
		--adversarial data/adversarial_suite.json \
		--k $(K)

## Generate eval_report.md from latest results
eval-report:
	$(PYTHON) model/eval_report.py \
		--results data/eval_results_latest.json \
		--output data/eval_report.md

## Regenerate training pairs from corpus
pairs:
	$(PYTHON) -u model/pairs.py \
		--max-pairs $(MAX_PAIRS) \
		--boundary-pairs $(BOUNDARY_PAIRS) \
		--output $(PAIRS_OUT)

## Train on current pairs
train:
	$(PYTHON) -u model/siamese.py train \
		--pairs $(PAIRS_OUT) \
		--max-pairs 1000000 \
		--epochs 15

## Show accuracy trend from eval history
trend:
	$(PYTHON) -c "\
import json; \
rows = [json.loads(l) for l in open('data/eval_history.jsonl')]; \
print(f\"{'Run':25s} {'Acc':>6} {'FN T3+T4':>9} {'FP T0':>7} {'corpus':>8}\"); \
print('-'*60); \
prev = None; \
[print(f\"{r['run_id'][:25]} {r['overall_acc']:6.3f} {r['fn_rate_t3_t4']:9.3f} {r['fp_rate_t0_t1']:7.3f} {r['corpus_size']:8,}\") for r in rows]"

help:
	@echo "Targets: eval adversarial eval-report pairs train trend"
	@echo "Overrides: CHECKPOINT=... CORPUS=... K=..."
