PYTHON      := model/.venv/bin/python3
CHECKPOINT  ?= model/checkpoints/best.pt
CORPUS      ?= data/synthetic/scores-cache.jsonl
SCENARIOS   ?= data/test_scenarios.json
PAIRS_OUT   ?= data/synthetic/pairs-v3.jsonl
IMAGE       ?= alignlayer-scorer
MAX_PAIRS   ?= 2000000
BOUNDARY_PAIRS ?= 50000
K           ?= 5
PID_FILE    := data/traces/.serve.pid
PORT        ?= 8000

.PHONY: eval eval-report adversarial pairs train trend serve serve-bg serve-stop docker-build docker-run help

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

## Run scoring server locally (foreground)
serve:
	ALIGNLAYER_CHECKPOINT=$(CHECKPOINT) \
	ALIGNLAYER_CORPUS=$(CORPUS) \
	PORT=$(PORT) \
	$(PYTHON) model/serve.py

## Start scoring server in background with PID tracking
serve-bg:
	@if [ -f $(PID_FILE) ] && kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
		echo "Server already running (PID $$(cat $(PID_FILE)))"; \
	else \
		mkdir -p $$(dirname $(PID_FILE)); \
		ALIGNLAYER_CHECKPOINT=$(CHECKPOINT) \
		ALIGNLAYER_CORPUS=$(CORPUS) \
		PORT=$(PORT) \
		nohup $(PYTHON) model/serve.py > data/traces/serve.log 2>&1 & \
		echo $$! > $(PID_FILE); \
		echo "Server started (PID $$(cat $(PID_FILE))), log → data/traces/serve.log"; \
		sleep 2; \
		if kill -0 $$(cat $(PID_FILE)) 2>/dev/null; then \
			echo "Health: $$(curl -s http://localhost:$(PORT)/health)"; \
		else \
			echo "Server failed to start. Check data/traces/serve.log"; \
			rm -f $(PID_FILE); \
			exit 1; \
		fi \
	fi

## Stop background scoring server
serve-stop:
	@if [ -f $(PID_FILE) ]; then \
		PID=$$(cat $(PID_FILE)); \
		if kill -0 $$PID 2>/dev/null; then \
			kill $$PID; \
			echo "Stopped server (PID $$PID)"; \
		else \
			echo "PID $$PID not running (stale PID file)"; \
		fi; \
		rm -f $(PID_FILE); \
	else \
		echo "No PID file found at $(PID_FILE)"; \
	fi

## Build Docker image
docker-build:
	docker build -t $(IMAGE) .

## Run scoring server in Docker (corpus mounted from host)
docker-run:
	docker run -d \
		--name alignlayer-scorer \
		--restart unless-stopped \
		-p 8000:8000 \
		-v $(CURDIR)/$(CORPUS):/data/scores-cache.jsonl:ro \
		-e ALIGNLAYER_CHECKPOINT=model/checkpoints/best.pt \
		$(IMAGE)

help:
	@echo "Targets: eval adversarial eval-report pairs train trend serve serve-bg serve-stop docker-build docker-run"
	@echo "Overrides: CHECKPOINT=... CORPUS=... K=... PORT=... IMAGE=..."
