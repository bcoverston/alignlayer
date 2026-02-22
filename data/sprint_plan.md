# AlignLayer Sprint Plan -- Round 2

## Program Manager Summary

Round 1 showed 70.4% overall accuracy with a 48% false negative rate on T3+T4 -- the model systematically under-scores dangerous commands. The Round 2 strategy attacks this from three angles simultaneously: expand the corpus to fill the documented gaps (AWS/k8s/SSH/dry-run), add a deterministic heuristic layer for compound commands and dry-run flags that the char-level CNN cannot reliably learn, and retrain with weighted loss to penalize catastrophic misses. Target: 85%+ overall accuracy, FN rate on T3+T4 below 20%.

## Sequencing and Dependencies

```
Week 1                    Week 2                    Week 3
───────────────────────── ───────────────────────── ─────────────────

WS1: Corpus Expansion ──────────────┐
                                    ├──→ WS4: Architecture (retrain)
WS2: Heuristic Layer ───────────────┤         │
                                    │         ▼
WS3: k-NN Tuning ──────────────────►├──→ Integration + Round 2 Eval
                                    │
WS5: Eval Pipeline ────────────────►┘
```

- WS1 (Corpus) and WS2 (Heuristic) run in parallel from day 1. No dependencies.
- WS3 (k-NN) runs immediately against the existing checkpoint. No blockers.
- WS5 (Eval) runs in parallel with everything. Ships incrementally.
- WS4 (Architecture) blocks on WS1 completing -- new data required to validate architecture changes. Also consumes WS3 findings (optimal k) for the retrained model.
- Integration testing uses WS5's automated pipeline.

## Workstream 1: Corpus Expansion

**Owner:** TBD
**Blocking:** WS4 (Architecture retrain requires expanded corpus)
**ETA:** 5 days

### Objective

Add 400+ labeled commands targeting the 8 corpus gaps identified in the eval report. These gaps account for ~60% of all errors.

### Plan

1. **AWS read/write discrimination (~150 commands).** Cover `describe-*`/`get-*`/`list-*` (T0) vs `set-*`/`update-*`/`delete-*` (T3+) across EC2, RDS, S3, IAM, ECS, Lambda, AutoScaling. The model currently collapses all AWS commands to T-2.

2. **kubectl verb discrimination (~100 commands).** `get`/`describe`/`logs` = T0, `apply`/`patch`/`rollout restart` = T3, `delete` = T3-T4. Vary namespace, resource type, flags.

3. **SSH subcommand variation (~100 commands).** SSH+read (T0: `uname`, `cat`, `tail -f`), SSH+write (T3: `cp`, `mv`, config changes), SSH+destructive (T4: `rm -rf`, service stops). The model currently defaults all SSH to T3.

4. **Dry-run flag variants (~100 commands).** Pair each dangerous command with its `--dry-run`/`--check`/`plan` variant labeled T-1. Include: `terraform plan` vs `apply`, `npm publish --dry-run` vs `publish`, `kubectl delete --dry-run=client` vs `delete`.

5. **Compound/chained commands (~80 commands).** `&&`, `||`, `;`, `|`, `$()` subshells. Label at `max(component_tiers)`. Priority: chains where the dangerous subcommand is not the first token.

6. **T-2 exfiltration patterns (~50 commands).** `curl | bash`, `eval $(curl)`, credential piping, reverse shells, typosquat installs. Currently near-zero representation.

7. **Airflow/orchestrator (~40 commands).** `dags trigger` (T3), `dags delete` (T4), `dags test` (T-1), `dags list` (T0).

8. **npm/package manager publish (~30 commands).** `npm publish` (T3), `npm dist-tag` (T3), `npm unpublish` (T4).

9. **Ingest the 71 corrected commands** from the eval report's misclassification table directly into `scores-cache.jsonl`.

### Generation approach

Use the existing `pairs.py` pipeline with targeted category prompts. Run `--n-per-category` at 50-60 for each gap category. Score with LLM (not heuristic-only) to maintain label quality. Manual review pass on T-2 and T4 labels before merge.

### Conflicts

- The new corpus must use the same JSONL schema as `scores-cache.jsonl`. No schema changes.
- WS4 depends on this completing before retraining. If corpus work slips, WS4 slips.
- WS2's dry-run heuristic overlaps with the dry-run training data. Both should proceed -- the heuristic is a safety net; the corpus teaches the model the pattern natively.

## Workstream 2: Heuristic Layer

**Owner:** TBD
**Blocking:** Nothing (post-processing layer, can deploy independently of retraining)
**ETA:** 3 days

### Objective

Add two deterministic post-processing rules that fix classes of errors the char-level CNN structurally cannot handle well.

### Plan

1. **Dry-run flag detection.** Before returning the k-NN prediction, scan the input command for `--dry-run`, `--check`, `--dry-run=client`, `-n` (in make/rsync context), `--simulate`, `--preview`, `--no-act`, and `terraform plan`. If detected, cap the predicted tier at T-1. This alone fixes 6 errors from Round 1 (T-1 accuracy from 45% to ~91%).

   Implementation: Add a `apply_heuristic_overrides(cmd: str, prediction: dict) -> dict` function in `siamese.py` that wraps `predict_risk()`. Keep the raw prediction in the output for diagnostics.

2. **Compound command decomposition.** Split on `&&`, `||`, `;`, and `|`. Score each subcommand independently via the same k-NN path. Return `max(tiers)` as the final tier. This fixes the `git commit && git push --force` miss (delta=-6) and similar chains.

   Edge case: pipe chains where the pipe itself changes semantics (e.g., `kubectl get ... | jq` is still T0). Heuristic: only the *last* command in a pipe inherits write risk; earlier commands in a pipe are read-side. For `&&`/`||`/`;`, take the max.

3. **AWS/kubectl verb override (stretch goal).** If the command starts with `aws` and the subcommand matches `describe-*`/`get-*`/`list-*`, floor the tier at T0. Similarly for `kubectl get/describe/logs`. This is a targeted fix for pattern #6 in the eval report. Only implement if time permits -- the corpus expansion (WS1) should teach this natively.

### Conflicts

- The heuristic layer must not interfere with training or embedding. It is inference-only post-processing.
- The `_DRY_FLAGS` set already exists in `pairs.py`'s heuristic scorer. Reuse the same set for consistency.
- WS3 (k-NN tuning) changes the prediction path. Coordinate: heuristic overrides apply *after* k-NN returns a prediction, regardless of k value.

## Workstream 3: k-NN Tuning

**Owner:** TBD
**Blocking:** Nothing (runs against existing checkpoint immediately)
**ETA:** 2 days

### Objective

Optimize the k-NN inference layer. Current k=5 with a 21K corpus is sensitive to outlier neighbors.

### Plan

1. **k sweep.** Test k in {3, 5, 7, 9, 11, 15, 21} against the Round 1 benchmark scenarios. Measure overall accuracy, FN rate on T3+T4, and FP rate on T0. The eval report recommends k=11; validate empirically.

2. **Distance-weighted voting.** Replace uniform k-NN voting with inverse-distance-weighted voting: `weight_i = 1 / (dist_i + epsilon)`. This reduces the influence of distant neighbors in the top-k. Implement in `predict_risk()` as an option.

3. **Tier voting strategy.** Current: `max(set(votes), key=votes.count)` (mode). Test alternatives:
   - Weighted mode (using distance weights above)
   - Risk-average then map to tier (current approach for `avg_risk`)
   - Conservative: if any neighbor is T3+, bias the vote upward

4. **Embedding index optimization (stretch).** Current brute-force `torch.norm` over 21K vectors is fast enough but won't scale. Evaluate FAISS or annoy for approximate NN if corpus grows past 50K. Not blocking for Round 2.

### Deliverables

- Optimal k value with benchmark evidence
- Distance-weighted voting implementation (behind a flag)
- Recommendation for WS4 on which k and voting strategy to use post-retrain

### Conflicts

- WS2's heuristic layer wraps the k-NN output. Ensure the `predict_risk()` return schema is stable.
- After WS4 retrains, k may need re-tuning. Plan for a second quick sweep post-retrain.

## Workstream 4: Architecture

**Owner:** TBD
**Blocking:** WS1 (needs expanded corpus for validation), WS3 (needs optimal k recommendation)
**ETA:** 5 days (starts after WS1 completes, ~day 6)

### Objective

Modify the model architecture and loss function to reduce catastrophic misses (T3/T4 under-scored by 2+ tiers).

### Plan

1. **Wider CNN kernels.** Add kernel size 16 to the existing `[3, 5, 7, 11]` set. The current max kernel (11 chars) cannot span `--force` in context when it appears 30+ chars from the command verb. A width-16 kernel gives the model a better shot at capturing long-range flag interactions.

   Impact: ~10K additional parameters (64 filters x 16 x 32 embed_dim). Still under 100K total.

2. **Weighted contrastive loss.** Scale the loss by `|tier_a - tier_b|^2` for dissimilar pairs. A T4-vs-T1 miss (delta=3) should contribute 9x the loss of a T1-vs-T0 miss (delta=1). This directly addresses the eval finding that the model treats large and small tier gaps similarly.

   Implementation: Modify `contrastive_loss()` to accept tier metadata. The pair dataset already carries tier info in `action_a`/`action_b`.

3. **Training on expanded corpus.** Regenerate pairs from the WS1-expanded `scores-cache.jsonl`. Use `--boundary-pairs 500` to force dissimilar pairs at the T2/T3 and T3/T4 boundaries where the model is weakest.

4. **Hyperparameter sweep (stretch).** If time permits, sweep learning rate {5e-4, 1e-3, 2e-3}, batch size {256, 512, 1024}, and margin {0.8, 1.0, 1.2}. Use WS5's automated eval to compare.

### What we are NOT doing

- Two-stage model (verb+scope classifier feeding into the embedding). Too much complexity for Round 2. Revisit if Round 2 accuracy plateaus below 85%.
- Attention layer over CNN features. Same rationale -- the wider kernel is a simpler intervention with less risk.

### Conflicts

- Changing `contrastive_loss()` signature affects training but not inference. No conflict with WS2/WS3.
- Adding a kernel changes the model checkpoint format. Old checkpoints will not load. Document this clearly.

## Workstream 5: Eval Pipeline

**Owner:** TBD
**Blocking:** Nothing (runs in parallel with all other workstreams)
**ETA:** 3 days (incremental delivery)

### Objective

Automate the eval pipeline and improve scenario coverage so Round 2 results are statistically meaningful and reproducible.

### Plan

1. **`make eval` target.** Single command that: generates/loads scenarios, runs inference, produces the eval report. Eliminate the manual 4-batch process from Round 1. Wire up `eval_harness.py` as the backbone.

2. **Stratified scenario generation.** Round 1 had 100 T0 commands but only 3 T-2 and 11 T-1. Generate ~30 commands per tier for balanced per-tier accuracy measurement. Use the same Ollama generation pipeline from `pairs.py` with tier-specific prompts.

3. **Adversarial scenarios.** Include 2-3 adversarial scenarios per round (expand beyond s20). Categories: AI agent attacks, social engineering commands, exfiltration chains, typosquat installs.

4. **Structured output format.** Persist eval results as JSON alongside the markdown report. Schema: per-command predictions, per-tier accuracy, per-scenario accuracy, delta distribution. Enable cross-round accuracy trend tracking.

5. **Regression test suite.** The 71 misclassified commands from Round 1 become a fixed regression set. Every future eval run checks these first and reports how many are now correct.

### Conflicts

- Scenario generation uses the same Ollama endpoint as WS1 corpus generation. If both run simultaneously, they compete for GPU. Sequence: WS1 generates first (it has more commands), WS5 generates after, or run on different machines.
- The eval pipeline must support both the current checkpoint and the WS4-retrained checkpoint. Parameterize the checkpoint path.

## Integrated Timeline

| Day | WS1: Corpus | WS2: Heuristic | WS3: k-NN | WS4: Architecture | WS5: Eval |
|-----|-------------|----------------|------------|--------------------|-----------|
| 1   | Generate AWS/kubectl commands | Implement dry-run detection | k sweep {3..21} | -- | `make eval` target |
| 2   | Generate SSH/dry-run commands | Implement compound decomposition | Distance-weighted voting | -- | Stratified scenario gen |
| 3   | Generate T-2/chain/airflow/npm | Test heuristic on Round 1 misses | Benchmark all strategies | -- | Adversarial scenarios |
| 4   | LLM scoring + manual review | Ship heuristic layer | Write up k recommendation | -- | JSON output format |
| 5   | Ingest 71 corrected commands, final QA | -- | -- | -- | Regression test suite |
| 6   | -- | -- | -- | Add kernel-16 + weighted loss | -- |
| 7   | -- | -- | -- | Retrain on expanded corpus | -- |
| 8   | -- | -- | -- | Retrain completes | -- |
| 9   | -- | -- | k re-sweep post-retrain | Integrate heuristic layer | Full Round 2 eval |
| 10  | -- | -- | -- | Final checkpoint | Round 2 report |

## Success Criteria for Round 2

| Metric | Round 1 | Round 2 Target |
|--------|---------|----------------|
| Overall accuracy | 70.4% | >= 85% |
| FN rate (T3+T4 under-scored by >1 tier) | 48.0% | <= 20% |
| FP rate (T0/T-1 over-scored by >1 tier) | 20.2% | <= 15% |
| T-1 accuracy (dry-run commands) | 45% | >= 85% |
| T-2 accuracy (exfiltration/attack) | 33% | >= 60% |
| T3 accuracy (external write) | 55% | >= 75% |
| T4 accuracy (catastrophic) | 42% | >= 70% |
| Worst single miss (delta) | -6 | <= |3| |
| Regression set (71 Round 1 misses) | 0/71 | >= 55/71 |

## Risks and Mitigations

**1. Corpus quality -- LLM mislabeling propagates into training data.**
The Ollama 14b scorer has unknown accuracy on adversarial/edge-case commands. A T-2 command mislabeled as T1 teaches the model to under-score exfiltration.
*Mitigation:* Manual review pass on all T-2 and T4 labels. Cross-check against the heuristic scorer -- large heuristic/LLM disagreements get flagged for human review.

**2. Architecture change invalidates existing checkpoint.**
Adding kernel-16 changes the model state dict. All downstream tooling that loads `best.pt` breaks until retrain completes.
*Mitigation:* Version checkpoints (`best-v1.pt`, `best-v2.pt`). Keep v1 as fallback. Update `eval_harness.py` and `predict_risk()` to accept checkpoint path as parameter (already supported).

**3. Overfitting to the Round 1 benchmark.**
Ingesting the 71 misclassified commands as training data and building a regression suite from them risks teaching to the test. Round 2 accuracy on these specific commands will look good but may not generalize.
*Mitigation:* The 71 commands are a small fraction of the expanded corpus (~10% of new data). WS5's stratified scenario generation creates fresh eval commands not seen in training. Report accuracy on both the regression set and the fresh set separately.
