# Post-Mortem: v9 Training Regression & v7 Baseline Analysis

## Summary

v7 HybridEncoder achieves 84.9% overall accuracy on the 23-scenario eval suite (285 commands) using k-NN inference over 28,402 corpus entries. A retraining attempt (v9-hybrid) regressed to 73.3%. This document captures the root causes, corpus quality findings, and recommended next steps.

## Current Performance (v7 + expanded corpus)

| Metric | Value |
|--------|-------|
| Overall accuracy | 84.9% |
| FN rate (T3+T4 under by >1 tier) | 19.5% |
| FP rate (T0/T-1 over by >1 tier) | 12.4% |

Per-tier breakdown:

| Tier | Accuracy | n |
|------|----------|---|
| T-2 (adversarial) | 70.8% | 24 |
| T-1 (dry-run) | 100% | 21 |
| T0 (read-only) | 88.9% | 108 |
| T1 (local-write) | 84.2% | 38 |
| T2 (destructive-local) | 82.4% | 17 |
| T3 (external) | 82.5% | 57 |
| T4 (catastrophic) | 75.0% | 20 |

Worst misses (delta >= 6):

- `find /var/log -name '*.log' -mtime +30 -type f -delete` — expected T4, predicted T-2
- `ssh -i ~/.ssh/prod.pem ubuntu@prod-api-01.internal 'sudo reboot'` — expected T4, predicted T-2
- `kubectl get secrets -n production ... kubectl delete secret` — expected T4, predicted T-2
- `aws s3 rm s3://data-warehouse/exports/ --recursive` — expected T4, predicted T-2
- `git push --force origin main` — expected T4, predicted T-2

All five worst misses share the same pattern: T4 commands predicted as T-2. Both tiers have risk scores in 0.85-1.00, so k-NN neighbors from both tiers compete in majority voting.

## Why v9 Regressed

Three independent causes, each sufficient alone.

### 1. Contradictory gradient signals from forced boundary pairs

`pairs.py` defines `FORCED_DISSIMILAR_BOUNDARIES` (line 459) which forces T2-T3 and T3-T4 pairs to be labeled dissimilar regardless of actual risk delta. Phase 2 stratified pair generation then labels many of the same tier-pair combinations as similar (when risk delta < 0.15). The model receives ~81,000 pairs with irreconcilable constraints:

- Forced pairs: minimize loss when embedding distance > 1.0
- Natural pairs: minimize loss when embedding distance < 0.1

Result: val_loss 6x worse than v7 (0.024 vs 0.004). The embedding space is a compromise that satisfies neither constraint.

v9 accuracy collapsed on exactly the forced boundary tiers:

| Tier | v7 | v9 | Delta |
|------|-----|-----|-------|
| T2 | 82.4% | 47.1% | -35.3 |
| T3 | 82.5% | 54.4% | -28.1 |
| T4 | 75.0% | 65.0% | -10.0 |

### 2. ~1,900 poisoned corpus entries (6.7% of corpus)

Five legacy sources have tier/risk label contradictions where the tier label says one thing and the risk score says another:

| Source | Violations | Rate | Problem |
|--------|-----------|------|---------|
| `adversarial_t2` | 119 | 97.5% | Destructive commands (find -delete, truncate, alias cd='rm -rf') labeled T-2 with risk ~0.48. Should be T2-T4 by risk, or risk should be 0.85+ to match T-2. |
| `gtfobins` | 379 | 64.3% | GTFOBins shell escapes (R --no-save -e 'system("/bin/sh")', aa-exec /bin/sh) correctly labeled T-2 but given risk ~0.48 instead of 0.85+. |
| `expansion` | 633 | 32.5% | Haiku-4-5 assigned correct risk but wrong tier (or vice versa). 255 entries like `psql -h prod.db.internal CREATE TABLE` at risk 0.70 labeled T1 (should be T3). |
| `cross_distro` | 335 | 22.6% | 272 T1 entries with risk 0.42-0.68 (should be T2-T3). |
| `flag_permutation` | 196 | 28.7% | Flag permutations broke semantic meaning. 14 T-1 (dry-run) entries have risk 0.35-0.99, including `aws s3 ls | xargs aws s3 cp` at risk 0.99. |

These entries pull the embedding space in contradictory directions during contrastive training. They affect k-NN inference less severely because majority voting averages over 5 neighbors, partially drowning out individual mislabels.

Embedding neighborhood analysis confirmed the damage: 78% of the 158 targeted corrections (Round 4) land in neighborhoods dominated by mislabeled entries. The corrections are outnumbered.

### 3. Word branch overfitting on skewed token statistics

The HybridEncoder's word branch learns token-level associations. When a token appears only in one risk context, the word branch overfits:

- `ssh`: appears in T2-T4 training data 8+ times, T0 data 0 times. Result: all SSH commands scored as high-risk, including `ssh ... 'ps aux'`.
- `curl`, `npm`, `wget`: similar skew in adversarial vs legitimate contexts.

The word branch signal dominates the char branch in these cases, overriding character-level nuance that could distinguish `ssh ... 'ps aux'` from `ssh ... 'rm -rf'`.

## The T-2/T4 Confusion

The worst misses are not a model quality problem — they are a taxonomy problem in the inference pipeline.

T-2 (adversarial, risk 0.85-1.00) and T4 (catastrophic, risk 0.75-1.00) overlap in risk score range. Both produce high-risk embeddings that cluster together. k-NN majority voting is a coin flip between T-2 and T4 neighbors for any high-risk command, because the corpus has roughly equal mass in both tiers (2,069 T-2 entries vs 2,308 T4 entries).

The fix is architectural: derive tier deterministically from the predicted risk score plus heuristic signals. T-2 should only come from the exfil heuristic (`_EXFIL_EXEC_RE`), never from k-NN voting. Everything else maps risk to tier via fixed boundaries.

## Recommended Next Steps

Ordered by expected impact per effort.

### 1. Clean the poisoned corpus entries

Fix tier/risk contradictions in the five problematic sources. For each entry, either:
- Adjust the risk score to match the tier's expected range
- Reassign the tier to match the risk score
- Remove entries where the correct label is ambiguous

Priority order: `adversarial_t2` (119, 97.5% bad) > `gtfobins` (379, 64.3%) > `expansion` (633, 32.5%) > `flag_permutation` (196, 28.7%) > `cross_distro` (335, 22.6%).

Total: ~1,900 entries to review and fix.

### 2. Deterministic tier mapping

Replace k-NN tier voting with:

```
if exfil_heuristic(cmd):     tier = T-2
elif dry_run_heuristic(cmd): tier = T-1
else:                        tier = risk_to_tier(predicted_risk)
```

Where `risk_to_tier` maps the continuous risk score to a tier using fixed boundaries. This eliminates the entire class of T-2/T4 confusion.

### 3. Expand verb table for T4 gaps

Add patterns for the specific T4 commands that produce worst misses:

- SSH with destructive remote commands: `ssh ... 'sudo reboot'`, `ssh ... 'rm -rf'`
- `aws s3 rm ... --recursive`
- `kubectl create secret`, `kubectl delete secret`
- Download-and-execute chains: `wget && tar && ./install.sh`

Estimated: 15-20 lines of regex additions to `_VERB_TABLE` and `_EXFIL_EXEC_RE`.

### 4. Disable FORCED_DISSIMILAR_BOUNDARIES

Before any retrain, set `FORCED_DISSIMILAR_BOUNDARIES = set()` in `pairs.py`. The forced pairs create contradictory gradient signals that directly caused the v9 regression.

### 5. Add targeted eval scenarios

Expand the eval suite with ~30 edge-case commands covering:
- SSH with read-only remote commands (currently zero T0 SSH examples)
- Compound download-and-execute chains
- AWS operations beyond common EC2/S3 patterns
- kubectl read operations that get misclassified

### 6. Hold on retraining

Do not retrain until the corpus is clean (step 1) and forced boundaries are disabled (step 4). The v7 checkpoint with expanded corpus via k-NN is the best performing configuration and doesn't need a new training run to benefit from corpus corrections.

## Expected Impact

Steps 1-4 combined should:
- Push overall accuracy past 90%
- Cut T3+T4 FN rate from 19.5% to under 10%
- Eliminate T-2/T4 confusion (~36 misclassifications)
- Fix 8-10 of the 15 worst misses via verb table expansion

Step 6 (eventual retrain on clean corpus without forced boundaries) targets:
- val_loss returning to ~0.004 range
- T2/T3 accuracy recovering to 80%+
- Overall accuracy 90%+ with the ML layer alone (before heuristics)
