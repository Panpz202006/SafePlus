# Synthetic performance review — 2026-09-24

All 20 runs completed. Under the repository's current settings, GRU has the highest mean test AUPRC. SafePlus improves on SAFE but trails GRU and SAFE-r, with substantial seed sensitivity. These results establish a runnable synthetic baseline, not a real-data validation of the proposed method.

## Setup

- Existing synthetic dataset: 2,000 entities, 24 steps, 8 features; 532 observed detections and 703 known onsets.
- Four models × seeds 0–4. Matching seeded entity splits: 1,400 training, 200 validation, 400 test entities per run.
- Hidden size 32, batch size 64, Adam learning rate 0.001, up to 40 epochs, early-stopping patience 7. Validation AUPRC selects checkpoints; validation F1 selects thresholds.
- CPU, one PyTorch compute thread and one interop thread; PyTorch 2.10.0, Python 3.11.9, macOS ARM64. Dataset hash and runtime environment are saved in environment.json.
- The corrected early-detection metric requires an alarm strictly before observed detection. No test-driven tuning was performed.

## Predictive quality

Mean ± sample standard deviation across five seeds:

- GRU: AUPRC **0.774 ± 0.040**, AUROC 0.946, F1 0.857.
- SAFE-r: AUPRC **0.760 ± 0.038**, AUROC 0.946, F1 0.835.
- SafePlus: AUPRC **0.696 ± 0.144**, AUROC 0.901, F1 0.737.
- SAFE: AUPRC **0.270 ± 0.036**, AUROC 0.511, F1 0.396.

SafePlus seed 0 stopped after eight epochs and restored its first-epoch checkpoint: test AUPRC was 0.440. The other four runs achieved 0.743–0.779. Validation AUPRC initially degraded even as likelihood loss improved; this warrants investigation using validation-only experiments. Several survival-model runs reached the 40-epoch limit, so these are bounded-training results rather than evidence of convergence.

SafePlus onset MAE was 9.44 ± 0.74 bins; onset NLL was 3.56 ± 0.44. This does not demonstrate accurate onset recovery. Posterior recovery metrics condition on observed detection/censoring and are separate from online risk prediction.

Mean early-detection rates were GRU 0.519, SAFE 0.870, SAFE-r 0.042, SafePlus 0.224. SAFE's high early rate accompanies poor discrimination and should not be interpreted as superior practical performance. Mean lead times include late alarms and can be negative.

Paired seed-bootstrap comparisons (10,000 resamples) give SafePlus-minus-GRU AUPRC difference −0.0784, with percentile interval [−0.1757, −0.0229]. These exploratory intervals describe only five seeds on one fixed synthetic dataset; they are not population-level confidence claims. All paired comparisons are in paired_comparisons.csv.

## Runtime

Mean end-to-end training-command time, including data loading, training, validation threshold selection, held-out evaluation and saving:

- GRU: 1.38 seconds/run, mean 11.2 epochs.
- SAFE: 2.45 seconds/run, mean 17.4 epochs.
- SAFE-r: 3.77 seconds/run, mean 38.8 epochs.
- SafePlus: 3.08 seconds/run, mean 29.0 epochs.

Different epoch counts prevent interpreting these totals as equal-work training-speed comparisons. Python startup and final cross-run aggregation are excluded.

Mean of per-checkpoint median forward-pass latency, batch size 64:

- GRU: 0.648 ms/batch.
- SAFE: 0.646 ms/batch.
- SAFE-r: 0.660 ms/batch.
- SafePlus: 0.678 ms/batch.

Each of the 20 checkpoints had 20 warm-up passes and 200 timed passes on the same resident CPU tensor (64 × 24 × 8). These measure model forward calls only, excluding data loading, checkpoint loading and prediction-file writing. Timing is sequential on this machine, without randomized model order; small differences should not be overinterpreted. Per-run medians and p95 values are saved in review_runs.csv.

## Scope and next experiments

The repository's discrimination labels are observed detection indicators: censored fraudulent entities count as negatives. SafePlus predicts commission risk, whereas SAFE-r predicts detection risk. Consequently this comparison evaluates the existing detection-label benchmark, not independently verified fraud status. Full-horizon scores can also use covariates after the operational detection time; early metrics must be considered separately.

Next investigate SafePlus checkpoint selection and training duration on validation data, assess SAFE's numerical saturation, and evaluate onset recovery against explicit onset baselines. A publication-level review still needs real-data adapters, independent dataset replicates, delay/censoring sweeps, calibration, and additional baselines from docs/PROTOCOL.md.

## Reproduction and artifacts

Run from the repository root. The runner refuses to overwrite nonempty run directories; select a new output directory for another experiment.

```bash
PYTHONPATH=. /opt/anaconda3/envs/transformer/bin/python -B \
  reports/performance_review_20260924/run_review.py \
  --output reports/performance_review_repeat
```

The original review directory contains all 20 checkpoints, effective configurations, epoch histories and held-out metrics under runs/. It also contains run.log, environment.json, timings.json, the repository's summary.csv/summary.tex, review_runs.csv, review_summary.csv and paired_comparisons.csv. summarize_review.py recreates the additional analysis and inference timings for runs in its own directory.
