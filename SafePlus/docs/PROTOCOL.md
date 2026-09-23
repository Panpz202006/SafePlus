# Performance-review protocol

## Research questions

- **RQ1 — discrimination:** Does SafePlus improve AUPRC/AUROC over classifiers,
  conventional survival models, and SAFE?
- **RQ2 — timeliness:** Does it detect fraudulent entities earlier than the observed
  blocking, suspension, or operational response time?
- **RQ3 — latent recovery:** When a known injected action time is concealed, does
  SafePlus recover it accurately and with a calibrated posterior?
- **RQ4 — robustness:** How do delay length, censoring rate, class imbalance, and
  sequence sparsity affect performance?

## Required comparisons

1. Static classifier: logistic regression or SVM on aggregated features.
2. Sequential classifier: GRU/LSTM.
3. Cox proportional hazards or DeepSurv.
4. SAFE-r: ordinary event-time survival objective.
5. SAFE: left-censoring/early-detection objective.
6. SafePlus: joint commission and detection hazards.

Recommended modern comparators, when compatible with the data contract, are
DeepHit, Dynamic-DeepHit, and a delayed-feedback classifier.

## Splits and leakage control

- Prefer calendar-time splits. If acquisition time is unavailable, use fixed seeded
  entity-level splits and state this limitation.
- Keep all events belonging to one entity in one split.
- Fit scaling, encoding, resampling, and thresholds on training/validation only.
- Never select a checkpoint or ablation using test performance.
- Run seeds `0, 1, 2, 3, 4`; report mean, standard deviation, and paired bootstrap
  confidence intervals.

## Metrics

### Discrimination

AUPRC is primary because fraud data are imbalanced. Also report AUROC, precision,
recall, F1, and accuracy. Report operating-point metrics at a validation-selected
threshold and, where meaningful, fixed alert-budget recall.

### Early detection

- F1 and accuracy at prefixes 1–5 for direct SAFE comparability;
- early-detected fraudster rate;
- lead time `t_d - t_alarm` among detected positives;
- recall at `k` steps before `t_d`.

### Latent onset

Only use these when `t_onset` is known independently of training supervision:

- posterior-mode onset MAE;
- posterior NLL at the true onset;
- 50%/80%/95% credible-interval coverage and width.

### Survival calibration

Report detection NLL, integrated Brier score, and calibration curves at prespecified
horizons. Account for censoring when computing time-dependent metrics.

## Controlled delayed-detection protocol

For a dataset with a known injected fraud-action time `t_a`:

1. Store `t_a` as evaluation-only `t_onset`.
2. Sample delay `Delta` from prespecified geometric, log-normal, or empirical laws.
3. Define `t_d=t_a+Delta`; right-censor when `t_d` exceeds the horizon or under a
   random censoring mechanism.
4. Train using only covariates, observed `t_d`, and censoring indicator.
5. Stratify results by delay quartile and censoring status.

Run a factorial sensitivity grid over delay mean `{1, 2, 4, 8}` and censoring rate
`{0.0, 0.2, 0.4, 0.6}`.

## Ablations

- joint model vs commission-only SAFE objective;
- exact marginalization vs variational posterior vs stochastic EM;
- irregular-time attention vs GRU encoder;
- local-only vs local-plus-global context;
- null-event insertion on/off;
- shared vs separate commission/detection encoders;
- known delay distribution vs learned detection hazard.

## Statistical reporting

Use paired comparisons on identical splits/seeds. Report the absolute difference,
95% paired bootstrap CI, and a corrected p-value across primary dataset–metric
families. Include runtime, peak GPU memory, parameter count, and convergence failures.

