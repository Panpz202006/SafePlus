# Frozen Exp05 method on the reviewed Wiki 300

Base: `exp05-io-contract-v1`, commit
`06fa34d2097b8ca4fa209dbf60ba551fef31a33d`. This is the frozen two-state
diagonal-Gaussian behavior EM / TRAIN global scale / delay prior / fixed-fusion
pipeline, not the repository's separate main-branch GRU model. Neither existing
tag is moved. The fusion remains `w=2/3`, `temperature=1/3` (old `alpha=2`).

## Length-only model adaptation

`initialize_emissions` and `train_emission_em` now accept optional `lengths`.
Observed prefixes end at each real sequence boundary for censored entities. The
fixed-length default code path and its calculation order remain unchanged.
Schema 1.0 still rejects unequal lengths. Schema 1.1 explicitly opts in with
`variable_length_policy=slice_before_fit_and_predict`. Storage padding must be
zero and never contributes to initialization, sufficient statistics or candidate
likelihoods. Inference receives the cropped real sequence. No Gaussian emission,
E-step/M-step equation, calibration formula or fusion formula is changed.

The recovered seed-0 synthetic x and injected onsets equal the tracked original
synthetic demo arrays. The corrected detector count is 678. The adapted runner
reproduced all 400 output rows and all 132 eligible point predictions from the
committed frozen run. Parameters, scale and numeric prediction fields agree to
`1e-10` (median/mode predictions agree exactly). MAE remains `1.2348484848`,
Within-1 `0.6439393939`, Within-2 `0.8712121212`. A compressed NPZ's byte hash
differs from the historical archive, so array equality and committed output/
parameter regression are recorded rather than asserting archive-byte identity.

## Fixed real-data preparation

The frozen 300 annotations are unchanged: 61 source EXACT points, 139 source UTC
`(lower,upper]` intervals and 100 reviewed-window controls. Feature input is the
previously cached published-contribution metadata, not annotation judgments,
block reasons, filter hits or evidence-fetch availability. The first version has
eight fixed metadata-state features, not a learned text representation.

The earlier generic I/O artifact computed elapsed time and gap to the block time
at an endpoint snapshot. For this frozen method td is only a boundary. The new
preparer instead carries the last known published-event state unchanged to the
endpoint. That snapshot is not an extra edit. Original records and UTC truth are
not modified. All 200 endpoint states were recomputed; normal windows are intact.

Input, fixed matched-group split and labels remain physically separate. TRAIN /
VAL / TEST = 226 / 24 / 50. The generator does not choose features or thresholds
using TEST onsets. TEST is evaluated once, after prediction. Normal controls have
no observed detection and receive no retrospective onset prediction.

## Explicit supervised prior exception

This run uses `empirical_train`, with the exact Tag-2 Laplace(1) estimator, using
36 TRAIN source EXACT points that match both a published revision ID and UTC.
The explicit `train_onset_for_prior.csv` lists only those TRAIN entities. No
interval is collapsed and no VAL/TEST onset estimates the prior. The support is
the frozen runner's positive integer event-position delay, not clock duration.

This **is partially onset-supervised through the delay prior**. The behavior EM
still never receives onset values. Do not call the entire pipeline completely
onset-unsupervised, and do not describe AI evidence adjudication as independent
human verification. The prior's small training cohort and edit-step time unit
must be reported with results.

## Commands

From a branch based on Tag 2 after applying the patch:

```bash
python scripts/49_exp05_prepare_wiki_real.py --source PATH_TO_RESTORED_BUNDLE --out datasets/wiki300_exp05_v1_1 --frozen PATH_TO_FROZEN_300_JSONL
python scripts/45_exp04_validate_io.py --bundle datasets/wiki300_exp05_v1_1
python scripts/47_exp04_run_bundle.py --bundle datasets/wiki300_exp05_v1_1 --output runs/wiki300_exp05_real_v1
python scripts/50_exp05_evaluate_wiki.py --bundle datasets/wiki300_exp05_v1_1 --predictions runs/wiki300_exp05_real_v1
```

The delivered prepared bundle can be used directly, without running preparation
again. Choose a new output directory for reproduction; existing output is not
overwritten. Script 47 never opens evaluation gold. It reads the explicitly
permitted TRAIN-only prior resource. Script 50 alone reads UTC/interval gold.

The original 48 evaluator remains unchanged for the synthetic regression.
For real Wiki data, 50 evaluates source EXACT points in UTC. Only source EXACT
points aligned to a published revision have exact event-position errors; an
exact UTC between observed edits is not an exact event-index target. Interval
points are evaluated using distance to, and membership in, `(lower,upper]`.
Distance zero at the open lower boundary is not a hit. Report all denominators.

## First real run (not a final paper claim)

EM converged at iteration 33, fitting TRAIN only. TEST has 50 entities:

* 9 EXACT, 29 INTERVAL and 12 right-censored controls;
* 38 detected entities received onset predictions; the 12 controls are explicitly
  unsupported for this retrospective task, not scored as correct classifiers;
* 9 EXACT UTC points: fused MAE 0.8840 days, behavior-only 0.7700 days,
  delay-only 3.5472 days;
* only 8 of those EXACT points have an exact published-event index; corresponding
  event-position MAE is 5.875, 5.5 and 5.125;
* fused interval point hit rate is 15/29 (51.72%); behavior-only also 15/29,
  delay-only 17/29;
* fused mean interval distance is 5.5788 days; behavior-only 0.4736 days and
  delay-only 10.7023 days. This is not a point-onset error metric.

This first real run does **not** demonstrate that fusion outperforms behavior-only
evidence. Paired UTC-error bootstrap intervals cross zero and the EXACT TEST
cohort is very small. No weights or features were selected from these TEST
results. Retain this sealed baseline; use VAL or a separately designed new held-
out protocol for later development, not repeated tuning against these TEST labels.

## Validation and limits

20 tests passed: 9 existing contract/stage-2 tests and 11 new length/scorer tests.
The variable runner can fit/predict with evaluation gold physically absent.
Changing storage padding cannot change fitted parameters. With all lengths equal,
the modified EM matches the original frozen source exactly on a test fixture.

Raw source labels and sampling limits remain unchanged. Metadata-state features
do not directly represent semantic text violations. The selected normal windows
and positive histories differ in length/calendar coverage. Matched grouping is
not proof that every unobserved related account or shared page is separated.
No independent human sign-off, random Wikipedia sampling or reliable general-
purpose real-data superiority is asserted.
