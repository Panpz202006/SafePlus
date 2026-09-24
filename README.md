# SafePlus Benchmark

Reproducible performance-review framework for **SafePlus**, the joint latent
fraud-commission and delayed-detection model proposed as a successor to SAFE.

The repository separates three empirical questions:

1. **Fraud discrimination** — does the model rank fraudulent entities well?
2. **Early detection** — does it identify them before the operational response time?
3. **Latent-onset recovery** — when ground-truth action time is available but hidden
   during training, does the posterior recover it?

## Model contract

For entity history `x[0:T]`, SafePlus predicts two discrete hazards:

- `h_f(t|x)`: fraud is commissioned at time `t`;
- `h_d(t|x)`: conditional detection hazard, active only after commission.
  The current head does not explicitly receive onset or elapsed delay; conditioning
  on onset determines which detection-survival factors are active.

For a detected entity, the observed likelihood marginalizes the latent onset:

```text
p(t_d=d|x) = sum_{k<=d} p(t_f=k|x) p(t_d=d|t_f=k,x)
```

For an entity not detected by horizon `T`, it includes both legitimate/not-yet-
fraudulent histories and committed-but-undetected histories. The reference model
computes this marginal exactly in discrete time. A posterior over `t_f` is returned
for interpretability and controlled onset-recovery evaluation.

## Repository map

```text
configs/                 experiment YAMLs
safeplus/data/           canonical NPZ schema, loader, synthetic generator
safeplus/models/         GRU, SAFE-r, SAFE, SafePlus
safeplus/evaluation/     discrimination, prefix, early-alarm and onset metrics
safeplus/training/       seeded training and validation checkpoint selection
safeplus/cli.py          workflows and CSV/LaTeX run aggregation
scripts/                 train/evaluate/aggregate entry points
tests/                   likelihood, censoring, posterior and smoke tests
docs/                    benchmark protocol and dataset adapter guide
```

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
safeplus synthetic --output data/processed/synthetic_demo.npz
safeplus train --config configs/synthetic/safeplus.yaml --output-dir runs/first_run
safeplus evaluate --run-dir runs/first_run/seed_0
safeplus predict --run-dir runs/first_run/seed_0 \
  --data data/processed/synthetic_demo.npz --output predictions.npz
safeplus aggregate --runs runs --output reports
pytest -q
```

The same commands work as `python -m safeplus` without installing the console
entry point, provided dependencies are installed. Use `python -m safeplus --help`
or `python -m safeplus train --help` for options. Run from the repository root;
relative paths in YAML are resolved from the current working directory.

For a short training run with a separate output directory:

```bash
python -m safeplus train --config configs/synthetic/safeplus.yaml \
  --epochs 2 --batch-size 32 --device cpu --output-dir runs/quickcheck
```

Training saves `model.pt`, the effective `config.yaml` (including CLI overrides),
`history.json`, and `metrics.json` under `<output-dir>/seed_<seed>`. A nonempty run
directory is rejected to avoid mixing checkpoints and results; use another output
directory or seed for a new run. `evaluate` displays saved test metrics.

`predict` runs the checkpoint on an NPZ containing `x[N,L,D]` and `lengths[N]`;
`entity_id` is optional. Use the same features, ordering, and preprocessing as
training. Its output NPZ contains per-step `risk`, `final_risk`, `prediction`,
`threshold`, `lengths`, and `entity_id`. Predictions use the saved validation
threshold. Padding repeats the last valid risk. SAFE+ risk estimates commission;
SAFE-r risk estimates detection. No detection or onset labels are needed.

The original `scripts/*.py` entry points remain available.

For a smoke test without installing the package itself (dependencies, including
PyTorch and pytest, must already be available in the active Python environment):

```bash
bash scripts/smoke_test.sh
```

## Canonical data format

The training CLI loads one `.npz` and creates seeded train/validation/test subsets.
The file contains padded arrays:

| Key | Shape | Meaning |
|---|---:|---|
| `x` | `[N,L,D]` | time-varying covariates |
| `lengths` | `[N]` | valid sequence lengths |
| `detected` | `[N]` | whether detection/blocking occurred |
| `t_detect` | `[N]` | zero-based detection time; horizon if censored |
| `t_onset` | `[N]` | true onset for controlled evaluation, `-1` if unknown |
| `entity_id` | `[N]` | stable entity identifier |

Raw datasets are never committed. Adapters must preserve chronological ordering,
fit preprocessing on the training split only, and document whether a timestamp is
an action time (`t_a`), a detection time (`t_d`), or a proxy.

## Planned research review matrix

The following is a research plan; the listed real-data adapters, extra baselines,
calibration metrics, and interval coverage are not all implemented.

| Track | Datasets | Models | Primary metrics |
|---|---|---|---|
| SAFE reproduction | SAFE–Twitter, SAFE–Wiki | SVM, Cox, GRU, SAFE | F1/Accuracy@1…5, early-detected rate/steps |
| Delayed detection | Twitter, Wiki, transaction sequences | GRU, SAFE-r, SAFE, SafePlus | AUPRC, AUROC, detection NLL, lead time |
| Onset recovery | BankSim, PaySim, Sparkov, Handbook, synthetic | SAFE proxy, SafePlus | onset MAE, posterior NLL, interval coverage |
| Robustness | controlled data | SafePlus variants | metrics by delay/censoring bucket |

Use at least five seeds. Select thresholds only on validation data. Report bootstrap
95% confidence intervals and paired seed-level comparisons. Never describe a
transaction timestamp as the true onset of fraudulent intent.

## Adding a dataset

Implement an adapter that emits the canonical `.npz` format, then copy
`configs/template.yaml`. See `docs/DATASETS.md` and `docs/PROTOCOL.md`.

## Reproducibility

- configuration and seed are copied into every run directory;
- checkpoints are selected by validation AUPRC;
- test data are evaluated once after selection;
- results are stored as machine-readable JSON;
- aggregation emits `summary.csv` and `summary.tex`.

## Status

This is a research scaffold, not a claim that all ten candidate datasets provide
verified `t_f` and `t_d`. Twitter/Wiki provide operational endpoints; controlled
datasets enable direct recovery tests by hiding known injected action times and
simulating detection delays.


## SAFE+ inference options

The core joint model uses stable, linear-time exact inference on discrete bins.
It returns both onset probabilities and the no-commission posterior for censored
entities. Set `model.inference: stochastic_em` and `model.em_samples: 8` for
posterior-sampled training. Use `model.name: safe-r` for the ordinary detection-time
survival baseline. See [implementation details and scope](docs/IMPLEMENTATION.md)
for the connection to the draft and remaining research components.

## Implemented algorithms

All four models use a unidirectional GRU with input `x[B,L,D]`, hidden states
`[B,L,H]`, and a scalar risk per time bin `[B,L]`. Valid outputs are causal;
packing excludes padded observations from the recurrent state. Times are zero-based.
Below, `d` is observed detection time, `c = lengths - 1` is the last observed bin,
`h(t)` is a hazard, and `S(t) = product_{j=0..t}(1-h(j))`.

### GRU classifier — `model.name: gru`

[Implementation](safeplus/models/baselines.py). A linear head maps each hidden
state to a logit; `risk(t) = sigmoid(logit(t))`. Binary cross-entropy assigns the
entity's observed `detected` label to every valid prefix. Loss averages over valid
time steps, so longer sequences receive more weight. Scores can rise or fall;
there is no event-time likelihood, censoring model, or onset posterior. Neither
`t_detect` nor `t_onset` supervises this classifier.

This is a useful discrimination baseline. Here its label is eventual observed
detection, not verified fraud: censored fraudulent entities count as negatives.
Early-prefix labels can be positive even before true commission.

### SAFE-r — `model.name: safe-r`

[Implementation](safeplus/models/baselines.py). The head defines a detection
hazard. The objective is the negative mean log likelihood over entities:

```text
Detected at d: L = h(d) * product_{j<d}(1-h(j))
Censored at c: L = S(c)
Online risk:   1 - S(t), cumulative detection probability
```

Log-sigmoid and cumulative log survival stabilize training. A detected event
contributes survival strictly before `d`, followed by the event at `d`.
Censoring includes survival through `c`. This model does not recover commission
onset, and its risk has a different meaning from SafePlus commission risk.

### SAFE — `model.name: safe`

[Implementation](safeplus/models/safe.py). This discrete reference model rewards
an event anywhere up to the observed detection deadline:

```text
Detected by d: L = 1 - S(d)
Censored at c: L = S(c)
Online risk:   1 - S(t)
```

Unlike SAFE-r, positive examples do not require event mass exactly at `d`.
There is one hazard head and no separate delay model or onset posterior. The
implementation uses probability-space cumulative products, an upper hazard clamp,
and a likelihood floor of `1e-8`. Long horizons can saturate scores, and the floor
can suppress gradients. These are debugging targets, not confirmed explanations
for every poor run. Raw model output includes padded-bin scores; the trainer and
prediction CLI replace padding with the last valid score.

### SafePlus — `model.name: safeplus`

[Implementation](safeplus/models/safeplus.py). Two heads predict commission and
detection hazards. Define `p_f(k) = h_f(k) * product_{j<k}(1-h_f(j))`:

```text
Detected at d:
  L = sum_{k=0..d} p_f(k) * product_{j=k..d-1}(1-h_d(j)) * h_d(d)
Censored at c:
  L = product_{j=0..c}(1-h_f(j))
      + sum_{k=0..c} p_f(k) * product_{j=k..c}(1-h_d(j))
Online risk:
  1 - product_{j=0..t}(1-h_f(j))
```

An empty product is one, allowing commission and detection in the same bin.
Exact inference uses prefix commission survival, suffix detection survival, and
`logsumexp` in **O(BL)** time and storage for the inference arrays, excluding GRU
computation. Training averages marginal negative log likelihood over entities.

`loss()` also returns a posterior over `L` onset bins plus one no-commission
state. The full posterior sums to one; onset-only mass may be less than one for
censored entities. This posterior conditions on observed detection/censoring and
is for retrospective recovery, not online alarms. `t_onset` is evaluation-only.
The two latent mechanisms may not be identifiable from detection labels alone.

Optional `inference: stochastic_em` samples from the detached exact posterior
and optimizes sampled complete-data log likelihood. `em_samples` controls the
number of samples. Evaluation still uses exact marginal likelihood. This option
adds sampling noise and does not remove exact posterior computation.

## Run all four models

The synthetic configurations share hidden size 32, batch size 64, learning rate
0.001, a 40-epoch limit, and patience 7. Run from the repository root after setup:

```bash
for model in gru safe-r safeplus safe; do
  for seed in 0 1 2 3 4; do
    python -m safeplus train --config "configs/synthetic/$model.yaml" \
      --seed "$seed" --output-dir "runs/comparison/$model"
  done
done
python -m safeplus aggregate --runs runs/comparison --output reports/comparison
```

Choose a new output directory if a run already exists. Each seed controls both
initialization and the 70/10/20 entity split; matching seeds give matching splits
across models. This loader does not implement calendar-time or group-aware splits;
adapters must ensure one sequence per entity to avoid cross-split entity leakage.

Edit YAML for `hidden_dim`, `learning_rate`, `patience`, or model-specific options.
CLI overrides support `epochs`, `batch_size`, `device`, `seed`, and `output_dir`.
Only SafePlus accepts `inference` and `em_samples`. The current CLI does not expose
all `Trainer` options, such as weight decay. For CPU timing, control PyTorch thread
counts explicitly; the saved review runner uses one compute and one interop thread.

## Metrics and interpretation

- Checkpoint selection uses validation average precision (reported as AUPRC) at
  the last valid step. Strict improvements reset early-stopping patience.
- Threshold selection maximizes validation F1 over 181 thresholds from 0.05 to
  0.95; ties choose the lowest candidate. It is a bounded grid, not an exhaustive
  search over score values.
- Test metrics include AUPRC, AUROC, precision, recall, F1, accuracy, and prefix
  F1/accuracy at steps 1–5. Single-class AUPRC/AUROC are reported as NaN.
- Early-detected rate is the fraction of observed positives whose first alarm
  is strictly before `t_detect`. Mean lead time includes all observed positives
  that alarm, including late alarms with negative lead time.
- SafePlus adds posterior-mode onset MAE and posterior NLL for known onsets.
  Censored onset probabilities retain no-commission mass; they are not
  renormalized conditional on commission.

Full-horizon discrimination may include covariates after detection if the input
contains them. It is not a substitute for a pre-detection deployment evaluation.
All current discrimination metrics use `detected`, including for SafePlus's
commission score. Censored fraud therefore creates a target mismatch. Thresholds
optimize final F1, not early recall or a fixed alert budget.

`evaluate` prints saved held-out results; it does not rerun inference. `aggregate`
computes means and sample standard deviations grouped by dataset/model; separate
experiments by directory to avoid pooling different hyperparameters. It does not
compute bootstrap intervals. Detection NLL, Brier score, credible interval
coverage, and delay/censoring stratification are not currently exported by the
standard training workflow.

## Debugging and optimization

Start with [the algorithm debugging guide](docs/DEBUGGING.md), which covers
invariants, numerical failure modes, checkpoint selection, profiling, and
validation-only optimization experiments. Inline comments in the model, trainer,
data, metric, and CLI code explain the relevant contracts.

The [2026-09-24 synthetic review](reports/performance_review_20260924/REVIEW.md)
contains 20 completed runs, saved checkpoints, runtime methodology, and
[per-run results](reports/performance_review_20260924/review_runs.csv).
Mean AUPRC was GRU 0.774, SAFE-r 0.760, SafePlus 0.696, and SAFE 0.270.
SafePlus was seed-sensitive; these results do not establish a real-data advantage.
The review runner and summarizer are saved alongside its results for reproduction.

Use the same interpreter for installation and execution. If tests fail with
`ModuleNotFoundError: torch`, activate the environment where dependencies were
installed, or install with `python -m pip install -e ".[dev]"`. The local review
used `/opt/anaconda3/envs/transformer/bin/python`; this path is machine-specific.
