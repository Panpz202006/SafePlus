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
- `h_d(t|x,t_f)`: fraud is detected at `t`, active only after commission.

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
safeplus/data/           canonical schema, CSV loader, synthetic generator
safeplus/models/         SAFE, SafePlus, GRU classifier
safeplus/evaluation/     discrimination, prefix, onset and calibration metrics
safeplus/training/       deterministic trainer and checkpoint selection
safeplus/reporting/      aggregate runs and emit paper-ready CSV/LaTeX tables
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
safeplus train --config configs/synthetic/safeplus.yaml
safeplus evaluate --run-dir runs/synthetic/safeplus/seed_0
safeplus predict --run-dir runs/synthetic/safeplus/seed_0 \
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

For a no-install smoke test:

```bash
bash scripts/smoke_test.sh
```

## Canonical data format

Each `.npz` split contains padded arrays:

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

## Recommended review matrix

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
