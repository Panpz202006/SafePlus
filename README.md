# SAFEPlus

SAFEPlus studies sequential fraud detection with delayed observation of
fraud. The repository includes a frozen legacy SAFE baseline and a
synthetic likelihood pilot for the SAFEPlus formulation.

## Repository layout

- [`safeplus/`](safeplus/) contains reusable library code. `data.py` prepares
  legacy histories and checks exact cross-split feature overlap. `training.py`
  runs baseline training and validation-only checkpoint selection.
  `evaluation.py` selects validation thresholds and scores fixed test
  prefixes. The full SAFEPlus sequence model is pending.
- [`baselines/`](baselines/safe/PROVENANCE.md) contains the frozen legacy
  SAFE implementation, its input arrays, and source provenance. It supplies
  a reproducible comparator for upcoming SAFEPlus experiments.
- [`datasets/`](datasets/) holds the source archives for future preprocessing.
  The common dataset format and account-level split validation are pending.
- [`experiments/`](experiments/) contains runnable studies,
  protocols, and reference outputs for the paper:
  - [`comparison/`](experiments/comparison/) holds the audited SAFE
    rerun and the protocol for matched comparisons with SAFEPlus and other
    methods.
  - [Component ablations](#component-ablations) are planned for the full
    SAFEPlus model.
  - [`verification/`](experiments/verification/)
    contains the synthetic likelihood and identifiability pilot.
  - `runs/` holds generated results in Git-ignored directories.
- [`tests/`](tests/) contains automated checks for the synthetic likelihood,
  batching, split overlap handling, and prefix scoring.
- [`paper/`](paper/) contains the LaTeX draft and build
  configuration. Run `make -C paper` to build the PDF under `paper/build/`.

## Quick start

Run the standard-library likelihood pilot first. For the full test suite
and legacy SAFE runner, use Python 3.11 in a virtual environment and the
[verified baseline dependencies](experiments/comparison/safe_baseline/requirements.txt).

```sh
python3 -m experiments.verification.synthetic_likelihood.run
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r experiments/comparison/safe_baseline/requirements.txt
python -m unittest discover -s tests -v
```

The separate `data/supplychain-lifecycle-300-v1` history contains an earlier
SAFEPlus prototype and dataset adapters. Review those components against
the current data and evaluation protocol before porting them to `main`.

## Comparator protocol

Freeze account-disjoint splits and the observation cutoffs before training.
Fit preprocessing on training data, and select thresholds and checkpoints
using validation data. Score each method on the same held-out accounts
and prediction timestamps. Record the feature availability and label
availability checks with the results.

The bundled legacy SAFE results are exploratory. Account identity and
upstream feature timing are unverified for the legacy arrays. Other
state-of-the-art comparators and the SAFEPlus sequence model are pending
implementation and evaluation.


## Run the audited SAFE baseline

From the repository root, use the Python 3.11 environment from Quick start.
The pinned dependencies were verified on macOS arm64. The runner loads the
frozen input arrays bundled under `baselines/safe/`, some of which use
Python 2 pickle serialization.

```sh
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OPENBLAS_NUM_THREADS=1
python -m experiments.comparison.safe_baseline.run twitter
python -m experiments.comparison.safe_baseline.run wiki
```

The runner uses seed 7, filters exact full-feature matches across splits
without consulting labels, and preserves the legacy test splits. It selects
checkpoints and thresholds by mean validation accuracy over prefixes 1–5.
Test inference receives those five prefixes and scores every test record.
The [reference transcripts](experiments/comparison/safe_baseline/reference/)
report precision, recall, F1, and accuracy. Reruns write JSON summaries under
`experiments/runs/safe_baseline/<dataset>/<run-id>/`.

## Synthetic likelihood verification

The pilot uses constant onset and detection probabilities of 0.08 and 0.30
over 20 bins, with same-bin detection allowed and latent onset held out of
fitting. It runs five seeds with 5,000 training and 5,000 held-out accounts
per seed. The 189 likelihood checks compare explicit marginalization,
enumeration, and the linear-time recurrence. The pilot also checks posterior
normalization, gradients, and long-horizon numerical stability.

Swapping the hazards preserves the detection likelihood while changing
inferred onset risk. Supplying the true detection hazard recovers the
generating onset probability in this setting. The next verification step is
to test whether a small, randomly revealed subset of onset labels resolves
the ambiguity. See the
[archived design and results](experiments/verification/synthetic_likelihood/reference/).

```sh
python -m unittest discover -s tests -p 'test_synthetic_likelihood.py' -v
python -m experiments.verification.synthetic_likelihood.run --plot
```

Omit `--plot` to run with the standard library. Plotting requires Matplotlib.
Use `--out PATH` to choose a destination. By default each run writes to a
unique directory under `experiments/runs/synthetic_likelihood/`.

## Component ablations

Use full SAFEPlus as the reference and remove or replace one component at a
time. Keep the account splits, prefix cutoffs, outcome definitions, metrics,
selection budget, and seeds fixed across variants. Planned contrasts include
a single detection-time process versus separate onset and detection processes,
and a shared versus history-dependent delay if the full model includes that
component. Assess onset recovery using synthetic or independently audited
onset labels.

## Paper

Edit [`paper/main.tex`](paper/main.tex), the files in `paper/sections/`, and
`paper/references.bib`. Build with `make -C paper` using TeX Live or MacTeX
with `latexmk`. For Overleaf, upload the paper source and select `main.tex`.
The draft uses anonymous authors and review line numbers. Resolve the
provisional conceptual-note attribution before submission.

The LaTeX style and example files come from the
[IJCAI author kit](https://ijcai.org/authors_kit/). Check the
[IJCAI 2027 main-track call](https://2027.ijcai.org/ijcai-2027-call-for-papers-main-track/)
for submission requirements and deadlines.
