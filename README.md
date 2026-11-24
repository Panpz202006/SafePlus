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
- [`experiments/`](experiments/README.md) contains runnable studies,
  protocols, and reference outputs for the paper:
  - [`comparison/`](experiments/comparison/README.md) holds the audited SAFE
    rerun and the protocol for matched comparisons with SAFEPlus and other
    methods.
  - [`ablation/`](experiments/ablation/README.md) defines comparisons between
    full SAFEPlus and variants that change one component at a time.
    Ablation results are pending.
  - [`verification/`](experiments/verification/synthetic_likelihood/README.md)
    contains the synthetic likelihood and identifiability pilot.
  - `runs/` holds generated results in Git-ignored directories.
- [`tests/`](tests/) contains automated checks for the synthetic likelihood,
  batching, split overlap handling, and prefix scoring.
- [`paper/`](paper/README.md) contains the LaTeX draft and build
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
