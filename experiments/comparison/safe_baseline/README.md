# Audited legacy SAFE rerun

The runner uses the tracked snapshot in `baselines/safe/`. See its
[provenance note](../../../baselines/safe/PROVENANCE.md) for the source revision
and bundled inputs.

From the SAFEPlus root, run each dataset in a fresh Python 3.11 process.
The requirements file lists the versions verified on macOS arm64. The
runner reads the frozen, repository-bundled legacy NumPy arrays, some of
which use Python 2 pickle serialization.

```sh
python -m pip install -r experiments/comparison/safe_baseline/requirements.txt
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OPENBLAS_NUM_THREADS=1
python -m experiments.comparison.safe_baseline.run twitter
python -m experiments.comparison.safe_baseline.run wiki
```

The runner uses seed 7, filters exact full-feature matches across splits without
using labels, preserves the legacy test splits, and selects the checkpoint and
threshold by mean validation accuracy over prefixes 1–5. Test inference receives
only those five prefixes and includes all test records. The reference transcripts in `reference/`
report precision, recall, F1, and accuracy in that order.

The rerun is exploratory. The legacy test results informed the audit protocol,
and the bundled arrays lack IDs and feature timestamps for verifying account
separation and upstream feature timing.

Each rerun writes a machine-readable summary under
`experiments/runs/safe_baseline/<dataset>/<run-id>/`.
