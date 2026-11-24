# Frozen SAFE baseline

The included implementation is from [upstream SAFE](https://github.com/PanpanZheng/SAFE)
at revision `adf7f73980a68b5504b7ee3f820bcd8faaac2c91` with the stable loss
and cleanup from local revision `4112ee4c41c935da94ebd7047269a569307eecec`.
The [archived patch](legacy-safe.patch)
records that change. The snapshot includes only the baseline entry point,
its helper module, and the 18 legacy input arrays required by the runner.

Use `python -m experiments.comparison.safe_baseline.run twitter` or
`python -m experiments.comparison.safe_baseline.run wiki` from the repository root.
The historical upstream README describes additional programs in the full
upstream repository.
