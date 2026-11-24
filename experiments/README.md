# Experiments

## Paper results

- [`comparison/`](comparison/README.md) compares the full proposed method
  with SAFE and other relevant methods on the same evaluation protocol.
- [`ablation/`](ablation/README.md) measures the contribution of individual
  SAFEPlus components against the full method.

## Verification

[`verification/synthetic_likelihood/`](verification/synthetic_likelihood/README.md)
checks the likelihood and probes identifiability with constant hazards.

Each study keeps its protocol in `README.md`, its entry point in `run.py`,
and reviewed evidence in `reference/`. Generated output goes under
`runs/`, which Git ignores. Shared implementation lives in `safeplus/`
and tests live in `tests/`.

```sh
python -m experiments.comparison.safe_baseline.run wiki
python -m experiments.verification.synthetic_likelihood.run
python -m unittest discover -s tests -v
```
