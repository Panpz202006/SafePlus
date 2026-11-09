# SafePlus Supply-Chain Lifecycle 300

This directory contains a provisional package-level lifecycle dataset for SafePlus experiments.
Each malicious sequence covers the observable package history before the first known malicious
release (`t_f`) and the interval to a public detection signal (`t_d`). Normal controls contain
continuous registry histories and no `t_f`/`t_d` label.

## Current release

- 300 package sequences: 200 malicious and 100 normal controls.
- 42,669 version-level timeline rows.
- 692 evidence rows.
- R1 evidence audit completed for all 300 samples.
- R2 chronology and cross-table audit completed for all 300 samples.
- R1/R2 binary inclusion agreement: 300/300.
- P1 follow-up completed for all 15 samples with only two pre-`t_f` normal versions.
- P2 follow-up completed for all 107 samples without a release between `t_f` and proxy `t_d`.
- Final control screening completed for all 100 normal controls.
- Model-ready inclusion and a campaign-held-out split are available for all 300 samples.

## Important limitations

- The 200 malicious `t_d` values are public-advisory proxy times. They are not official npm or
  PyPI removal times.
- 107 malicious samples have no additional registry version between `t_f` and the observation
  endpoint. Detection is represented by the evidence table rather than a later package release.
- The 100 normal controls passed registry-history and known-malicious-record screening, but their
  status remains pending final manual confirmation.
- Most malicious samples come from two Shai-Hulud campaigns. Results must therefore report
  campaign concentration and use campaign-aware splitting.
- `SC-M-0187` depends on a preserved historical registry timestamp for one normal-prefix version
  that is no longer visible in the current registry response.

## Main files

- `exports/provisional_samples_300.csv`: one row per package sequence.
- `exports/provisional_timeline.csv`: complete version-event timelines used by the release.
- `exports/provisional_evidence.csv`: advisory and registry evidence.
- `exports/review_r1.csv`: first evidence audit.
- `exports/review_r2.csv`: second chronology and evidence cross-check.
- `exports/review_priority_p1_thin_prefix.csv`: official-registry follow-up for thin histories.
- `exports/review_priority_p2_no_post_tf_release.csv`: evidence closure for sequences without an interim release.
- `exports/review_priority_controls_final.csv`: official-registry and malicious-name cross-check for controls.
- `exports/model_ready_samples_v1.csv`: final inclusion status and deterministic train/validation/test split.
- `exports/completion_summary.json`: completion counts, split counts and remaining limitations.
- `reports/SafePlus_SupplyChain_R2_Audit.xlsx`: readable audit workbook.
- `scripts/supplychain_pipeline.py`: collection, export, validation, R1 and R2 pipeline.
- `scripts/review_priority_p1.py`: reproducible P1 follow-up check.
- `scripts/complete_remaining_audits.py`: P2 audit, control audit and model split generation.

## Reproduction

Run commands from this directory after installing the repository environment:

```bash
python scripts/supplychain_pipeline.py validate-provisional
python scripts/supplychain_pipeline.py review-provisional-r1
python scripts/supplychain_pipeline.py review-provisional-r2
python scripts/review_priority_p1.py
python scripts/complete_remaining_audits.py
```

The scripts expect the working database and registry cache used during collection. Those large
intermediate caches are intentionally excluded from Git. The exported CSV files and evidence
records are the versioned research artifacts.

## Model-ready split

- Train: 49 malicious and 68 normal samples.
- Validation: 22 malicious and 18 normal samples.
- Test: 129 malicious and 14 normal samples.
- All Shai-Hulud wave 2 malicious samples are isolated in the outer test set.

The automated evidence and registry audits are complete. The release remains a B-grade benchmark
because malicious `t_d` values are public-advisory proxies. Upgrading those labels requires official
platform action or removal timestamps that are not present in the current public evidence.
