# SafePlus combined controlled release (1000 sequences)

This folder combines ten prespecified 100-sequence batches into one model-ready release.
Every sequence has a normal prefix, a controlled onset, and an in-window controlled detection time.

## Provenance

- t_onset/tf is a controlled injected action index, not a real platform timestamp.
- t_detect/td is produced by a declared delay rule, not a real discovery or ban time.
- Samples were not selected according to model performance.

## Files

- model_input_all.npz: 1000 canonical model sequences.
- sequence_index_all.csv: entity-level protocol and provenance.
- timeline_evaluation_only_all.csv: 24000 evaluation-only time-step rows.
- batch_summary.csv: ten batch conditions.
- validation_report.json: release-level checks.
- manifest.json: release contract and limitations.

## Limitation

All 1000 sequences are detected positives. This release supports controlled onset-recovery studies but cannot alone provide valid discrimination metrics such as AUPRC, AUROC, precision, recall, or F1.
