# Wikipedia tf audit release (2026-10-21)

This directory records the completed Wikipedia `t_f` annotation and adjudication work used by the SafePlus dataset pipeline.

## Files

- `B_D2正式一致性_人工终审完成.xlsx`: final B/D2 agreement analysis, 99 resolved disagreements, and 410 unique labels.
- `SafePlus_tf人工审计交付包_20261021.zip`: annotation requirements, source audit batches, evidence records, and the final workbook.

## Review status

- Accounts reviewed: 410
- Pairwise disagreements: 99
- Disagreements resolved: 99
- Final reviewer identifier: `D2`
- Final unique labels: 410
- `positive_onset`: 125
- `right_censored_control`: 4
- `excluded`: 148
- `uncertain_exclude`: 133

Records labeled `uncertain_exclude` are retained for traceability but should not enter training, validation, or test sets.

## Integrity

SHA-256 for the delivery package:

```text
FC936634C40F66CAAAA8458C9FCF3021305947A0150448C4F362EAB15CC506E8
```
