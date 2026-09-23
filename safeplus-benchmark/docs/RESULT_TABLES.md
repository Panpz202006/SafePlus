# Paper-ready result tables

## Main comparison

| Dataset | Model | AUPRC ↑ | AUROC ↑ | F1 ↑ | Detection NLL ↓ | Lead time ↑ |
|---|---|---:|---:|---:|---:|---:|
| Twitter | SAFE | TBD | TBD | TBD | TBD | TBD |
| Twitter | SafePlus | TBD | TBD | TBD | TBD | TBD |
| Wiki | SAFE | TBD | TBD | TBD | TBD | TBD |
| Wiki | SafePlus | TBD | TBD | TBD | TBD | TBD |

## Controlled onset recovery

| Dataset | Model | Onset MAE ↓ | Onset NLL ↓ | 80% coverage ↑ | Width ↓ |
|---|---|---:|---:|---:|---:|
| BankSim | SAFE proxy | TBD | TBD | — | — |
| BankSim | SafePlus | TBD | TBD | TBD | TBD |

## Delay and censoring robustness

| Mean delay | Censoring | AUPRC ↑ | Onset MAE ↓ | Coverage ↑ |
|---:|---:|---:|---:|---:|
| 1 | 0.0 | TBD | TBD | TBD |
| 4 | 0.2 | TBD | TBD | TBD |
| 8 | 0.6 | TBD | TBD | TBD |

Populate tables through `scripts/aggregate.py`; do not manually copy single-run
results. Mark unavailable ground truth with an em dash rather than an estimated
proxy presented as truth.
