# Dataset adapters and semantic audit

The central benchmark risk is timestamp misinterpretation. Every adapter must fill
the audit fields below before results can enter a paper table.

| Dataset family | Observable endpoint | Safe use | Prohibited claim |
|---|---|---|---|
| SAFE–Twitter | suspension time `t_d` | real delayed-response early detection | verified commission onset |
| SAFE–Wiki | block/response endpoint `t_d` | real delayed-response early detection | exact first vandal intent |
| IEEE-CIS | transaction time + label | chronological action-level detection | operational detection delay without external data |
| ULB Credit Card | elapsed transaction time + label | imbalanced transaction ranking | customer-level onset recovery |
| BankSim/PaySim/Sparkov/Handbook | injected fraudulent action `t_a` | hide `t_a`, simulate `t_d`, score recovery | real operational latency |
| IBM AML/SAML-D | labeled illicit transfer/action | temporal graph/action benchmark | verified latent intent onset |

## Adapter checklist

- [ ] entity identifier is stable and does not leak the label;
- [ ] events are sorted and duplicate timestamps have a deterministic order;
- [ ] time units and zero point are documented;
- [ ] missing values and categorical vocabularies are fit on training data only;
- [ ] `t_detect` means detection/response, not merely a labeled action;
- [ ] `t_onset=-1` unless ground truth exists independently of model supervision;
- [ ] split is chronological or the reason it cannot be is documented;
- [ ] positive prevalence and censoring rate are reported per split;
- [ ] raw licenses permit the intended use and redistribution policy.

## SAFE reproduction

The 2019 draft reports:

- Twitter: 5,540 balanced entities, 21 timestamps, five change features, split 7:1:2;
- Wiki: 1,759 entities, sequence lengths 12–20, eight edit features;
- GRU hidden size 32, Adam learning rate `1e-3`, batch size 16;
- ten runs with mean and standard deviation;
- precision, recall, F1, accuracy, prefix metrics, early-detected rate and steps.

Preserve an explicit `legacy_reproduction` configuration with these choices. Run a
second modern protocol with chronological splits, imbalanced prevalence, AUPRC, and
confidence intervals. Do not mix the two result tables.

