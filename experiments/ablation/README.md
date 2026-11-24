# Component ablations

Use the full SAFEPlus implementation as the reference. For each planned
ablation, remove or replace one component and document the resulting model
and what the comparison can establish. Use the same data, prefix cutoffs,
outcome definitions, metrics, model selection budget, and seeds as the
full-model run. Select checkpoints and thresholds on validation data.

Planned contrasts include a single detection-time process to assess the
contribution of separating onset and detection, and a shared versus
history-dependent detection delay if the full model includes that
component. Onset recovery requires synthetic ground truth or independently
audited onset labels.

Ablation results are pending implementation of the full SAFEPlus model.
