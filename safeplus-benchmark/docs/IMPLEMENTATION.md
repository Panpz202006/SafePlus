# SAFE+ draft implementation

The implemented model follows Sections 2.3 and 3 of the supplied draft,
"Survival Analysis for Fraud Detection". It discretizes time into observed bins.
The causal GRU summarizes covariates up to the current bin. Separate sigmoid
heads predict commission and detection hazards. Detection starts in the onset
bin; no detection is possible before commission.

For onset k, its mass is h_f[k] times the product of (1-h_f[j]) for j < k.
For detection at d, multiply this by the product of (1-h_d[j]) for k <= j < d,
and by h_d[d]. Sum these joint masses over k <= d.
For censoring at c, sum onset masses times detection survival through c,
and add commission survival through c. This last term represents no commission
yet, not a confirmed legitimate entity.

Training uses log-sigmoid and log-sum-exp without clipping the likelihood.
Prefix commission survival and suffix detection survival make exact inference
O(batch size * padded length). This is exact for the discrete model, not an
exact solver for the continuous-time integrals in the draft.

`SafePlus.loss` returns:

- `log_likelihood`: observed-data log likelihood for each entity;
- `onset_posterior`: probabilities for each onset bin, zero beyond the endpoint;
- `no_commission_posterior`: probability of no commission by the censoring horizon;
- `latent_posterior`: onset bins followed by the no-commission state;
- `risk`: causal cumulative commission probability, without conditioning on a
  future detection label. Posterior onset inference does condition on that label
  and should not be used as an online alarm score.

The full posterior sums to one. For censored entities, the onset-only posterior
can sum to less than one. No ground-truth `t_onset` enters training.

Exact marginal likelihood is the default. To use the draft's stochastic-EM idea:

```yaml
model:
  name: safeplus
  hidden_dim: 32
  inference: stochastic_em
  em_samples: 8
```

Samples come from the detached exact discrete posterior; SGD optimizes the
sampled complete-data log likelihood. The final state includes censored entities
without commission. Evaluation always uses the exact observed likelihood.
This sampler still computes the full discrete posterior and offers no asymptotic
speed advantage over exact inference. Training losses under the two objectives
are not directly comparable.

The `safe-r` model implements ordinary event-time survival (Section 2.1), while
`safe` retains the left-censoring objective (Section 2.2).

## Scope and remaining research work

This implementation covers the core latent model and discrete inference.
Section 4's irregular-time attention, global/local representations, null-event
preprocessing, and Section 6's future distillation are not implemented. The
current dataset schema is a discrete observation grid. Continuous-time hazard
integration and approximate variational inference also remain separate work.
Real-data adapters and the paper's full multi-dataset experiments remain to be
completed; synthetic smoke tests are not evidence of benchmark performance or
identifiability of latent onset from detection labels alone.
