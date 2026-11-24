# SAFEPlus: first empirical verification

This is a restricted synthetic diagnostic, not a trained SAFEPlus system or a SAFE comparison. The supplied three-page draft was reviewed on 2026-11-10. The source PDF fingerprint is recorded in `results/summary.json`.

## Review priorities

1. Equations (10)-(14) give a coherent serial three-state model under the stated assumptions: initially fraud-free, absorbing fraud, no false detection, independent administrative censoring, and onset before detection within a bin. Validate this restricted likelihood before adding a GRU.
2. Identifiability is the main scientific issue. Detection-only likelihood cannot label the two constant hazards, and good detection prediction is not evidence of accurate latent-onset inference. Known delay is an oracle anchor, not proof the practical problem is solved.
3. Observed behavioral covariates may depend on latent fraud. A causal encoder alone does not justify treating such a covariate path as external; the conditional observation assumptions or an emission model need attention before real-data claims.
4. The likelihood should be positioned against latent multi-state/event-history methods. Two hazards and marginalization alone are not an established novelty claim. The supplied conceptual note's attribution/publication status remains unverified. SAFE already addresses delayed labels through its revised loss; avoid describing it as simply fitting exact fraud onset at suspension.

Primary references checked for this review:
- SAFE (AAAI 2019): https://ojs.aaai.org/index.php/AAAI/article/view/3923
- SmoothHazard and interval-censored illness-death modeling: https://www.jstatsoft.org/article/view/v079i07
- Joint latent disease/observation modeling: https://pmc.ncbi.nlm.nih.gov/articles/PMC4415156/

This is targeted positioning, not an exhaustive novelty review.

## Smallest experiment

Use constant per-bin onset probability `a=0.08` and detection probability `b=0.30`. Generate onset first and permit same-bin detection. Observe only `(min(Td,C), delta)` during fitting, with administrative horizon `C=20` bins. Latent onset is reserved for synthetic evaluation. There are no covariates, recurrent networks, or real accounts.

- Compare explicit marginalization, path enumeration, and the O(K) state recurrence.
- Check normalization, zero/one boundaries, posterior sums, separate detection/censoring indicators, and analytic two-parameter gradients versus central finite differences.
- Verify log-space probabilities remain finite at a 1,000-bin stress case.
- Fit the unordered two-hazard pair by a fixed grid and retain both labeled orientations. The low-a branch is an arbitrary display convention, not identification.
- Repeat with the correct detection probability supplied, using a scalar refinement after grid initialization.

Five seeds (7, 11, 19, 23, 31), with 5,000 train and 5,000 held-out accounts per seed. The same accounts are evaluated under each model. No hidden onset labels enter parameter fitting.

## Verified results

- 189 probability/recurrence/path checks; maximum error about 2.22e-16.
- Posterior normalization error about 4.44e-16.
- Analytic versus finite-difference gradient error about 2.03e-10.
- Swapping the generating hazards changes detection probabilities by at most 1.67e-16, but changes fraud risk among undetected accounts at bin 20 from 0.2022 to 0.9966.
- The two fitted unanchored branches have the same mean held-out NLL (2.80010 nats/account), while their mean undetected-fraud risks are 0.2009 and 0.9962. Similar fit does not select the correct branch.
- With the true delay fixed, mean estimated onset probability is 0.08045 (sample SD 0.00149), versus generating value 0.08. Mean undetected-fraud risk is 0.2035, versus the generating value 0.2022.

These are diagnostic results for one simulation setting. The spread across five seeds is sample SD, not a confidence interval. No comparison with SAFE, neural gradient validation, partial-audit recovery, posterior interval coverage, or real-data performance is claimed. Constant positive onset hazards also imply eventual onset for every simulated account; a cure fraction is not tested.

## Run

From the SAFEPlus project root:

```bash
python3 -m unittest discover -s tests -p 'test_synthetic_likelihood.py' -v
python3 -m experiments.verification.synthetic_likelihood.run
```

The experiment and tests require only Python's standard library. Plotting is optional and uses an already installed Matplotlib:

```bash
MPLCONFIGDIR=/tmp/safeplus-mpl /opt/anaconda3/bin/python \
  -m experiments.verification.synthetic_likelihood.run --plot
```

`reference/` preserves the first reviewed run. New runs write to unique directories under `experiments/runs/synthetic_likelihood/`. Use `--out PATH` to choose an explicit destination. `summary.json` contains the design and aggregated metrics; `replicates.csv` contains individual seed/model results; `diagnostic.png` shows the likelihood symmetry, overlapping detection CDFs, and latent-risk ambiguity.

Next small step: randomly reveal a small subset of onset labels, keep the model constant-hazard, and test whether that information selects the correct orientation. Only then add sequence features or compare against a validated legacy SAFE baseline.
