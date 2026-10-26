# SafePlus mixed controlled benchmark v1

This dataset contains 1000 independent controlled synthetic account sequences.
It includes normal controls, detected fraud sequences, and censored fraud
sequences. It is designed to exercise discrimination, censoring, delayed
detection, and latent-onset evaluation in one model-ready package.

## Composition

- normal controls: 650
- detected fraud: 280
- censored fraud: 70
- sequence shape: 1000 x 24 x 8
- split: stratified entity-level 70/10/20

## Important label distinction

`detected` is the observed event indicator used by the current model contract.
It is not the same as true fraud when a fraud sequence is censored. The true
fraud status and injected onset are stored only in evaluation metadata and are
not included in `x`.

## Provenance and limitation

All features, onsets, delays, and censoring decisions are generated under the
prespecified protocol in `manifest.json`. They are not Wikipedia, Twitter, or
other real-platform observations and must not be reported as manually verified
events. This benchmark supports controlled method checks, not real-world claims.
