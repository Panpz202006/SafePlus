# Wikipedia Account Lifecycle Dataset v1

This directory contains the public, quality-checked portion of a Wikipedia
account-lifecycle collection prepared for SafePlus experiments.

## Current release

- 3,500 unique registered accounts
- 245,884 public lifecycle events
- 226,163 published revisions
- 19,721 other public account-log events
- 3,667 public block-action records
- 7 non-overlapping collection batches

The following checks passed across all released batches:

- no duplicate account identifiers across batches;
- no event before the reported registration timestamp;
- no event after the selected public action candidate;
- no duplicate event key within a batch;
- no mismatch between entity-level and event-level counts;
- no discontinuous per-account sequence index;
- no missing registration timestamp;
- no left-censored or safety-cap-truncated account in the release tables;
- every released account reached the beginning of its publicly retrievable
  revision history.

## Lifecycle boundary

For each released account, the public observation sequence begins at the
reported registration time and the earliest retrievable public activity. It
ends at the earliest automatically selected relevant public block candidate.
Events after that candidate are not included.

This is a complete **publicly observable** lifecycle. Hidden, suppressed,
deleted, or otherwise unavailable platform data cannot be recovered and is not
inferred or fabricated.

## Label status

`td_candidate_utc` is an automatically selected review candidate. It remains
pending human confirmation as the first related platform action. This release
does not generate a fraud/malicious-behavior onset label (`t_f`) and is not yet
model-ready.

## Files in each batch

- `entities.csv`: one row per account and observation-boundary metadata;
- `events.csv`: ordered public revision and account-log events;
- `platform_actions.csv`: public block-action history;
- `td_review_queue.csv`: human-review queue for the selected action candidate;
- `crawl_requests.csv`: redacted request provenance and response hashes;
- `manifest.json`: collection contract, counts, and skipped-candidate summary.

## Not published

The collection workspace also contains raw API caches and a private mapping
between account names and stable hashed identifiers. Those files are excluded
from this repository. An in-progress eighth batch is also excluded until its
collection and quality checks finish.

## Remaining stages

1. Human verification of `td_candidate_utc`.
2. Collection of revision-diff evidence for onset review.
3. Construction of matched right-censored control accounts.
4. Human onset annotation where the public evidence supports it.
5. Causal feature extraction, temporal binning, and SafePlus-format conversion.

