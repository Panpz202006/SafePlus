# SafePlus model-ready controlled sequences v1

This release contains controlled temporal sequences for testing latent onset
recovery and delayed detection. It is organized in batches of 100 so that each
experimental condition can be validated before the next condition is added.

## Current release

The current branch contains 800 sequences:

| Batch | Behavior after tf | Detection delay | Model entity IDs |
| --- | --- | --- | --- |
| 001 | Immediate full-strength shift | Fixed at 4 steps | 1–100 |
| 002 | Immediate full-strength shift | Truncated geometric, 1–8 steps | 101–200 |
| 003 | Gradual shift beginning at tf | Fixed at 4 steps | 201–300 |
| 004 | Gradual shift beginning at tf | Truncated geometric, 1–8 steps | 301–400 |
| 005 | Immediate weak shift (35% signal strength) | Truncated geometric, 1–8 steps | 401–500 |
| 006 | Gradual weak shift (35% signal strength) | Truncated geometric, 1–8 steps | 501–600 |
| 007 | Immediate weak shift (35% signal strength) | Fixed at 4 steps | 601–700 |
| 008 | Gradual weak shift (35% signal strength) | Fixed at 4 steps | 701–800 |

Every sequence has 24 time steps, eight time-varying features, at least four
normal-prefix steps, a known controlled onset, and an in-window detection time.
Each batch uses a deterministic 70/10/20 entity-level split.

## Label provenance

- t_onset/tf is a controlled injected action time.
- t_detect/td is generated from a prespecified delay rule.
- These labels are not real platform timestamps and must not be described as
  manually verified Wikipedia fraud or platform detection times.
- t_onset and evaluation-only phase columns are not included in x.
- Samples were generated from fixed protocols, not selected by model scores.

## Files in each batch

- model_input.npz: canonical SafePlus arrays.
- sequence_index.csv: one row per sequence, including label provenance.
- timeline_evaluation_only.csv: step-level features and evaluation-only phases.
- manifest.json: generation protocol and model contract.
- validation_report.json: deterministic data checks.
- checksums_sha256.json: file hashes.
- README.md: batch-specific notes and limitations.

The canonical NPZ keys are x, lengths, detected, t_detect, t_onset, and
entity_id.

The files have been loaded with the repository's SequenceDataset. Entity IDs
are unique across all released batches.

## Scope limitation

The first eight batches contain detected positive sequences. They are suitable
for controlled onset-recovery experiments, the 2-by-2 comparison of behavior
alignment and delay stability, and a weak-signal comparison between batches 002
and 005, plus the corresponding gradual-signal comparison between batches 004
and 006. They are not a standalone discrimination
benchmark: AUPRC, AUROC, precision, recall, and F1 require negative or censored
controls.

## Reproduction

The generator is stored in scripts/generate_model_ready_batch.py and requires
Python and NumPy. Run it from an Anaconda environment and choose a new output
directory, batch ID, entity prefix, non-overlapping entity-ID offset, seed,
behavior mode, and delay mode.

Example for batch 004:

    python scripts/generate_model_ready_batch.py ^
      --output batch_004_gradual_random_delay_100 ^
      --batch-id batch_004_gradual_random_delay_100 ^
      --entity-prefix ctrl_b004 ^
      --entity-id-offset 300 ^
      --count 100 ^
      --seed 20261029 ^
      --length 24 ^
      --features 8 ^
      --behavior-mode gradual_ramp ^
      --delay-mode truncated_geometric ^
      --delay-max 8 ^
      --geometric-p 0.25 ^
      --min-prefix 4

Batch 005 uses the same aligned/random-delay condition as batch 002 but adds
`--signal-strength 0.35` to isolate the effect of weaker behavioral evidence.
Batch 006 applies the same 35% signal strength to the gradual/random-delay
condition from batch 004.
Batch 007 applies the same 35% signal strength to the aligned/fixed-delay
condition from batch 001.
Batch 008 applies the same 35% signal strength to the gradual/fixed-delay
condition from batch 003, completing the weak-signal 2-by-2 matrix.
