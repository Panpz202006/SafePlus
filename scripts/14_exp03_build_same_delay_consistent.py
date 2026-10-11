from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from safeplus.data import load_npz, make_loaders


# ============================================================
# Experiment 03H
# Same-delay paired censoring control
#
# Goal:
#
# Reconstruct the ORIGINAL synthetic data generation exactly,
# including every fraud entity's:
#
#   true t_f
#   detection delay
#   candidate t_d
#
# Then remove ONLY the random censoring coin.
#
# Therefore:
#
#   x                  unchanged
#   t_f                unchanged
#   delay              unchanged
#   candidate t_d      unchanged
#   train/val/test     unchanged
#
# ONLY:
#
#   in-window candidate t_d can no longer be randomly hidden.
#
# This creates a clean paired intervention on censoring.
# ============================================================


SOURCE_DATA = Path(
    r"data/processed/synthetic_demo.npz"
)

OUTPUT_DATA = Path(
    r"data/processed/synthetic_same_delay_consistent.npz"
)

REPORT_DIR = Path(
    r"reports/experiment03/same_delay_censor_control"
)


# ============================================================
# These MUST match the original synthetic generator
# ============================================================

N = 2000
LENGTH = 24
FEATURES = 8

FRAUD_RATE = 0.35
CENSOR_RATE = 0.20
DELAY_MEAN = 4.0

GENERATOR_SEED = 0

SPLIT = [0.7, 0.1, 0.2]
SPLIT_SEED = 0

BATCH_SIZE = 64


# ============================================================
# Replay ORIGINAL generator exactly
# ============================================================


def replay_original_generator():

    rng = np.random.default_rng(
        GENERATOR_SEED
    )

    x = rng.normal(
        0,
        1,
        size=(
            N,
            LENGTH,
            FEATURES,
        ),
    ).astype(
        np.float32
    )

    is_fraud = (
        rng.random(N)
        < FRAUD_RATE
    )

    onset = np.full(
        N,
        -1,
        dtype=np.int64,
    )

    original_detected = np.zeros(
        N,
        dtype=np.int64,
    )

    original_t_detect = np.full(
        N,
        LENGTH - 1,
        dtype=np.int64,
    )

    signal = rng.normal(
        0.9,
        0.15,
        size=FEATURES,
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Extra variables retained only for this audit.
    # --------------------------------------------------------

    detection_delay = np.full(
        N,
        -1,
        dtype=np.int64,
    )

    candidate_t_detect = np.full(
        N,
        -1,
        dtype=np.int64,
    )

    original_random_censored = np.zeros(
        N,
        dtype=bool,
    )

    original_natural_censored = np.zeros(
        N,
        dtype=bool,
    )

    censor_draws = np.full(
        N,
        np.nan,
        dtype=np.float64,
    )

    for i in np.flatnonzero(
        is_fraud
    ):

        onset[i] = rng.integers(
            max(
                2,
                LENGTH // 6,
            ),
            max(
                3,
                2 * LENGTH // 3,
            ),
        )

        ramp = np.linspace(
            0.25,
            1.0,
            LENGTH - onset[i],
            dtype=np.float32,
        )[:, None]

        x[
            i,
            onset[i] :,
        ] += (
            ramp
            * signal
        )

        delay = max(
            1,
            int(
                rng.exponential(
                    DELAY_MEAN
                )
            ),
        )

        detection_delay[i] = (
            delay
        )

        candidate = (
            onset[i]
            + delay
        )

        candidate_t_detect[i] = (
            candidate
        )

        if candidate < LENGTH:

            censor_draw = (
                rng.random()
            )

            censor_draws[i] = (
                censor_draw
            )

            if (
                censor_draw
                >= CENSOR_RATE
            ):

                original_detected[i] = 1

                original_t_detect[i] = (
                    candidate
                )

            else:

                original_random_censored[i] = True

        else:

            original_natural_censored[i] = True

    return {
        "x":
            x,

        "is_fraud":
            is_fraud,

        "onset":
            onset,

        "original_detected":
            original_detected,

        "original_t_detect":
            original_t_detect,

        "detection_delay":
            detection_delay,

        "candidate_t_detect":
            candidate_t_detect,

        "original_random_censored":
            original_random_censored,

        "original_natural_censored":
            original_natural_censored,

        "censor_draws":
            censor_draws,
    }


# ============================================================
# Main
# ============================================================


def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DATA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_dataset = load_npz(
        SOURCE_DATA
    )

    replay = (
        replay_original_generator()
    )

    # ========================================================
    # Verify exact reproduction BEFORE changing anything
    # ========================================================

    print(
        "=" * 80
    )

    print(
        "ORIGINAL GENERATOR REPRODUCTION CHECK"
    )

    print(
        "=" * 80
    )

    x_matches = np.array_equal(
        replay["x"],
        original_dataset.arrays[
            "x"
        ],
    )

    onset_matches = np.array_equal(
        replay["onset"],
        original_dataset.arrays[
            "t_onset"
        ],
    )

    detected_matches = np.array_equal(
        replay[
            "original_detected"
        ],
        original_dataset.arrays[
            "detected"
        ],
    )

    t_detect_matches = np.array_equal(
        replay[
            "original_t_detect"
        ],
        original_dataset.arrays[
            "t_detect"
        ],
    )

    print(
        "x matches:",
        x_matches,
    )

    print(
        "t_onset matches:",
        onset_matches,
    )

    print(
        "detected matches:",
        detected_matches,
    )

    print(
        "t_detect matches:",
        t_detect_matches,
    )

    if not all(
        [
            x_matches,
            onset_matches,
            detected_matches,
            t_detect_matches,
        ]
    ):

        raise RuntimeError(
            "Original synthetic generator was not reproduced exactly."
        )

    # ========================================================
    # Construct corrected labels
    #
    # IMPORTANT:
    #
    # candidate td is NOT regenerated.
    #
    # We simply reveal every candidate td that was already
    # inside the observation window.
    # ========================================================

    corrected_detected = np.zeros(
        N,
        dtype=np.int64,
    )

    corrected_t_detect = np.full(
        N,
        LENGTH - 1,
        dtype=np.int64,
    )

    fraud = (
        replay[
            "onset"
        ] >= 0
    )

    candidate = replay[
        "candidate_t_detect"
    ]

    in_window_detection = (
        fraud
        & (
            candidate
            < LENGTH
        )
    )

    natural_censor = (
        fraud
        & (
            candidate
            >= LENGTH
        )
    )

    corrected_detected[
        in_window_detection
    ] = 1

    corrected_t_detect[
        in_window_detection
    ] = candidate[
        in_window_detection
    ]

    # ========================================================
    # Save corrected dataset
    # ========================================================

    if "entity_id" in (
        original_dataset.arrays
    ):

        entity_id = np.asarray(
            original_dataset.arrays[
                "entity_id"
            ],
            dtype=np.int64,
        )

    else:

        entity_id = np.arange(
            N,
            dtype=np.int64,
        )

    lengths = np.asarray(
        original_dataset.arrays[
            "lengths"
        ],
        dtype=np.int64,
    )

    np.savez_compressed(
        OUTPUT_DATA,

        x=np.asarray(
            original_dataset.arrays[
                "x"
            ],
            dtype=np.float32,
        ),

        lengths=lengths,

        detected=corrected_detected,

        t_detect=corrected_t_detect,

        t_onset=np.asarray(
            original_dataset.arrays[
                "t_onset"
            ],
            dtype=np.int64,
        ),

        entity_id=entity_id,
    )

    corrected_dataset = load_npz(
        OUTPUT_DATA
    )

    # ========================================================
    # Invariance check
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "PAIRED DATA INVARIANCE CHECK"
    )

    print(
        "=" * 80
    )

    print(
        "x unchanged:",
        np.array_equal(
            original_dataset.arrays[
                "x"
            ],
            corrected_dataset.arrays[
                "x"
            ],
        ),
    )

    print(
        "lengths unchanged:",
        np.array_equal(
            original_dataset.arrays[
                "lengths"
            ],
            corrected_dataset.arrays[
                "lengths"
            ],
        ),
    )

    print(
        "t_onset unchanged:",
        np.array_equal(
            original_dataset.arrays[
                "t_onset"
            ],
            corrected_dataset.arrays[
                "t_onset"
            ],
        ),
    )

    print(
        "entity_id unchanged:",
        np.array_equal(
            entity_id,
            corrected_dataset.arrays[
                "entity_id"
            ],
        ),
    )

    print()

    print(
        "Detection delay unchanged by construction."
    )

    print(
        "Candidate t_d unchanged by construction."
    )

    # ========================================================
    # Recover same split
    # ========================================================

    (
        train_loader,
        val_loader,
        test_loader,
    ) = make_loaders(
        corrected_dataset,
        BATCH_SIZE,
        SPLIT,
        SPLIT_SEED,
    )

    print()

    print(
        "=" * 80
    )

    print(
        "SPLIT SIZES"
    )

    print(
        "=" * 80
    )

    print(
        "Train:",
        len(
            train_loader.dataset
        ),
    )

    print(
        "Validation:",
        len(
            val_loader.dataset
        ),
    )

    print(
        "Test:",
        len(
            test_loader.dataset
        ),
    )

    if not hasattr(
        test_loader.dataset,
        "indices",
    ):

        raise RuntimeError(
            "Could not recover test indices."
        )

    test_indices = np.asarray(
        test_loader.dataset.indices,
        dtype=np.int64,
    )

    # ========================================================
    # Test-set paired analysis
    # ========================================================

    test_tf = replay[
        "onset"
    ][
        test_indices
    ]

    test_original_detected = replay[
        "original_detected"
    ][
        test_indices
    ]

    test_corrected_detected = corrected_detected[
        test_indices
    ]

    test_candidate = replay[
        "candidate_t_detect"
    ][
        test_indices
    ]

    test_delay = replay[
        "detection_delay"
    ][
        test_indices
    ]

    test_random_censored = replay[
        "original_random_censored"
    ][
        test_indices
    ]

    test_natural_censored = replay[
        "original_natural_censored"
    ][
        test_indices
    ]

    test_fraud = (
        test_tf >= 0
    )

    # --------------------------------------------------------
    # Paired intervention group
    #
    # These exact same entities were:
    #
    # old dataset:
    #     detected = 0
    #
    # corrected dataset:
    #     detected = 1
    #
    # Nothing else changed.
    # --------------------------------------------------------

    paired_revealed = (
        test_random_censored
    )

    corrected_detected_fraud = (
        test_fraud
        & (
            test_corrected_detected
            == 1
        )
    )

    corrected_natural_censored = (
        test_fraud
        & (
            test_corrected_detected
            == 0
        )
    )

    print()

    print(
        "=" * 80
    )

    print(
        "TEST-SET SAME-DELAY CENSOR CONTROL"
    )

    print(
        "=" * 80
    )

    print(
        "Test entities:",
        len(
            test_indices
        ),
    )

    print(
        "Normal:",
        int(
            (
                ~test_fraud
            ).sum()
        ),
    )

    print(
        "Fraud:",
        int(
            test_fraud.sum()
        ),
    )

    print()

    print(
        "OLD dataset:"
    )

    print(
        "  detected fraud:",
        int(
            (
                test_fraud
                & (
                    test_original_detected
                    == 1
                )
            ).sum()
        ),
    )

    print(
        "  censored fraud:",
        int(
            (
                test_fraud
                & (
                    test_original_detected
                    == 0
                )
            ).sum()
        ),
    )

    print()

    print(
        "CORRECTED same-delay dataset:"
    )

    print(
        "  detected fraud:",
        int(
            corrected_detected_fraud.sum()
        ),
    )

    print(
        "  naturally censored fraud:",
        int(
            corrected_natural_censored.sum()
        ),
    )

    print()

    print(
        "Paired random-censored -> revealed:",
        int(
            paired_revealed.sum()
        ),
    )

    print(
        "Natural censored retained:",
        int(
            test_natural_censored.sum()
        ),
    )

    # ========================================================
    # Exact pair examples
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "PAIRED INTERVENTION EXAMPLES"
    )

    print(
        "=" * 80
    )

    paired_local_indices = np.where(
        paired_revealed
    )[0]

    for local_index in (
        paired_local_indices[
            :10
        ]
    ):

        global_index = (
            test_indices[
                local_index
            ]
        )

        print(
            f"entity="
            f"{entity_id[global_index]}, "
            f"tf="
            f"{test_tf[local_index]}, "
            f"delay="
            f"{test_delay[local_index]}, "
            f"candidate td="
            f"{test_candidate[local_index]}, "
            f"old detected="
            f"{test_original_detected[local_index]}, "
            f"new detected="
            f"{test_corrected_detected[local_index]}"
        )

    # ========================================================
    # Consistency check
    # ========================================================

    invalid_new_censored = (
        corrected_natural_censored
        & (
            test_candidate
            < LENGTH
        )
    )

    print()

    print(
        "=" * 80
    )

    print(
        "MODEL-CONSISTENCY CHECK"
    )

    print(
        "=" * 80
    )

    print(
        "New censored fraud with candidate td < sequence length:",
        int(
            invalid_new_censored.sum()
        ),
    )

    print(
        "New censored fraud with candidate td >= sequence length:",
        int(
            corrected_natural_censored.sum()
        ),
    )

    if (
        invalid_new_censored.sum()
        != 0
    ):

        raise RuntimeError(
            "Same-delay corrected dataset is still inconsistent."
        )

    print()

    print(
        "PASS: same-delay corrected dataset contains "
        "only natural right censoring."
    )

    # ========================================================
    # Save paired audit
    # ========================================================

    np.savez_compressed(
        REPORT_DIR
        / "paired_audit.npz",

        test_indices=
            test_indices,

        entity_id=
            entity_id[
                test_indices
            ],

        t_onset=
            test_tf,

        detection_delay=
            test_delay,

        candidate_t_detect=
            test_candidate,

        old_detected=
            test_original_detected,

        new_detected=
            test_corrected_detected,

        paired_random_censored_revealed=
            paired_revealed,

        natural_censored=
            test_natural_censored,
    )

    metadata = {
        "source_dataset":
            str(
                SOURCE_DATA
            ),

        "corrected_dataset":
            str(
                OUTPUT_DATA
            ),

        "x_unchanged":
            True,

        "t_onset_unchanged":
            True,

        "detection_delay_unchanged":
            True,

        "candidate_t_detect_unchanged":
            True,

        "split_unchanged":
            True,

        "old_test_detected_fraud":
            int(
                (
                    test_fraud
                    & (
                        test_original_detected
                        == 1
                    )
                ).sum()
            ),

        "old_test_censored_fraud":
            int(
                (
                    test_fraud
                    & (
                        test_original_detected
                        == 0
                    )
                ).sum()
            ),

        "new_test_detected_fraud":
            int(
                corrected_detected_fraud.sum()
            ),

        "new_test_natural_censored_fraud":
            int(
                corrected_natural_censored.sum()
            ),

        "paired_random_censored_revealed":
            int(
                paired_revealed.sum()
            ),
    }

    (
        REPORT_DIR
        / "dataset_summary.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()

    print(
        "=" * 80
    )

    print(
        "SAVED"
    )

    print(
        "=" * 80
    )

    print(
        OUTPUT_DATA
    )

    print(
        REPORT_DIR
        / "dataset_summary.json"
    )

    print(
        REPORT_DIR
        / "paired_audit.npz"
    )


if __name__ == "__main__":
    main()