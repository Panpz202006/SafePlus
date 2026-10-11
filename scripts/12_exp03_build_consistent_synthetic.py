from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from safeplus.data import load_npz, make_loaders


# ============================================================
# Experiment 03G
# Build a model-consistent synthetic benchmark
#
# IMPORTANT:
#
# We KEEP the original:
#
#   x
#   t_onset
#   entity_id
#   sequence lengths
#
# Therefore the behavioral trajectories and true t_f values
# are exactly the same as in the previous experiments.
#
# We ONLY regenerate the detection process.
#
#
# Old synthetic censoring:
#
#   candidate td < T
#       but detection could still be randomly hidden.
#
# This creates label-missingness that is NOT the same as the
# right-censoring assumption used by the current joint model.
#
#
# Corrected detection process:
#
#   td = tf + delay
#
#   if td is inside the observation window:
#       detected = 1
#
#   if td lies beyond the observation window:
#       detected = 0
#
# Thus every undetected fraud entity satisfies:
#
#       tf < T < td
#
# up to the discrete boundary convention.
# ============================================================


SOURCE_DATA = Path(
    r"data/processed/synthetic_demo.npz"
)

OUTPUT_DATA = Path(
    r"data/processed/synthetic_consistent_delay8.npz"
)

REPORT_DIR = Path(
    r"reports/experiment03/consistent_synthetic"
)

# ------------------------------------------------------------
# Detection process
# ------------------------------------------------------------

DELAY_MEAN = 8.0

# Use a separate fixed RNG so that x and tf remain untouched.
DETECTION_SEED = 20261009

# ------------------------------------------------------------
# Same experimental split as all previous experiments
# ------------------------------------------------------------

SPLIT_SEED = 0
BATCH_SIZE = 64
SPLIT = [0.7, 0.1, 0.2]


def main():

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DATA.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Load ORIGINAL benchmark
    # ========================================================

    original = load_npz(
        SOURCE_DATA
    )

    x = np.asarray(
        original.arrays["x"],
        dtype=np.float32,
    ).copy()

    lengths = np.asarray(
        original.arrays["lengths"],
        dtype=np.int64,
    ).copy()

    t_onset = np.asarray(
        original.arrays["t_onset"],
        dtype=np.int64,
    ).copy()

    if "entity_id" in original.arrays:

        entity_id = np.asarray(
            original.arrays["entity_id"],
            dtype=np.int64,
        ).copy()

    else:

        entity_id = np.arange(
            len(x),
            dtype=np.int64,
        )

    n = len(x)

    max_length = x.shape[1]

    if not np.all(
        lengths == max_length
    ):

        raise ValueError(
            "This diagnostic currently expects "
            "equal full sequence lengths."
        )

    horizon = (
        max_length - 1
    )

    fraud = (
        t_onset >= 0
    )

    # ========================================================
    # Corrected detection process
    # ========================================================

    detected = np.zeros(
        n,
        dtype=np.int64,
    )

    t_detect = np.full(
        n,
        horizon,
        dtype=np.int64,
    )

    # Keep latent candidate detection time for audit only.
    candidate_t_detect = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    detection_delay = np.full(
        n,
        -1,
        dtype=np.int64,
    )

    rng = np.random.default_rng(
        DETECTION_SEED
    )

    fraud_indices = np.flatnonzero(
        fraud
    )

    for i in fraud_indices:

        delay = max(
            1,
            int(
                rng.exponential(
                    DELAY_MEAN
                )
            ),
        )

        candidate = (
            int(t_onset[i])
            + delay
        )

        detection_delay[i] = (
            delay
        )

        candidate_t_detect[i] = (
            candidate
        )

        # ----------------------------------------------------
        # MODEL-CONSISTENT CENSORING
        #
        # No random deletion of an in-window detection.
        # ----------------------------------------------------

        if candidate < max_length:

            detected[i] = 1

            t_detect[i] = (
                candidate
            )

        else:

            detected[i] = 0

            # For compatibility with the current dataset format,
            # undetected users store the observation endpoint.
            t_detect[i] = (
                horizon
            )

    # ========================================================
    # Save corrected dataset
    # ========================================================

    np.savez_compressed(
        OUTPUT_DATA,

        x=x,

        lengths=lengths,

        detected=detected,

        t_detect=t_detect,

        t_onset=t_onset,

        entity_id=entity_id,
    )

    # ========================================================
    # Verify x and tf were NOT changed
    # ========================================================

    corrected = load_npz(
        OUTPUT_DATA
    )

    print(
        "=" * 80
    )

    print(
        "DATA INVARIANCE CHECK"
    )

    print(
        "=" * 80
    )

    print(
        "x unchanged:",
        np.array_equal(
            original.arrays["x"],
            corrected.arrays["x"],
        ),
    )

    print(
        "lengths unchanged:",
        np.array_equal(
            original.arrays["lengths"],
            corrected.arrays["lengths"],
        ),
    )

    print(
        "t_onset unchanged:",
        np.array_equal(
            original.arrays["t_onset"],
            corrected.arrays["t_onset"],
        ),
    )

    print(
        "entity_id unchanged:",
        np.array_equal(
            entity_id,
            corrected.arrays["entity_id"],
        ),
    )

    # ========================================================
    # Global statistics
    # ========================================================

    global_fraud = (
        t_onset >= 0
    )

    global_detected_fraud = (
        global_fraud
        & (
            detected == 1
        )
    )

    global_censored_fraud = (
        global_fraud
        & (
            detected == 0
        )
    )

    print()

    print(
        "=" * 80
    )

    print(
        "GLOBAL CORRECTED DATASET"
    )

    print(
        "=" * 80
    )

    print(
        "Entities:",
        n,
    )

    print(
        "Fraud:",
        int(
            global_fraud.sum()
        ),
    )

    print(
        "Detected fraud:",
        int(
            global_detected_fraud.sum()
        ),
    )

    print(
        "Naturally censored fraud:",
        int(
            global_censored_fraud.sum()
        ),
    )

    print(
        "Fraud censor fraction:",
        float(
            global_censored_fraud.sum()
            / max(
                1,
                global_fraud.sum(),
            )
        ),
    )

    # ========================================================
    # Recover EXACT same train/val/test split
    # ========================================================

    (
        train_loader,
        val_loader,
        test_loader,
    ) = make_loaders(
        corrected,
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
            "Cannot recover held-out test indices."
        )

    test_indices = np.asarray(
        test_loader.dataset.indices,
        dtype=np.int64,
    )

    test_tf = t_onset[
        test_indices
    ]

    test_detected = detected[
        test_indices
    ]

    test_candidate = (
        candidate_t_detect[
            test_indices
        ]
    )

    test_delay = (
        detection_delay[
            test_indices
        ]
    )

    test_fraud = (
        test_tf >= 0
    )

    test_detected_fraud = (
        test_fraud
        & (
            test_detected == 1
        )
    )

    test_censored_fraud = (
        test_fraud
        & (
            test_detected == 0
        )
    )

    # ========================================================
    # Test statistics
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "TEST-SET CORRECTED CENSORING"
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
            (~test_fraud).sum()
        ),
    )

    print(
        "Fraud:",
        int(
            test_fraud.sum()
        ),
    )

    print(
        "Detected fraud:",
        int(
            test_detected_fraud.sum()
        ),
    )

    print(
        "Naturally censored fraud:",
        int(
            test_censored_fraud.sum()
        ),
    )

    print(
        "Natural censor fraction among fraud:",
        float(
            test_censored_fraud.sum()
            / max(
                1,
                test_fraud.sum(),
            )
        ),
    )

    # ========================================================
    # Critical consistency check
    #
    # Every censored fraud must have candidate td OUTSIDE
    # the observed sequence.
    # ========================================================

    invalid_censored = (
        test_censored_fraud
        & (
            test_candidate
            < max_length
        )
    )

    valid_natural_censored = (
        test_censored_fraud
        & (
            test_candidate
            >= max_length
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
        "Censored fraud with candidate td < sequence length:",
        int(
            invalid_censored.sum()
        ),
    )

    print(
        "Censored fraud with candidate td >= sequence length:",
        int(
            valid_natural_censored.sum()
        ),
    )

    if (
        invalid_censored.sum()
        != 0
    ):

        raise RuntimeError(
            "Corrected benchmark still contains "
            "in-window hidden detections."
        )

    print()

    print(
        "PASS: every censored fraud entity is "
        "naturally right-censored."
    )

    # ========================================================
    # Delay statistics
    # ========================================================

    fraud_delays = (
        test_delay[
            test_fraud
        ]
    )

    print()

    print(
        "=" * 80
    )

    print(
        "TEST FRAUD DELAY STATISTICS"
    )

    print(
        "=" * 80
    )

    print(
        "Mean delay:",
        float(
            fraud_delays.mean()
        ),
    )

    print(
        "Median delay:",
        float(
            np.median(
                fraud_delays
            )
        ),
    )

    print(
        "Min delay:",
        int(
            fraud_delays.min()
        ),
    )

    print(
        "Max delay:",
        int(
            fraud_delays.max()
        ),
    )

    # ========================================================
    # Examples
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "NATURALLY CENSORED EXAMPLES"
    )

    print(
        "=" * 80
    )

    examples = np.where(
        test_censored_fraud
    )[0][
        :10
    ]

    for local_index in examples:

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
            f"latent td="
            f"{test_candidate[local_index]}, "
            f"observation horizon="
            f"{horizon}"
        )

    # ========================================================
    # Save audit metadata
    # ========================================================

    metadata = {
        "source_dataset":
            str(
                SOURCE_DATA
            ),

        "output_dataset":
            str(
                OUTPUT_DATA
            ),

        "behavior_x_unchanged":
            True,

        "true_tf_unchanged":
            True,

        "detection_seed":
            DETECTION_SEED,

        "delay_distribution":
            "integer floor of exponential",

        "delay_mean":
            DELAY_MEAN,

        "random_detection_censoring":
            False,

        "observation_horizon":
            int(
                horizon
            ),

        "test_entities":
            int(
                len(
                    test_indices
                )
            ),

        "test_normal":
            int(
                (
                    ~test_fraud
                ).sum()
            ),

        "test_fraud":
            int(
                test_fraud.sum()
            ),

        "test_detected_fraud":
            int(
                test_detected_fraud.sum()
            ),

        "test_naturally_censored_fraud":
            int(
                test_censored_fraud.sum()
            ),

        "test_natural_censor_fraction":
            float(
                test_censored_fraud.sum()
                / max(
                    1,
                    test_fraud.sum(),
                )
            ),

        "invalid_hidden_in_window_detection_count":
            int(
                invalid_censored.sum()
            ),

        "test_mean_detection_delay":
            float(
                fraud_delays.mean()
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

    np.savez_compressed(
        REPORT_DIR
        / "detection_audit.npz",

        entity_id=
            entity_id,

        t_onset=
            t_onset,

        candidate_t_detect=
            candidate_t_detect,

        detection_delay=
            detection_delay,

        detected=
            detected,

        t_detect=
            t_detect,
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
        / "detection_audit.npz"
    )


if __name__ == "__main__":
    main()