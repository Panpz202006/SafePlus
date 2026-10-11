from __future__ import annotations

import numpy as np

from safeplus.data import load_npz, make_loaders


DATA_PATH = "data/processed/synthetic_demo.npz"

N = 2000
LENGTH = 24
FEATURES = 8
FRAUD_RATE = 0.35
CENSOR_RATE = 0.20
DELAY_MEAN = 4.0
SEED = 0

BATCH_SIZE = 64
SPLIT = [0.7, 0.1, 0.2]


def regenerate_with_audit():
    """
    Reproduce synthetic.py exactly, but additionally retain
    the latent candidate detection time and the reason why
    a fraudulent entity was not observed as detected.
    """

    rng = np.random.default_rng(SEED)

    x = rng.normal(
        0,
        1,
        size=(N, LENGTH, FEATURES),
    ).astype(np.float32)

    is_fraud = (
        rng.random(N)
        < FRAUD_RATE
    )

    onset = np.full(
        N,
        -1,
        dtype=np.int64,
    )

    detected = np.zeros(
        N,
        dtype=np.int64,
    )

    t_detect = np.full(
        N,
        LENGTH - 1,
        dtype=np.int64,
    )

    signal = rng.normal(
        0.9,
        0.15,
        size=FEATURES,
    ).astype(np.float32)

    candidate_detect = np.full(
        N,
        -1,
        dtype=np.int64,
    )

    random_censored = np.zeros(
        N,
        dtype=bool,
    )

    horizon_censored = np.zeros(
        N,
        dtype=bool,
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

        candidate = (
            onset[i]
            + delay
        )

        candidate_detect[i] = (
            candidate
        )

        # IMPORTANT:
        # Match the original generator exactly.
        if candidate < LENGTH:

            censor_draw = (
                rng.random()
            )

            if (
                censor_draw
                >= CENSOR_RATE
            ):

                detected[i] = 1

                t_detect[i] = (
                    candidate
                )

            else:

                # Detection candidate was inside the
                # observation horizon, but the random
                # censoring coin hid it.
                random_censored[i] = True

        else:

            # Detection naturally falls beyond
            # the observation horizon.
            horizon_censored[i] = True

    return {
        "x":
            x,

        "onset":
            onset,

        "detected":
            detected,

        "t_detect":
            t_detect,

        "candidate_detect":
            candidate_detect,

        "random_censored":
            random_censored,

        "horizon_censored":
            horizon_censored,
    }


def main():

    audit = (
        regenerate_with_audit()
    )

    dataset = load_npz(
        DATA_PATH
    )

    # --------------------------------------------------------
    # First verify that our reproduction exactly corresponds
    # to the dataset being used by the experiments.
    # --------------------------------------------------------

    print(
        "=" * 80
    )

    print(
        "DATA REPRODUCTION CHECK"
    )

    print(
        "=" * 80
    )

    print(
        "x matches:",
        np.allclose(
            audit["x"],
            dataset.arrays[
                "x"
            ],
        ),
    )

    print(
        "t_onset matches:",
        np.array_equal(
            audit[
                "onset"
            ],
            dataset.arrays[
                "t_onset"
            ],
        ),
    )

    print(
        "detected matches:",
        np.array_equal(
            audit[
                "detected"
            ],
            dataset.arrays[
                "detected"
            ],
        ),
    )

    print(
        "t_detect matches:",
        np.array_equal(
            audit[
                "t_detect"
            ],
            dataset.arrays[
                "t_detect"
            ],
        ),
    )

    # --------------------------------------------------------
    # Recover EXACT same test split
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
    ) = make_loaders(
        dataset,
        BATCH_SIZE,
        SPLIT,
        SEED,
    )

    if not hasattr(
        test_loader.dataset,
        "indices",
    ):

        raise RuntimeError(
            "Cannot recover test subset indices."
        )

    test_indices = np.asarray(
        test_loader.dataset.indices
    )

    onset = audit[
        "onset"
    ][
        test_indices
    ]

    detected = audit[
        "detected"
    ][
        test_indices
    ]

    candidate = audit[
        "candidate_detect"
    ][
        test_indices
    ]

    random_censored = audit[
        "random_censored"
    ][
        test_indices
    ]

    horizon_censored = audit[
        "horizon_censored"
    ][
        test_indices
    ]

    fraud = (
        onset >= 0
    )

    censored_fraud = (
        fraud
        & (
            detected == 0
        )
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()

    print(
        "=" * 80
    )

    print(
        "TEST-SET CENSORING AUDIT"
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
        "Fraud entities:",
        int(
            fraud.sum()
        ),
    )

    print(
        "Detected fraud:",
        int(
            (
                fraud
                & (
                    detected == 1
                )
            ).sum()
        ),
    )

    print(
        "Censored fraud:",
        int(
            censored_fraud.sum()
        ),
    )

    print()

    print(
        "Among censored fraud:"
    )

    print(
        "  Randomly censored despite candidate td < T:",
        int(
            random_censored.sum()
        ),
    )

    print(
        "  Naturally censored because candidate td >= T:",
        int(
            horizon_censored.sum()
        ),
    )

    print()

    if (
        censored_fraud.sum()
        > 0
    ):

        print(
            "Random-censor fraction:",
            float(
                random_censored.sum()
                / censored_fraud.sum()
            ),
        )

        print(
            "Natural horizon-censor fraction:",
            float(
                horizon_censored.sum()
                / censored_fraud.sum()
            ),
        )

    print()

    print(
        "Random-censored examples:"
    )

    random_indices = np.where(
        random_censored
    )[0][:10]

    for local_index in (
        random_indices
    ):

        print(
            f"  entity="
            f"{test_indices[local_index]}, "
            f"tf="
            f"{onset[local_index]}, "
            f"latent candidate td="
            f"{candidate[local_index]}, "
            f"observed detected=0"
        )

    print()

    print(
        "Natural horizon-censored examples:"
    )

    natural_indices = np.where(
        horizon_censored
    )[0][:10]

    for local_index in (
        natural_indices
    ):

        print(
            f"  entity="
            f"{test_indices[local_index]}, "
            f"tf="
            f"{onset[local_index]}, "
            f"candidate td="
            f"{candidate[local_index]}, "
            f"T="
            f"{LENGTH - 1}"
        )


if __name__ == "__main__":
    main()