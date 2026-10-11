from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Experiment 03Y
# Behavior-Likelihood Identification
#
# 中文：
# 行为似然能否缓解 latent tf 不可识别问题？
#
#
# Core idea:
#
# Previous:
#
#     candidate tf mainly explains observed td.
#
#
# Now:
#
#     candidate tf must also explain
#     the observed behavior trajectory x.
#
#
# Posterior score:
#
#   score(k)
#
#   =
#
#   alpha * log p(x_0:td | tf=k)
#
#   +
#
#   log p(delay = td-k)
#
#
#
# IMPORTANT:
#
# No true tf is used for:
#
#   - estimating normal anchor
#   - estimating fraud anchor
#   - candidate scoring
#   - posterior computation
#
#
# true tf is ONLY used for final synthetic evaluation.
#
#
# This is a controlled diagnostic,
# NOT the final neural model.
# ============================================================


DATA_PATH = Path(
    r"data/processed/synthetic_same_delay_consistent.npz"
)

OUTPUT_DIR = Path(
    r"reports/experiment03/"
    r"behavior_likelihood_identification"
)


SEED = 0

DELAY_MEAN = 4.0


# ============================================================
# Behavior weights
#
# alpha = 0:
#     delay evidence only
#
# Larger alpha:
#     behavior likelihood gets more influence.
# ============================================================


ALPHA_VALUES = [
    0.0,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
]


# ============================================================
# Split
# ============================================================


def split_indices(
    n,
    seed=0,
):

    rng = np.random.default_rng(
        seed
    )

    indices = rng.permutation(
        n
    )

    n_train = int(
        0.7 * n
    )

    n_val = int(
        0.1 * n
    )

    train_idx = indices[
        :n_train
    ]

    val_idx = indices[
        n_train:
        n_train + n_val
    ]

    test_idx = indices[
        n_train + n_val:
    ]

    return (
        train_idx,
        val_idx,
        test_idx,
    )


# ============================================================
# Synthetic delay prior
#
# delay = max(
#     1,
#     int(Exponential(scale=4))
# )
# ============================================================


def delay_probability(
    delay,
):

    if delay < 1:

        return 0.0

    if delay == 1:

        return (
            1.0
            -
            math.exp(
                -2.0
                / DELAY_MEAN
            )
        )

    return (
        math.exp(
            -delay
            / DELAY_MEAN
        )
        -
        math.exp(
            -(delay + 1)
            / DELAY_MEAN
        )
    )


# ============================================================
# Robust median / MAD
# ============================================================


def robust_location_scale(
    observations,
):

    median = np.median(
        observations,
        axis=0,
    )

    mad = np.median(
        np.abs(
            observations
            - median
        ),
        axis=0,
    )

    # Gaussian-consistent MAD scale
    scale = (
        1.4826
        * mad
    )

    scale = np.maximum(
        scale,
        0.25,
    )

    return (
        median,
        scale,
    )


# ============================================================
# Estimate behavior anchors using TRAIN DATA ONLY
#
#
# Normal-like anchor:
#
#   all time points from detected == 0 users.
#
# This pool may contain a small amount of natural censoring,
# therefore robust median is used.
#
#
# Fraud anchor:
#
#   x_td from detected users.
#
# At td, fraud must already have happened,
# so this is observable weak supervision.
# ============================================================


def estimate_behavior_anchors(
    x,
    detected,
    td,
    train_idx,
):

    normal_like = []

    fraud_endpoint = []

    for idx in train_idx:

        if detected[idx] == 0:

            normal_like.append(
                x[idx]
            )

        else:

            d = int(
                td[idx]
            )

            fraud_endpoint.append(
                x[
                    idx,
                    d,
                    :
                ]
            )

    normal_like = np.concatenate(
        normal_like,
        axis=0,
    )

    fraud_endpoint = np.stack(
        fraud_endpoint,
        axis=0,
    )

    (
        normal_center,
        normal_scale,
    ) = robust_location_scale(
        normal_like
    )

    fraud_center = np.median(
        fraud_endpoint,
        axis=0,
    )

    return {
        "normal_center":
            normal_center,

        "fraud_center":
            fraud_center,

        "scale":
            normal_scale,

        "n_normal_like_points":
            len(
                normal_like
            ),

        "n_fraud_endpoints":
            len(
                fraud_endpoint
            ),
    }


# ============================================================
# Candidate-conditioned behavior model
#
#
# Candidate:
#
#     tf = k
#
#
# Before k:
#
#     expected behavior ~= normal_center
#
#
# From k through td:
#
#     expected behavior gradually moves
#
#       normal_center
#
#              ->
#
#       fraud_center
#
#
# The gradual path is used because onset corresponds to
# the BEGINNING of behavioral drift, not necessarily the
# point of largest instantaneous change.
#
#
# No true tf is involved.
# ============================================================


def behavior_log_likelihood(
    observed_x,
    candidate_tf,
    observed_td,
    normal_center,
    fraud_center,
    scale,
):

    total = 0.0

    direction = (
        fraud_center
        -
        normal_center
    )

    denominator = max(
        observed_td
        - candidate_tf
        + 1,
        1,
    )

    for t in range(
        observed_td + 1
    ):

        if t < candidate_tf:

            expected = (
                normal_center
            )

        else:

            # ---------------------------------------------
            # Gradual progression.
            #
            # At onset:
            #   small departure from normal.
            #
            # At td:
            #   reaches fraud-side anchor.
            # ---------------------------------------------

            progress = (
                t
                - candidate_tf
                + 1
            ) / denominator

            expected = (
                normal_center
                +
                progress
                * direction
            )

        residual = (
            observed_x[t]
            - expected
        ) / scale

        total += (
            -0.5
            * float(
                np.sum(
                    residual ** 2
                )
            )
        )

    return total


# ============================================================
# Stable softmax
# ============================================================


def softmax(
    scores,
):

    scores = np.asarray(
        scores,
        dtype=np.float64,
    )

    scores = (
        scores
        -
        np.max(
            scores
        )
    )

    probabilities = np.exp(
        scores
    )

    probabilities = (
        probabilities
        /
        probabilities.sum()
    )

    return probabilities


# ============================================================
# Evaluate one alpha
# ============================================================


def evaluate_alpha(
    alpha,
    x,
    detected,
    td,
    true_tf,
    entity_id,
    test_idx,
    anchors,
):

    rows = []

    normal_center = anchors[
        "normal_center"
    ]

    fraud_center = anchors[
        "fraud_center"
    ]

    scale = anchors[
        "scale"
    ]

    for idx in test_idx:

        if detected[idx] != 1:

            continue

        d = int(
            td[idx]
        )

        tf = int(
            true_tf[idx]
        )

        # Legal:
        #
        # tf < td
        candidates = np.arange(
            0,
            d,
            dtype=np.int64,
        )

        behavior_scores = []

        delay_scores = []

        total_scores = []

        for k in candidates:

            behavior_ll = (
                behavior_log_likelihood(
                    observed_x=x[idx],
                    candidate_tf=int(k),
                    observed_td=d,
                    normal_center=normal_center,
                    fraud_center=fraud_center,
                    scale=scale,
                )
            )

            delay = (
                d
                - int(k)
            )

            delay_prob = max(
                delay_probability(
                    delay
                ),
                1e-30,
            )

            delay_ll = math.log(
                delay_prob
            )

            behavior_scores.append(
                behavior_ll
            )

            delay_scores.append(
                delay_ll
            )

        behavior_scores = np.asarray(
            behavior_scores,
            dtype=np.float64,
        )

        delay_scores = np.asarray(
            delay_scores,
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # Normalize behavior evidence per user.
        #
        # Raw Gaussian likelihood accumulates across
        # time × features and therefore has a different
        # numerical scale from delay log-probability.
        #
        # We standardize candidate behavior scores WITHOUT
        # using true tf.
        # ----------------------------------------------------

        behavior_std = float(
            np.std(
                behavior_scores
            )
        )

        if behavior_std < 1e-8:

            normalized_behavior = (
                behavior_scores
                * 0.0
            )

        else:

            normalized_behavior = (
                behavior_scores
                -
                np.mean(
                    behavior_scores
                )
            ) / behavior_std

        total_scores = (
            alpha
            * normalized_behavior
            +
            delay_scores
        )

        posterior = softmax(
            total_scores
        )

        posterior_mean_tf = float(
            np.sum(
                posterior
                * candidates
            )
        )

        posterior_mode_tf = int(
            candidates[
                np.argmax(
                    posterior
                )
            ]
        )

        posterior_mean_delay = (
            d
            - posterior_mean_tf
        )

        # Behavior-only diagnostic
        behavior_posterior = softmax(
            normalized_behavior
        )

        behavior_mean_tf = float(
            np.sum(
                behavior_posterior
                * candidates
            )
        )

        behavior_mode_tf = int(
            candidates[
                np.argmax(
                    behavior_posterior
                )
            ]
        )

        rows.append(
            {
                "entity_id":
                    int(
                        entity_id[idx]
                    ),

                "alpha":
                    alpha,

                "true_tf":
                    tf,

                "observed_td":
                    d,

                "true_delay":
                    d - tf,

                "posterior_mean_tf":
                    posterior_mean_tf,

                "posterior_mode_tf":
                    posterior_mode_tf,

                "posterior_mean_delay":
                    posterior_mean_delay,

                "behavior_mean_tf":
                    behavior_mean_tf,

                "behavior_mode_tf":
                    behavior_mode_tf,
            }
        )

    df = pd.DataFrame(
        rows
    )

    mean_error = (
        df[
            "posterior_mean_tf"
        ]
        -
        df[
            "true_tf"
        ]
    )

    mode_error = (
        df[
            "posterior_mode_tf"
        ]
        -
        df[
            "true_tf"
        ]
    )

    behavior_error = (
        df[
            "behavior_mean_tf"
        ]
        -
        df[
            "true_tf"
        ]
    )

    result = {
        "alpha":
            alpha,

        "n":
            len(
                df
            ),

        "posterior_mean_tf_mae":
            float(
                np.mean(
                    np.abs(
                        mean_error
                    )
                )
            ),

        "posterior_mode_tf_mae":
            float(
                np.mean(
                    np.abs(
                        mode_error
                    )
                )
            ),

        "posterior_signed_error":
            float(
                np.mean(
                    mean_error
                )
            ),

        "posterior_mean_delay":
            float(
                df[
                    "posterior_mean_delay"
                ].mean()
            ),

        "true_mean_delay":
            float(
                df[
                    "true_delay"
                ].mean()
            ),

        "within_1_rate":
            float(
                (
                    np.abs(
                        mean_error
                    )
                    <= 1
                ).mean()
            ),

        "within_2_rate":
            float(
                (
                    np.abs(
                        mean_error
                    )
                    <= 2
                ).mean()
            ),

        "behavior_only_mean_tf_mae":
            float(
                np.mean(
                    np.abs(
                        behavior_error
                    )
                )
            ),
    }

    return (
        result,
        df,
    )


# ============================================================
# Main
# ============================================================


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80
    )

    print(
        "EXPERIMENT 03Y:"
    )

    print(
        "BEHAVIOR-LIKELIHOOD IDENTIFICATION"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Question:"
    )

    print(
        "Can explicit p(x | tf) provide"
    )

    print(
        "candidate-specific evidence that td"
    )

    print(
        "likelihood alone is missing?"
    )

    print()

    print(
        "No true tf is used to construct"
    )

    print(
        "the behavior likelihood."
    )

    # ========================================================
    # Load
    # ========================================================

    data = np.load(
        DATA_PATH,
        allow_pickle=True,
    )

    x = data[
        "x"
    ].astype(
        np.float64
    )

    detected = data[
        "detected"
    ].astype(
        np.int64
    )

    td = data[
        "t_detect"
    ].astype(
        np.int64
    )

    true_tf = data[
        "t_onset"
    ].astype(
        np.int64
    )

    if "entity_id" in data:

        entity_id = data[
            "entity_id"
        ].astype(
            np.int64
        )

    else:

        entity_id = np.arange(
            len(
                x
            )
        )

    (
        train_idx,
        val_idx,
        test_idx,
    ) = split_indices(
        len(
            x
        ),
        seed=SEED,
    )

    # ========================================================
    # Estimate weakly supervised behavior anchors
    # ========================================================

    anchors = estimate_behavior_anchors(
        x,
        detected,
        td,
        train_idx,
    )

    print()

    print(
        "Behavior anchors estimated from TRAIN only:"
    )

    print(
        "  normal-like points:",
        anchors[
            "n_normal_like_points"
        ],
    )

    print(
        "  detected fraud endpoints:",
        anchors[
            "n_fraud_endpoints"
        ],
    )

    print()

    print(
        "Normal center:"
    )

    print(
        np.round(
            anchors[
                "normal_center"
            ],
            3,
        )
    )

    print()

    print(
        "Fraud endpoint center:"
    )

    print(
        np.round(
            anchors[
                "fraud_center"
            ],
            3,
        )
    )

    print()

    print(
        "Fraud-normal direction:"
    )

    print(
        np.round(
            anchors[
                "fraud_center"
            ]
            -
            anchors[
                "normal_center"
            ],
            3,
        )
    )

    # ========================================================
    # Evaluate fixed alpha sweep
    #
    # IMPORTANT:
    #
    # We report every alpha.
    # We do NOT select alpha using true tf.
    # ========================================================

    results = []

    all_entity_results = []

    for alpha in ALPHA_VALUES:

        result, entity_df = (
            evaluate_alpha(
                alpha=alpha,
                x=x,
                detected=detected,
                td=td,
                true_tf=true_tf,
                entity_id=entity_id,
                test_idx=test_idx,
                anchors=anchors,
            )
        )

        results.append(
            result
        )

        all_entity_results.append(
            entity_df
        )

    summary = pd.DataFrame(
        results
    )

    entity_results = pd.concat(
        all_entity_results,
        ignore_index=True,
    )

    print()

    print(
        "=" * 80
    )

    print(
        "FINAL COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # ========================================================
    # Entity 1067
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "ENTITY 1067"
    )

    print(
        "=" * 80
    )

    entity_1067 = (
        entity_results[
            entity_results[
                "entity_id"
            ] == 1067
        ]
    )

    if len(
        entity_1067
    ):

        print(
            entity_1067[
                [
                    "alpha",
                    "true_tf",
                    "observed_td",
                    "posterior_mean_tf",
                    "posterior_mode_tf",
                    "posterior_mean_delay",
                    "behavior_mean_tf",
                    "behavior_mode_tf",
                ]
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # Save
    # ========================================================

    summary.to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False,
    )

    entity_results.to_csv(
        OUTPUT_DIR
        / "entity_diagnostics.csv",
        index=False,
    )

    np.savez(
        OUTPUT_DIR
        / "behavior_anchors.npz",

        normal_center=
            anchors[
                "normal_center"
            ],

        fraud_center=
            anchors[
                "fraud_center"
            ],

        scale=
            anchors[
                "scale"
            ],
    )

    print()

    print(
        "=" * 80
    )

    print(
        "HOW TO INTERPRET"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "alpha = 0:"
    )

    print(
        "  delay-only reference."
    )

    print()

    print(
        "If alpha > 0 lowers tf MAE:"
    )

    print(
        "  explicit behavior evidence helps"
    )

    print(
        "  identify user-level tf."
    )

    print()

    print(
        "If behavior-only is useful but"
    )

    print(
        "  behavior + delay becomes worse:"
    )

    print(
        "  the two evidence terms require"
    )

    print(
        "  better calibration."
    )

    print()

    print(
        "If all alpha values fail:"
    )

    print(
        "  this simple global progression"
    )

    print(
        "  likelihood is insufficient."
    )

    print(
        "  Then move to a learned"
    )

    print(
        "  latent-state emission model."
    )


if __name__ == "__main__":
    main()