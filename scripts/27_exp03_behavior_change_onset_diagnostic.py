from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Experiment 03T
# Behavior-Change Onset Diagnostic
#
# 中文：
# 行为变化起点诊断。
#
#
# 核心问题：
#
# 不使用真实 t_f 训练，
# 不查看 t_d 之后的未来行为，
#
# 只根据 x_0 ... x_td，
#
# 能否找到一个与真实 t_f 接近的
# behavior change point（行为变化点）？
#
#
# IMPORTANT:
#
# true t_f 只在最后用于 synthetic evaluation。
#
# 它绝不参与候选点计算。
#
#
# 这不是最终模型。
# 只是验证：
#
#   x 本身能不能提供 user-level onset anchor。
# ============================================================


DATA_PATH = Path(
    r"data/processed/synthetic_same_delay_consistent.npz"
)

OUTPUT_DIR = Path(
    r"reports/experiment03/"
    r"behavior_change_onset_diagnostic"
)

SEED = 0


# ============================================================
# Same train / val / test split
# ============================================================


def split_indices(
    n: int,
    seed: int = 0,
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
# Normalized mean-change score
#
#
# Compare:
#
#   left segment
#
# vs
#
#   right segment
#
#
# Score:
#
#   sqrt(nL*nR / (nL+nR))
#
#   *
#
#   || mean(right) - mean(left) ||_2
#
#
# The sqrt term avoids always favoring
# extremely tiny segments.
# ============================================================


def mean_change_score(
    left,
    right,
):

    n_left = len(
        left
    )

    n_right = len(
        right
    )

    if (
        n_left == 0
        or n_right == 0
    ):

        return -np.inf

    left_mean = left.mean(
        axis=0
    )

    right_mean = right.mean(
        axis=0
    )

    difference = np.linalg.norm(
        right_mean
        - left_mean
    )

    balance = np.sqrt(
        (
            n_left
            * n_right
        )
        /
        (
            n_left
            + n_right
        )
    )

    return float(
        balance
        * difference
    )


# ============================================================
# Method A
#
# Prefix-Suffix change point
#
#
# Candidate k:
#
#   left:
#       x_0 ... x_{k-1}
#
#   right:
#       x_k ... x_td
#
#
# Only observed behavior up to td is used.
# ============================================================


def prefix_suffix_prediction(
    observed_x,
):

    length = len(
        observed_x
    )

    best_k = None

    best_score = -np.inf

    # At least 2 observations
    # on each side.
    for k in range(
        2,
        length - 1,
    ):

        left = observed_x[
            :k
        ]

        right = observed_x[
            k:
        ]

        score = mean_change_score(
            left,
            right,
        )

        if score > best_score:

            best_score = (
                score
            )

            best_k = (
                k
            )

    return (
        best_k,
        best_score,
    )


# ============================================================
# Method B / C
#
# Local window change point
#
#
# Candidate k:
#
#   left:
#       previous window steps
#
#   right:
#       k and following window steps
#
# ============================================================


def local_window_prediction(
    observed_x,
    window,
):

    length = len(
        observed_x
    )

    best_k = None

    best_score = -np.inf

    for k in range(
        2,
        length - 1,
    ):

        left_start = max(
            0,
            k - window,
        )

        right_end = min(
            length,
            k + window,
        )

        left = observed_x[
            left_start:k
        ]

        right = observed_x[
            k:right_end
        ]

        # Require at least 2 points
        # on both sides.
        if (
            len(left) < 2
            or len(right) < 2
        ):

            continue

        score = mean_change_score(
            left,
            right,
        )

        if score > best_score:

            best_score = (
                score
            )

            best_k = (
                k
            )

    return (
        best_k,
        best_score,
    )


# ============================================================
# Metrics
# ============================================================


def summarize(
    df,
    prediction_column,
    name,
):

    valid = df[
        df[
            prediction_column
        ].notna()
    ].copy()

    true_tf = valid[
        "true_tf"
    ].to_numpy()

    pred_tf = valid[
        prediction_column
    ].to_numpy()

    error = (
        pred_tf
        - true_tf
    )

    abs_error = np.abs(
        error
    )

    summary = {
        "method":
            name,

        "n":
            int(
                len(
                    valid
                )
            ),

        "mae":
            float(
                abs_error.mean()
            ),

        "signed_error":
            float(
                error.mean()
            ),

        "before_tf_rate":
            float(
                (
                    error < 0
                ).mean()
            ),

        "exact_tf_rate":
            float(
                (
                    error == 0
                ).mean()
            ),

        "after_tf_rate":
            float(
                (
                    error > 0
                ).mean()
            ),

        "within_1_rate":
            float(
                (
                    abs_error <= 1
                ).mean()
            ),

        "within_2_rate":
            float(
                (
                    abs_error <= 2
                ).mean()
            ),
    }

    return summary


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
        "EXPERIMENT 03T:"
    )

    print(
        "BEHAVIOR-CHANGE ONSET DIAGNOSTIC"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "核心问题："
    )

    print(
        "不依赖真实 t_f 训练，"
    )

    print(
        "只观察 detection 之前的行为，"
    )

    print(
        "行为变化本身能否提供 onset 锚点？"
    )

    print()

    print(
        "This is a diagnostic, not the final model."
    )

    print(
        "=" * 80
    )

    # ========================================================
    # Load data
    # ========================================================

    data = np.load(
        DATA_PATH,
        allow_pickle=True,
    )

    print()

    print(
        "Dataset keys:",
        list(
            data.keys()
        ),
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

    if "t_onset" in data:

        true_tf = data[
            "t_onset"
        ].astype(
            np.int64
        )

    elif "onset" in data:

        true_tf = data[
            "onset"
        ].astype(
            np.int64
        )

    else:

        raise KeyError(
            "Missing t_onset/onset"
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
        _,
        _,
        test_idx,
    ) = split_indices(
        len(
            x
        ),
        seed=SEED,
    )

    # ========================================================
    # Same 132 detected fraud users
    # ========================================================

    test_detected_fraud = [
        int(idx)
        for idx in test_idx
        if (
            detected[idx] == 1
            and true_tf[idx] >= 0
        )
    ]

    print()

    print(
        "Detected fraud test users:",
        len(
            test_detected_fraud
        ),
    )

    rows = []

    for idx in test_detected_fraud:

        d = int(
            td[idx]
        )

        tf = int(
            true_tf[idx]
        )

        # ----------------------------------------------------
        # STRICT CAUSAL:
        #
        # Only behavior from 0 through td.
        #
        # Nothing after td is used.
        # ----------------------------------------------------

        observed_x = x[
            idx,
            :d + 1,
            :,
        ]

        (
            prefix_tf,
            prefix_score,
        ) = prefix_suffix_prediction(
            observed_x
        )

        (
            local2_tf,
            local2_score,
        ) = local_window_prediction(
            observed_x,
            window=2,
        )

        (
            local3_tf,
            local3_score,
        ) = local_window_prediction(
            observed_x,
            window=3,
        )

        (
            local4_tf,
            local4_score,
        ) = local_window_prediction(
            observed_x,
            window=4,
        )

        rows.append(
            {
                "entity_id":
                    int(
                        entity_id[idx]
                    ),

                "true_tf":
                    tf,

                "observed_td":
                    d,

                "true_delay":
                    d - tf,

                # ---------------------------------------------
                # Trivial short-delay baseline
                # ---------------------------------------------

                "td_minus_1":
                    d - 1,

                # ---------------------------------------------
                # Behavior change predictions
                # ---------------------------------------------

                "prefix_suffix_tf":
                    prefix_tf,

                "prefix_suffix_score":
                    prefix_score,

                "local_w2_tf":
                    local2_tf,

                "local_w2_score":
                    local2_score,

                "local_w3_tf":
                    local3_tf,

                "local_w3_score":
                    local3_score,

                "local_w4_tf":
                    local4_tf,

                "local_w4_score":
                    local4_score,
            }
        )

    df = pd.DataFrame(
        rows
    )

    # ========================================================
    # Summary
    # ========================================================

    summaries = []

    summaries.append(
        summarize(
            df,
            "td_minus_1",
            "td_minus_1_baseline",
        )
    )

    summaries.append(
        summarize(
            df,
            "prefix_suffix_tf",
            "prefix_suffix_change",
        )
    )

    summaries.append(
        summarize(
            df,
            "local_w2_tf",
            "local_window_2",
        )
    )

    summaries.append(
        summarize(
            df,
            "local_w3_tf",
            "local_window_3",
        )
    )

    summaries.append(
        summarize(
            df,
            "local_w4_tf",
            "local_window_4",
        )
    )

    summary_df = pd.DataFrame(
        summaries
    )

    print()

    print(
        "=" * 80
    )

    print(
        "RESULT"
    )

    print(
        "=" * 80
    )

    print()

    print(
        summary_df.to_string(
            index=False
        )
    )

    # ========================================================
    # By true delay
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "PREFIX-SUFFIX BY TRUE DELAY"
    )

    print(
        "=" * 80
    )

    df[
        "prefix_abs_error"
    ] = np.abs(
        df[
            "prefix_suffix_tf"
        ]
        - df[
            "true_tf"
        ]
    )

    by_delay = (
        df
        .groupby(
            "true_delay",
            as_index=False,
        )
        .agg(
            n_users=(
                "entity_id",
                "count",
            ),

            predicted_tf_mae=(
                "prefix_abs_error",
                "mean",
            ),

            predicted_tf_mean=(
                "prefix_suffix_tf",
                "mean",
            ),

            true_tf_mean=(
                "true_tf",
                "mean",
            ),
        )
    )

    print(
        by_delay.to_string(
            index=False
        )
    )

    # ========================================================
    # Representative entity 1067
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

    entity = df[
        df[
            "entity_id"
        ] == 1067
    ]

    if len(
        entity
    ):

        print(
            entity.to_string(
                index=False
            )
        )

    else:

        print(
            "Entity 1067 not found in selected test users."
        )

    # ========================================================
    # Save
    # ========================================================

    df.to_csv(
        OUTPUT_DIR
        / "entity_diagnostics.csv",
        index=False,
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False,
    )

    by_delay.to_csv(
        OUTPUT_DIR
        / "prefix_suffix_by_true_delay.csv",
        index=False,
    )

    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT_DIR
        / "summary.csv"
    )


if __name__ == "__main__":
    main()