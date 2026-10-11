from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Experiment 03Z
# Behavior Evidence Robustness Audit
#
# 中文：
# 行为 evidence 正向结果的稳健性检查。
#
#
# We DO NOT train a new model.
#
# We reconstruct the posterior from Experiment 03Y
# and answer:
#
# 1. Is improvement broad across users?
#
# 2. Does it also help long-delay users?
#
# 3. Are behavior-only and delay-only errors complementary?
#
# 4. Mean / Median / Mode:
#    which posterior point estimator is better for tf MAE?
#
#
# IMPORTANT:
#
# true tf is ONLY used for post-hoc evaluation.
# It never enters posterior construction.
# ============================================================


BASE_SCRIPT = Path(
    r"scripts/32_exp03_behavior_likelihood_identification.py"
)

OUTPUT_DIR = Path(
    r"reports/experiment03/"
    r"behavior_evidence_robustness"
)


SEED = 0

ALPHAS = [
    0.0,
    0.5,
    1.0,
]


BOOTSTRAP_SAMPLES = 5000


# ============================================================
# Load Experiment 03Y
# ============================================================


if not BASE_SCRIPT.exists():

    raise FileNotFoundError(
        BASE_SCRIPT
    )


spec = importlib.util.spec_from_file_location(
    "exp03y",
    BASE_SCRIPT,
)

if (
    spec is None
    or spec.loader is None
):

    raise RuntimeError(
        "Could not load Experiment 03Y."
    )


previous = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    previous
)


# ============================================================
# Posterior median
#
# Smallest candidate k such that:
#
#     P(tf <= k | x, td) >= 0.5
#
#
# For absolute-error loss,
# posterior median is the natural Bayes estimator.
# ============================================================


def posterior_median(
    candidates,
    posterior,
):

    cumulative = np.cumsum(
        posterior
    )

    index = int(
        np.searchsorted(
            cumulative,
            0.5,
        )
    )

    index = min(
        index,
        len(candidates) - 1,
    )

    return int(
        candidates[index]
    )


# ============================================================
# Build posterior exactly as 03Y
# ============================================================


def infer_one_user(
    x_user,
    observed_td,
    anchors,
    alpha,
):

    candidates = np.arange(
        0,
        observed_td,
        dtype=np.int64,
    )

    behavior_scores = []

    delay_scores = []

    for k in candidates:

        behavior_ll = (
            previous.behavior_log_likelihood(
                observed_x=x_user,
                candidate_tf=int(k),
                observed_td=observed_td,
                normal_center=anchors[
                    "normal_center"
                ],
                fraud_center=anchors[
                    "fraud_center"
                ],
                scale=anchors[
                    "scale"
                ],
            )
        )

        delay = (
            observed_td
            - int(k)
        )

        delay_prob = max(
            previous.delay_probability(
                delay
            ),
            1e-30,
        )

        behavior_scores.append(
            behavior_ll
        )

        delay_scores.append(
            np.log(
                delay_prob
            )
        )

    behavior_scores = np.asarray(
        behavior_scores,
        dtype=np.float64,
    )

    delay_scores = np.asarray(
        delay_scores,
        dtype=np.float64,
    )

    # Same normalization as 03Y.
    behavior_std = float(
        np.std(
            behavior_scores
        )
    )

    if behavior_std < 1e-8:

        normalized_behavior = (
            np.zeros_like(
                behavior_scores
            )
        )

    else:

        normalized_behavior = (
            behavior_scores
            -
            behavior_scores.mean()
        ) / behavior_std

    total_scores = (
        alpha
        * normalized_behavior
        +
        delay_scores
    )

    posterior = previous.softmax(
        total_scores
    )

    posterior_mean = float(
        np.sum(
            posterior
            * candidates
        )
    )

    posterior_mode = int(
        candidates[
            np.argmax(
                posterior
            )
        ]
    )

    posterior_med = posterior_median(
        candidates,
        posterior,
    )

    # ========================================================
    # Behavior-only posterior
    # ========================================================

    behavior_posterior = (
        previous.softmax(
            normalized_behavior
        )
    )

    behavior_mean = float(
        np.sum(
            behavior_posterior
            * candidates
        )
    )

    return {
        "posterior_mean_tf":
            posterior_mean,

        "posterior_median_tf":
            posterior_med,

        "posterior_mode_tf":
            posterior_mode,

        "behavior_only_mean_tf":
            behavior_mean,
    }


# ============================================================
# MAE summary
# ============================================================


def summarize_estimator(
    df,
    prediction_column,
):

    error = (
        df[
            prediction_column
        ].to_numpy()
        -
        df[
            "true_tf"
        ].to_numpy()
    )

    absolute_error = np.abs(
        error
    )

    return {
        "mae":
            float(
                absolute_error.mean()
            ),

        "signed_error":
            float(
                error.mean()
            ),

        "within_1":
            float(
                (
                    absolute_error <= 1
                ).mean()
            ),

        "within_2":
            float(
                (
                    absolute_error <= 2
                ).mean()
            ),
    }


# ============================================================
# Paired bootstrap CI
#
#
# Improvement:
#
#     baseline absolute error
#
#          -
#
#     new absolute error
#
#
# Positive:
#
#     new method is better.
# ============================================================


def bootstrap_improvement_ci(
    baseline_error,
    new_error,
    n_bootstrap=5000,
    seed=0,
):

    rng = np.random.default_rng(
        seed
    )

    improvement = (
        baseline_error
        -
        new_error
    )

    n = len(
        improvement
    )

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=np.float64,
    )

    for i in range(
        n_bootstrap
    ):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        bootstrap_means[i] = (
            improvement[
                indices
            ].mean()
        )

    lower = float(
        np.quantile(
            bootstrap_means,
            0.025,
        )
    )

    upper = float(
        np.quantile(
            bootstrap_means,
            0.975,
        )
    )

    mean = float(
        improvement.mean()
    )

    return (
        mean,
        lower,
        upper,
    )


# ============================================================
# Delay bucket
# ============================================================


def delay_bucket(
    delay,
):

    if delay <= 2:

        return "delay_1_2"

    if delay <= 5:

        return "delay_3_5"

    return "delay_6_plus"


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
        "EXPERIMENT 03Z:"
    )

    print(
        "BEHAVIOR EVIDENCE ROBUSTNESS AUDIT"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "No new model is trained."
    )

    print(
        "true tf is used ONLY for evaluation."
    )

    # ========================================================
    # Load data
    # ========================================================

    data = np.load(
        previous.DATA_PATH,
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
    ) = previous.split_indices(
        len(
            x
        ),
        seed=SEED,
    )

    # ========================================================
    # Same TRAIN-only anchors as 03Y
    # ========================================================

    anchors = (
        previous.estimate_behavior_anchors(
            x,
            detected,
            td,
            train_idx,
        )
    )

    # ========================================================
    # Reconstruct test posterior
    # ========================================================

    rows = []

    for idx in test_idx:

        if detected[idx] != 1:

            continue

        d = int(
            td[idx]
        )

        tf = int(
            true_tf[idx]
        )

        for alpha in ALPHAS:

            prediction = infer_one_user(
                x_user=x[idx],
                observed_td=d,
                anchors=anchors,
                alpha=alpha,
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

                    "delay_bucket":
                        delay_bucket(
                            d - tf
                        ),

                    **prediction,
                }
            )

    df = pd.DataFrame(
        rows
    )

    # ========================================================
    # 1. Mean / Median / Mode
    # ========================================================

    estimator_rows = []

    for alpha in ALPHAS:

        sub = df[
            df[
                "alpha"
            ] == alpha
        ].copy()

        for estimator in [
            "posterior_mean_tf",
            "posterior_median_tf",
            "posterior_mode_tf",
        ]:

            metrics = summarize_estimator(
                sub,
                estimator,
            )

            estimator_rows.append(
                {
                    "alpha":
                        alpha,

                    "estimator":
                        estimator,

                    **metrics,
                }
            )

    estimator_summary = pd.DataFrame(
        estimator_rows
    )

    print()

    print(
        "=" * 80
    )

    print(
        "MEAN vs MEDIAN vs MODE"
    )

    print(
        "=" * 80
    )

    print(
        estimator_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # 2. User-level improvement
    # ========================================================

    wide = df.pivot(
        index=[
            "entity_id",
            "true_tf",
            "observed_td",
            "true_delay",
            "delay_bucket",
        ],
        columns="alpha",
        values="posterior_mean_tf",
    ).reset_index()

    wide = wide.rename(
        columns={
            0.0:
                "tf_alpha_0",

            0.5:
                "tf_alpha_05",

            1.0:
                "tf_alpha_10",
        }
    )

    wide[
        "error_alpha_0"
    ] = np.abs(
        wide[
            "tf_alpha_0"
        ]
        -
        wide[
            "true_tf"
        ]
    )

    wide[
        "error_alpha_05"
    ] = np.abs(
        wide[
            "tf_alpha_05"
        ]
        -
        wide[
            "true_tf"
        ]
    )

    wide[
        "error_alpha_10"
    ] = np.abs(
        wide[
            "tf_alpha_10"
        ]
        -
        wide[
            "true_tf"
        ]
    )

    comparison_rows = []

    for label in [
        "05",
        "10",
    ]:

        baseline_error = wide[
            "error_alpha_0"
        ].to_numpy()

        new_error = wide[
            f"error_alpha_{label}"
        ].to_numpy()

        difference = (
            baseline_error
            -
            new_error
        )

        (
            mean_gain,
            ci_low,
            ci_high,
        ) = bootstrap_improvement_ci(
            baseline_error,
            new_error,
            n_bootstrap=
                BOOTSTRAP_SAMPLES,
            seed=SEED,
        )

        comparison_rows.append(
            {
                "alpha":
                    (
                        0.5
                        if label == "05"
                        else 1.0
                    ),

                "improved_rate":
                    float(
                        (
                            difference > 1e-10
                        ).mean()
                    ),

                "unchanged_rate":
                    float(
                        (
                            np.abs(
                                difference
                            )
                            <= 1e-10
                        ).mean()
                    ),

                "worsened_rate":
                    float(
                        (
                            difference < -1e-10
                        ).mean()
                    ),

                "mean_mae_gain":
                    mean_gain,

                "bootstrap_ci_low":
                    ci_low,

                "bootstrap_ci_high":
                    ci_high,
            }
        )

    comparison_summary = pd.DataFrame(
        comparison_rows
    )

    print()

    print(
        "=" * 80
    )

    print(
        "USER-LEVEL EFFECT vs DELAY-ONLY"
    )

    print(
        "=" * 80
    )

    print(
        comparison_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # 3. By delay bucket
    # ========================================================

    bucket_rows = []

    for bucket in [
        "delay_1_2",
        "delay_3_5",
        "delay_6_plus",
    ]:

        bucket_df = wide[
            wide[
                "delay_bucket"
            ] == bucket
        ]

        for alpha_label, column in [
            (
                0.0,
                "error_alpha_0",
            ),
            (
                0.5,
                "error_alpha_05",
            ),
            (
                1.0,
                "error_alpha_10",
            ),
        ]:

            bucket_rows.append(
                {
                    "delay_bucket":
                        bucket,

                    "alpha":
                        alpha_label,

                    "n_users":
                        len(
                            bucket_df
                        ),

                    "mae":
                        float(
                            bucket_df[
                                column
                            ].mean()
                        ),
                }
            )

    bucket_summary = pd.DataFrame(
        bucket_rows
    )

    print()

    print(
        "=" * 80
    )

    print(
        "BY TRUE DELAY"
    )

    print(
        "=" * 80
    )

    print(
        bucket_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # 4. Behavior-only vs delay-only error correlation
    # ========================================================

    reference = df[
        df[
            "alpha"
        ] == 0.0
    ].copy()

    delay_signed_error = (
        reference[
            "posterior_mean_tf"
        ].to_numpy()
        -
        reference[
            "true_tf"
        ].to_numpy()
    )

    behavior_signed_error = (
        reference[
            "behavior_only_mean_tf"
        ].to_numpy()
        -
        reference[
            "true_tf"
        ].to_numpy()
    )

    correlation = float(
        np.corrcoef(
            delay_signed_error,
            behavior_signed_error,
        )[0, 1]
    )

    behavior_mae = float(
        np.mean(
            np.abs(
                behavior_signed_error
            )
        )
    )

    behavior_signed_mean = float(
        behavior_signed_error.mean()
    )

    print()

    print(
        "=" * 80
    )

    print(
        "EVIDENCE COMPLEMENTARITY"
    )

    print(
        "=" * 80
    )

    print(
        "Behavior-only MAE:",
        behavior_mae,
    )

    print(
        "Behavior-only signed error:",
        behavior_signed_mean,
    )

    print(
        "Signed-error correlation"
    )

    print(
        "(delay-only vs behavior-only):",
        correlation,
    )

    print()

    print(
        "Negative / weak correlation means:"
    )

    print(
        "the two evidence sources make"
    )

    print(
        "different mistakes and can complement each other."
    )

    # ========================================================
    # 5. Largest improvements / failures
    # ========================================================

    wide[
        "gain_alpha_10"
    ] = (
        wide[
            "error_alpha_0"
        ]
        -
        wide[
            "error_alpha_10"
        ]
    )

    print()

    print(
        "=" * 80
    )

    print(
        "TOP IMPROVEMENTS: alpha=1.0"
    )

    print(
        "=" * 80
    )

    print(
        wide.sort_values(
            "gain_alpha_10",
            ascending=False,
        )
        .head(
            15
        )
        [
            [
                "entity_id",
                "true_tf",
                "observed_td",
                "true_delay",
                "error_alpha_0",
                "error_alpha_10",
                "gain_alpha_10",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()

    print(
        "=" * 80
    )

    print(
        "WORST FAILURES: alpha=1.0"
    )

    print(
        "=" * 80
    )

    print(
        wide.sort_values(
            "gain_alpha_10",
            ascending=True,
        )
        .head(
            15
        )
        [
            [
                "entity_id",
                "true_tf",
                "observed_td",
                "true_delay",
                "error_alpha_0",
                "error_alpha_10",
                "gain_alpha_10",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Save
    # ========================================================

    df.to_csv(
        OUTPUT_DIR
        / "posterior_estimators.csv",
        index=False,
    )

    estimator_summary.to_csv(
        OUTPUT_DIR
        / "estimator_summary.csv",
        index=False,
    )

    comparison_summary.to_csv(
        OUTPUT_DIR
        / "user_level_comparison.csv",
        index=False,
    )

    bucket_summary.to_csv(
        OUTPUT_DIR
        / "delay_bucket_summary.csv",
        index=False,
    )

    wide.to_csv(
        OUTPUT_DIR
        / "paired_entity_analysis.csv",
        index=False,
    )

    print()

    print(
        "=" * 80
    )

    print(
        "FINAL DECISION RULE"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Strong evidence for behavior likelihood if:"
    )

    print(
        "1. Most users improve;"
    )

    print(
        "2. Bootstrap MAE-gain CI stays above 0;"
    )

    print(
        "3. Improvement remains for delay >= 6;"
    )

    print(
        "4. Gain is not limited to one posterior estimator."
    )


if __name__ == "__main__":
    main()