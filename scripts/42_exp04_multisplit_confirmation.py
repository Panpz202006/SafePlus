from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Experiment 04H-1
# Multi-split Robustness Confirmation
#
# IMPORTANT:
# Final model is FROZEN before this experiment:
#
#   Convex Fusion:
#       [w * Behavior + (1-w) * Delay] / T
#
#   w = 0.7
#   T = 1/3
#
# No parameter selection is allowed here.
#
# This experiment asks:
#
#   Does the final 04F model remain better when we change
#   train / validation / test split seeds?
#
# This is NOT yet fresh synthetic generation.
# It is split-robustness on the current fixed dataset.
# ============================================================


BASE_04A = Path(
    r"scripts/34_exp04_learned_behavior_evidence.py"
)

BASE_03Y = Path(
    r"scripts/32_exp03_behavior_likelihood_identification.py"
)

BASE_04F = Path(
    r"scripts/40_exp04_convex_fusion_audit.py"
)

OUTPUT_DIR = Path(
    r"reports/experiment04/multisplit_confirmation"
)


FINAL_W = 0.7
FINAL_T = 1.0 / 3.0

OLD_ALPHA = 2.0

SPLIT_SEEDS = list(
    range(1, 11)
)

BOOTSTRAP_SAMPLES = 10000


# ============================================================
# Dynamic import
# ============================================================


def load_module(
    path,
    module_name,
):

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not load {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# Bootstrap over user-level pooled errors
# ============================================================


def bootstrap_gain_ci(
    baseline_error,
    final_error,
    n_bootstrap=10000,
    seed=12345,
):

    baseline_error = np.asarray(
        baseline_error,
        dtype=np.float64,
    )

    final_error = np.asarray(
        final_error,
        dtype=np.float64,
    )

    gain = (
        baseline_error
        - final_error
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(
        gain
    )

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=np.float64,
    )

    for i in range(
        n_bootstrap
    ):

        sampled = rng.integers(
            0,
            n,
            size=n,
        )

        bootstrap_means[
            i
        ] = gain[
            sampled
        ].mean()

    return {
        "mean_gain":
            float(
                gain.mean()
            ),

        "ci_low":
            float(
                np.quantile(
                    bootstrap_means,
                    0.025,
                )
            ),

        "ci_high":
            float(
                np.quantile(
                    bootstrap_means,
                    0.975,
                )
            ),

        "improved_rate":
            float(
                (
                    gain > 1e-10
                ).mean()
            ),

        "unchanged_rate":
            float(
                (
                    np.abs(
                        gain
                    )
                    <= 1e-10
                ).mean()
            ),

        "worsened_rate":
            float(
                (
                    gain < -1e-10
                ).mean()
            ),
    }


# ============================================================
# One seed
# ============================================================


def run_one_seed(
    split_seed,
    x,
    detected,
    td,
    true_tf,
    entity_id,
    fraud_label,
    exp04a,
    exp03y,
    exp04f,
):

    print()
    print(
        "=" * 80
    )

    print(
        f"SPLIT SEED {split_seed}"
    )

    print(
        "=" * 80
    )

    (
        train_idx,
        val_idx,
        test_idx,
    ) = exp04a.split_indices(
        len(
            x
        ),
        seed=
            split_seed,
    )

    print(
        f"train={len(train_idx)} "
        f"val={len(val_idx)} "
        f"test={len(test_idx)}"
    )

    # --------------------------------------------------------
    # Retrain Behavior emission using this TRAIN split.
    #
    # Final fusion parameters remain completely frozen.
    # --------------------------------------------------------

    (
        behavior_params,
        em_history,
    ) = exp04a.train_emission_em(
        x=x,
        fraud_label=
            fraud_label,
        detected=
            detected,
        td=
            td,
        train_idx=
            train_idx,
    )

    global_scale = (
        exp04f.estimate_global_behavior_scale(
            x=x,
            fraud_label=
                fraud_label,
            detected=
                detected,
            td=
                td,
            train_idx=
                train_idx,
            behavior_params=
                behavior_params,
            exp04a=
                exp04a,
        )
    )

    method_configs = [
        (
            "behavior_raw",
            {
                "mode":
                    "behavior_raw",
            },
        ),

        (
            "delay_only",
            {
                "mode":
                    "delay_only",
            },
        ),

        (
            "old_fixed_alpha",
            {
                "mode":
                    "old_alpha",

                "alpha":
                    OLD_ALPHA,
            },
        ),

        (
            "final_convex",
            {
                "mode":
                    "convex",

                "w":
                    FINAL_W,

                "temperature":
                    FINAL_T,
            },
        ),
    ]

    summaries = []
    prediction_frames = {}

    for method_name, kwargs in method_configs:

        (
            summary,
            df,
        ) = exp04f.evaluate_config(
            split_name=
                f"split_seed_{split_seed}",
            indices=
                test_idx,
            x=x,
            detected=
                detected,
            td=
                td,
            true_tf=
                true_tf,
            entity_id=
                entity_id,
            behavior_params=
                behavior_params,
            global_scale=
                global_scale,
            exp04a=
                exp04a,
            exp03y=
                exp03y,
            **kwargs,
        )

        summaries.append(
            {
                "split_seed":
                    split_seed,

                "method":
                    method_name,

                **summary,
            }
        )

        df[
            "split_seed"
        ] = split_seed

        df[
            "method_name"
        ] = method_name

        prediction_frames[
            method_name
        ] = df

        print(
            f"{method_name:>18} | "
            f"MAE={summary['mae']:.6f} | "
            f"signed={summary['signed_error']:.6f} | "
            f"within1={summary['within_1']:.6f} | "
            f"within2={summary['within_2']:.6f}"
        )

    return (
        summaries,
        prediction_frames,
        em_history,
        global_scale,
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
        "EXPERIMENT 04H-1: "
        "MULTI-SPLIT ROBUSTNESS CONFIRMATION"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "FINAL MODEL IS FROZEN:"
    )

    print(
        f"  w = {FINAL_W}"
    )

    print(
        f"  T = {FINAL_T}"
    )

    print()

    print(
        "No validation selection is performed."
    )

    print(
        "No parameter is changed across split seeds."
    )

    print()

    print(
        "MAE = Mean Absolute Error "
        "（平均绝对误差，越小越好）"
    )

    print(
        "Signed Error = predicted tf - true tf"
    )

    print(
        "Negative = 偏早；Positive = 偏晚"
    )

    # --------------------------------------------------------
    # Load modules
    # --------------------------------------------------------

    exp04a = load_module(
        BASE_04A,
        "exp04a",
    )

    exp03y = load_module(
        BASE_03Y,
        "exp03y",
    )

    exp04f = load_module(
        BASE_04F,
        "exp04f",
    )

    # --------------------------------------------------------
    # Load fixed dataset
    # --------------------------------------------------------

    data = np.load(
        exp04a.DATA_PATH,
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

    fraud_label = (
        exp04a.get_entity_fraud_label(
            data
        )
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

    # --------------------------------------------------------
    # Run all seeds
    # --------------------------------------------------------

    all_summaries = []

    all_predictions = []

    scale_rows = []

    for split_seed in SPLIT_SEEDS:

        (
            summaries,
            prediction_frames,
            em_history,
            global_scale,
        ) = run_one_seed(
            split_seed=
                split_seed,
            x=x,
            detected=
                detected,
            td=
                td,
            true_tf=
                true_tf,
            entity_id=
                entity_id,
            fraud_label=
                fraud_label,
            exp04a=
                exp04a,
            exp03y=
                exp03y,
            exp04f=
                exp04f,
        )

        all_summaries.extend(
            summaries
        )

        for df in prediction_frames.values():

            all_predictions.append(
                df
            )

        scale_rows.append(
            {
                "split_seed":
                    split_seed,

                "global_behavior_scale":
                    global_scale,
            }
        )

        em_history.to_csv(
            OUTPUT_DIR
            / (
                f"em_history_seed_"
                f"{split_seed}.csv"
            ),
            index=False,
        )

    summary_df = pd.DataFrame(
        all_summaries
    )

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    scales_df = pd.DataFrame(
        scale_rows
    )

    # ========================================================
    # Aggregate mean ± std across split seeds
    # ========================================================

    aggregate_rows = []

    methods = [
        "behavior_raw",
        "delay_only",
        "old_fixed_alpha",
        "final_convex",
    ]

    for method in methods:

        sub = summary_df[
            summary_df[
                "method"
            ] == method
        ]

        aggregate_rows.append(
            {
                "method":
                    method,

                "mae_mean":
                    float(
                        sub[
                            "mae"
                        ].mean()
                    ),

                "mae_std":
                    float(
                        sub[
                            "mae"
                        ].std(
                            ddof=1
                        )
                    ),

                "signed_mean":
                    float(
                        sub[
                            "signed_error"
                        ].mean()
                    ),

                "signed_std":
                    float(
                        sub[
                            "signed_error"
                        ].std(
                            ddof=1
                        )
                    ),

                "within1_mean":
                    float(
                        sub[
                            "within_1"
                        ].mean()
                    ),

                "within2_mean":
                    float(
                        sub[
                            "within_2"
                        ].mean()
                    ),
            }
        )

    aggregate_df = pd.DataFrame(
        aggregate_rows
    )

    print()
    print(
        "=" * 80
    )

    print(
        "MULTI-SPLIT AGGREGATE"
    )

    print(
        "=" * 80
    )

    print(
        aggregate_df.to_string(
            index=False
        )
    )

    # ========================================================
    # Per-seed win table
    # ========================================================

    pivot = (
        summary_df
        .pivot(
            index=
                "split_seed",
            columns=
                "method",
            values=
                "mae",
        )
        .reset_index()
    )

    pivot[
        "final_beats_behavior"
    ] = (
        pivot[
            "final_convex"
        ]
        <
        pivot[
            "behavior_raw"
        ]
    )

    pivot[
        "final_beats_delay"
    ] = (
        pivot[
            "final_convex"
        ]
        <
        pivot[
            "delay_only"
        ]
    )

    pivot[
        "final_beats_old"
    ] = (
        pivot[
            "final_convex"
        ]
        <
        pivot[
            "old_fixed_alpha"
        ]
    )

    print()
    print(
        "=" * 80
    )

    print(
        "PER-SEED MAE"
    )

    print(
        "=" * 80
    )

    print(
        pivot.to_string(
            index=False
        )
    )

    print()

    print(
        "Final Convex beats Behavior on"
    )

    print(
        f"  {int(pivot['final_beats_behavior'].sum())}"
        f" / {len(pivot)} split seeds"
    )

    print(
        "Final Convex beats Delay on"
    )

    print(
        f"  {int(pivot['final_beats_delay'].sum())}"
        f" / {len(pivot)} split seeds"
    )

    print(
        "Final Convex beats Old Fixed Alpha on"
    )

    print(
        f"  {int(pivot['final_beats_old'].sum())}"
        f" / {len(pivot)} split seeds"
    )

    # ========================================================
    # Pooled paired Bootstrap CI
    # ========================================================

    # Rows from different split seeds may include the same
    # original entity in different test sets.
    #
    # Therefore this pooled CI is a robustness diagnostic,
    # not the final independent-sample paper CI.
    # ========================================================

    paired_rows = []

    final_df = predictions_df[
        predictions_df[
            "method_name"
        ] == "final_convex"
    ].copy()

    final_df = final_df.sort_values(
        [
            "split_seed",
            "index",
        ]
    )

    final_error = np.abs(
        final_df[
            "median_tf"
        ].to_numpy()
        -
        final_df[
            "true_tf"
        ].to_numpy()
    )

    for baseline_method in [
        "behavior_raw",
        "delay_only",
        "old_fixed_alpha",
    ]:

        baseline_df = predictions_df[
            predictions_df[
                "method_name"
            ] == baseline_method
        ].copy()

        baseline_df = baseline_df.sort_values(
            [
                "split_seed",
                "index",
            ]
        )

        if not np.array_equal(
            final_df[
                [
                    "split_seed",
                    "index",
                ]
            ].to_numpy(),
            baseline_df[
                [
                    "split_seed",
                    "index",
                ]
            ].to_numpy(),
        ):

            raise RuntimeError(
                "Paired prediction alignment failed."
            )

        baseline_error = np.abs(
            baseline_df[
                "median_tf"
            ].to_numpy()
            -
            baseline_df[
                "true_tf"
            ].to_numpy()
        )

        metrics = bootstrap_gain_ci(
            baseline_error=
                baseline_error,
            final_error=
                final_error,
            n_bootstrap=
                BOOTSTRAP_SAMPLES,
            seed=
                2026,
        )

        paired_rows.append(
            {
                "comparison":
                    (
                        "final_convex_vs_"
                        + baseline_method
                    ),

                **metrics,
            }
        )

    paired_df = pd.DataFrame(
        paired_rows
    )

    print()
    print(
        "=" * 80
    )

    print(
        "POOLED PAIRED ROBUSTNESS DIAGNOSTIC"
    )

    print(
        "=" * 80
    )

    print(
        paired_df.to_string(
            index=False
        )
    )

    # ========================================================
    # Delay-bucket aggregate
    # ========================================================

    bucket_rows = []

    for split_seed in SPLIT_SEEDS:

        seed_df = predictions_df[
            predictions_df[
                "split_seed"
            ] == split_seed
        ]

        for method in methods:

            method_df = seed_df[
                seed_df[
                    "method_name"
                ] == method
            ]

            for bucket in [
                "delay_1_2",
                "delay_3_5",
                "delay_6_plus",
            ]:

                sub = method_df[
                    method_df[
                        "delay_bucket"
                    ] == bucket
                ]

                if len(
                    sub
                ) == 0:
                    continue

                error = np.abs(
                    sub[
                        "median_tf"
                    ].to_numpy()
                    -
                    sub[
                        "true_tf"
                    ].to_numpy()
                )

                bucket_rows.append(
                    {
                        "split_seed":
                            split_seed,

                        "method":
                            method,

                        "delay_bucket":
                            bucket,

                        "n":
                            len(
                                sub
                            ),

                        "mae":
                            float(
                                error.mean()
                            ),
                    }
                )

    bucket_df = pd.DataFrame(
        bucket_rows
    )

    bucket_aggregate = (
        bucket_df
        .groupby(
            [
                "method",
                "delay_bucket",
            ],
            as_index=False,
        )
        .agg(
            mae_mean=(
                "mae",
                "mean",
            ),
            mae_std=(
                "mae",
                "std",
            ),
        )
    )

    print()
    print(
        "=" * 80
    )

    print(
        "DELAY-BUCKET MULTI-SPLIT AGGREGATE"
    )

    print(
        "=" * 80
    )

    print(
        bucket_aggregate.to_string(
            index=False
        )
    )

    # ========================================================
    # Save
    # ========================================================

    summary_df.to_csv(
        OUTPUT_DIR
        / "per_seed_summary.csv",
        index=False,
    )

    aggregate_df.to_csv(
        OUTPUT_DIR
        / "aggregate_summary.csv",
        index=False,
    )

    pivot.to_csv(
        OUTPUT_DIR
        / "per_seed_mae_comparison.csv",
        index=False,
    )

    predictions_df.to_csv(
        OUTPUT_DIR
        / "all_predictions.csv",
        index=False,
    )

    paired_df.to_csv(
        OUTPUT_DIR
        / "pooled_pairwise_diagnostic.csv",
        index=False,
    )

    bucket_df.to_csv(
        OUTPUT_DIR
        / "per_seed_delay_buckets.csv",
        index=False,
    )

    bucket_aggregate.to_csv(
        OUTPUT_DIR
        / "delay_bucket_aggregate.csv",
        index=False,
    )

    scales_df.to_csv(
        OUTPUT_DIR
        / "global_behavior_scales.csv",
        index=False,
    )

    print()
    print(
        "=" * 80
    )

    print(
        "04H-1 INTERPRETATION"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Strong robustness evidence:"
    )

    print(
        "  Final Convex has the lowest mean MAE,"
    )

    print(
        "  beats both single-evidence methods on most/all seeds,"
    )

    print(
        "  and retains reasonable short/mid/long-delay behavior."
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This is multi-split robustness on ONE generated dataset."
    )

    print(
        "It is not a replacement for fresh synthetic generations"
    )

    print(
        "or the final real-data holdout."
    )


if __name__ == "__main__":
    main()