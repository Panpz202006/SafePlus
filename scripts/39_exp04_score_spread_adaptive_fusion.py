from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Experiment 04E
# Score-Spread Adaptive Fusion
#
# Motivation:
# Experiment 04D found that score_spread had the strongest
# validation correlation with Behavior Advantage.
#
# Idea:
# larger score spread
# -> Behavior distinguishes candidate tf more strongly
# -> larger behavior weight alpha_i
#
# Protocol:
# 1. Train behavior emission model on TRAIN only.
# 2. Estimate one global behavior score scale on TRAIN only.
# 3. Build TRAIN reference distribution of score_spread.
# 4. Convert each user's score_spread to a TRAIN-based percentile.
# 5. Map percentile -> adaptive alpha_i.
# 6. Select alpha_min / alpha_max on VALIDATION only.
# 7. Evaluate frozen rule on TEST.
# ============================================================


BASE_04A = Path(
    r"scripts/34_exp04_learned_behavior_evidence.py"
)

BASE_03Y = Path(
    r"scripts/32_exp03_behavior_likelihood_identification.py"
)

OUTPUT_DIR = Path(
    r"reports/experiment04/score_spread_adaptive_fusion"
)

SEED = 0

FIXED_BETA_VALUES = [
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
]

ALPHA_MIN_VALUES = [
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
]

ALPHA_MAX_VALUES = [
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
]

BOOTSTRAP_SAMPLES = 5000


# ============================================================
# Dynamic import
# ============================================================


def load_module(path, module_name):

    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
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
# Helpers
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


def delay_bucket(delay):

    if delay <= 2:
        return "delay_1_2"

    if delay <= 5:
        return "delay_3_5"

    return "delay_6_plus"


def bootstrap_improvement_ci(
    baseline_error,
    new_error,
    n_bootstrap=5000,
    seed=0,
):

    baseline_error = np.asarray(
        baseline_error,
        dtype=np.float64,
    )

    new_error = np.asarray(
        new_error,
        dtype=np.float64,
    )

    improvement = (
        baseline_error
        - new_error
    )

    rng = np.random.default_rng(
        seed
    )

    n = len(
        improvement
    )

    boot = np.empty(
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

        boot[i] = improvement[
            indices
        ].mean()

    return {
        "mean_gain":
            float(
                improvement.mean()
            ),

        "ci_low":
            float(
                np.quantile(
                    boot,
                    0.025,
                )
            ),

        "ci_high":
            float(
                np.quantile(
                    boot,
                    0.975,
                )
            ),

        "improved_rate":
            float(
                (
                    improvement > 1e-10
                ).mean()
            ),

        "unchanged_rate":
            float(
                (
                    np.abs(
                        improvement
                    )
                    <= 1e-10
                ).mean()
            ),

        "worsened_rate":
            float(
                (
                    improvement < -1e-10
                ).mean()
            ),
    }


# ============================================================
# Candidate scores
# ============================================================


def behavior_candidate_scores(
    x_user,
    observed_td,
    behavior_params,
    exp04a,
):

    d = int(
        observed_td
    )

    observed_x = x_user[
        : d + 1
    ]

    candidates = np.arange(
        0,
        d,
        dtype=np.int64,
    )

    scores = []

    for k in candidates:

        score = (
            exp04a.candidate_behavior_log_likelihood(
                observed_x=observed_x,
                candidate_tf=int(k),
                params=behavior_params,
            )
        )

        scores.append(
            score
        )

    return (
        candidates,
        np.asarray(
            scores,
            dtype=np.float64,
        ),
    )


def delay_candidate_scores(
    candidates,
    observed_td,
    exp03y,
):

    d = int(
        observed_td
    )

    scores = []

    for k in candidates:

        delay = (
            d - int(k)
        )

        probability = max(
            exp03y.delay_probability(
                delay
            ),
            1e-30,
        )

        scores.append(
            math.log(
                probability
            )
        )

    return np.asarray(
        scores,
        dtype=np.float64,
    )


# ============================================================
# TRAIN-only calibration statistics
# ============================================================


def estimate_global_behavior_scale(
    x,
    fraud_label,
    detected,
    td,
    train_idx,
    behavior_params,
    exp04a,
):

    per_user_std = []

    for idx in train_idx:

        if fraud_label[idx] != 1:
            continue

        if detected[idx] != 1:
            continue

        (
            candidates,
            raw_scores,
        ) = behavior_candidate_scores(
            x_user=x[idx],
            observed_td=td[idx],
            behavior_params=behavior_params,
            exp04a=exp04a,
        )

        if len(candidates) <= 1:
            continue

        score_std = float(
            np.std(
                raw_scores
            )
        )

        if score_std > 1e-8:
            per_user_std.append(
                score_std
            )

    if not per_user_std:
        raise RuntimeError(
            "Could not estimate global behavior scale."
        )

    return max(
        float(
            np.median(
                np.asarray(
                    per_user_std,
                    dtype=np.float64,
                )
            )
        ),
        1e-8,
    )


def build_train_score_spread_reference(
    x,
    fraud_label,
    detected,
    td,
    train_idx,
    behavior_params,
    global_scale,
    exp04a,
):

    spreads = []

    for idx in train_idx:

        if fraud_label[idx] != 1:
            continue

        if detected[idx] != 1:
            continue

        (
            candidates,
            raw_scores,
        ) = behavior_candidate_scores(
            x_user=x[idx],
            observed_td=td[idx],
            behavior_params=behavior_params,
            exp04a=exp04a,
        )

        if len(candidates) <= 1:
            continue

        spread = float(
            (
                np.max(raw_scores)
                - np.min(raw_scores)
            )
            / global_scale
        )

        spreads.append(
            spread
        )

    if not spreads:
        raise RuntimeError(
            "No TRAIN score spreads were available."
        )

    return np.sort(
        np.asarray(
            spreads,
            dtype=np.float64,
        )
    )


def spread_percentile(
    score_spread,
    train_spreads,
):

    count = np.searchsorted(
        train_spreads,
        score_spread,
        side="right",
    )

    return float(
        count
        / len(train_spreads)
    )


# ============================================================
# One-user inference
# ============================================================


def infer_one(
    x_user,
    observed_td,
    behavior_params,
    global_scale,
    train_spreads,
    exp04a,
    exp03y,
    mode,
    fixed_beta=None,
    alpha_min=None,
    alpha_max=None,
):

    (
        candidates,
        raw_behavior,
    ) = behavior_candidate_scores(
        x_user=x_user,
        observed_td=observed_td,
        behavior_params=behavior_params,
        exp04a=exp04a,
    )

    delay_scores = (
        delay_candidate_scores(
            candidates=candidates,
            observed_td=observed_td,
            exp03y=exp03y,
        )
    )

    behavior_scaled = (
        raw_behavior
        - raw_behavior.mean()
    ) / global_scale

    score_spread = float(
        np.max(
            behavior_scaled
        )
        - np.min(
            behavior_scaled
        )
    )

    spread_confidence = (
        spread_percentile(
            score_spread=
                score_spread,
            train_spreads=
                train_spreads,
        )
    )

    if mode == "delay_only":

        alpha_i = 0.0
        total_scores = delay_scores

    elif mode == "behavior_only":

        alpha_i = np.nan
        total_scores = behavior_scaled

    elif mode == "fixed":

        if fixed_beta is None:
            raise ValueError(
                "fixed_beta is required."
            )

        alpha_i = float(
            fixed_beta
        )

        total_scores = (
            alpha_i
            * behavior_scaled
            +
            delay_scores
        )

    elif mode == "adaptive":

        if (
            alpha_min is None
            or alpha_max is None
        ):
            raise ValueError(
                "alpha_min and alpha_max are required."
            )

        alpha_i = (
            float(alpha_min)
            +
            (
                float(alpha_max)
                - float(alpha_min)
            )
            * spread_confidence
        )

        total_scores = (
            alpha_i
            * behavior_scaled
            +
            delay_scores
        )

    else:

        raise ValueError(
            mode
        )

    posterior = (
        exp04a.stable_softmax(
            total_scores
        )
    )

    mean_tf = float(
        np.sum(
            posterior
            * candidates
        )
    )

    median_tf = (
        posterior_median(
            candidates,
            posterior,
        )
    )

    mode_tf = int(
        candidates[
            np.argmax(
                posterior
            )
        ]
    )

    return {
        "mean_tf":
            mean_tf,

        "median_tf":
            median_tf,

        "mode_tf":
            mode_tf,

        "score_spread":
            score_spread,

        "spread_confidence":
            spread_confidence,

        "alpha_i":
            alpha_i,
    }


# ============================================================
# Evaluate one configuration
# ============================================================


def evaluate_config(
    split_name,
    indices,
    x,
    detected,
    td,
    true_tf,
    entity_id,
    behavior_params,
    global_scale,
    train_spreads,
    exp04a,
    exp03y,
    mode,
    fixed_beta=None,
    alpha_min=None,
    alpha_max=None,
):

    rows = []

    for idx in indices:

        if detected[idx] != 1:
            continue

        tf = int(
            true_tf[idx]
        )

        d = int(
            td[idx]
        )

        pred = infer_one(
            x_user=x[idx],
            observed_td=d,
            behavior_params=behavior_params,
            global_scale=global_scale,
            train_spreads=train_spreads,
            exp04a=exp04a,
            exp03y=exp03y,
            mode=mode,
            fixed_beta=fixed_beta,
            alpha_min=alpha_min,
            alpha_max=alpha_max,
        )

        rows.append(
            {
                "split":
                    split_name,

                "index":
                    int(idx),

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

                "delay_bucket":
                    delay_bucket(
                        d - tf
                    ),

                "method":
                    mode,

                "fixed_beta":
                    fixed_beta,

                "alpha_min":
                    alpha_min,

                "alpha_max":
                    alpha_max,

                **pred,
            }
        )

    df = pd.DataFrame(
        rows
    )

    if len(df) == 0:
        raise RuntimeError(
            f"No detected fraud users in {split_name}."
        )

    error = (
        df[
            "median_tf"
        ].to_numpy()
        -
        df[
            "true_tf"
        ].to_numpy()
    )

    summary = {
        "n":
            len(df),

        "mae":
            float(
                np.mean(
                    np.abs(
                        error
                    )
                )
            ),

        "signed_error":
            float(
                np.mean(
                    error
                )
            ),

        "within_1":
            float(
                (
                    np.abs(
                        error
                    )
                    <= 1
                ).mean()
            ),

        "within_2":
            float(
                (
                    np.abs(
                        error
                    )
                    <= 2
                ).mean()
            ),

        "mean_alpha":
            (
                float(
                    np.nanmean(
                        df[
                            "alpha_i"
                        ].to_numpy(
                            dtype=np.float64
                        )
                    )
                )
                if not np.all(
                    pd.isna(
                        df[
                            "alpha_i"
                        ]
                    )
                )
                else np.nan
            ),

        "mean_spread_confidence":
            float(
                df[
                    "spread_confidence"
                ].mean()
            ),
    }

    return (
        summary,
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

    print("=" * 80)
    print(
        "EXPERIMENT 04E: "
        "SCORE-SPREAD ADAPTIVE FUSION"
    )
    print("=" * 80)

    print()
    print(
        "Score Spread = 行为分数跨度；"
        "候选 tf 的最好分数和最差分数之间的差距"
    )

    print(
        "Adaptive Fusion = 自适应融合；"
        "每个用户拥有自己的 alpha_i"
    )

    print(
        "MAE = Mean Absolute Error（平均绝对误差，越小越好）"
    )

    print(
        "Signed Error = predicted tf - true tf "
        "（负数偏早，正数偏晚）"
    )

    exp04a = load_module(
        BASE_04A,
        "exp04a",
    )

    exp03y = load_module(
        BASE_03Y,
        "exp03y",
    )

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
            len(x)
        )

    (
        train_idx,
        val_idx,
        test_idx,
    ) = exp04a.split_indices(
        len(x),
        seed=SEED,
    )

    print()
    print(
        f"Split: "
        f"train={len(train_idx)} "
        f"val={len(val_idx)} "
        f"test={len(test_idx)}"
    )

    (
        behavior_params,
        em_history,
    ) = exp04a.train_emission_em(
        x=x,
        fraud_label=fraud_label,
        detected=detected,
        td=td,
        train_idx=train_idx,
    )

    global_scale = (
        estimate_global_behavior_scale(
            x=x,
            fraud_label=fraud_label,
            detected=detected,
            td=td,
            train_idx=train_idx,
            behavior_params=behavior_params,
            exp04a=exp04a,
        )
    )

    train_spreads = (
        build_train_score_spread_reference(
            x=x,
            fraud_label=fraud_label,
            detected=detected,
            td=td,
            train_idx=train_idx,
            behavior_params=behavior_params,
            global_scale=global_scale,
            exp04a=exp04a,
        )
    )

    print()
    print(
        "Global behavior scale from TRAIN =",
        f"{global_scale:.6f}",
    )

    print(
        "TRAIN score-spread reference:"
    )

    print(
        "  n =",
        len(train_spreads),
    )

    print(
        "  median =",
        float(
            np.median(
                train_spreads
            )
        ),
    )

    print(
        "  q25 =",
        float(
            np.quantile(
                train_spreads,
                0.25,
            )
        ),
    )

    print(
        "  q75 =",
        float(
            np.quantile(
                train_spreads,
                0.75,
            )
        ),
    )

    fixed_rows = []

    for beta in FIXED_BETA_VALUES:

        (
            summary,
            _,
        ) = evaluate_config(
            split_name="validation",
            indices=val_idx,
            x=x,
            detected=detected,
            td=td,
            true_tf=true_tf,
            entity_id=entity_id,
            behavior_params=behavior_params,
            global_scale=global_scale,
            train_spreads=train_spreads,
            exp04a=exp04a,
            exp03y=exp03y,
            mode="fixed",
            fixed_beta=beta,
        )

        fixed_rows.append(
            {
                "beta":
                    beta,

                **summary,
            }
        )

    fixed_validation = pd.DataFrame(
        fixed_rows
    )

    fixed_validation[
        "abs_signed_error"
    ] = np.abs(
        fixed_validation[
            "signed_error"
        ]
    )

    fixed_selected = (
        fixed_validation
        .sort_values(
            [
                "mae",
                "abs_signed_error",
                "beta",
            ]
        )
        .iloc[0]
    )

    selected_fixed_beta = float(
        fixed_selected[
            "beta"
        ]
    )

    adaptive_rows = []

    print()
    print("=" * 80)
    print(
        "VALIDATION SCORE-SPREAD ADAPTIVE GRID"
    )
    print("=" * 80)

    for alpha_min in ALPHA_MIN_VALUES:

        for alpha_max in ALPHA_MAX_VALUES:

            if alpha_max < alpha_min:
                continue

            (
                summary,
                _,
            ) = evaluate_config(
                split_name="validation",
                indices=val_idx,
                x=x,
                detected=detected,
                td=td,
                true_tf=true_tf,
                entity_id=entity_id,
                behavior_params=behavior_params,
                global_scale=global_scale,
                train_spreads=train_spreads,
                exp04a=exp04a,
                exp03y=exp03y,
                mode="adaptive",
                alpha_min=alpha_min,
                alpha_max=alpha_max,
            )

            adaptive_rows.append(
                {
                    "alpha_min":
                        alpha_min,

                    "alpha_max":
                        alpha_max,

                    **summary,
                }
            )

            print(
                f"amin={alpha_min:>4} "
                f"amax={alpha_max:>4} | "
                f"MAE={summary['mae']:.6f} | "
                f"signed={summary['signed_error']:.6f} | "
                f"mean alpha={summary['mean_alpha']:.6f} | "
                f"mean confidence={summary['mean_spread_confidence']:.6f}"
            )

    adaptive_validation = pd.DataFrame(
        adaptive_rows
    )

    adaptive_validation[
        "abs_signed_error"
    ] = np.abs(
        adaptive_validation[
            "signed_error"
        ]
    )

    adaptive_selected = (
        adaptive_validation
        .sort_values(
            [
                "mae",
                "abs_signed_error",
                "alpha_min",
                "alpha_max",
            ]
        )
        .iloc[0]
    )

    selected_alpha_min = float(
        adaptive_selected[
            "alpha_min"
        ]
    )

    selected_alpha_max = float(
        adaptive_selected[
            "alpha_max"
        ]
    )

    print()
    print(
        "VALIDATION-SELECTED FIXED:"
    )

    print(
        f"  beta={selected_fixed_beta} | "
        f"MAE={float(fixed_selected['mae']):.6f} | "
        f"signed={float(fixed_selected['signed_error']):.6f}"
    )

    print()
    print(
        "VALIDATION-SELECTED SCORE-SPREAD ADAPTIVE:"
    )

    print(
        f"  alpha_min={selected_alpha_min} "
        f"alpha_max={selected_alpha_max} | "
        f"MAE={float(adaptive_selected['mae']):.6f} | "
        f"signed={float(adaptive_selected['signed_error']):.6f}"
    )

    method_configs = [
        (
            "delay_only",
            {
                "mode":
                    "delay_only",
            },
        ),
        (
            "behavior_only",
            {
                "mode":
                    "behavior_only",
            },
        ),
        (
            "fixed_global_fusion",
            {
                "mode":
                    "fixed",

                "fixed_beta":
                    selected_fixed_beta,
            },
        ),
        (
            "score_spread_adaptive_fusion",
            {
                "mode":
                    "adaptive",

                "alpha_min":
                    selected_alpha_min,

                "alpha_max":
                    selected_alpha_max,
            },
        ),
    ]

    test_results = {}
    test_frames = {}

    for method, kwargs in method_configs:

        (
            summary,
            df,
        ) = evaluate_config(
            split_name="test",
            indices=test_idx,
            x=x,
            detected=detected,
            td=td,
            true_tf=true_tf,
            entity_id=entity_id,
            behavior_params=behavior_params,
            global_scale=global_scale,
            train_spreads=train_spreads,
            exp04a=exp04a,
            exp03y=exp03y,
            **kwargs,
        )

        test_results[
            method
        ] = summary

        df[
            "method_label"
        ] = method

        test_frames[
            method
        ] = df

    overall_rows = []

    for method, _ in method_configs:

        overall_rows.append(
            {
                "method":
                    method,

                **test_results[
                    method
                ],
            }
        )

    overall_summary = pd.DataFrame(
        overall_rows
    )

    print()
    print("=" * 80)
    print(
        "FIXED TEST OVERALL"
    )
    print("=" * 80)

    print(
        overall_summary[
            [
                "method",
                "n",
                "mae",
                "signed_error",
                "within_1",
                "within_2",
                "mean_alpha",
                "mean_spread_confidence",
            ]
        ].to_string(
            index=False
        )
    )

    bucket_rows = []

    for method, df in test_frames.items():

        for bucket in [
            "delay_1_2",
            "delay_3_5",
            "delay_6_plus",
        ]:

            sub = df[
                df[
                    "delay_bucket"
                ] == bucket
            ]

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
                    "delay_bucket":
                        bucket,

                    "method":
                        method,

                    "n":
                        len(sub),

                    "mae":
                        float(
                            error.mean()
                        ),

                    "mean_alpha":
                        (
                            float(
                                np.nanmean(
                                    sub[
                                        "alpha_i"
                                    ].to_numpy(
                                        dtype=np.float64
                                    )
                                )
                            )
                            if not np.all(
                                pd.isna(
                                    sub[
                                        "alpha_i"
                                    ]
                                )
                            )
                            else np.nan
                        ),

                    "mean_spread_confidence":
                        float(
                            sub[
                                "spread_confidence"
                            ].mean()
                        ),
                }
            )

    bucket_summary = pd.DataFrame(
        bucket_rows
    )

    print()
    print("=" * 80)
    print(
        "TEST BY TRUE DELAY"
    )
    print("=" * 80)

    print(
        bucket_summary.to_string(
            index=False
        )
    )

    adaptive_df = test_frames[
        "score_spread_adaptive_fusion"
    ]

    adaptive_error = np.abs(
        adaptive_df[
            "median_tf"
        ].to_numpy()
        -
        adaptive_df[
            "true_tf"
        ].to_numpy()
    )

    paired_rows = []

    for baseline_method in [
        "delay_only",
        "behavior_only",
        "fixed_global_fusion",
    ]:

        baseline_df = test_frames[
            baseline_method
        ]

        baseline_error = np.abs(
            baseline_df[
                "median_tf"
            ].to_numpy()
            -
            baseline_df[
                "true_tf"
            ].to_numpy()
        )

        metrics = (
            bootstrap_improvement_ci(
                baseline_error=
                    baseline_error,
                new_error=
                    adaptive_error,
                n_bootstrap=
                    BOOTSTRAP_SAMPLES,
                seed=SEED,
            )
        )

        paired_rows.append(
            {
                "comparison":
                    f"adaptive_vs_{baseline_method}",

                **metrics,
            }
        )

    paired_summary = pd.DataFrame(
        paired_rows
    )

    print()
    print("=" * 80)
    print(
        "PAIRED USER-LEVEL IMPROVEMENT"
    )
    print("=" * 80)

    print(
        paired_summary.to_string(
            index=False
        )
    )

    em_history.to_csv(
        OUTPUT_DIR
        / "em_history.csv",
        index=False,
    )

    fixed_validation.to_csv(
        OUTPUT_DIR
        / "validation_fixed_fusion.csv",
        index=False,
    )

    adaptive_validation.to_csv(
        OUTPUT_DIR
        / "validation_score_spread_grid.csv",
        index=False,
    )

    overall_summary.to_csv(
        OUTPUT_DIR
        / "test_overall_summary.csv",
        index=False,
    )

    bucket_summary.to_csv(
        OUTPUT_DIR
        / "test_delay_bucket_summary.csv",
        index=False,
    )

    paired_summary.to_csv(
        OUTPUT_DIR
        / "test_user_level_improvement.csv",
        index=False,
    )

    np.savez(
        OUTPUT_DIR
        / "train_score_spread_reference.npz",
        train_spreads=
            train_spreads,
        global_scale=
            np.asarray(
                [global_scale],
                dtype=np.float64,
            ),
    )

    for method, df in test_frames.items():

        df.to_csv(
            OUTPUT_DIR
            / f"test_{method}_predictions.csv",
            index=False,
        )

    print()
    print("=" * 80)
    print(
        "HOW TO INTERPRET 04E"
    )
    print("=" * 80)

    print()
    print(
        "Score Spread（行为分数跨度）:"
    )

    print(
        "  larger spread means Behavior separates "
        "candidate tf more strongly."
    )

    print()
    print(
        "Adaptive alpha_i:"
    )

    print(
        "  larger TRAIN-relative spread percentile "
        "gives larger Behavior weight."
    )

    print()
    print(
        "Strong positive evidence if:"
    )

    print(
        "  1. adaptive fusion beats fixed global fusion;"
    )

    print(
        "  2. mid/long-delay buckets improve;"
    )

    print(
        "  3. signed error remains close to zero;"
    )

    print(
        "  4. bootstrap CI vs fixed fusion stays above zero."
    )


if __name__ == "__main__":
    main()