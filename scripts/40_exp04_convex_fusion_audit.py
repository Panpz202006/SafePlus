from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE_04A = Path(
    r"scripts/34_exp04_learned_behavior_evidence.py"
)

BASE_03Y = Path(
    r"scripts/32_exp03_behavior_likelihood_identification.py"
)

OUTPUT_DIR = Path(
    r"reports/experiment04/convex_fusion_audit"
)

SEED = 0

OLD_ALPHA_VALUES = [
    0.0,
    0.1,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
]

W_VALUES = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    2.0 / 3.0,
    0.7,
    0.8,
    0.9,
    1.0,
]

TEMPERATURE_VALUES = [
    0.2,
    0.25,
    1.0 / 3.0,
    0.5,
    0.75,
    1.0,
    1.5,
    2.0,
    4.0,
]

BOOTSTRAP_SAMPLES = 5000


# ============================================================
# Dynamic import
# ============================================================


def load_module(
    path: Path,
    module_name: str,
):

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    spec = (
        importlib.util.spec_from_file_location(
            module_name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Could not load {path}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# Basic helpers
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


def delay_bucket(
    delay,
):

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
            d
            - int(k)
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
# TRAIN-only Behavior scale
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

        if fraud_label[
            idx
        ] != 1:
            continue

        if detected[
            idx
        ] != 1:
            continue

        (
            candidates,
            raw_scores,
        ) = behavior_candidate_scores(
            x_user=x[
                idx
            ],
            observed_td=td[
                idx
            ],
            behavior_params=
                behavior_params,
            exp04a=exp04a,
        )

        if len(
            candidates
        ) <= 1:
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


# ============================================================
# One-user inference
# ============================================================


def infer_one(
    x_user,
    observed_td,
    behavior_params,
    global_scale,
    exp04a,
    exp03y,
    mode,
    alpha=None,
    w=None,
    temperature=None,
):

    (
        candidates,
        raw_behavior,
    ) = behavior_candidate_scores(
        x_user=x_user,
        observed_td=observed_td,
        behavior_params=
            behavior_params,
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

    if mode == "behavior_raw":

        logits = (
            raw_behavior
            - raw_behavior.mean()
        )

    elif mode == "behavior_global":

        logits = (
            behavior_scaled
        )

    elif mode == "delay_only":

        logits = (
            delay_scores
        )

    elif mode == "old_alpha":

        if alpha is None:
            raise ValueError(
                "alpha is required."
            )

        logits = (
            float(alpha)
            * behavior_scaled
            +
            delay_scores
        )

    elif mode == "convex":

        if w is None:
            raise ValueError(
                "w is required."
            )

        if temperature is None:
            raise ValueError(
                "temperature is required."
            )

        if temperature <= 0:
            raise ValueError(
                "temperature must be positive."
            )

        mixed_score = (
            float(w)
            * behavior_scaled
            +
            (
                1.0
                - float(w)
            )
            * delay_scores
        )

        logits = (
            mixed_score
            / float(
                temperature
            )
        )

    else:

        raise ValueError(
            mode
        )

    posterior = (
        exp04a.stable_softmax(
            logits
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
    }


# ============================================================
# Evaluation
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
    exp04a,
    exp03y,
    mode,
    alpha=None,
    w=None,
    temperature=None,
):

    rows = []

    for idx in indices:

        if detected[
            idx
        ] != 1:
            continue

        tf = int(
            true_tf[
                idx
            ]
        )

        d = int(
            td[
                idx
            ]
        )

        pred = infer_one(
            x_user=x[
                idx
            ],
            observed_td=d,
            behavior_params=
                behavior_params,
            global_scale=
                global_scale,
            exp04a=exp04a,
            exp03y=exp03y,
            mode=mode,
            alpha=alpha,
            w=w,
            temperature=
                temperature,
        )

        rows.append(
            {
                "split":
                    split_name,

                "index":
                    int(
                        idx
                    ),

                "entity_id":
                    int(
                        entity_id[
                            idx
                        ]
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

                "mode":
                    mode,

                "alpha":
                    alpha,

                "w":
                    w,

                "temperature":
                    temperature,

                **pred,
            }
        )

    df = pd.DataFrame(
        rows
    )

    if len(
        df
    ) == 0:
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
            len(
                df
            ),

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
        "EXPERIMENT 04F: "
        "CONVEX FUSION STRUCTURAL AUDIT"
    )

    print("=" * 80)

    print()

    print(
        "Old fusion:"
    )

    print(
        "  alpha * Behavior + Delay"
    )

    print()

    print(
        "New fusion:"
    )

    print(
        "  [w * Behavior + (1-w) * Delay] / Temperature"
    )

    print()

    print(
        "w = Evidence Weight（证据权重）"
    )

    print(
        "Temperature = 温度参数，控制 Posterior 尖锐程度"
    )

    print(
        "MAE = Mean Absolute Error（平均绝对误差，越小越好）"
    )

    print(
        "Signed Error = predicted tf - true tf"
    )

    # ========================================================
    # Load modules and data
    # ========================================================

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

    print(
        "NOTE: current TEST is development diagnostic only."
    )

    # ========================================================
    # Train Behavior model
    # ========================================================

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
            behavior_params=
                behavior_params,
            exp04a=exp04a,
        )
    )

    print()

    print(
        "Global behavior scale from TRAIN =",
        f"{global_scale:.6f}",
    )

    # ========================================================
    # Validation: OLD alpha
    # ========================================================

    print()
    print("=" * 80)
    print(
        "VALIDATION: OLD ALPHA FUSION"
    )
    print("=" * 80)

    old_rows = []

    for alpha in OLD_ALPHA_VALUES:

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
            behavior_params=
                behavior_params,
            global_scale=
                global_scale,
            exp04a=exp04a,
            exp03y=exp03y,
            mode="old_alpha",
            alpha=alpha,
        )

        old_rows.append(
            {
                "alpha":
                    alpha,

                **summary,
            }
        )

        print(
            f"alpha={alpha:>4} | "
            f"MAE={summary['mae']:.6f} | "
            f"signed={summary['signed_error']:.6f}"
        )

    old_validation = pd.DataFrame(
        old_rows
    )

    old_validation[
        "abs_signed_error"
    ] = np.abs(
        old_validation[
            "signed_error"
        ]
    )

    old_selected = (
        old_validation
        .sort_values(
            [
                "mae",
                "abs_signed_error",
                "alpha",
            ]
        )
        .iloc[0]
    )

    selected_alpha = float(
        old_selected[
            "alpha"
        ]
    )

    # ========================================================
    # Validation: Convex fusion
    # ========================================================

    print()
    print("=" * 80)
    print(
        "VALIDATION: CONVEX FUSION GRID"
    )
    print("=" * 80)

    convex_rows = []

    for w in W_VALUES:

        for temperature in TEMPERATURE_VALUES:

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
                behavior_params=
                    behavior_params,
                global_scale=
                    global_scale,
                exp04a=exp04a,
                exp03y=exp03y,
                mode="convex",
                w=w,
                temperature=
                    temperature,
            )

            convex_rows.append(
                {
                    "w":
                        w,

                    "temperature":
                        temperature,

                    **summary,
                }
            )

    convex_validation = pd.DataFrame(
        convex_rows
    )

    convex_validation[
        "abs_signed_error"
    ] = np.abs(
        convex_validation[
            "signed_error"
        ]
    )

    convex_validation[
        "temperature_distance"
    ] = np.abs(
        np.log(
            convex_validation[
                "temperature"
            ]
        )
    )

    convex_validation[
        "weight_distance"
    ] = np.abs(
        convex_validation[
            "w"
        ]
        - 0.5
    )

    convex_selected = (
        convex_validation
        .sort_values(
            [
                "mae",
                "abs_signed_error",
                "temperature_distance",
                "weight_distance",
            ]
        )
        .iloc[0]
    )

    selected_w = float(
        convex_selected[
            "w"
        ]
    )

    selected_temperature = float(
        convex_selected[
            "temperature"
        ]
    )

    # ========================================================
    # IMPORTANT FIX:
    # sort first, select display columns afterwards
    # ========================================================

    print()

    print(
        "TOP 10 CONVEX VALIDATION CONFIGURATIONS"
    )

    top10 = (
        convex_validation
        .sort_values(
            [
                "mae",
                "abs_signed_error",
            ]
        )
        .head(10)
    )

    print(
        top10[
            [
                "w",
                "temperature",
                "mae",
                "signed_error",
                "within_1",
                "within_2",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "VALIDATION-SELECTED OLD ALPHA:"
    )

    print(
        f"  alpha={selected_alpha} | "
        f"MAE={float(old_selected['mae']):.6f} | "
        f"signed={float(old_selected['signed_error']):.6f}"
    )

    print()

    print(
        "VALIDATION-SELECTED CONVEX:"
    )

    print(
        f"  w={selected_w:.6f} | "
        f"T={selected_temperature:.6f} | "
        f"MAE={float(convex_selected['mae']):.6f} | "
        f"signed={float(convex_selected['signed_error']):.6f}"
    )

    # ========================================================
    # Structural equivalence
    # ========================================================

    equivalent_w = (
        selected_alpha
        / (
            selected_alpha
            + 1.0
        )
    )

    equivalent_temperature = (
        1.0
        / (
            selected_alpha
            + 1.0
        )
    )

    (
        _,
        old_equiv_df,
    ) = evaluate_config(
        split_name="validation",
        indices=val_idx,
        x=x,
        detected=detected,
        td=td,
        true_tf=true_tf,
        entity_id=entity_id,
        behavior_params=
            behavior_params,
        global_scale=
            global_scale,
        exp04a=exp04a,
        exp03y=exp03y,
        mode="old_alpha",
        alpha=
            selected_alpha,
    )

    (
        _,
        convex_equiv_df,
    ) = evaluate_config(
        split_name="validation",
        indices=val_idx,
        x=x,
        detected=detected,
        td=td,
        true_tf=true_tf,
        entity_id=entity_id,
        behavior_params=
            behavior_params,
        global_scale=
            global_scale,
        exp04a=exp04a,
        exp03y=exp03y,
        mode="convex",
        w=
            equivalent_w,
        temperature=
            equivalent_temperature,
    )

    prediction_mismatch = int(
        np.sum(
            old_equiv_df[
                "median_tf"
            ].to_numpy()
            !=
            convex_equiv_df[
                "median_tf"
            ].to_numpy()
        )
    )

    print()
    print("=" * 80)
    print(
        "STRUCTURAL EQUIVALENCE CHECK"
    )
    print("=" * 80)

    print(
        f"selected old alpha = "
        f"{selected_alpha}"
    )

    print(
        f"equivalent w = "
        f"{equivalent_w:.6f}"
    )

    print(
        f"equivalent T = "
        f"{equivalent_temperature:.6f}"
    )

    print(
        "median prediction mismatch count =",
        prediction_mismatch,
    )

    # ========================================================
    # Development test
    # ========================================================

    method_configs = [
        (
            "behavior_raw_04A_style",
            {
                "mode":
                    "behavior_raw",
            },
        ),

        (
            "behavior_global_scaled",
            {
                "mode":
                    "behavior_global",
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
                    selected_alpha,
            },
        ),

        (
            "fixed_convex_fusion",
            {
                "mode":
                    "convex",

                "w":
                    selected_w,

                "temperature":
                    selected_temperature,
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
            split_name=
                "development_test",
            indices=test_idx,
            x=x,
            detected=detected,
            td=td,
            true_tf=true_tf,
            entity_id=entity_id,
            behavior_params=
                behavior_params,
            global_scale=
                global_scale,
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
        "DEVELOPMENT TEST OVERALL"
    )
    print("=" * 80)

    print(
        overall_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # Delay buckets
    # ========================================================

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
                ]
                == bucket
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
                        len(
                            sub
                        ),

                    "mae":
                        float(
                            error.mean()
                        ),
                }
            )

    bucket_summary = pd.DataFrame(
        bucket_rows
    )

    print()
    print("=" * 80)
    print(
        "DEVELOPMENT TEST BY TRUE DELAY"
    )
    print("=" * 80)

    print(
        bucket_summary.to_string(
            index=False
        )
    )

    # ========================================================
    # Paired bootstrap
    # ========================================================

    convex_df = test_frames[
        "fixed_convex_fusion"
    ]

    convex_error = np.abs(
        convex_df[
            "median_tf"
        ].to_numpy()
        -
        convex_df[
            "true_tf"
        ].to_numpy()
    )

    paired_rows = []

    for baseline_method in [
        "delay_only",
        "behavior_raw_04A_style",
        "old_fixed_alpha",
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
                    convex_error,
                n_bootstrap=
                    BOOTSTRAP_SAMPLES,
                seed=
                    SEED,
            )
        )

        paired_rows.append(
            {
                "comparison":
                    f"convex_vs_{baseline_method}",

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

    # ========================================================
    # Save results
    # ========================================================

    em_history.to_csv(
        OUTPUT_DIR
        / "em_history.csv",
        index=False,
    )

    old_validation.to_csv(
        OUTPUT_DIR
        / "validation_old_alpha.csv",
        index=False,
    )

    convex_validation.to_csv(
        OUTPUT_DIR
        / "validation_convex_grid.csv",
        index=False,
    )

    overall_summary.to_csv(
        OUTPUT_DIR
        / "development_test_overall.csv",
        index=False,
    )

    bucket_summary.to_csv(
        OUTPUT_DIR
        / "development_test_delay_buckets.csv",
        index=False,
    )

    paired_summary.to_csv(
        OUTPUT_DIR
        / "development_test_pairwise.csv",
        index=False,
    )

    for method, df in test_frames.items():

        df.to_csv(
            OUTPUT_DIR
            / f"{method}_predictions.csv",
            index=False,
        )

    print()
    print("=" * 80)
    print(
        "04F INTERPRETATION GUIDE"
    )
    print("=" * 80)

    print()

    print(
        "If convex fusion improves over old alpha fusion,"
    )

    print(
        "separating evidence weight w from Temperature is useful."
    )

    print()

    print(
        "If selected w and T reproduce the old alpha,"
    )

    print(
        "the old formula was already sufficient "
        "at the fixed-weight level."
    )

    print()

    print(
        "The next experiment is user-specific w_i,"
    )

    print(
        "not further manual alpha tuning."
    )


if __name__ == "__main__":
    main()