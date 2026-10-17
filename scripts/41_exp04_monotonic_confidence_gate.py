from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Experiment 04G
# Monotonic Confidence-aware Gate
#
# Fixed Convex Fusion from 04F:
#
#   score = [w * Behavior + (1-w) * Delay] / T
#
# 04G changes ONLY:
#
#   fixed w
#
# into:
#
#   w_i = sigmoid(a * z_i + b)
#
# where z_i is TRAIN-robust-scaled Behavior Score Spread.
#
# Temperature stays fixed at 1/3,
# as selected by Experiment 04F.
#
# This makes the ablation clean:
#
#   fixed evidence weight
#          vs
#   user-specific evidence weight
#
# IMPORTANT:
# Current test split has already been inspected many times.
# Therefore it is DEVELOPMENT TEST only.
# Final claims require fresh untouched evaluation later.
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
    r"reports/experiment04/monotonic_confidence_gate"
)

SEED = 0

# 04F selected this value.
FIXED_TEMPERATURE = 1.0 / 3.0

# 04F selected fixed w = 0.7.
FIXED_W = 0.7


# a >= 0 enforces monotonicity:
#
# larger Score Spread
# -> larger Behavior weight
#
A_VALUES = [
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
]

# sigmoid(0.85) is approximately 0.70,
# so b=0.85 includes the 04F fixed solution neighborhood.
B_VALUES = [
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    0.85,
    1.0,
    1.5,
    2.0,
]

BOOTSTRAP_SAMPLES = 5000


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
# Gate helpers
# ============================================================


def sigmoid(x):

    x = np.clip(
        x,
        -50.0,
        50.0,
    )

    return float(
        1.0
        / (
            1.0
            + np.exp(-x)
        )
    )


def robust_z(
    value,
    median,
    iqr,
):
    """
    Robust Scaling（鲁棒标准化）

    z = (value - TRAIN median) / TRAIN IQR

    IQR = Interquartile Range（四分位距）

    Compared with mean/std,
    median/IQR is less sensitive to extreme users.
    """

    return float(
        (
            value
            - median
        )
        / max(
            iqr,
            1e-8,
        )
    )


# ============================================================
# TRAIN reference for Score Spread
# ============================================================


def build_train_spread_reference(
    x,
    fraud_label,
    detected,
    td,
    train_idx,
    behavior_params,
    global_scale,
    exp04a,
    exp04f,
):

    spreads = []

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
            raw_behavior,
        ) = exp04f.behavior_candidate_scores(
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

        behavior_scaled = (
            raw_behavior
            - raw_behavior.mean()
        ) / global_scale

        spread = float(
            np.max(
                behavior_scaled
            )
            - np.min(
                behavior_scaled
            )
        )

        spreads.append(
            spread
        )

    if not spreads:
        raise RuntimeError(
            "No TRAIN Score Spread values found."
        )

    spreads = np.asarray(
        spreads,
        dtype=np.float64,
    )

    median = float(
        np.median(
            spreads
        )
    )

    q25 = float(
        np.quantile(
            spreads,
            0.25,
        )
    )

    q75 = float(
        np.quantile(
            spreads,
            0.75,
        )
    )

    iqr = (
        q75
        - q25
    )

    return (
        spreads,
        median,
        iqr,
    )


# ============================================================
# One-user inference
# ============================================================


def infer_one(
    x_user,
    observed_td,
    behavior_params,
    global_scale,
    spread_median,
    spread_iqr,
    exp04a,
    exp03y,
    exp04f,
    mode,
    a=None,
    b=None,
):

    (
        candidates,
        raw_behavior,
    ) = exp04f.behavior_candidate_scores(
        x_user=x_user,
        observed_td=observed_td,
        behavior_params=
            behavior_params,
        exp04a=exp04a,
    )

    delay_scores = (
        exp04f.delay_candidate_scores(
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

    spread_z = robust_z(
        value=
            score_spread,
        median=
            spread_median,
        iqr=
            spread_iqr,
    )

    if mode == "behavior_raw":

        logits = (
            raw_behavior
            - raw_behavior.mean()
        )

        w_i = np.nan

    elif mode == "delay_only":

        logits = (
            delay_scores
        )

        w_i = 0.0

    elif mode == "fixed_gate":

        w_i = float(
            FIXED_W
        )

        mixed_score = (
            w_i
            * behavior_scaled
            +
            (
                1.0
                - w_i
            )
            * delay_scores
        )

        logits = (
            mixed_score
            / FIXED_TEMPERATURE
        )

    elif mode == "monotonic_gate":

        if a is None:
            raise ValueError(
                "a is required."
            )

        if b is None:
            raise ValueError(
                "b is required."
            )

        w_i = sigmoid(
            float(a)
            * spread_z
            +
            float(b)
        )

        mixed_score = (
            w_i
            * behavior_scaled
            +
            (
                1.0
                - w_i
            )
            * delay_scores
        )

        logits = (
            mixed_score
            / FIXED_TEMPERATURE
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
        exp04f.posterior_median(
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

        "spread_z":
            spread_z,

        "w_i":
            w_i,
    }


# ============================================================
# Evaluation
# ============================================================


def evaluate(
    split_name,
    indices,
    x,
    detected,
    td,
    true_tf,
    entity_id,
    behavior_params,
    global_scale,
    spread_median,
    spread_iqr,
    exp04a,
    exp03y,
    exp04f,
    mode,
    a=None,
    b=None,
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
            spread_median=
                spread_median,
            spread_iqr=
                spread_iqr,
            exp04a=exp04a,
            exp03y=exp03y,
            exp04f=exp04f,
            mode=mode,
            a=a,
            b=b,
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
                    exp04f.delay_bucket(
                        d - tf
                    ),

                "mode":
                    mode,

                "a":
                    a,

                "b":
                    b,

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

        "mean_w":
            (
                float(
                    np.nanmean(
                        df[
                            "w_i"
                        ].to_numpy(
                            dtype=np.float64
                        )
                    )
                )
                if not np.all(
                    pd.isna(
                        df[
                            "w_i"
                        ]
                    )
                )
                else np.nan
            ),

        "min_w":
            (
                float(
                    np.nanmin(
                        df[
                            "w_i"
                        ].to_numpy(
                            dtype=np.float64
                        )
                    )
                )
                if not np.all(
                    pd.isna(
                        df[
                            "w_i"
                        ]
                    )
                )
                else np.nan
            ),

        "max_w":
            (
                float(
                    np.nanmax(
                        df[
                            "w_i"
                        ].to_numpy(
                            dtype=np.float64
                        )
                    )
                )
                if not np.all(
                    pd.isna(
                        df[
                            "w_i"
                        ]
                    )
                )
                else np.nan
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
        "EXPERIMENT 04G: "
        "MONOTONIC CONFIDENCE-AWARE GATE"
    )

    print("=" * 80)

    print()

    print(
        "Gate:"
    )

    print(
        "  w_i = sigmoid(a * z_i + b)"
    )

    print()

    print(
        "z_i = TRAIN-robust-scaled Score Spread"
    )

    print(
        "a >= 0 guarantees monotonicity"
    )

    print(
        "Temperature is fixed at 1/3 from Experiment 04F"
    )

    print()

    print(
        "MAE = Mean Absolute Error（平均绝对误差，越小越好）"
    )

    print(
        "Signed Error = predicted tf - true tf"
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
    # Load data
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

    # --------------------------------------------------------
    # Train Behavior model on TRAIN
    # --------------------------------------------------------

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
        exp04f.estimate_global_behavior_scale(
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

    (
        train_spreads,
        spread_median,
        spread_iqr,
    ) = build_train_spread_reference(
        x=x,
        fraud_label=fraud_label,
        detected=detected,
        td=td,
        train_idx=train_idx,
        behavior_params=
            behavior_params,
        global_scale=
            global_scale,
        exp04a=exp04a,
        exp04f=exp04f,
    )

    print()

    print(
        "Global behavior scale from TRAIN =",
        f"{global_scale:.6f}",
    )

    print()

    print(
        "TRAIN Score Spread reference:"
    )

    print(
        "  n =",
        len(
            train_spreads
        ),
    )

    print(
        "  median =",
        f"{spread_median:.6f}",
    )

    print(
        "  IQR =",
        f"{spread_iqr:.6f}",
    )

    # ========================================================
    # Fixed gate validation baseline
    # ========================================================

    (
        fixed_val_summary,
        _,
    ) = evaluate(
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
        spread_median=
            spread_median,
        spread_iqr=
            spread_iqr,
        exp04a=exp04a,
        exp03y=exp03y,
        exp04f=exp04f,
        mode="fixed_gate",
    )

    print()

    print(
        "FIXED 04F VALIDATION BASELINE:"
    )

    print(
        f"  w={FIXED_W:.3f} | "
        f"T={FIXED_TEMPERATURE:.6f} | "
        f"MAE={fixed_val_summary['mae']:.6f} | "
        f"signed={fixed_val_summary['signed_error']:.6f}"
    )

    # ========================================================
    # Validation Gate grid
    # ========================================================

    gate_rows = []

    print()

    print("=" * 80)

    print(
        "VALIDATION MONOTONIC GATE GRID"
    )

    print("=" * 80)

    for a in A_VALUES:

        for b in B_VALUES:

            (
                summary,
                _,
            ) = evaluate(
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
                spread_median=
                    spread_median,
                spread_iqr=
                    spread_iqr,
                exp04a=exp04a,
                exp03y=exp03y,
                exp04f=exp04f,
                mode=
                    "monotonic_gate",
                a=a,
                b=b,
            )

            gate_rows.append(
                {
                    "a":
                        a,

                    "b":
                        b,

                    **summary,
                }
            )

    gate_validation = pd.DataFrame(
        gate_rows
    )

    gate_validation[
        "abs_signed_error"
    ] = np.abs(
        gate_validation[
            "signed_error"
        ]
    )

    # Prefer lower MAE first.
    # If tied:
    #   lower bias
    #   simpler/smoother gate (smaller a)
    #   smaller |b|
    selected = (
        gate_validation
        .sort_values(
            [
                "mae",
                "abs_signed_error",
                "a",
                "b",
            ]
        )
        .iloc[0]
    )

    selected_a = float(
        selected[
            "a"
        ]
    )

    selected_b = float(
        selected[
            "b"
        ]
    )

    print()

    print(
        "TOP 10 MONOTONIC GATE CONFIGURATIONS"
    )

    top10 = (
        gate_validation
        .sort_values(
            [
                "mae",
                "abs_signed_error",
                "a",
            ]
        )
        .head(10)
    )

    print(
        top10[
            [
                "a",
                "b",
                "mae",
                "signed_error",
                "within_1",
                "within_2",
                "mean_w",
                "min_w",
                "max_w",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "VALIDATION-SELECTED MONOTONIC GATE:"
    )

    print(
        f"  a={selected_a:.6f} | "
        f"b={selected_b:.6f} | "
        f"MAE={float(selected['mae']):.6f} | "
        f"signed={float(selected['signed_error']):.6f}"
    )

    print(
        f"  mean_w={float(selected['mean_w']):.6f} | "
        f"min_w={float(selected['min_w']):.6f} | "
        f"max_w={float(selected['max_w']):.6f}"
    )

    if selected_a == 0.0:

        print()

        print(
            "WARNING:"
        )

        print(
            "Validation selected a=0."
        )

        print(
            "This means user-specific Score Spread "
            "did NOT improve over a fixed gate."
        )

    else:

        print()

        print(
            "Validation selected a>0."
        )

        print(
            "This supports user-specific "
            "confidence-aware weighting."
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
            "delay_only",
            {
                "mode":
                    "delay_only",
            },
        ),

        (
            "fixed_convex_gate",
            {
                "mode":
                    "fixed_gate",
            },
        ),

        (
            "monotonic_confidence_gate",
            {
                "mode":
                    "monotonic_gate",

                "a":
                    selected_a,

                "b":
                    selected_b,
            },
        ),
    ]

    test_results = {}
    test_frames = {}

    for method, kwargs in method_configs:

        (
            summary,
            df,
        ) = evaluate(
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
            spread_median=
                spread_median,
            spread_iqr=
                spread_iqr,
            exp04a=exp04a,
            exp03y=exp03y,
            exp04f=exp04f,
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
    # Delay buckets + gate behavior
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

            mean_w = (
                float(
                    np.nanmean(
                        sub[
                            "w_i"
                        ].to_numpy(
                            dtype=np.float64
                        )
                    )
                )
                if not np.all(
                    pd.isna(
                        sub[
                            "w_i"
                        ]
                    )
                )
                else np.nan
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

                    "mean_w":
                        mean_w,
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
    # Paired Bootstrap CI
    # ========================================================

    gate_df = test_frames[
        "monotonic_confidence_gate"
    ]

    gate_error = np.abs(
        gate_df[
            "median_tf"
        ].to_numpy()
        -
        gate_df[
            "true_tf"
        ].to_numpy()
    )

    paired_rows = []

    for baseline_method in [
        "delay_only",
        "behavior_raw_04A_style",
        "fixed_convex_gate",
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
            exp04f.bootstrap_improvement_ci(
                baseline_error=
                    baseline_error,
                new_error=
                    gate_error,
                n_bootstrap=
                    BOOTSTRAP_SAMPLES,
                seed=
                    SEED,
            )
        )

        paired_rows.append(
            {
                "comparison":
                    f"gate_vs_{baseline_method}",

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
    # Save
    # ========================================================

    em_history.to_csv(
        OUTPUT_DIR
        / "em_history.csv",
        index=False,
    )

    gate_validation.to_csv(
        OUTPUT_DIR
        / "validation_gate_grid.csv",
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
        "04G INTERPRETATION GUIDE"
    )

    print("=" * 80)

    print()

    print(
        "Key question 1:"
    )

    print(
        "  Is selected a > 0?"
    )

    print(
        "  If yes, validation prefers user-specific weighting."
    )

    print()

    print(
        "Key question 2:"
    )

    print(
        "  Does Gate beat Fixed Convex Fusion?"
    )

    print()

    print(
        "Key question 3:"
    )

    print(
        "  Do long-delay users receive larger w_i?"
    )

    print(
        "  If yes, the Gate is learning to suppress unreliable Delay evidence."
    )

    print()

    print(
        "Key question 4:"
    )

    print(
        "  Is Bootstrap CI vs fixed gate above zero?"
    )

    print(
        "  If not, improvement is not yet statistically stable."
    )


if __name__ == "__main__":
    main()