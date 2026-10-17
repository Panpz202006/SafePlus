from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Experiment 04H-2
# Fresh Synthetic Generation Confirmation
#
# FINAL FUSION PARAMETERS ARE FROZEN:
#
#   w = 2/3
#   T = 1/3
#
# These are exactly equivalent to:
#
#   alpha = 2
#
# in the old fusion:
#
#   alpha * Behavior + Delay
#
# No parameter selection is allowed in this experiment.
#
# We generate completely NEW synthetic datasets by changing
# only the generator seed.
#
# x, fraud labels, tf, delays and td are all freshly generated.
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
    r"reports/experiment04/fresh_generation_confirmation"
)


# ============================================================
# Original synthetic generator parameters
# ============================================================

N = 2000
LENGTH = 24
FEATURES = 8

FRAUD_RATE = 0.35
DELAY_MEAN = 4.0

# Completely fresh generator seeds.
GENERATOR_SEEDS = list(
    range(
        101,
        111,
    )
)

# Same split mechanism for every independently generated dataset.
SPLIT_SEED = 0


# ============================================================
# Frozen final fusion
# ============================================================

FINAL_W = (
    2.0 / 3.0
)

FINAL_T = (
    1.0 / 3.0
)

EQUIVALENT_ALPHA = 2.0

BOOTSTRAP_SAMPLES = 20000


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
# Fresh generator
# ============================================================


def generate_same_delay_consistent(
    seed,
):
    """
    Freshly generate the SAME mechanism used by
    Experiment 03H.

    Important:

    We still consume the original random censoring draw
    whenever candidate td lies inside the window.

    However, we IGNORE that censoring coin when constructing
    the corrected labels.

    Therefore:

        candidate td < LENGTH
            -> detected = 1

        candidate td >= LENGTH
            -> detected = 0

    This allows seed=0 to reproduce the current
    synthetic_same_delay_consistent dataset exactly.
    """

    rng = np.random.default_rng(
        seed
    )

    # --------------------------------------------------------
    # Normal baseline behavior
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Fraud entity assignment
    # --------------------------------------------------------

    is_fraud = (
        rng.random(
            N
        )
        < FRAUD_RATE
    )

    t_onset = np.full(
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

    # --------------------------------------------------------
    # Shared fraud signal
    # --------------------------------------------------------

    signal = rng.normal(
        0.9,
        0.15,
        size=FEATURES,
    ).astype(
        np.float32
    )

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

    # --------------------------------------------------------
    # Fraud trajectories
    # --------------------------------------------------------

    for i in np.flatnonzero(
        is_fraud
    ):

        # Same tf generation rule as original generator.
        t_onset[i] = rng.integers(
            max(
                2,
                LENGTH // 6,
            ),
            max(
                3,
                2 * LENGTH // 3,
            ),
        )

        # Gradual behavior drift starts exactly at tf.
        ramp = np.linspace(
            0.25,
            1.0,
            LENGTH
            - t_onset[i],
            dtype=np.float32,
        )[:, None]

        x[
            i,
            t_onset[i]:,
        ] += (
            ramp
            * signal
        )

        # Same original detection-delay distribution.
        delay = max(
            1,
            int(
                rng.exponential(
                    DELAY_MEAN
                )
            ),
        )

        detection_delay[
            i
        ] = delay

        candidate = (
            int(
                t_onset[i]
            )
            + delay
        )

        candidate_t_detect[
            i
        ] = candidate

        if candidate < LENGTH:

            # ------------------------------------------------
            # VERY IMPORTANT:
            #
            # Original generator drew a random censoring coin
            # here.
            #
            # We still consume that RNG draw so seed=0 follows
            # exactly the same random-number stream.
            #
            # But the corrected dataset IGNORES the coin.
            # ------------------------------------------------

            _ = rng.random()

            detected[
                i
            ] = 1

            t_detect[
                i
            ] = candidate

        else:

            # Natural right censoring only.
            detected[
                i
            ] = 0

            t_detect[
                i
            ] = (
                LENGTH - 1
            )

    lengths = np.full(
        N,
        LENGTH,
        dtype=np.int64,
    )

    entity_id = np.arange(
        N,
        dtype=np.int64,
    )

    return {
        "x":
            x,

        "lengths":
            lengths,

        # IMPORTANT:
        # Explicit fraud label is now retained.
        "is_fraud":
            is_fraud.astype(
                np.int64
            ),

        "detected":
            detected,

        "t_detect":
            t_detect,

        "t_onset":
            t_onset,

        "entity_id":
            entity_id,

        "detection_delay":
            detection_delay,

        "candidate_t_detect":
            candidate_t_detect,
    }


# ============================================================
# Seed-0 reproduction test
# ============================================================


def seed_zero_reproduction_check(
    exp04a,
):

    current = np.load(
        exp04a.DATA_PATH,
        allow_pickle=True,
    )

    replay = (
        generate_same_delay_consistent(
            seed=0
        )
    )

    checks = {
        "x":
            np.array_equal(
                replay[
                    "x"
                ],
                current[
                    "x"
                ],
            ),

        "lengths":
            np.array_equal(
                replay[
                    "lengths"
                ],
                current[
                    "lengths"
                ],
            ),

        "detected":
            np.array_equal(
                replay[
                    "detected"
                ],
                current[
                    "detected"
                ],
            ),

        "t_detect":
            np.array_equal(
                replay[
                    "t_detect"
                ],
                current[
                    "t_detect"
                ],
            ),

        "t_onset":
            np.array_equal(
                replay[
                    "t_onset"
                ],
                current[
                    "t_onset"
                ],
            ),

        "entity_id":
            np.array_equal(
                replay[
                    "entity_id"
                ],
                current[
                    "entity_id"
                ],
            ),
    }

    print()

    print(
        "=" * 80
    )

    print(
        "SEED-0 GENERATOR REPRODUCTION CHECK"
    )

    print(
        "=" * 80
    )

    for name, passed in checks.items():

        print(
            f"{name:>10}: {passed}"
        )

    if not all(
        checks.values()
    ):

        raise RuntimeError(
            "Fresh generator does not reproduce "
            "the current seed-0 benchmark exactly."
        )

    print()

    print(
        "PASS: generator exactly reproduces "
        "the current seed-0 benchmark."
    )


# ============================================================
# Generation-level Bootstrap CI
# ============================================================


def bootstrap_seed_gain_ci(
    baseline_mae,
    final_mae,
    n_bootstrap=
        BOOTSTRAP_SAMPLES,
    seed=
        20261017,
):

    baseline_mae = np.asarray(
        baseline_mae,
        dtype=np.float64,
    )

    final_mae = np.asarray(
        final_mae,
        dtype=np.float64,
    )

    gain = (
        baseline_mae
        - final_mae
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

        "wins":
            int(
                (
                    gain > 1e-12
                ).sum()
            ),

        "ties":
            int(
                (
                    np.abs(
                        gain
                    )
                    <= 1e-12
                ).sum()
            ),

        "losses":
            int(
                (
                    gain < -1e-12
                ).sum()
            ),
    }


# ============================================================
# Run one independent generated dataset
# ============================================================


def run_one_generation(
    generation_seed,
    exp04a,
    exp03y,
    exp04f,
):

    print()

    print(
        "=" * 80
    )

    print(
        f"FRESH GENERATION SEED "
        f"{generation_seed}"
    )

    print(
        "=" * 80
    )

    arrays = (
        generate_same_delay_consistent(
            seed=generation_seed
        )
    )

    x = arrays[
        "x"
    ].astype(
        np.float64
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We now use an explicit binary fraud label.
    # We no longer recover fraud labels from numeric tf.
    # --------------------------------------------------------

    fraud_label = arrays[
        "is_fraud"
    ].astype(
        np.int64
    )

    detected = arrays[
        "detected"
    ].astype(
        np.int64
    )

    td = arrays[
        "t_detect"
    ].astype(
        np.int64
    )

    true_tf = arrays[
        "t_onset"
    ].astype(
        np.int64
    )

    entity_id = arrays[
        "entity_id"
    ].astype(
        np.int64
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
            SPLIT_SEED,
    )

    print(
        f"train={len(train_idx)} "
        f"val={len(val_idx)} "
        f"test={len(test_idx)}"
    )

    print(
        "fraud entities =",
        int(
            fraud_label.sum()
        ),
    )

    print(
        "detected fraud =",
        int(
            (
                fraud_label
                * detected
            ).sum()
        ),
    )

    # --------------------------------------------------------
    # Retrain Behavior Evidence model from scratch.
    #
    # Numeric tf is NOT used for training.
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

    # --------------------------------------------------------
    # Frozen comparisons
    # --------------------------------------------------------

    configs = [
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
            "old_alpha_equivalent",
            {
                "mode":
                    "old_alpha",

                "alpha":
                    EQUIVALENT_ALPHA,
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

    for method, kwargs in configs:

        (
            summary,
            df,
        ) = exp04f.evaluate_config(
            split_name=
                f"fresh_seed_{generation_seed}",
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
                "generation_seed":
                    generation_seed,

                "method":
                    method,

                **summary,
            }
        )

        df[
            "generation_seed"
        ] = generation_seed

        df[
            "method_name"
        ] = method

        prediction_frames[
            method
        ] = df

        print(
            f"{method:>22} | "
            f"MAE={summary['mae']:.6f} | "
            f"signed={summary['signed_error']:.6f} | "
            f"within1={summary['within_1']:.6f} | "
            f"within2={summary['within_2']:.6f}"
        )

    # --------------------------------------------------------
    # Exact equivalence sanity check
    # --------------------------------------------------------

    old_df = prediction_frames[
        "old_alpha_equivalent"
    ]

    convex_df = prediction_frames[
        "final_convex"
    ]

    mismatch = int(
        np.sum(
            old_df[
                "median_tf"
            ].to_numpy()
            !=
            convex_df[
                "median_tf"
            ].to_numpy()
        )
    )

    print(
        "equivalence mismatch =",
        mismatch,
    )

    if mismatch != 0:

        raise RuntimeError(
            "alpha=2 and convex w=2/3,T=1/3 "
            "are not equivalent."
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
        "EXPERIMENT 04H-2: "
        "FRESH SYNTHETIC GENERATION CONFIRMATION"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "FINAL MODEL PARAMETERS ARE FROZEN:"
    )

    print(
        f"  w = {FINAL_W}"
    )

    print(
        f"  T = {FINAL_T}"
    )

    print(
        f"  equivalent alpha = "
        f"{EQUIVALENT_ALPHA}"
    )

    print()

    print(
        "Generation seeds:",
        GENERATOR_SEEDS,
    )

    print(
        "Split seed:",
        SPLIT_SEED,
    )

    print()

    print(
        "NO validation selection is performed."
    )

    print(
        "NO fusion parameter is changed."
    )

    # --------------------------------------------------------
    # Modules
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
    # Critical reproduction check
    # --------------------------------------------------------

    seed_zero_reproduction_check(
        exp04a=
            exp04a,
    )

    # --------------------------------------------------------
    # Fresh generations
    # --------------------------------------------------------

    all_summaries = []

    all_predictions = []

    scale_rows = []

    for generation_seed in GENERATOR_SEEDS:

        (
            summaries,
            prediction_frames,
            em_history,
            global_scale,
        ) = run_one_generation(
            generation_seed=
                generation_seed,
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
                "generation_seed":
                    generation_seed,

                "global_behavior_scale":
                    global_scale,
            }
        )

        em_history.to_csv(
            OUTPUT_DIR
            / (
                f"em_history_generation_"
                f"{generation_seed}.csv"
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
    # Aggregate across independent generations
    # ========================================================

    methods = [
        "behavior_raw",
        "delay_only",
        "old_alpha_equivalent",
        "final_convex",
    ]

    aggregate_rows = []

    for method in methods:

        sub = summary_df[
            summary_df[
                "method"
            ]
            == method
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
        "FRESH-GENERATION AGGREGATE"
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
    # Per-generation table
    # ========================================================

    pivot = (
        summary_df
        .pivot(
            index=
                "generation_seed",
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

    print()

    print(
        "=" * 80
    )

    print(
        "PER-GENERATION MAE"
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
        "Final Convex beats Behavior:"
    )

    print(
        f"  "
        f"{int(pivot['final_beats_behavior'].sum())}"
        f" / {len(pivot)} generations"
    )

    print(
        "Final Convex beats Delay:"
    )

    print(
        f"  "
        f"{int(pivot['final_beats_delay'].sum())}"
        f" / {len(pivot)} generations"
    )

    # ========================================================
    # Generation-level Bootstrap CI
    # ========================================================

    final_seed_mae = pivot[
        "final_convex"
    ].to_numpy(
        dtype=np.float64
    )

    generation_pair_rows = []

    for baseline in [
        "behavior_raw",
        "delay_only",
    ]:

        metrics = bootstrap_seed_gain_ci(
            baseline_mae=
                pivot[
                    baseline
                ].to_numpy(
                    dtype=np.float64
                ),

            final_mae=
                final_seed_mae,

            n_bootstrap=
                BOOTSTRAP_SAMPLES,
        )

        generation_pair_rows.append(
            {
                "comparison":
                    (
                        "final_convex_vs_"
                        + baseline
                    ),

                **metrics,
            }
        )

    generation_pair_df = pd.DataFrame(
        generation_pair_rows
    )

    print()

    print(
        "=" * 80
    )

    print(
        "GENERATION-LEVEL PAIRED BOOTSTRAP"
    )

    print(
        "=" * 80
    )

    print(
        generation_pair_df.to_string(
            index=False
        )
    )

    # ========================================================
    # Pooled independent user-level Bootstrap CI
    # ========================================================

    final_df = predictions_df[
        predictions_df[
            "method_name"
        ]
        == "final_convex"
    ].copy()

    final_df = final_df.sort_values(
        [
            "generation_seed",
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

    pooled_rows = []

    for baseline in [
        "behavior_raw",
        "delay_only",
    ]:

        baseline_df = predictions_df[
            predictions_df[
                "method_name"
            ]
            == baseline
        ].copy()

        baseline_df = baseline_df.sort_values(
            [
                "generation_seed",
                "index",
            ]
        )

        if not np.array_equal(
            final_df[
                [
                    "generation_seed",
                    "index",
                ]
            ].to_numpy(),
            baseline_df[
                [
                    "generation_seed",
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

        metrics = (
            exp04f.bootstrap_improvement_ci(
                baseline_error=
                    baseline_error,

                new_error=
                    final_error,

                n_bootstrap=
                    BOOTSTRAP_SAMPLES,

                seed=
                    20261017,
            )
        )

        pooled_rows.append(
            {
                "comparison":
                    (
                        "final_convex_vs_"
                        + baseline
                    ),

                **metrics,
            }
        )

    pooled_df = pd.DataFrame(
        pooled_rows
    )

    print()

    print(
        "=" * 80
    )

    print(
        "POOLED USER-LEVEL PAIRED BOOTSTRAP"
    )

    print(
        "=" * 80
    )

    print(
        pooled_df.to_string(
            index=False
        )
    )

    # ========================================================
    # Delay buckets
    # ========================================================

    bucket_rows = []

    for generation_seed in GENERATOR_SEEDS:

        seed_df = predictions_df[
            predictions_df[
                "generation_seed"
            ]
            == generation_seed
        ]

        for method in [
            "behavior_raw",
            "delay_only",
            "final_convex",
        ]:

            method_df = seed_df[
                seed_df[
                    "method_name"
                ]
                == method
            ]

            for bucket in [
                "delay_1_2",
                "delay_3_5",
                "delay_6_plus",
            ]:

                sub = method_df[
                    method_df[
                        "delay_bucket"
                    ]
                    == bucket
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
                        "generation_seed":
                            generation_seed,

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
        "FRESH-GENERATION DELAY-BUCKET AGGREGATE"
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
        / "per_generation_summary.csv",
        index=False,
    )

    aggregate_df.to_csv(
        OUTPUT_DIR
        / "aggregate_summary.csv",
        index=False,
    )

    pivot.to_csv(
        OUTPUT_DIR
        / "per_generation_mae.csv",
        index=False,
    )

    generation_pair_df.to_csv(
        OUTPUT_DIR
        / "generation_level_bootstrap.csv",
        index=False,
    )

    pooled_df.to_csv(
        OUTPUT_DIR
        / "pooled_user_bootstrap.csv",
        index=False,
    )

    bucket_df.to_csv(
        OUTPUT_DIR
        / "per_generation_delay_buckets.csv",
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

    # ========================================================
    # Automatic decision
    # ========================================================

    final_row = aggregate_df[
        aggregate_df[
            "method"
        ]
        == "final_convex"
    ].iloc[
        0
    ]

    behavior_row = aggregate_df[
        aggregate_df[
            "method"
        ]
        == "behavior_raw"
    ].iloc[
        0
    ]

    delay_row = aggregate_df[
        aggregate_df[
            "method"
        ]
        == "delay_only"
    ].iloc[
        0
    ]

    beats_behavior = (
        float(
            final_row[
                "mae_mean"
            ]
        )
        <
        float(
            behavior_row[
                "mae_mean"
            ]
        )
    )

    beats_delay = (
        float(
            final_row[
                "mae_mean"
            ]
        )
        <
        float(
            delay_row[
                "mae_mean"
            ]
        )
    )

    print()

    print(
        "=" * 80
    )

    print(
        "04H-2 DECISION RULE"
    )

    print(
        "=" * 80
    )

    print(
        "Final Convex mean MAE < "
        "Behavior mean MAE:",
        beats_behavior,
    )

    print(
        "Final Convex mean MAE < "
        "Delay mean MAE:",
        beats_delay,
    )

    if (
        beats_behavior
        and beats_delay
    ):

        print()

        print(
            "PASS:"
        )

        print(
            "Frozen fusion is robust across "
            "fresh synthetic generations."
        )

    else:

        print()

        print(
            "NOT PASSED:"
        )

        print(
            "Do not claim final synthetic robustness."
        )


if __name__ == "__main__":
    main()