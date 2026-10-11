from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Experiment 03I
# Paired censoring effect on latent t_f recovery
#
# Compare the SAME entities under:
#
# OLD:
#   candidate t_d existed inside the observation horizon
#   but was randomly hidden.
#
# NEW:
#   the same candidate t_d is revealed.
#
# For every paired entity:
#
#   x              same
#   true t_f       same
#   delay          same
#   candidate t_d  same
#   split          same
#
# We compare latent-t_f posterior recovery before/after
# correcting censoring.
#
# IMPORTANT:
# Models were retrained on the corresponding old/new datasets.
# Therefore this is a paired dataset-level intervention, not
# inference with identical frozen model weights.
# ============================================================


AUDIT_PATH = Path(
    r"reports/experiment03/"
    r"same_delay_censor_control/"
    r"paired_audit.npz"
)

OLD_ROOT = Path(
    r"reports/experiment03/"
    r"latent_decoupled_joint"
)

NEW_ROOT = Path(
    r"reports/experiment03/"
    r"latent_decoupled_same_delay_consistent"
)

OUTPUT_DIR = Path(
    r"reports/experiment03/"
    r"paired_censoring_effect"
)

INFERENCE_VARIANTS = (
    "exact",
    "stochastic_em",
)


def safe_mean(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return float("nan")

    return float(
        values.mean()
    )


def safe_median(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return float("nan")

    return float(
        np.median(values)
    )


def bootstrap_mean_ci(
    values,
    samples=10000,
    seed=20261009,
):
    """
    Bootstrap 95% confidence interval for the mean paired improvement.

    Positive improvement means:
        old error - new error > 0
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return (
            float("nan"),
            float("nan"),
        )

    rng = np.random.default_rng(
        seed
    )

    boot_means = np.empty(
        samples,
        dtype=np.float64,
    )

    for i in range(samples):

        draw = rng.choice(
            values,
            size=len(values),
            replace=True,
        )

        boot_means[i] = (
            draw.mean()
        )

    low = float(
        np.percentile(
            boot_means,
            2.5,
        )
    )

    high = float(
        np.percentile(
            boot_means,
            97.5,
        )
    )

    return (
        low,
        high,
    )


def load_paired_audit():

    if not AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing paired audit: {AUDIT_PATH}"
        )

    with np.load(
        AUDIT_PATH,
        allow_pickle=False,
    ) as data:

        audit = {
            key: np.asarray(
                data[key]
            )
            for key in data.files
        }

    required = {
        "entity_id",
        "t_onset",
        "detection_delay",
        "candidate_t_detect",
        "old_detected",
        "new_detected",
        "paired_random_censored_revealed",
    }

    missing = (
        required
        - set(
            audit.keys()
        )
    )

    if missing:
        raise KeyError(
            f"Audit is missing fields: {sorted(missing)}"
        )

    paired_mask = audit[
        "paired_random_censored_revealed"
    ].astype(bool)

    paired = pd.DataFrame(
        {
            "entity_id":
                audit[
                    "entity_id"
                ][
                    paired_mask
                ],

            "true_tf_audit":
                audit[
                    "t_onset"
                ][
                    paired_mask
                ],

            "detection_delay":
                audit[
                    "detection_delay"
                ][
                    paired_mask
                ],

            "candidate_td":
                audit[
                    "candidate_t_detect"
                ][
                    paired_mask
                ],

            "old_detected_audit":
                audit[
                    "old_detected"
                ][
                    paired_mask
                ],

            "new_detected_audit":
                audit[
                    "new_detected"
                ][
                    paired_mask
                ],
        }
    )

    return paired


def load_entity_results(
    inference,
):

    old_path = (
        OLD_ROOT
        / inference
        / "entity_diagnostics.csv"
    )

    new_path = (
        NEW_ROOT
        / inference
        / "entity_diagnostics.csv"
    )

    if not old_path.exists():
        raise FileNotFoundError(
            f"Missing old results: {old_path}"
        )

    if not new_path.exists():
        raise FileNotFoundError(
            f"Missing new results: {new_path}"
        )

    old = pd.read_csv(
        old_path
    )

    new = pd.read_csv(
        new_path
    )

    return (
        old,
        new,
    )


def prepare_variant(
    inference,
    paired_audit,
):

    old, new = (
        load_entity_results(
            inference
        )
    )

    # ========================================================
    # Select only the 36 paired entities
    # ========================================================

    old_pair = old.merge(
        paired_audit,
        on="entity_id",
        how="inner",
    )

    new_pair = new.merge(
        paired_audit,
        on="entity_id",
        how="inner",
    )

    if len(old_pair) != len(
        paired_audit
    ):
        raise RuntimeError(
            f"{inference}: old results contain "
            f"{len(old_pair)} paired entities, expected "
            f"{len(paired_audit)}."
        )

    if len(new_pair) != len(
        paired_audit
    ):
        raise RuntimeError(
            f"{inference}: new results contain "
            f"{len(new_pair)} paired entities, expected "
            f"{len(paired_audit)}."
        )

    # ========================================================
    # Align by entity ID
    # ========================================================

    old_pair = old_pair.sort_values(
        "entity_id"
    ).reset_index(
        drop=True
    )

    new_pair = new_pair.sort_values(
        "entity_id"
    ).reset_index(
        drop=True
    )

    if not np.array_equal(
        old_pair[
            "entity_id"
        ].to_numpy(),
        new_pair[
            "entity_id"
        ].to_numpy(),
    ):
        raise RuntimeError(
            "Old/new entity alignment failed."
        )

    # ========================================================
    # Sanity checks
    # ========================================================

    if not np.array_equal(
        old_pair[
            "true_tf"
        ].to_numpy(),
        new_pair[
            "true_tf"
        ].to_numpy(),
    ):
        raise RuntimeError(
            "true_tf changed between paired datasets."
        )

    if not np.array_equal(
        old_pair[
            "true_tf"
        ].to_numpy(),
        old_pair[
            "true_tf_audit"
        ].to_numpy(),
    ):
        raise RuntimeError(
            "Audit true_tf does not match experiment results."
        )

    if not np.all(
        old_pair[
            "detected"
        ].to_numpy()
        == 0
    ):
        raise RuntimeError(
            "Some old paired entities were unexpectedly detected."
        )

    if not np.all(
        new_pair[
            "detected"
        ].to_numpy()
        == 1
    ):
        raise RuntimeError(
            "Some corrected paired entities were not revealed."
        )

    # ========================================================
    # Build paired table
    # ========================================================

    result = pd.DataFrame(
        {
            "entity_id":
                old_pair[
                    "entity_id"
                ],

            "true_tf":
                old_pair[
                    "true_tf"
                ],

            "detection_delay":
                old_pair[
                    "detection_delay"
                ],

            "candidate_td":
                old_pair[
                    "candidate_td"
                ],

            # ------------------------------------------------
            # Old observation
            # ------------------------------------------------

            "old_observed_td":
                old_pair[
                    "observed_td"
                ],

            "old_no_commission":
                old_pair[
                    "no_commission_posterior"
                ],

            "old_mode_tf":
                old_pair[
                    "posterior_mode_tf"
                ],

            "old_mean_tf":
                old_pair[
                    "posterior_mean_tf"
                ],

            "old_true_tf_prob_cond":
                old_pair[
                    "true_tf_probability_given_commission"
                ],

            "old_expected_abs_error":
                old_pair[
                    "expected_abs_error"
                ],

            "old_sampled_tf_mae":
                old_pair[
                    "sampled_tf_mae"
                ],

            # ------------------------------------------------
            # Corrected observation
            # ------------------------------------------------

            "new_observed_td":
                new_pair[
                    "observed_td"
                ],

            "new_no_commission":
                new_pair[
                    "no_commission_posterior"
                ],

            "new_mode_tf":
                new_pair[
                    "posterior_mode_tf"
                ],

            "new_mean_tf":
                new_pair[
                    "posterior_mean_tf"
                ],

            "new_true_tf_prob_cond":
                new_pair[
                    "true_tf_probability_given_commission"
                ],

            "new_expected_abs_error":
                new_pair[
                    "expected_abs_error"
                ],

            "new_sampled_tf_mae":
                new_pair[
                    "sampled_tf_mae"
                ],
        }
    )

    # ========================================================
    # Explicit errors
    # ========================================================

    result[
        "old_mean_abs_error"
    ] = np.abs(
        result[
            "old_mean_tf"
        ]
        - result[
            "true_tf"
        ]
    )

    result[
        "new_mean_abs_error"
    ] = np.abs(
        result[
            "new_mean_tf"
        ]
        - result[
            "true_tf"
        ]
    )

    result[
        "old_mode_abs_error"
    ] = np.abs(
        result[
            "old_mode_tf"
        ]
        - result[
            "true_tf"
        ]
    )

    result[
        "new_mode_abs_error"
    ] = np.abs(
        result[
            "new_mode_tf"
        ]
        - result[
            "true_tf"
        ]
    )

    # ========================================================
    # Positive number = corrected censoring improved recovery
    # ========================================================

    result[
        "mean_error_improvement"
    ] = (
        result[
            "old_mean_abs_error"
        ]
        - result[
            "new_mean_abs_error"
        ]
    )

    result[
        "mode_error_improvement"
    ] = (
        result[
            "old_mode_abs_error"
        ]
        - result[
            "new_mode_abs_error"
        ]
    )

    result[
        "sampled_mae_improvement"
    ] = (
        result[
            "old_sampled_tf_mae"
        ]
        - result[
            "new_sampled_tf_mae"
        ]
    )

    result[
        "expected_error_improvement"
    ] = (
        result[
            "old_expected_abs_error"
        ]
        - result[
            "new_expected_abs_error"
        ]
    )

    result[
        "no_commission_reduction"
    ] = (
        result[
            "old_no_commission"
        ]
        - result[
            "new_no_commission"
        ]
    )

    result[
        "true_tf_prob_gain"
    ] = (
        result[
            "new_true_tf_prob_cond"
        ]
        - result[
            "old_true_tf_prob_cond"
        ]
    )

    return result


def summarize_metric(
    paired,
    old_column,
    new_column,
    improvement_column,
    label,
):

    old_values = paired[
        old_column
    ].to_numpy(
        dtype=np.float64
    )

    new_values = paired[
        new_column
    ].to_numpy(
        dtype=np.float64
    )

    improvement = paired[
        improvement_column
    ].to_numpy(
        dtype=np.float64
    )

    ci_low, ci_high = (
        bootstrap_mean_ci(
            improvement
        )
    )

    improved = int(
        np.sum(
            improvement > 0
        )
    )

    unchanged = int(
        np.sum(
            np.isclose(
                improvement,
                0.0,
            )
        )
    )

    worsened = int(
        np.sum(
            improvement < 0
        )
    )

    return {
        "metric":
            label,

        "old_mean":
            safe_mean(
                old_values
            ),

        "new_mean":
            safe_mean(
                new_values
            ),

        "mean_improvement":
            safe_mean(
                improvement
            ),

        "median_improvement":
            safe_median(
                improvement
            ),

        "bootstrap_95_ci_low":
            ci_low,

        "bootstrap_95_ci_high":
            ci_high,

        "improved_entities":
            improved,

        "unchanged_entities":
            unchanged,

        "worsened_entities":
            worsened,
    }


def analyze_variant(
    inference,
    paired_audit,
):

    paired = prepare_variant(
        inference,
        paired_audit,
    )

    summaries = []

    summaries.append(
        summarize_metric(
            paired,
            "old_mean_abs_error",
            "new_mean_abs_error",
            "mean_error_improvement",
            "posterior_mean_absolute_error",
        )
    )

    summaries.append(
        summarize_metric(
            paired,
            "old_mode_abs_error",
            "new_mode_abs_error",
            "mode_error_improvement",
            "posterior_mode_absolute_error",
        )
    )

    summaries.append(
        summarize_metric(
            paired,
            "old_sampled_tf_mae",
            "new_sampled_tf_mae",
            "sampled_mae_improvement",
            "sampled_tf_mae",
        )
    )

    summaries.append(
        summarize_metric(
            paired,
            "old_expected_abs_error",
            "new_expected_abs_error",
            "expected_error_improvement",
            "posterior_expected_absolute_error",
        )
    )

    # ========================================================
    # Posterior-state changes
    # ========================================================

    no_commission_reduction = paired[
        "no_commission_reduction"
    ].to_numpy(
        dtype=np.float64
    )

    true_tf_gain = paired[
        "true_tf_prob_gain"
    ].to_numpy(
        dtype=np.float64
    )

    posterior_summary = {
        "old_mean_no_commission":
            safe_mean(
                paired[
                    "old_no_commission"
                ]
            ),

        "new_mean_no_commission":
            safe_mean(
                paired[
                    "new_no_commission"
                ]
            ),

        "mean_no_commission_reduction":
            safe_mean(
                no_commission_reduction
            ),

        "old_mean_true_tf_probability_given_commission":
            safe_mean(
                paired[
                    "old_true_tf_prob_cond"
                ]
            ),

        "new_mean_true_tf_probability_given_commission":
            safe_mean(
                paired[
                    "new_true_tf_prob_cond"
                ]
            ),

        "mean_true_tf_probability_gain":
            safe_mean(
                true_tf_gain
            ),
    }

    # ========================================================
    # Save
    # ========================================================

    variant_dir = (
        OUTPUT_DIR
        / inference
    )

    variant_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paired.to_csv(
        variant_dir
        / "paired_entities.csv",
        index=False,
    )

    pd.DataFrame(
        summaries
    ).to_csv(
        variant_dir
        / "paired_metric_summary.csv",
        index=False,
    )

    summary = {
        "inference":
            inference,

        "n_paired_entities":
            int(
                len(
                    paired
                )
            ),

        "metrics":
            {
                row["metric"]:
                    row
                for row in summaries
            },

        "posterior_state_change":
            posterior_summary,
    }

    (
        variant_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return (
        paired,
        summary,
    )


def print_variant_summary(
    inference,
    summary,
):

    print()

    print(
        "=" * 80
    )

    print(
        f"PAIRED RESULT: {inference}"
    )

    print(
        "=" * 80
    )

    print(
        "Paired entities:",
        summary[
            "n_paired_entities"
        ],
    )

    print()

    for metric_name, row in (
        summary[
            "metrics"
        ].items()
    ):

        print(
            metric_name
        )

        print(
            "  old mean:",
            row[
                "old_mean"
            ],
        )

        print(
            "  new mean:",
            row[
                "new_mean"
            ],
        )

        print(
            "  mean improvement:",
            row[
                "mean_improvement"
            ],
        )

        print(
            "  median improvement:",
            row[
                "median_improvement"
            ],
        )

        print(
            "  bootstrap 95% CI:",
            (
                row[
                    "bootstrap_95_ci_low"
                ],
                row[
                    "bootstrap_95_ci_high"
                ],
            ),
        )

        print(
            "  improved / unchanged / worsened:",
            (
                row[
                    "improved_entities"
                ],
                row[
                    "unchanged_entities"
                ],
                row[
                    "worsened_entities"
                ],
            ),
        )

        print()

    state = summary[
        "posterior_state_change"
    ]

    print(
        "Posterior-state change"
    )

    print(
        "  old mean no-commission:",
        state[
            "old_mean_no_commission"
        ],
    )

    print(
        "  new mean no-commission:",
        state[
            "new_mean_no_commission"
        ],
    )

    print(
        "  mean no-commission reduction:",
        state[
            "mean_no_commission_reduction"
        ],
    )

    print(
        "  old P(true tf | commission):",
        state[
            "old_mean_true_tf_probability_given_commission"
        ],
    )

    print(
        "  new P(true tf | commission):",
        state[
            "new_mean_true_tf_probability_given_commission"
        ],
    )

    print(
        "  mean probability gain:",
        state[
            "mean_true_tf_probability_gain"
        ],
    )


def print_top_examples(
    inference,
    paired,
):

    print()

    print(
        "=" * 80
    )

    print(
        f"TOP PAIRED IMPROVEMENTS: {inference}"
    )

    print(
        "=" * 80
    )

    ordered = paired.sort_values(
        "mean_error_improvement",
        ascending=False,
    )

    columns = [
        "entity_id",
        "true_tf",
        "detection_delay",
        "candidate_td",
        "old_mean_tf",
        "new_mean_tf",
        "old_mean_abs_error",
        "new_mean_abs_error",
        "mean_error_improvement",
        "old_no_commission",
        "new_no_commission",
    ]

    print(
        ordered[
            columns
        ].head(
            10
        ).to_string(
            index=False
        )
    )

    print()

    print(
        "=" * 80
    )

    print(
        f"WORST / NON-IMPROVED PAIRS: {inference}"
    )

    print(
        "=" * 80
    )

    print(
        ordered[
            columns
        ].tail(
            10
        ).to_string(
            index=False
        )
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paired_audit = (
        load_paired_audit()
    )

    print(
        "=" * 80
    )

    print(
        "EXPERIMENT 03I: PAIRED CENSORING EFFECT"
    )

    print(
        "=" * 80
    )

    print(
        "Paired random-censored users:",
        len(
            paired_audit
        ),
    )

    print()

    print(
        "For every paired user:"
    )

    print(
        "  same x"
    )

    print(
        "  same true tf"
    )

    print(
        "  same delay"
    )

    print(
        "  same candidate td"
    )

    print(
        "  old dataset hides td"
    )

    print(
        "  corrected dataset reveals td"
    )

    all_summaries = {}

    for inference in (
        INFERENCE_VARIANTS
    ):

        paired, summary = (
            analyze_variant(
                inference,
                paired_audit,
            )
        )

        all_summaries[
            inference
        ] = summary

        print_variant_summary(
            inference,
            summary,
        )

        print_top_examples(
            inference,
            paired,
        )

    # ========================================================
    # Save combined summary
    # ========================================================

    (
        OUTPUT_DIR
        / "summary.json"
    ).write_text(
        json.dumps(
            all_summaries,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Compact final comparison
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "FINAL PAIRED COMPARISON"
    )

    print(
        "=" * 80
    )

    for inference in (
        INFERENCE_VARIANTS
    ):

        summary = all_summaries[
            inference
        ]

        mean_row = summary[
            "metrics"
        ][
            "posterior_mean_absolute_error"
        ]

        sample_row = summary[
            "metrics"
        ][
            "sampled_tf_mae"
        ]

        state = summary[
            "posterior_state_change"
        ]

        print()

        print(
            inference
        )

        print(
            "  posterior-mean MAE:",
            mean_row[
                "old_mean"
            ],
            "->",
            mean_row[
                "new_mean"
            ],
        )

        print(
            "  mean MAE reduction:",
            mean_row[
                "mean_improvement"
            ],
        )

        print(
            "  sampled-tf MAE:",
            sample_row[
                "old_mean"
            ],
            "->",
            sample_row[
                "new_mean"
            ],
        )

        print(
            "  no-commission posterior:",
            state[
                "old_mean_no_commission"
            ],
            "->",
            state[
                "new_mean_no_commission"
            ],
        )

        print(
            "  P(true tf | commission):",
            state[
                "old_mean_true_tf_probability_given_commission"
            ],
            "->",
            state[
                "new_mean_true_tf_probability_given_commission"
            ],
        )

    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT_DIR
        / "summary.json"
    )


if __name__ == "__main__":
    main()