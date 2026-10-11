from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Experiment 03L
# Detection-Time Collapse Audit
#
# 中文：
# 检查 latent t_f 是否塌缩到 observed t_d 附近。
#
#
# 对 detected fraud 用户：
#
# True detection delay:
#
#     true_delay = t_d - true_t_f
#
#
# Model-implied delay using posterior mode:
#
#     mode_delay = t_d - posterior_mode_t_f
#
#
# Model-implied delay using posterior mean:
#
#     mean_delay = t_d - posterior_mean_t_f
#
#
# 如果：
#
#     true_delay 明显 > 0
#
# 但：
#
#     predicted delay ≈ 0
#
# 那么模型实际上是在采用捷径：
#
#     t_f ≈ t_d
#
# 这就是 detection-time collapse（检测时间塌缩）。
# ============================================================


OUTPUT_DIR = Path(
    r"reports/experiment03/detection_time_collapse_audit"
)


# ============================================================
# Three model versions
#
# non_strict:
#     曾经可以利用 t_d 后面的未来行为。
#
# strict_local:
#     严格禁止 t_d 后信息，
#     timing 候选主要依赖候选时刻本身。
#
# endpoint_conditioned:
#     严格禁止 t_d 后信息，
#     每个候选 t_f 可以结合截至 t_d 的完整证据。
# ============================================================


EXPERIMENTS = {
    "non_strict": Path(
        r"reports/experiment03/"
        r"latent_decoupled_same_delay_consistent"
    ),

    "strict_local": Path(
        r"reports/experiment03/"
        r"strict_causal_same_delay_consistent"
    ),

    "endpoint_conditioned": Path(
        r"reports/experiment03/"
        r"endpoint_conditioned_same_delay_consistent"
    ),
}


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


def load_detected_fraud(
    experiment_name,
    root,
    inference,
):

    path = (
        root
        / inference
        / "entity_diagnostics.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing file: {path}"
        )

    frame = pd.read_csv(
        path
    )

    required = {
        "entity_id",
        "true_tf",
        "observed_td",
        "detected",
        "posterior_mode_tf",
        "posterior_mean_tf",
    }

    missing = (
        required
        - set(
            frame.columns
        )
    )

    if missing:

        raise KeyError(
            f"{path} missing columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Only detected fraud users.
    #
    # We need known synthetic true tf only for post-hoc audit.
    # --------------------------------------------------------

    detected_fraud = frame[
        (
            frame[
                "true_tf"
            ] >= 0
        )
        &
        (
            frame[
                "detected"
            ] == 1
        )
    ].copy()

    detected_fraud[
        "experiment"
    ] = experiment_name

    detected_fraud[
        "inference"
    ] = inference

    return detected_fraud


def analyze(
    experiment_name,
    root,
    inference,
):

    frame = load_detected_fraud(
        experiment_name,
        root,
        inference,
    )

    # ========================================================
    # True delay
    #
    # 真正从 fraud onset 到 detection 的时间差。
    # ========================================================

    frame[
        "true_delay"
    ] = (
        frame[
            "observed_td"
        ]
        - frame[
            "true_tf"
        ]
    )

    # ========================================================
    # Predicted delay using posterior mode
    #
    # posterior mode:
    # 后验概率最大的位置。
    # ========================================================

    frame[
        "mode_predicted_delay"
    ] = (
        frame[
            "observed_td"
        ]
        - frame[
            "posterior_mode_tf"
        ]
    )

    # ========================================================
    # Predicted delay using posterior mean
    #
    # posterior mean:
    # 后验概率分布的重心。
    # ========================================================

    frame[
        "mean_predicted_delay"
    ] = (
        frame[
            "observed_td"
        ]
        - frame[
            "posterior_mean_tf"
        ]
    )

    # ========================================================
    # Distances of tf estimate from td
    #
    # If these values are near zero,
    # model is placing tf very close to td.
    # ========================================================

    frame[
        "mode_distance_to_td"
    ] = np.abs(
        frame[
            "posterior_mode_tf"
        ]
        - frame[
            "observed_td"
        ]
    )

    frame[
        "mean_distance_to_td"
    ] = np.abs(
        frame[
            "posterior_mean_tf"
        ]
        - frame[
            "observed_td"
        ]
    )

    # ========================================================
    # tf localization errors
    # ========================================================

    frame[
        "mode_tf_error"
    ] = np.abs(
        frame[
            "posterior_mode_tf"
        ]
        - frame[
            "true_tf"
        ]
    )

    frame[
        "mean_tf_error"
    ] = np.abs(
        frame[
            "posterior_mean_tf"
        ]
        - frame[
            "true_tf"
        ]
    )

    # ========================================================
    # Delay errors
    #
    # Mathematically these are closely related to tf errors,
    # but printing them makes the collapse interpretation
    # much easier.
    # ========================================================

    frame[
        "mode_delay_error"
    ] = np.abs(
        frame[
            "mode_predicted_delay"
        ]
        - frame[
            "true_delay"
        ]
    )

    frame[
        "mean_delay_error"
    ] = np.abs(
        frame[
            "mean_predicted_delay"
        ]
        - frame[
            "true_delay"
        ]
    )

    # ========================================================
    # Direct collapse indicators
    # ========================================================

    mode_equals_td = (
        frame[
            "posterior_mode_tf"
        ]
        == frame[
            "observed_td"
        ]
    )

    mode_within_1_td = (
        frame[
            "mode_distance_to_td"
        ] <= 1
    )

    mode_within_2_td = (
        frame[
            "mode_distance_to_td"
        ] <= 2
    )

    mean_within_1_td = (
        frame[
            "mean_distance_to_td"
        ] <= 1
    )

    mean_within_2_td = (
        frame[
            "mean_distance_to_td"
        ] <= 2
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "experiment":
            experiment_name,

        "inference":
            inference,

        "n_detected_fraud":
            int(
                len(
                    frame
                )
            ),

        # ----------------------------------------------------
        # True detection delay
        # ----------------------------------------------------

        "true_delay_mean":
            safe_mean(
                frame[
                    "true_delay"
                ]
            ),

        "true_delay_median":
            safe_median(
                frame[
                    "true_delay"
                ]
            ),

        "true_delay_min":
            float(
                frame[
                    "true_delay"
                ].min()
            ),

        "true_delay_max":
            float(
                frame[
                    "true_delay"
                ].max()
            ),

        # ----------------------------------------------------
        # Model-implied delay
        # ----------------------------------------------------

        "mode_predicted_delay_mean":
            safe_mean(
                frame[
                    "mode_predicted_delay"
                ]
            ),

        "mode_predicted_delay_median":
            safe_median(
                frame[
                    "mode_predicted_delay"
                ]
            ),

        "mean_predicted_delay_mean":
            safe_mean(
                frame[
                    "mean_predicted_delay"
                ]
            ),

        "mean_predicted_delay_median":
            safe_median(
                frame[
                    "mean_predicted_delay"
                ]
            ),

        # ----------------------------------------------------
        # Collapse indicators
        # ----------------------------------------------------

        "mode_equals_td_rate":
            float(
                mode_equals_td.mean()
            ),

        "mode_within_1_of_td_rate":
            float(
                mode_within_1_td.mean()
            ),

        "mode_within_2_of_td_rate":
            float(
                mode_within_2_td.mean()
            ),

        "posterior_mean_within_1_of_td_rate":
            float(
                mean_within_1_td.mean()
            ),

        "posterior_mean_within_2_of_td_rate":
            float(
                mean_within_2_td.mean()
            ),

        # ----------------------------------------------------
        # tf error
        # ----------------------------------------------------

        "mode_tf_mae":
            safe_mean(
                frame[
                    "mode_tf_error"
                ]
            ),

        "mean_tf_mae":
            safe_mean(
                frame[
                    "mean_tf_error"
                ]
            ),

        # ----------------------------------------------------
        # delay error
        # ----------------------------------------------------

        "mode_delay_mae":
            safe_mean(
                frame[
                    "mode_delay_error"
                ]
            ),

        "mean_delay_mae":
            safe_mean(
                frame[
                    "mean_delay_error"
                ]
            ),
    }

    return (
        frame,
        summary,
    )


def print_summary(
    summary,
):

    print()

    print(
        "=" * 80
    )

    print(
        f"{summary['experiment']} | "
        f"{summary['inference']}"
    )

    print(
        "=" * 80
    )

    print(
        "Detected fraud users:",
        summary[
            "n_detected_fraud"
        ],
    )

    print()

    print(
        "TRUE DETECTION DELAY"
    )

    print(
        "  mean:",
        summary[
            "true_delay_mean"
        ],
    )

    print(
        "  median:",
        summary[
            "true_delay_median"
        ],
    )

    print(
        "  min / max:",
        (
            summary[
                "true_delay_min"
            ],
            summary[
                "true_delay_max"
            ],
        ),
    )

    print()

    print(
        "MODEL-PREDICTED DELAY"
    )

    print(
        "  mode-based mean:",
        summary[
            "mode_predicted_delay_mean"
        ],
    )

    print(
        "  mode-based median:",
        summary[
            "mode_predicted_delay_median"
        ],
    )

    print(
        "  posterior-mean based mean:",
        summary[
            "mean_predicted_delay_mean"
        ],
    )

    print(
        "  posterior-mean based median:",
        summary[
            "mean_predicted_delay_median"
        ],
    )

    print()

    print(
        "DETECTION-TIME COLLAPSE INDICATORS"
    )

    print(
        "  posterior mode exactly equals td:",
        summary[
            "mode_equals_td_rate"
        ],
    )

    print(
        "  posterior mode within 1 step of td:",
        summary[
            "mode_within_1_of_td_rate"
        ],
    )

    print(
        "  posterior mode within 2 steps of td:",
        summary[
            "mode_within_2_of_td_rate"
        ],
    )

    print(
        "  posterior mean within 1 step of td:",
        summary[
            "posterior_mean_within_1_of_td_rate"
        ],
    )

    print(
        "  posterior mean within 2 steps of td:",
        summary[
            "posterior_mean_within_2_of_td_rate"
        ],
    )

    print()

    print(
        "t_f ERROR"
    )

    print(
        "  mode MAE:",
        summary[
            "mode_tf_mae"
        ],
    )

    print(
        "  posterior mean MAE:",
        summary[
            "mean_tf_mae"
        ],
    )


def print_examples(
    frame,
    experiment,
    inference,
):

    columns = [
        "entity_id",
        "true_tf",
        "observed_td",
        "true_delay",
        "posterior_mode_tf",
        "mode_predicted_delay",
        "posterior_mean_tf",
        "mean_predicted_delay",
    ]

    # --------------------------------------------------------
    # Sort by how close mode is to td.
    # --------------------------------------------------------

    ordered = frame.sort_values(
        [
            "mode_distance_to_td",
            "entity_id",
        ],
        ascending=[
            True,
            True,
        ],
    )

    print()

    print(
        "-" * 80
    )

    print(
        f"Examples closest to td: "
        f"{experiment} | {inference}"
    )

    print(
        "-" * 80
    )

    print(
        ordered[
            columns
        ]
        .head(
            10
        )
        .to_string(
            index=False
        )
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 80
    )

    print(
        "EXPERIMENT 03L: DETECTION-TIME COLLAPSE AUDIT"
    )

    print(
        "=" * 80
    )

    print(
        "Detection-time = 检测时间 t_d。"
    )

    print(
        "Collapse = 塌缩；模型是否把 t_f 直接推到 t_d 附近。"
    )

    print(
        "Audit = 审计、检查。"
    )

    print()

    print(
        "核心检查："
    )

    print(
        "真实 delay = t_d - true t_f"
    )

    print(
        "模型 delay = t_d - predicted t_f"
    )

    print()

    print(
        "如果真实 delay 明显大于 0，"
    )

    print(
        "但模型 delay 接近 0，"
    )

    print(
        "就说明模型在走 t_f ≈ t_d 的捷径。"
    )

    all_results = {}

    rows = []

    for experiment_name, root in (
        EXPERIMENTS.items()
    ):

        all_results[
            experiment_name
        ] = {}

        for inference in (
            INFERENCE_VARIANTS
        ):

            frame, summary = analyze(
                experiment_name,
                root,
                inference,
            )

            all_results[
                experiment_name
            ][
                inference
            ] = summary

            rows.append(
                summary
            )

            print_summary(
                summary
            )

            print_examples(
                frame,
                experiment_name,
                inference,
            )

            variant_dir = (
                OUTPUT_DIR
                / experiment_name
                / inference
            )

            variant_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            frame.to_csv(
                variant_dir
                / "detected_fraud_audit.csv",
                index=False,
            )

    # ========================================================
    # Save summaries
    # ========================================================

    (
        OUTPUT_DIR
        / "summary.json"
    ).write_text(
        json.dumps(
            all_results,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    pd.DataFrame(
        rows
    ).to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False,
    )

    # ========================================================
    # Compact comparison
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "COMPACT COMPARISON"
    )

    print(
        "=" * 80
    )

    for experiment_name in (
        EXPERIMENTS
    ):

        print()

        print(
            experiment_name
        )

        for inference in (
            INFERENCE_VARIANTS
        ):

            summary = all_results[
                experiment_name
            ][
                inference
            ]

            print(
                f"  {inference}:"
            )

            print(
                "    true delay mean =",
                summary[
                    "true_delay_mean"
                ],
            )

            print(
                "    predicted mode delay mean =",
                summary[
                    "mode_predicted_delay_mean"
                ],
            )

            print(
                "    predicted mean delay =",
                summary[
                    "mean_predicted_delay_mean"
                ],
            )

            print(
                "    mode == td rate =",
                summary[
                    "mode_equals_td_rate"
                ],
            )

            print(
                "    mode within 1 of td =",
                summary[
                    "mode_within_1_of_td_rate"
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

    print(
        OUTPUT_DIR
        / "summary.csv"
    )


if __name__ == "__main__":
    main()