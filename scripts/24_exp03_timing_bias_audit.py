from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Experiment 03Q
# Timing Bias Audit
#
# 中文：
# 检查 learned Timing 是否系统性地把 latent t_f
# 往 observed t_d 附近推。
#
#
# 数据来源：
#
# Experiment 03P 已经保存：
#
#   prior-only
#   oracle-prior + learned Timing
#
# 对每个 detected fraud 用户的结果。
#
#
# 核心比较：
#
# 真实 delay:
#
#     td - true_tf
#
# Prior-only 预测 delay:
#
#     td - prior_mean_tf
#
# Learned Timing 后预测 delay:
#
#     td - learned_mean_tf
#
#
# 如果：
#
#     prior-only 已经接近真实 delay
#
# 但 learned Timing 后明显缩短，
#
# 就说明 Timing 在系统性推动短 delay。
# ============================================================


ROOT = Path(
    r"reports/experiment03/prior_timing_ablation"
)

OUTPUT_DIR = Path(
    r"reports/experiment03/timing_bias_audit"
)


def audit(
    inference: str,
):

    path = (
        ROOT
        / f"{inference}_comparison.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    df = pd.read_csv(
        path
    )

    # ========================================================
    # How much does learned Timing change the delay?
    #
    # negative:
    #     Timing makes predicted delay shorter.
    #
    # positive:
    #     Timing makes predicted delay longer.
    # ========================================================

    df[
        "timing_delay_shift"
    ] = (
        df[
            "learned_mean_delay"
        ]
        - df[
            "prior_mean_delay"
        ]
    )

    # ========================================================
    # Distance to true delay
    # ========================================================

    df[
        "prior_delay_abs_error"
    ] = np.abs(
        df[
            "prior_mean_delay"
        ]
        - df[
            "true_delay"
        ]
    )

    df[
        "learned_delay_abs_error"
    ] = np.abs(
        df[
            "learned_mean_delay"
        ]
        - df[
            "true_delay"
        ]
    )

    # ========================================================
    # Did Timing improve or worsen this user?
    # ========================================================

    df[
        "timing_error_change"
    ] = (
        df[
            "learned_delay_abs_error"
        ]
        - df[
            "prior_delay_abs_error"
        ]
    )

    improved = (
        df[
            "timing_error_change"
        ] < 0
    )

    same = np.isclose(
        df[
            "timing_error_change"
        ],
        0.0,
        atol=1e-8,
    )

    worsened = (
        df[
            "timing_error_change"
        ] > 0
    )

    # ========================================================
    # Group users by true delay
    # ========================================================

    group = (
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

            prior_mean_delay=(
                "prior_mean_delay",
                "mean",
            ),

            learned_mean_delay=(
                "learned_mean_delay",
                "mean",
            ),

            prior_delay_mae=(
                "prior_delay_abs_error",
                "mean",
            ),

            learned_delay_mae=(
                "learned_delay_abs_error",
                "mean",
            ),

            mean_timing_shift=(
                "timing_delay_shift",
                "mean",
            ),
        )
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "inference":
            inference,

        "n_users":
            int(
                len(
                    df
                )
            ),

        "true_delay_mean":
            float(
                df[
                    "true_delay"
                ].mean()
            ),

        "prior_mean_delay":
            float(
                df[
                    "prior_mean_delay"
                ].mean()
            ),

        "learned_mean_delay":
            float(
                df[
                    "learned_mean_delay"
                ].mean()
            ),

        "mean_timing_delay_shift":
            float(
                df[
                    "timing_delay_shift"
                ].mean()
            ),

        "prior_delay_mae":
            float(
                df[
                    "prior_delay_abs_error"
                ].mean()
            ),

        "learned_delay_mae":
            float(
                df[
                    "learned_delay_abs_error"
                ].mean()
            ),

        "improved_rate":
            float(
                improved.mean()
            ),

        "same_rate":
            float(
                same.mean()
            ),

        "worsened_rate":
            float(
                worsened.mean()
            ),

        "timing_shortens_delay_rate":
            float(
                (
                    df[
                        "timing_delay_shift"
                    ] < 0
                ).mean()
            ),

        "timing_shortens_by_gt_1_rate":
            float(
                (
                    df[
                        "timing_delay_shift"
                    ] < -1
                ).mean()
            ),
    }

    # ========================================================
    # Print
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        inference
    )

    print(
        "=" * 80
    )

    print(
        "Detected fraud users:",
        summary[
            "n_users"
        ],
    )

    print()

    print(
        "AVERAGE DELAY"
    )

    print(
        "  true:",
        summary[
            "true_delay_mean"
        ],
    )

    print(
        "  prior-only:",
        summary[
            "prior_mean_delay"
        ],
    )

    print(
        "  + learned Timing:",
        summary[
            "learned_mean_delay"
        ],
    )

    print()

    print(
        "TIMING SHIFT"
    )

    print(
        "  mean shift:",
        summary[
            "mean_timing_delay_shift"
        ],
    )

    print(
        "  Timing makes delay shorter rate:",
        summary[
            "timing_shortens_delay_rate"
        ],
    )

    print(
        "  Timing shortens delay by >1 step rate:",
        summary[
            "timing_shortens_by_gt_1_rate"
        ],
    )

    print()

    print(
        "DELAY ERROR"
    )

    print(
        "  prior-only MAE:",
        summary[
            "prior_delay_mae"
        ],
    )

    print(
        "  learned Timing MAE:",
        summary[
            "learned_delay_mae"
        ],
    )

    print()

    print(
        "USER-LEVEL EFFECT"
    )

    print(
        "  improved:",
        summary[
            "improved_rate"
        ],
    )

    print(
        "  unchanged:",
        summary[
            "same_rate"
        ],
    )

    print(
        "  worsened:",
        summary[
            "worsened_rate"
        ],
    )

    print()

    print(
        "BY TRUE DELAY"
    )

    print(
        group.to_string(
            index=False
        )
    )

    print()

    # ========================================================
    # Interesting examples
    # ========================================================

    columns = [
        "entity_id",
        "true_tf",
        "observed_td",
        "true_delay",
        "prior_mean_delay",
        "learned_mean_delay",
        "timing_delay_shift",
        "prior_delay_abs_error",
        "learned_delay_abs_error",
    ]

    # Timing hurts most
    worst = df.sort_values(
        "timing_error_change",
        ascending=False,
    )

    print(
        "-" * 80
    )

    print(
        "Examples where learned Timing hurts most:"
    )

    print(
        "-" * 80
    )

    print(
        worst[
            columns
        ]
        .head(
            15
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # Save
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_DIR
        / f"{inference}_user_level.csv",
        index=False,
    )

    group.to_csv(
        OUTPUT_DIR
        / f"{inference}_by_true_delay.csv",
        index=False,
    )

    return summary


def main():

    print(
        "=" * 80
    )

    print(
        "EXPERIMENT 03Q: TIMING BIAS AUDIT"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "核心问题："
    )

    print(
        "Prior-only 已经给出较合理的平均 delay，"
    )

    print(
        "为什么加入 learned Timing 后反而变短？"
    )

    print()

    summaries = []

    for inference in (
        "exact",
        "stochastic_em",
    ):

        summary = audit(
            inference
        )

        summaries.append(
            summary
        )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False,
    )

    print()

    print(
        "=" * 80
    )

    print(
        "FINAL INTERPRETATION"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "如果大多数用户的 timing_delay_shift < 0，"
    )

    print(
        "并且加入 Timing 后 delay MAE 变差，"
    )

    print(
        "则可以认为 learned Timing 存在系统性的"
    )

    print(
        "short-delay bias（短延迟偏置）。"
    )

    print()

    print(
        "这时下一步重点应该是修改 Timing/objective，"
    )

    print(
        "而不是继续增加 global delay prior。"
    )


if __name__ == "__main__":
    main()