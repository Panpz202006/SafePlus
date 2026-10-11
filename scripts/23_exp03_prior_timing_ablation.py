from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Experiment 03P
# Prior-only vs Learned Timing Diagnostic
#
# 目的：
#
# 03O 中 Oracle delay prior + learned Timing 后，
# 所有 posterior mode delay 都等于 1。
#
# 但 synthetic 的真实 delay prior 本身 mode 就是 1。
#
# 所以这一步不重新训练模型，而是直接比较：
#
# A. Prior-only:
#       只使用真实总体 delay prior
#
# B. Oracle-prior + learned Timing:
#       使用 03O 已训练结果
#
# 看 learned Timing 到底有没有提供额外的
# 用户级 onset 信息。
# ============================================================


DELAY_MEAN = 4.0

ROOT = Path(
    r"reports/experiment03/oracle_delay_prior"
)

OUTPUT_DIR = Path(
    r"reports/experiment03/prior_timing_ablation"
)


# ============================================================
# True synthetic delay PMF
#
# Generator:
#
#   E ~ Exponential(scale=4)
#   D = max(1, int(E))
# ============================================================


def delay_probability(delay: int) -> float:

    if delay < 1:
        return 0.0

    mu = DELAY_MEAN

    # delay = 1:
    #
    # max(1, int(E)) = 1
    #
    # includes:
    #
    #   0 <= E < 2

    if delay == 1:

        return (
            1.0
            - math.exp(
                -2.0 / mu
            )
        )

    # delay >= 2:
    #
    # d <= E < d+1

    return (
        math.exp(
            -delay / mu
        )
        -
        math.exp(
            -(delay + 1) / mu
        )
    )


# ============================================================
# Prior-only posterior for one detected user
#
# observed td = d
#
# candidate:
#
#   tf = 0 ... d-1
#
# equivalently:
#
#   delay = d ... 1
#
# We condition the global delay prior on the legal candidates.
# ============================================================


def prior_only_posterior(td: int):

    candidate_tf = np.arange(
        0,
        td,
        dtype=np.int64,
    )

    delays = (
        td
        - candidate_tf
    )

    probabilities = np.array(
        [
            delay_probability(
                int(delay)
            )
            for delay in delays
        ],
        dtype=np.float64,
    )

    total = probabilities.sum()

    if total <= 0:
        raise RuntimeError(
            f"Invalid prior mass for td={td}"
        )

    probabilities /= total

    return (
        candidate_tf,
        delays,
        probabilities,
    )


# ============================================================
# Analyze one inference result
# ============================================================


def analyze(
    inference: str,
):

    csv_path = (
        ROOT
        / inference
        / "entity_diagnostics.csv"
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            csv_path
        )

    df = pd.read_csv(
        csv_path
    )

    detected_fraud = df[
        (
            df["true_tf"] >= 0
        )
        &
        (
            df["detected"] == 1
        )
    ].copy()

    rows = []

    for _, row in detected_fraud.iterrows():

        entity_id = int(
            row["entity_id"]
        )

        true_tf = int(
            row["true_tf"]
        )

        td = int(
            row["observed_td"]
        )

        (
            candidate_tf,
            delays,
            probability,
        ) = prior_only_posterior(
            td
        )

        # ----------------------------------------------------
        # Prior-only posterior mode
        # ----------------------------------------------------

        mode_index = int(
            np.argmax(
                probability
            )
        )

        prior_mode_tf = int(
            candidate_tf[
                mode_index
            ]
        )

        prior_mode_delay = (
            td
            - prior_mode_tf
        )

        # ----------------------------------------------------
        # Prior-only posterior mean
        # ----------------------------------------------------

        prior_mean_tf = float(
            (
                candidate_tf
                * probability
            ).sum()
        )

        prior_mean_delay = float(
            (
                delays
                * probability
            ).sum()
        )

        # ----------------------------------------------------
        # Probability at true tf
        # ----------------------------------------------------

        match = np.where(
            candidate_tf
            == true_tf
        )[0]

        if len(match) == 1:

            true_tf_probability = float(
                probability[
                    match[0]
                ]
            )

        else:

            true_tf_probability = 0.0

        # ----------------------------------------------------
        # Current 03O learned-Timing result
        # ----------------------------------------------------

        learned_mode_tf = float(
            row[
                "posterior_mode_tf"
            ]
        )

        learned_mean_tf = float(
            row[
                "posterior_mean_tf"
            ]
        )

        rows.append(
            {
                "entity_id":
                    entity_id,

                "true_tf":
                    true_tf,

                "observed_td":
                    td,

                "true_delay":
                    td - true_tf,

                "prior_mode_tf":
                    prior_mode_tf,

                "prior_mode_delay":
                    prior_mode_delay,

                "prior_mean_tf":
                    prior_mean_tf,

                "prior_mean_delay":
                    prior_mean_delay,

                "prior_true_tf_probability":
                    true_tf_probability,

                "prior_mode_tf_abs_error":
                    abs(
                        prior_mode_tf
                        - true_tf
                    ),

                "prior_mean_tf_abs_error":
                    abs(
                        prior_mean_tf
                        - true_tf
                    ),

                "learned_mode_tf":
                    learned_mode_tf,

                "learned_mean_tf":
                    learned_mean_tf,

                "learned_mode_delay":
                    td
                    - learned_mode_tf,

                "learned_mean_delay":
                    td
                    - learned_mean_tf,

                "learned_mode_tf_abs_error":
                    abs(
                        learned_mode_tf
                        - true_tf
                    ),

                "learned_mean_tf_abs_error":
                    abs(
                        learned_mean_tf
                        - true_tf
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    # ========================================================
    # Summary
    # ========================================================

    summary = {
        "inference":
            inference,

        "n_detected_fraud":
            int(
                len(
                    result
                )
            ),

        "true_delay_mean":
            float(
                result[
                    "true_delay"
                ].mean()
            ),

        # ----------------------------------------------------
        # Prior-only
        # ----------------------------------------------------

        "prior_mode_delay_mean":
            float(
                result[
                    "prior_mode_delay"
                ].mean()
            ),

        "prior_mean_delay_mean":
            float(
                result[
                    "prior_mean_delay"
                ].mean()
            ),

        "prior_mode_tf_mae":
            float(
                result[
                    "prior_mode_tf_abs_error"
                ].mean()
            ),

        "prior_mean_tf_mae":
            float(
                result[
                    "prior_mean_tf_abs_error"
                ].mean()
            ),

        "prior_true_tf_probability_mean":
            float(
                result[
                    "prior_true_tf_probability"
                ].mean()
            ),

        # ----------------------------------------------------
        # Oracle prior + learned Timing
        # ----------------------------------------------------

        "learned_mode_delay_mean":
            float(
                result[
                    "learned_mode_delay"
                ].mean()
            ),

        "learned_mean_delay_mean":
            float(
                result[
                    "learned_mean_delay"
                ].mean()
            ),

        "learned_mode_tf_mae":
            float(
                result[
                    "learned_mode_tf_abs_error"
                ].mean()
            ),

        "learned_mean_tf_mae":
            float(
                result[
                    "learned_mean_tf_abs_error"
                ].mean()
            ),
    }

    return (
        result,
        summary,
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
        "EXPERIMENT 03P:"
    )

    print(
        "PRIOR-ONLY vs LEARNED TIMING"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "核心问题："
    )

    print(
        "Oracle prior 本身的 mode 就是 delay=1。"
    )

    print(
        "因此不能只看到 mode=1 就判断模型 collapse。"
    )

    print()

    print(
        "这次比较："
    )

    print(
        "A. 只使用总体 delay prior"
    )

    print(
        "B. 总体 delay prior + learned Timing"
    )

    print()

    # --------------------------------------------------------
    # Print theoretical prior
    # --------------------------------------------------------

    print(
        "Synthetic delay prior:"
    )

    for delay in range(
        1,
        11,
    ):

        print(
            f"  P(delay={delay}) = "
            f"{delay_probability(delay):.6f}"
        )

    theoretical_mean = sum(
        delay
        * delay_probability(
            delay
        )
        for delay in range(
            1,
            200,
        )
    )

    print()

    print(
        "Theoretical delay mean:",
        theoretical_mean,
    )

    print()

    summaries = []

    for inference in (
        "exact",
        "stochastic_em",
    ):

        result, summary = analyze(
            inference
        )

        summaries.append(
            summary
        )

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
                "n_detected_fraud"
            ],
        )

        print()

        print(
            "TRUE DATA:"
        )

        print(
            "  true delay mean:",
            summary[
                "true_delay_mean"
            ],
        )

        print()

        print(
            "PRIOR ONLY:"
        )

        print(
            "  mode delay mean:",
            summary[
                "prior_mode_delay_mean"
            ],
        )

        print(
            "  posterior-mean delay:",
            summary[
                "prior_mean_delay_mean"
            ],
        )

        print(
            "  mode tf MAE:",
            summary[
                "prior_mode_tf_mae"
            ],
        )

        print(
            "  posterior mean tf MAE:",
            summary[
                "prior_mean_tf_mae"
            ],
        )

        print(
            "  P(true tf):",
            summary[
                "prior_true_tf_probability_mean"
            ],
        )

        print()

        print(
            "ORACLE PRIOR + LEARNED TIMING:"
        )

        print(
            "  mode delay mean:",
            summary[
                "learned_mode_delay_mean"
            ],
        )

        print(
            "  posterior-mean delay:",
            summary[
                "learned_mean_delay_mean"
            ],
        )

        print(
            "  mode tf MAE:",
            summary[
                "learned_mode_tf_mae"
            ],
        )

        print(
            "  posterior mean tf MAE:",
            summary[
                "learned_mean_tf_mae"
            ],
        )

        print()

        print(
            "CHANGE AFTER ADDING LEARNED TIMING:"
        )

        print(
            "  posterior mean tf MAE:",
            summary[
                "prior_mean_tf_mae"
            ],
            "->",
            summary[
                "learned_mean_tf_mae"
            ],
        )

        print(
            "  posterior-mean delay:",
            summary[
                "prior_mean_delay_mean"
            ],
            "->",
            summary[
                "learned_mean_delay_mean"
            ],
        )

        print()

        output_path = (
            OUTPUT_DIR
            / f"{inference}_comparison.csv"
        )

        result.to_csv(
            output_path,
            index=False,
        )

        print(
            "Saved:",
            output_path,
        )

        print()

    pd.DataFrame(
        summaries
    ).to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False,
    )

    print(
        "=" * 80
    )

    print(
        "INTERPRETATION GUIDE"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "如果 Prior-only 的 posterior mean delay "
        "本来接近真实 3~4，"
    )

    print(
        "但加入 learned Timing 后掉到 1~2，"
    )

    print(
        "说明 Timing/objective 在进一步推动短 delay。"
    )

    print()

    print(
        "如果 Prior-only 自己的 posterior mean delay "
        "也只有 1~2，"
    )

    print(
        "说明总体 delay prior 本身不足以恢复用户级 tf。"
    )

    print()

    print(
        "最终要解决的就不是简单加入 global prior，"
    )

    print(
        "而是学习 user-conditioned delay:"
    )

    print(
        "  p(delay | x)"
    )


if __name__ == "__main__":
    main()