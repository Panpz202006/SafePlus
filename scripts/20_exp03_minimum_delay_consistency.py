from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch


# ============================================================
# Experiment 03M
# Minimum-delay consistency check
#
# 中文：
# 最小检测延迟一致性检查。
#
#
# 数据生成规则：
#
#   delay >= 1
#
# 因此 detected 用户一定满足：
#
#   t_f < t_d
#
# 而不是：
#
#   t_f <= t_d
#
#
# 本实验保持上一轮 endpoint-conditioned
# （基于检测前完整证据）的模型不变，
# 只修改 latent joint（潜变量联合概率）：
#
#   detected 用户禁止候选 t_f = t_d
#
#
# 目的：
#
# 检查之前的 detection-time collapse
# （检测时间塌缩）是不是主要来自
# 模型错误允许了 delay=0。
#
#
# 如果修正后模型只是从：
#
#   t_f = t_d
#
# 变成：
#
#   t_f = t_d - 1
#
# 那说明问题更深：
#
# 模型仍然倾向于让 t_f 尽量贴近 t_d，
# 需要显式建模 detection delay。
# ============================================================


BASE_SCRIPT = Path(
    r"scripts/18_exp03_endpoint_conditioned_timing.py"
)

DATA_PATH = Path(
    r"data/processed/synthetic_same_delay_consistent.npz"
)

OUTPUT_DIR = Path(
    r"reports/experiment03/"
    r"minimum_delay_consistency"
)

RUN_ROOT = Path(
    r"runs/synthetic/exp03/"
    r"minimum-delay-consistency"
)

PREVIOUS_RESULT = Path(
    r"reports/experiment03/"
    r"endpoint_conditioned_same_delay_consistent/"
    r"summary.json"
)


# ============================================================
# Load Experiment 03K
# ============================================================


if not BASE_SCRIPT.exists():
    raise FileNotFoundError(
        f"Missing base script: {BASE_SCRIPT}"
    )


spec = importlib.util.spec_from_file_location(
    "exp03_endpoint_conditioned",
    BASE_SCRIPT,
)

if (
    spec is None
    or spec.loader is None
):
    raise RuntimeError(
        "Could not load Experiment 03K."
    )


endpoint_module = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    endpoint_module
)


# Experiment 03K internally loaded the original training pipeline
exp = endpoint_module.exp


# ============================================================
# Minimum-delay-consistent model
# ============================================================


class MinimumDelayEndpointModel(
    endpoint_module.EndpointConditionedLatentModel
):
    """
    Same endpoint-conditioned strict-causal model.

    ONLY difference:

    For detected users:

        old:
            candidate k <= d

        new:
            candidate k < d

    where:
        k = candidate t_f
        d = observed t_d

    Therefore detection delay is always at least 1 step.
    """

    def latent_joint(
        self,
        batch,
        out,
    ):
        detected = batch[
            "detected"
        ].bool()

        lengths = batch[
            "lengths"
        ]

        t_detect = batch[
            "t_detect"
        ]

        batch_size = len(
            lengths
        )

        max_length = out[
            "timing_probability"
        ].shape[1]

        rows = []

        for i in range(
            batch_size
        ):

            length_i = int(
                lengths[i].item()
            )

            detected_i = bool(
                detected[i].item()
            )

            state_log_mass = out[
                "log_pi"
            ].new_full(
                (max_length + 1,),
                -torch.inf,
            )

            # =================================================
            # DETECTED USER
            #
            # IMPORTANT:
            #
            # candidate t_f must satisfy:
            #
            #     t_f < t_d
            #
            # Therefore:
            #
            #     k = 0 ... d-1
            #
            # t_f = t_d is forbidden.
            # =================================================

            if detected_i:

                d = int(
                    t_detect[i].item()
                )

                if (
                    d <= 0
                    or d >= length_i
                ):
                    raise ValueError(
                        "Detected user must have "
                        "an in-range t_detect >= 1."
                    )

                # ---------------------------------------------
                # KEY CHANGE:
                #
                # range(d)
                #
                # instead of:
                #
                # range(d + 1)
                # ---------------------------------------------

                for k in range(
                    d
                ):

                    # Detection must NOT occur from k through d-1.
                    #
                    # Fraud occurs at k,
                    # detection finally happens at d.
                    log_survive_detection = (
                        out[
                            "log_detection_survival"
                        ][
                            i,
                            k:d,
                        ].sum()
                    )

                    # Detection event at d.
                    log_detect_at_d = out[
                        "log_detection_hazard"
                    ][
                        i,
                        d,
                    ]

                    state_log_mass[
                        k
                    ] = (
                        out[
                            "log_pi"
                        ][i]
                        +
                        out[
                            "log_timing"
                        ][
                            i,
                            k,
                        ]
                        +
                        log_survive_detection
                        +
                        log_detect_at_d
                    )

                # Detection was observed,
                # so "no fraud" is impossible.
                state_log_mass[
                    max_length
                ] = -torch.inf

            # =================================================
            # UNDETECTED / CENSORED USER
            #
            # 与上一版完全相同。
            #
            # 两种可能：
            #
            # 1. 根本没有 fraud
            #
            # 2. fraud 已经发生，
            #    但 detection 到观察结束仍未发生
            # =================================================

            else:

                horizon = (
                    length_i - 1
                )

                for k in range(
                    length_i
                ):

                    log_survive_detection = (
                        out[
                            "log_detection_survival"
                        ][
                            i,
                            k : horizon + 1,
                        ].sum()
                    )

                    state_log_mass[
                        k
                    ] = (
                        out[
                            "log_pi"
                        ][i]
                        +
                        out[
                            "log_timing"
                        ][
                            i,
                            k,
                        ]
                        +
                        log_survive_detection
                    )

                state_log_mass[
                    max_length
                ] = out[
                    "log_one_minus_pi"
                ][i]

            rows.append(
                state_log_mass
            )

        joint = torch.stack(
            rows,
            dim=0,
        )

        # -----------------------------------------------------
        # Exact observed-data likelihood
        #
        # 把所有合法 latent t_f 的概率全部加起来。
        # -----------------------------------------------------

        log_likelihood = (
            torch.logsumexp(
                joint,
                dim=1,
            )
        )

        # -----------------------------------------------------
        # Posterior
        #
        # 每一个候选 t_f 在当前观测下的后验概率。
        # -----------------------------------------------------

        posterior = (
            joint
            - log_likelihood[
                :,
                None,
            ]
        ).exp()

        return (
            joint,
            log_likelihood,
            posterior,
        )


# ============================================================
# Replace model in original training pipeline
# ============================================================


exp.LatentDecoupledJointModel = (
    MinimumDelayEndpointModel
)


# ============================================================
# Same dataset / hyperparameters
# ============================================================


exp.DATA_PATH = (
    DATA_PATH
)

exp.OUTPUT_DIR = (
    OUTPUT_DIR
)

exp.RUN_ROOT = (
    RUN_ROOT
)


# ============================================================
# Print experimental contract
# ============================================================


print(
    "=" * 80
)

print(
    "EXPERIMENT 03M: MINIMUM-DELAY CONSISTENCY CHECK"
)

print(
    "=" * 80
)

print(
    "Minimum = 最小。"
)

print(
    "Delay = detection delay，"
    "即从 t_f 到 t_d 的检测延迟。"
)

print(
    "Consistency = 模型定义和数据定义保持一致。"
)

print()

print(
    "DATA RULE:"
)

print(
    "  synthetic generator guarantees delay >= 1"
)

print()

print(
    "THEREFORE:"
)

print(
    "  detected user must satisfy t_f < t_d"
)

print(
    "  t_f = t_d is now forbidden"
)

print()

print(
    "UNCHANGED:"
)

print(
    "  same synthetic dataset"
)

print(
    "  same x"
)

print(
    "  same hidden true t_f"
)

print(
    "  same observed t_d"
)

print(
    "  same train/validation/test split"
)

print(
    "  same GRU encoder"
)

print(
    "  same occurrence head"
)

print(
    "  same endpoint-conditioned timing head"
)

print(
    "  same detection head"
)

print(
    "  same optimizer and hyperparameters"
)

print(
    "  true t_f remains hidden during training"
)

print()

print(
    "ONLY CHANGE:"
)

print(
    "  detected-user candidate t_f can only be 0 ... t_d-1"
)

print(
    "=" * 80
)

print()


# ============================================================
# Run exact + stochastic EM
# ============================================================


exp.main()


# ============================================================
# Compare against previous endpoint-conditioned model
# ============================================================


NEW_RESULT = (
    OUTPUT_DIR
    / "summary.json"
)


if (
    PREVIOUS_RESULT.exists()
    and NEW_RESULT.exists()
):

    previous = json.loads(
        PREVIOUS_RESULT.read_text(
            encoding="utf-8"
        )
    )

    current = json.loads(
        NEW_RESULT.read_text(
            encoding="utf-8"
        )
    )

    print()

    print(
        "=" * 80
    )

    print(
        "BEFORE vs MINIMUM-DELAY-CONSISTENT"
    )

    print(
        "=" * 80
    )

    for inference in (
        "exact",
        "stochastic_em",
    ):

        old = previous[
            inference
        ]

        new = current[
            inference
        ]

        print()

        print(
            inference
        )

        print(
            "  posterior mean MAE:",
            old[
                "posterior_mean_mae"
            ],
            "->",
            new[
                "posterior_mean_mae"
            ],
        )

        print(
            "  posterior mode MAE:",
            old[
                "posterior_mode_mae"
            ],
            "->",
            new[
                "posterior_mode_mae"
            ],
        )

        print(
            "  sampled tf MAE:",
            old[
                "sampled_tf_mae_given_commission"
            ],
            "->",
            new[
                "sampled_tf_mae_given_commission"
            ],
        )

        print(
            "  detected fraud mean MAE:",
            old[
                "detected_fraud_mean_mae"
            ],
            "->",
            new[
                "detected_fraud_mean_mae"
            ],
        )

        print(
            "  P(true tf | commission):",
            old[
                "mean_true_tf_probability_given_commission"
            ],
            "->",
            new[
                "mean_true_tf_probability_given_commission"
            ],
        )

        print(
            "  occurrence AUROC:",
            old[
                "occurrence_auroc"
            ],
            "->",
            new[
                "occurrence_auroc"
            ],
        )


print()

print(
    "Saved:"
)

print(
    NEW_RESULT
)