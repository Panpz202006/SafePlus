from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from safeplus.models.common import time_mask


# ============================================================
# Experiment 03O
# Oracle Detection-Delay Prior
#
# 中文：
# 使用 synthetic 数据真实的“总体 detection-delay 生成规律”
# 来约束 latent t_f 推断。
#
#
# IMPORTANT:
#
# 不给模型任何具体用户的真实 t_f。
#
# 给模型的只有总体 delay 规律：
#
#     delay = max(1, int(Exponential(scale=4)))
#
#
# 目的：
#
# 检查如果 detection delay 被正确建模，
# latent t_f 是否能够明显恢复。
#
#
# 这只是 Oracle（上帝视角）诊断，
# 不是最终方法。
# ============================================================


BASE_SCRIPT = Path(
    r"scripts/10_exp03_latent_decoupled_joint.py"
)

DATA_PATH = Path(
    r"data/processed/synthetic_same_delay_consistent.npz"
)

OUTPUT_DIR = Path(
    r"reports/experiment03/oracle_delay_prior"
)

RUN_ROOT = Path(
    r"runs/synthetic/exp03/oracle-delay-prior"
)

REFERENCE_RESULT = Path(
    r"reports/experiment03/"
    r"minimum_delay_consistency/"
    r"summary.json"
)


# ============================================================
# Synthetic generator's true delay parameter
#
# delay = max(1, int(exponential(4)))
# ============================================================

DELAY_MEAN = 4.0


# ============================================================
# Load original Experiment 03 latent-decoupled pipeline
# ============================================================


if not BASE_SCRIPT.exists():
    raise FileNotFoundError(
        f"Missing base script: {BASE_SCRIPT}"
    )


spec = importlib.util.spec_from_file_location(
    "exp03_latent_base",
    BASE_SCRIPT,
)

if (
    spec is None
    or spec.loader is None
):
    raise RuntimeError(
        "Could not load Experiment 03 base script."
    )


exp = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    exp
)


# ============================================================
# Oracle Delay Prior Model
# ============================================================


class OracleDelayPriorModel(
    exp.LatentDecoupledJointModel
):
    """
    Strict-causal latent model with a known delay prior.

    Detected user:
        endpoint = t_d

    Undetected user:
        endpoint = T

    No information after the endpoint is used.

    Candidate t_f=k is evaluated using:

        occurrence probability
        × timing probability
        × oracle delay probability

    instead of allowing the detection process to freely
    collapse toward t_d.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ):
        super().__init__(
            input_dim,
            hidden_dim,
        )

        # ----------------------------------------------------
        # Endpoint-conditioned timing scorer
        #
        # Candidate k is judged together with all evidence
        # observed up to t_d.
        # ----------------------------------------------------

        self.endpoint_timing_head = nn.Sequential(
            nn.Linear(
                hidden_dim * 2,
                hidden_dim,
            ),
            nn.Tanh(),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    # ========================================================
    # Oracle delay distribution
    # ========================================================

    @staticmethod
    def oracle_delay_log_pmf(
        delay: int,
        ref_tensor: torch.Tensor,
    ):
        """
        Synthetic generator:

            E ~ Exponential(scale=4)
            D = max(1, int(E))

        D must be >= 1.

        Return:
            log P(D = delay)
        """

        if delay < 1:
            return ref_tensor.new_tensor(
                -torch.inf
            )

        mu = DELAY_MEAN

        # ----------------------------------------------------
        # Because:
        #
        # max(1, int(E))
        #
        # delay=1 contains:
        #     0 <= E < 2
        # ----------------------------------------------------

        if delay == 1:

            probability = (
                1.0
                - math.exp(
                    -2.0 / mu
                )
            )

        else:

            probability = (
                math.exp(
                    -delay / mu
                )
                -
                math.exp(
                    -(delay + 1) / mu
                )
            )

        probability = max(
            probability,
            1e-30,
        )

        return ref_tensor.new_tensor(
            math.log(
                probability
            )
        )

    @staticmethod
    def oracle_delay_log_survival(
        observed_steps_after_onset: int,
        ref_tensor: torch.Tensor,
    ):
        """
        Return:

            log P(D > observed_steps_after_onset)

        Example:

        onset at k,
        observation ends at T.

        If user is still undetected at T:

            D > T-k
        """

        h = int(
            observed_steps_after_onset
        )

        if h < 1:

            probability = 1.0

        else:

            # D > h
            #
            # For D=max(1,floor(E)),
            # this means E >= h+1.
            probability = math.exp(
                -(h + 1)
                / DELAY_MEAN
            )

        probability = max(
            probability,
            1e-30,
        )

        return ref_tensor.new_tensor(
            math.log(
                probability
            )
        )

    # ========================================================
    # Strict causal forward
    # ========================================================

    def forward(
        self,
        x,
        lengths,
        endpoints,
    ):

        hidden = self.encoder(
            x,
            lengths,
        )

        batch_size = (
            hidden.shape[0]
        )

        seq_len = (
            hidden.shape[1]
        )

        sequence_mask = time_mask(
            lengths,
            seq_len,
        )

        steps = torch.arange(
            seq_len,
            device=x.device,
        )[None, :]

        endpoint_mask = (
            steps
            <= endpoints[:, None]
        )

        valid_mask = (
            sequence_mask
            & endpoint_mask
        )

        batch_indices = torch.arange(
            batch_size,
            device=x.device,
        )

        endpoint_hidden = hidden[
            batch_indices,
            endpoints,
        ]

        # ====================================================
        # Occurrence
        #
        # 是否发生 fraud
        # ====================================================

        occurrence_logit = (
            self.occurrence_head(
                endpoint_hidden
            ).squeeze(-1)
        )

        log_pi = F.logsigmoid(
            occurrence_logit
        )

        log_one_minus_pi = (
            F.logsigmoid(
                -occurrence_logit
            )
        )

        occurrence_probability = (
            occurrence_logit.sigmoid()
        )

        # ====================================================
        # Timing
        #
        # fraud 什么时候发生
        # ====================================================

        endpoint_expanded = (
            endpoint_hidden[
                :,
                None,
                :,
            ]
            .expand(
                -1,
                seq_len,
                -1,
            )
        )

        timing_pair = torch.cat(
            [
                hidden,
                endpoint_expanded,
            ],
            dim=-1,
        )

        timing_logits = (
            self.endpoint_timing_head(
                timing_pair
            ).squeeze(-1)
        )

        timing_logits = (
            timing_logits.masked_fill(
                ~valid_mask,
                -torch.inf,
            )
        )

        log_timing = F.log_softmax(
            timing_logits,
            dim=1,
        )

        timing_probability = (
            log_timing.exp()
        )

        # ====================================================
        # Detection head is kept for architectural consistency,
        # but the latent likelihood below uses the ORACLE
        # delay distribution instead.
        # ====================================================

        detection_logits = (
            self.detection_head(
                hidden
            ).squeeze(-1)
        )

        detection_hazard = (
            detection_logits.sigmoid()
        )

        detection_hazard = (
            detection_hazard.masked_fill(
                ~valid_mask,
                0.0,
            )
        )

        return {
            "hidden":
                hidden,

            "endpoint_hidden":
                endpoint_hidden,

            "valid_mask":
                valid_mask,

            "occurrence_logit":
                occurrence_logit,

            "log_pi":
                log_pi,

            "log_one_minus_pi":
                log_one_minus_pi,

            "occurrence_probability":
                occurrence_probability,

            "timing_logits":
                timing_logits,

            "log_timing":
                log_timing,

            "timing_probability":
                timing_probability,

            "detection_logits":
                detection_logits,

            "detection_hazard":
                detection_hazard,
        }

    # ========================================================
    # Oracle-delay latent joint
    # ========================================================

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
                        "Invalid detected t_d."
                    )

                # ---------------------------------------------
                # Candidate onset:
                #
                #     k < d
                #
                # because true delay >= 1.
                # ---------------------------------------------

                for k in range(
                    d
                ):

                    delay = (
                        d - k
                    )

                    log_delay = (
                        self.oracle_delay_log_pmf(
                            delay,
                            out["log_pi"][i],
                        )
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
                        log_delay
                    )

                # Detected => no-fraud impossible
                state_log_mass[
                    max_length
                ] = -torch.inf

            # =================================================
            # UNDETECTED USER
            # =================================================

            else:

                horizon = (
                    length_i - 1
                )

                # ---------------------------------------------
                # Possibility 1:
                # fraud occurred at k,
                # but detection delay extends beyond T.
                # ---------------------------------------------

                for k in range(
                    length_i
                ):

                    observed_after_onset = (
                        horizon - k
                    )

                    log_survival = (
                        self.oracle_delay_log_survival(
                            observed_after_onset,
                            out["log_pi"][i],
                        )
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
                        log_survival
                    )

                # ---------------------------------------------
                # Possibility 2:
                # no fraud occurred.
                # ---------------------------------------------

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

        log_likelihood = (
            torch.logsumexp(
                joint,
                dim=1,
            )
        )

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

    # ========================================================
    # Loss
    # ========================================================

    def loss(
        self,
        batch,
        inference="exact",
        em_samples=1,
        training=True,
    ):

        detected = batch[
            "detected"
        ].bool()

        endpoints = torch.where(
            detected,
            batch["t_detect"],
            batch["lengths"] - 1,
        )

        out = self.forward(
            batch["x"],
            batch["lengths"],
            endpoints,
        )

        (
            joint,
            log_likelihood,
            posterior,
        ) = self.latent_joint(
            batch,
            out,
        )

        # ====================================================
        # Exact / stochastic EM
        # ====================================================

        if (
            training
            and inference
            == "stochastic_em"
        ):

            samples = torch.multinomial(
                posterior.detach(),
                em_samples,
                replacement=True,
            )

            selected = joint.gather(
                1,
                samples,
            )

            train_loss = (
                -selected.mean()
            )

        else:

            train_loss = (
                -log_likelihood.mean()
            )

        max_length = out[
            "timing_probability"
        ].shape[1]

        onset_posterior = (
            posterior[
                :,
                :max_length,
            ]
        )

        no_commission_posterior = (
            posterior[
                :,
                max_length,
            ]
        )

        out.update(
            observed_endpoint=
                endpoints,

            latent_joint=
                joint,

            log_likelihood=
                log_likelihood,

            latent_posterior=
                posterior,

            onset_posterior=
                onset_posterior,

            no_commission_posterior=
                no_commission_posterior,
        )

        return (
            train_loss,
            out,
        )


# ============================================================
# Replace base model
# ============================================================


exp.LatentDecoupledJointModel = (
    OracleDelayPriorModel
)

exp.DATA_PATH = DATA_PATH

exp.OUTPUT_DIR = OUTPUT_DIR

exp.RUN_ROOT = RUN_ROOT


# ============================================================
# Experimental contract
# ============================================================


print(
    "=" * 80
)

print(
    "EXPERIMENT 03O: ORACLE DELAY PRIOR"
)

print(
    "=" * 80
)

print(
    "Oracle = 上帝视角诊断，不是最终方法。"
)

print()

print(
    "The model DOES NOT receive any user's true t_f."
)

print(
    "模型不知道任何具体用户的真实 t_f。"
)

print()

print(
    "Oracle information:"
)

print(
    "  synthetic delay generation rule:"
)

print(
    "  delay = max(1, int(Exponential(scale=4)))"
)

print()

print(
    "Strict causal rule:"
)

print(
    "  detected users only use behavior up to t_d"
)

print(
    "  no behavior after t_d is used"
)

print()

print(
    "Question:"
)

print(
    "  If the delay mechanism is known correctly,"
)

print(
    "  can latent t_f be recovered?"
)

print(
    "=" * 80
)

print()


# ============================================================
# Train Exact + Stochastic EM
# ============================================================


exp.main()


# ============================================================
# Compare against Minimum-delay model
# ============================================================


NEW_RESULT = (
    OUTPUT_DIR
    / "summary.json"
)


if (
    REFERENCE_RESULT.exists()
    and NEW_RESULT.exists()
):

    old = json.loads(
        REFERENCE_RESULT.read_text(
            encoding="utf-8"
        )
    )

    new = json.loads(
        NEW_RESULT.read_text(
            encoding="utf-8"
        )
    )

    print()

    print(
        "=" * 80
    )

    print(
        "MINIMUM-DELAY vs ORACLE-DELAY"
    )

    print(
        "=" * 80
    )

    for inference in (
        "exact",
        "stochastic_em",
    ):

        print()

        print(
            inference
        )

        print(
            "  posterior mean MAE:",
            old[
                inference
            ][
                "posterior_mean_mae"
            ],
            "->",
            new[
                inference
            ][
                "posterior_mean_mae"
            ],
        )

        print(
            "  posterior mode MAE:",
            old[
                inference
            ][
                "posterior_mode_mae"
            ],
            "->",
            new[
                inference
            ][
                "posterior_mode_mae"
            ],
        )

        print(
            "  sampled tf MAE:",
            old[
                inference
            ][
                "sampled_tf_mae_given_commission"
            ],
            "->",
            new[
                inference
            ][
                "sampled_tf_mae_given_commission"
            ],
        )

        print(
            "  detected fraud mean MAE:",
            old[
                inference
            ][
                "detected_fraud_mean_mae"
            ],
            "->",
            new[
                inference
            ][
                "detected_fraud_mean_mae"
            ],
        )

        print(
            "  P(true tf | commission):",
            old[
                inference
            ][
                "mean_true_tf_probability_given_commission"
            ],
            "->",
            new[
                inference
            ][
                "mean_true_tf_probability_given_commission"
            ],
        )


# ============================================================
# Delay-distribution audit
# ============================================================


print()

print(
    "=" * 80
)

print(
    "ORACLE DELAY DISTRIBUTION AUDIT"
)

print(
    "=" * 80
)


for inference in (
    "exact",
    "stochastic_em",
):

    csv_path = (
        OUTPUT_DIR
        / inference
        / "entity_diagnostics.csv"
    )

    if not csv_path.exists():

        print(
            f"Missing: {csv_path}"
        )

        continue

    df = pd.read_csv(
        csv_path
    )

    fraud = df[
        (
            df["true_tf"] >= 0
        )
        &
        (
            df["detected"] == 1
        )
    ].copy()

    fraud[
        "true_delay"
    ] = (
        fraud[
            "observed_td"
        ]
        - fraud[
            "true_tf"
        ]
    )

    fraud[
        "mode_delay"
    ] = (
        fraud[
            "observed_td"
        ]
        - fraud[
            "posterior_mode_tf"
        ]
    )

    fraud[
        "mean_delay"
    ] = (
        fraud[
            "observed_td"
        ]
        - fraud[
            "posterior_mean_tf"
        ]
    )

    print()

    print(
        inference
    )

    print(
        "  true delay mean:",
        float(
            fraud[
                "true_delay"
            ].mean()
        ),
    )

    print(
        "  predicted mode delay mean:",
        float(
            fraud[
                "mode_delay"
            ].mean()
        ),
    )

    print(
        "  predicted posterior-mean delay:",
        float(
            fraud[
                "mean_delay"
            ].mean()
        ),
    )

    print(
        "  mode delay == 1 rate:",
        float(
            (
                fraud[
                    "mode_delay"
                ]
                == 1
            ).mean()
        ),
    )

    print(
        "  mode delay <= 2 rate:",
        float(
            (
                fraud[
                    "mode_delay"
                ]
                <= 2
            ).mean()
        ),
    )

    print()

    print(
        "  predicted mode delay distribution:"
    )

    print(
        fraud[
            "mode_delay"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    fraud.to_csv(
        OUTPUT_DIR
        / inference
        / "oracle_delay_audit.csv",
        index=False,
    )


print()

print(
    "Saved:"
)

print(
    NEW_RESULT
)