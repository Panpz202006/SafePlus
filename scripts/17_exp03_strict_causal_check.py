from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from safeplus.models.common import time_mask


# ============================================================
# Experiment 03J
# Strict-Causal Latent Decoupled Joint Inference
#
# 中文：
# 严格因果版 latent（潜变量）解耦联合推断。
#
# 目的：
# 排除 detection（检测）发生之后的未来行为对
# latent t_f（隐藏欺诈开始时间）推断造成的信息泄露。
#
#
# Detected user:
#
#   observed endpoint = t_d
#
#   occurrence head:
#       only summarizes x_0 ... x_td
#
#   timing head:
#       only normalizes candidate t_f over 0 ... t_d
#
#   detection head:
#       only contributes through t_d
#
#
# Undetected user:
#
#   endpoint = T
#
#   full observed sequence up to T is allowed.
#
#
# True t_f is NEVER used in training.
# It is used only after training for synthetic evaluation.
# ============================================================


BASE_SCRIPT = Path(
    r"scripts/10_exp03_latent_decoupled_joint.py"
)

DATA_PATH = Path(
    r"data/processed/synthetic_same_delay_consistent.npz"
)

OUTPUT_DIR = Path(
    r"reports/experiment03/"
    r"strict_causal_same_delay_consistent"
)

RUN_ROOT = Path(
    r"runs/synthetic/exp03/"
    r"strict-causal-same-delay-consistent"
)

OLD_RESULT = Path(
    r"reports/experiment03/"
    r"latent_decoupled_same_delay_consistent/"
    r"summary.json"
)


# ============================================================
# Load the original latent-decoupled experiment implementation
# ============================================================


if not BASE_SCRIPT.exists():
    raise FileNotFoundError(
        f"Missing base script: {BASE_SCRIPT}"
    )


spec = importlib.util.spec_from_file_location(
    "exp03_latent_decoupled_base",
    BASE_SCRIPT,
)

if (
    spec is None
    or spec.loader is None
):
    raise RuntimeError(
        "Could not load Experiment 03F base script."
    )


exp = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    exp
)


# ============================================================
# Strict causal model
# ============================================================


class StrictCausalLatentDecoupledJointModel(
    exp.LatentDecoupledJointModel
):
    """
    Same model family as Experiment 03F.

    Only difference:
    for a detected user, anything after t_detect is prevented
    from influencing occurrence/timing inference.
    """

    def forward(
        self,
        x,
        lengths,
        endpoints,
    ):
        # ----------------------------------------------------
        # GRU itself remains causal.
        #
        # hidden[:, t] only depends on x_0 ... x_t.
        # ----------------------------------------------------

        hidden = self.encoder(
            x,
            lengths,
        )

        batch_size = hidden.shape[0]
        seq_len = hidden.shape[1]

        # ----------------------------------------------------
        # Original valid sequence mask
        # ----------------------------------------------------

        sequence_mask = time_mask(
            lengths,
            seq_len,
        )

        # ----------------------------------------------------
        # Strict observation mask
        #
        # detected:
        #     endpoint = t_detect
        #
        # undetected:
        #     endpoint = T
        # ----------------------------------------------------

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

        # ====================================================
        # 1. Occurrence head
        #
        # Occurrence = 是否发生 fraud
        #
        # OLD:
        #     hidden at the full sequence end.
        #
        # NEW:
        #     hidden at the observed endpoint only.
        # ====================================================

        batch_indices = torch.arange(
            batch_size,
            device=x.device,
        )

        endpoint_hidden = hidden[
            batch_indices,
            endpoints,
        ]

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
        # 2. Timing head
        #
        # Timing = 如果 fraud，什么时候发生
        #
        # Only candidate times up to the observed endpoint
        # participate in Softmax normalization.
        # ====================================================

        timing_logits = (
            self.timing_head(
                hidden
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
        # 3. Detection head
        #
        # Detection = 欺诈发生后什么时候被检测
        #
        # Only states up to endpoint are relevant.
        # ====================================================

        detection_logits = (
            self.detection_head(
                hidden
            ).squeeze(-1)
        )

        log_detection_hazard = (
            F.logsigmoid(
                detection_logits
            )
        )

        log_detection_survival = (
            F.logsigmoid(
                -detection_logits
            )
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

            "log_detection_hazard":
                log_detection_hazard,

            "log_detection_survival":
                log_detection_survival,

            "detection_hazard":
                detection_hazard,
        }

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

        # ----------------------------------------------------
        # Strict observed endpoint
        # ----------------------------------------------------

        endpoints = torch.where(
            detected,
            batch["t_detect"],
            batch["lengths"] - 1,
        )

        if (
            (endpoints < 0)
            | (
                endpoints
                >= batch["lengths"]
            )
        ).any():
            raise ValueError(
                "Invalid observed endpoint."
            )

        # ----------------------------------------------------
        # Forward pass with strict causal endpoint
        # ----------------------------------------------------

        out = self.forward(
            batch["x"],
            batch["lengths"],
            endpoints,
        )

        # ----------------------------------------------------
        # Reuse the SAME latent joint likelihood from
        # Experiment 03F.
        # ----------------------------------------------------

        (
            joint,
            log_likelihood,
            posterior,
        ) = self.latent_joint(
            batch,
            out,
        )

        # ====================================================
        # Exact or stochastic-EM training
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
# Replace ONLY the model class used by the original pipeline
# ============================================================


exp.LatentDecoupledJointModel = (
    StrictCausalLatentDecoupledJointModel
)


# ============================================================
# Same corrected dataset and same hyperparameters
# ============================================================


exp.DATA_PATH = DATA_PATH

exp.OUTPUT_DIR = OUTPUT_DIR

exp.RUN_ROOT = RUN_ROOT


# ============================================================
# Explain experimental contract
# ============================================================


print(
    "=" * 80
)

print(
    "EXPERIMENT 03J: STRICT CAUSAL CHECK"
)

print(
    "=" * 80
)

print(
    "Strict Causal = 严格因果。"
)

print(
    "Information Leakage = 信息泄露。"
)

print()

print(
    "Dataset:",
    DATA_PATH,
)

print()

print(
    "UNCHANGED:"
)

print(
    "  synthetic behavior x"
)

print(
    "  hidden true t_f"
)

print(
    "  observed t_d"
)

print(
    "  train/validation/test split"
)

print(
    "  GRU encoder"
)

print(
    "  occurrence/timing/detection heads"
)

print(
    "  latent likelihood"
)

print(
    "  exact inference"
)

print(
    "  stochastic EM"
)

print(
    "  optimizer and hyperparameters"
)

print()

print(
    "ONLY CHANGE:"
)

print(
    "  For detected users, no information after t_d "
    "may influence occurrence or timing inference."
)

print()

print(
    "Detected user with t_d=d:"
)

print(
    "  model may use x_0 ... x_d"
)

print(
    "  model may NOT use x_{d+1} ... x_T"
)

print()

print(
    "True t_f remains hidden from training."
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
# Compare old non-strict vs strict-causal results
# ============================================================


NEW_RESULT = (
    OUTPUT_DIR
    / "summary.json"
)


if (
    OLD_RESULT.exists()
    and NEW_RESULT.exists()
):

    old = json.loads(
        OLD_RESULT.read_text(
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
        "NON-STRICT vs STRICT-CAUSAL COMPARISON"
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

        old_result = old[
            inference
        ]

        new_result = new[
            inference
        ]

        print(
            "  posterior mean MAE:",
            old_result[
                "posterior_mean_mae"
            ],
            "->",
            new_result[
                "posterior_mean_mae"
            ],
        )

        print(
            "  posterior mode MAE:",
            old_result[
                "posterior_mode_mae"
            ],
            "->",
            new_result[
                "posterior_mode_mae"
            ],
        )

        print(
            "  sampled tf MAE:",
            old_result[
                "sampled_tf_mae_given_commission"
            ],
            "->",
            new_result[
                "sampled_tf_mae_given_commission"
            ],
        )

        print(
            "  detected fraud mean MAE:",
            old_result[
                "detected_fraud_mean_mae"
            ],
            "->",
            new_result[
                "detected_fraud_mean_mae"
            ],
        )

        print(
            "  occurrence AUROC:",
            old_result[
                "occurrence_auroc"
            ],
            "->",
            new_result[
                "occurrence_auroc"
            ],
        )

        print(
            "  P(true tf | commission):",
            old_result[
                "mean_true_tf_probability_given_commission"
            ],
            "->",
            new_result[
                "mean_true_tf_probability_given_commission"
            ],
        )


print()

print(
    "Saved:"
)

print(
    NEW_RESULT
)