from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from safeplus.models.common import time_mask


# ============================================================
# Experiment 03K
# Endpoint-conditioned strict-causal timing
#
# 中文：
# 基于检测前完整证据的严格因果 t_f 定位。
#
#
# 核心原则：
#
# Detected user:
#   observed endpoint = t_d
#
# 模型可以使用：
#   x_0 ... x_td
#
# 模型绝不能使用：
#   x_{td+1} ... x_T
#
#
# 当判断候选 t_f=k 时：
#
#   不只看 h_k
#
#   还同时看 h_td
#
# h_k:
#   候选起点 k 当时的局部状态
#
# h_td:
#   截止检测时间 t_d 的完整历史总结
#
#
# True t_f NEVER participates in training.
# True t_f is used only for post-hoc evaluation.
# ============================================================


BASE_SCRIPT = Path(
    r"scripts/10_exp03_latent_decoupled_joint.py"
)

DATA_PATH = Path(
    r"data/processed/synthetic_same_delay_consistent.npz"
)

OUTPUT_DIR = Path(
    r"reports/experiment03/"
    r"endpoint_conditioned_same_delay_consistent"
)

RUN_ROOT = Path(
    r"runs/synthetic/exp03/"
    r"endpoint-conditioned-same-delay-consistent"
)

NON_STRICT_RESULT = Path(
    r"reports/experiment03/"
    r"latent_decoupled_same_delay_consistent/"
    r"summary.json"
)

STRICT_RESULT = Path(
    r"reports/experiment03/"
    r"strict_causal_same_delay_consistent/"
    r"summary.json"
)


# ============================================================
# Load Experiment 03F base implementation
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
        "Could not load Experiment 03F."
    )


exp = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    exp
)


# ============================================================
# Endpoint-conditioned model
# ============================================================


class EndpointConditionedLatentModel(
    exp.LatentDecoupledJointModel
):
    """
    Strict-causal latent model.

    Difference from previous strict model:

    Previous:
        timing score for candidate k mainly used h_k.

    Current:
        timing score for candidate k uses BOTH:

            h_k
            h_endpoint

    where endpoint is:
        t_d for detected users
        T   for undetected users.

    No information after endpoint is used.
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

        self.hidden_dim = (
            hidden_dim
        )

        # ----------------------------------------------------
        # Pairwise timing scorer
        #
        # Pairwise = 成对关系打分。
        #
        # Input:
        #   candidate hidden h_k
        #   endpoint hidden h_endpoint
        #
        # A nonlinear layer is necessary so that endpoint
        # information can actually change the relative score
        # among different candidate times.
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

    def forward(
        self,
        x,
        lengths,
        endpoints,
    ):
        # ====================================================
        # Causal GRU
        #
        # h_t depends only on x_0 ... x_t.
        # ====================================================

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

        # ====================================================
        # Strict causal observation mask
        #
        # No candidate time after the observed endpoint
        # is allowed.
        # ====================================================

        endpoint_mask = (
            steps
            <= endpoints[:, None]
        )

        valid_mask = (
            sequence_mask
            & endpoint_mask
        )

        # ====================================================
        # Endpoint representation
        #
        # Detected:
        #   endpoint = t_d
        #
        # Undetected:
        #   endpoint = T
        #
        # This hidden vector summarizes all observed evidence
        # up to the endpoint, but nothing after it.
        # ====================================================

        batch_indices = torch.arange(
            batch_size,
            device=x.device,
        )

        endpoint_hidden = hidden[
            batch_indices,
            endpoints,
        ]

        # ====================================================
        # 1. Occurrence head
        #
        # Occurrence = 是否发生欺诈
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
        # 2. Endpoint-conditioned Timing
        #
        # Timing = 欺诈什么时候开始
        #
        # For each candidate k:
        #
        #   candidate evidence:
        #       h_k
        #
        #   full observed-prefix evidence:
        #       h_endpoint
        #
        # They are considered together.
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

        # Candidate t_f cannot occur after the observed endpoint.
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
        # Detection = 欺诈发生以后什么时候被检测
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

        # ====================================================
        # Observed endpoint
        # ====================================================

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

        # ====================================================
        # Strict-causal forward
        # ====================================================

        out = self.forward(
            batch["x"],
            batch["lengths"],
            endpoints,
        )

        # ====================================================
        # Reuse SAME latent joint likelihood
        # ====================================================

        (
            joint,
            log_likelihood,
            posterior,
        ) = self.latent_joint(
            batch,
            out,
        )

        # ====================================================
        # Exact / Sampling
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
# Replace model used by base training pipeline
# ============================================================


exp.LatentDecoupledJointModel = (
    EndpointConditionedLatentModel
)


# ============================================================
# Dataset / outputs
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
    "EXPERIMENT 03K: ENDPOINT-CONDITIONED TIMING"
)

print(
    "=" * 80
)

print(
    "Endpoint = 观察终点；detected 用户中就是 t_d。"
)

print(
    "Conditioned = 以某些已观察信息作为条件。"
)

print(
    "Timing = 定位 fraud 的开始时间 t_f。"
)

print()

print(
    "CORE RULE:"
)

print(
    "  Candidate t_f=k may use all observed evidence "
    "from x_0 through x_td."
)

print(
    "  Candidate t_f=k may NOT use any evidence after t_d."
)

print()

print(
    "Detected user:"
)

print(
    "  allowed: x_0 ... x_td"
)

print(
    "  forbidden: x_{td+1} ... x_T"
)

print()

print(
    "True t_f remains completely hidden during training."
)

print()

print(
    "UNCHANGED:"
)

print(
    "  dataset"
)

print(
    "  true t_f"
)

print(
    "  observed t_d"
)

print(
    "  split"
)

print(
    "  GRU encoder"
)

print(
    "  occurrence head"
)

print(
    "  detection head"
)

print(
    "  latent joint likelihood"
)

print(
    "  Exact inference"
)

print(
    "  stochastic EM sampling"
)

print(
    "  optimizer / learning rate / batch size"
)

print()

print(
    "CHANGED:"
)

print(
    "  timing candidate k is scored together with "
    "the endpoint representation."
)

print(
    "=" * 80
)

print()


# ============================================================
# Run
# ============================================================


exp.main()


# ============================================================
# Compare three versions
# ============================================================


NEW_RESULT = (
    OUTPUT_DIR
    / "summary.json"
)


def load_json(
    path,
):
    if not path.exists():
        return None

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


non_strict = load_json(
    NON_STRICT_RESULT
)

strict = load_json(
    STRICT_RESULT
)

endpoint = load_json(
    NEW_RESULT
)


if endpoint is not None:

    print()

    print(
        "=" * 80
    )

    print(
        "THREE-WAY COMPARISON"
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

        if non_strict is not None:

            print(
                "  Non-strict posterior mean MAE:",
                non_strict[
                    inference
                ][
                    "posterior_mean_mae"
                ],
            )

        if strict is not None:

            print(
                "  Strict-local posterior mean MAE:",
                strict[
                    inference
                ][
                    "posterior_mean_mae"
                ],
            )

        print(
            "  Endpoint-conditioned posterior mean MAE:",
            endpoint[
                inference
            ][
                "posterior_mean_mae"
            ],
        )

        print()

        if non_strict is not None:

            print(
                "  Non-strict sampled tf MAE:",
                non_strict[
                    inference
                ][
                    "sampled_tf_mae_given_commission"
                ],
            )

        if strict is not None:

            print(
                "  Strict-local sampled tf MAE:",
                strict[
                    inference
                ][
                    "sampled_tf_mae_given_commission"
                ],
            )

        print(
            "  Endpoint-conditioned sampled tf MAE:",
            endpoint[
                inference
            ][
                "sampled_tf_mae_given_commission"
            ],
        )

        print()

        print(
            "  Endpoint-conditioned mode MAE:",
            endpoint[
                inference
            ][
                "posterior_mode_mae"
            ],
        )

        print(
            "  Endpoint-conditioned P(true tf | commission):",
            endpoint[
                inference
            ][
                "mean_true_tf_probability_given_commission"
            ],
        )

        print(
            "  Endpoint-conditioned detected-fraud MAE:",
            endpoint[
                inference
            ][
                "detected_fraud_mean_mae"
            ],
        )

        print(
            "  Endpoint-conditioned occurrence AUROC:",
            endpoint[
                inference
            ][
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