from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.nn import functional as F

from safeplus.data import load_npz, make_loaders
from safeplus.models.common import SequenceEncoder, time_mask
from safeplus.utils import seed_everything


# ============================================================
# Experiment 03F
# Latent Decoupled Joint Inference
#
# IMPORTANT:
#
# TRUE t_f IS NOT USED IN TRAINING.
#
# The model observes only:
#
#   x
#   detected
#   t_detect
#
# The hidden synthetic t_onset is used ONLY after training for
# diagnostic evaluation.
#
#
# Model:
#
#   Shared causal GRU
#
#       ├── occurrence head:
#       │     pi(x) = P(fraud occurs within horizon | x)
#       │
#       ├── timing head:
#       │     q(t_f = t | fraud, x)
#       │
#       └── detection head:
#             h_d(t)
#
#
# Detected entity:
#
#   p(t_d=d | x)
#   =
#   pi(x)
#   sum_{k <= d}
#       q(k | fraud,x)
#       p(t_d=d | t_f=k,x)
#
#
# Censored / undetected entity:
#
#   p(t_d > T | x)
#   =
#   (1-pi(x))
#   +
#   pi(x)
#   sum_k
#       q(k | fraud,x)
#       P(t_d>T | t_f=k,x)
#
#
# Two training variants:
#
#   1. exact
#      marginalize all possible latent t_f
#
#   2. stochastic_em
#      compute posterior p(t_f | observation,x),
#      sample latent state, and optimize complete-data log mass
#
# ============================================================


DATA_PATH = Path(
    r"data/processed/synthetic_demo.npz"
)

OUTPUT_DIR = Path(
    r"reports/experiment03/latent_decoupled_joint"
)

RUN_ROOT = Path(
    r"runs/synthetic/exp03/latent-decoupled-joint"
)

SEED = 0
DEVICE = "cpu"

HIDDEN_DIM = 32
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

MAX_EPOCHS = 40
PATIENCE = 7

SPLIT = [0.7, 0.1, 0.2]

EM_SAMPLES = 1

DIAGNOSTIC_SAMPLES = 1000

DIAGNOSTIC_SAMPLE_SEED = 20261009


# ============================================================
# Model
# ============================================================


class LatentDecoupledJointModel(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.encoder = SequenceEncoder(
            input_dim,
            hidden_dim,
        )

        # ----------------------------------------------------
        # Does fraud occur within the observed horizon?
        # ----------------------------------------------------

        self.occurrence_head = nn.Linear(
            hidden_dim,
            1,
        )

        # ----------------------------------------------------
        # Conditional timing:
        #
        # q(t_f=t | fraud,x)
        # ----------------------------------------------------

        self.timing_head = nn.Linear(
            hidden_dim,
            1,
        )

        # ----------------------------------------------------
        # Detection hazard h_d(t)
        # ----------------------------------------------------

        self.detection_head = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(
        self,
        x,
        lengths,
    ):
        hidden = self.encoder(
            x,
            lengths,
        )

        batch_size = hidden.shape[0]
        seq_len = hidden.shape[1]

        valid_mask = time_mask(
            lengths,
            seq_len,
        )

        # ====================================================
        # Occurrence
        # ====================================================

        batch_indices = torch.arange(
            batch_size,
            device=hidden.device,
        )

        last_hidden = hidden[
            batch_indices,
            lengths - 1,
        ]

        occurrence_logit = (
            self.occurrence_head(
                last_hidden
            ).squeeze(-1)
        )

        log_pi = F.logsigmoid(
            occurrence_logit
        )

        log_one_minus_pi = F.logsigmoid(
            -occurrence_logit
        )

        occurrence_probability = (
            occurrence_logit.sigmoid()
        )

        # ====================================================
        # Timing
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
        # Detection
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

    # ========================================================
    # Construct latent joint states
    #
    # Output shape:
    #
    #   [B, L+1]
    #
    # states 0 ... L-1:
    #     fraud occurs at t_f = k
    #
    # final state L:
    #     no fraud occurs inside horizon
    #
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

            # -----------------------------------------------
            # Initialize all onset states to impossible.
            # -----------------------------------------------

            state_log_mass = out[
                "log_pi"
            ].new_full(
                (max_length + 1,),
                -torch.inf,
            )

            # -----------------------------------------------
            # DETECTED
            #
            # Detection at d implies fraud must have occurred
            # at some k <= d.
            # -----------------------------------------------

            if detected_i:

                d = int(
                    t_detect[i].item()
                )

                if (
                    d < 0
                    or d >= length_i
                ):
                    raise ValueError(
                        "Invalid detection time."
                    )

                for k in range(
                    d + 1
                ):

                    # Detection survival:
                    #
                    # k ... d-1
                    if k < d:

                        log_survive_detection = (
                            out[
                                "log_detection_survival"
                            ][
                                i,
                                k:d,
                            ].sum()
                        )

                    else:

                        log_survive_detection = (
                            out[
                                "log_pi"
                            ].new_zeros(())
                        )

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

                # No-fraud state is impossible if detection
                # was observed.
                state_log_mass[
                    max_length
                ] = -torch.inf

            # -----------------------------------------------
            # UNDETECTED / CENSORED
            #
            # Two possible explanations:
            #
            # A) fraud never occurred
            #
            # B) fraud occurred at k, but detection did not
            #    happen through observation horizon.
            # -----------------------------------------------

            else:

                horizon = (
                    length_i - 1
                )

                for k in range(
                    length_i
                ):

                    # Survival of detection from k THROUGH T.
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

                # No fraud inside the horizon.
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
            - log_likelihood[:, None]
        ).exp()

        return (
            joint,
            log_likelihood,
            posterior,
        )

    # ========================================================
    # Training / evaluation loss
    # ========================================================

    def loss(
        self,
        batch,
        inference="exact",
        em_samples=1,
        training=True,
    ):
        out = self(
            batch["x"],
            batch["lengths"],
        )

        (
            joint,
            log_likelihood,
            posterior,
        ) = self.latent_joint(
            batch,
            out,
        )

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
# Utilities
# ============================================================


def move_batch(
    batch,
):
    return {
        key: value.to(
            DEVICE
        )
        for key, value
        in batch.items()
    }


def safe_mean(
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:
        return float("nan")

    return float(
        values.mean()
    )


def safe_auc(
    function,
    labels,
    scores,
):
    if len(
        np.unique(
            labels
        )
    ) < 2:
        return float("nan")

    return float(
        function(
            labels,
            scores,
        )
    )


# ============================================================
# Epoch runner
# ============================================================


def run_epoch(
    model,
    loader,
    inference,
    optimizer=None,
):
    training = (
        optimizer
        is not None
    )

    model.train(
        training
    )

    total_training_loss = 0.0
    total_exact_nll = 0.0
    total_count = 0

    occurrence_probabilities = []

    onset_posteriors = []

    no_commission_posteriors = []

    latent_posteriors = []

    detected_all = []

    t_detect_all = []

    t_onset_all = []

    lengths_all = []

    for raw_batch in loader:

        batch = move_batch(
            raw_batch
        )

        with torch.set_grad_enabled(
            training
        ):

            train_loss, out = (
                model.loss(
                    batch,
                    inference=inference,
                    em_samples=
                        EM_SAMPLES,
                    training=
                        training,
                )
            )

            exact_nll = (
                -out[
                    "log_likelihood"
                ].mean()
            )

            if training:

                optimizer.zero_grad()

                train_loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    5.0,
                )

                optimizer.step()

        n = len(
            batch["x"]
        )

        total_training_loss += (
            float(
                train_loss.detach()
            )
            * n
        )

        total_exact_nll += (
            float(
                exact_nll.detach()
            )
            * n
        )

        total_count += n

        occurrence_probabilities.append(
            out[
                "occurrence_probability"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        onset_posteriors.append(
            out[
                "onset_posterior"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        no_commission_posteriors.append(
            out[
                "no_commission_posterior"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        latent_posteriors.append(
            out[
                "latent_posterior"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        detected_all.append(
            batch[
                "detected"
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
        )

        t_detect_all.append(
            batch[
                "t_detect"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        t_onset_all.append(
            batch[
                "t_onset"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        lengths_all.append(
            batch[
                "lengths"
            ]
            .detach()
            .cpu()
            .numpy()
        )

    return {
        "training_loss":
            total_training_loss
            / total_count,

        "exact_nll":
            total_exact_nll
            / total_count,

        "occurrence_probability":
            np.concatenate(
                occurrence_probabilities
            ),

        "onset_posterior":
            np.concatenate(
                onset_posteriors
            ),

        "no_commission_posterior":
            np.concatenate(
                no_commission_posteriors
            ),

        "latent_posterior":
            np.concatenate(
                latent_posteriors
            ),

        "detected":
            np.concatenate(
                detected_all
            ),

        "t_detect":
            np.concatenate(
                t_detect_all
            ),

        "t_onset":
            np.concatenate(
                t_onset_all
            ),

        "lengths":
            np.concatenate(
                lengths_all
            ),
    }


# ============================================================
# Posterior diagnostics
#
# IMPORTANT:
# t_onset is used HERE only.
# It is NOT used in the model loss.
# ============================================================


def evaluate_posterior(
    payload,
):
    posterior = payload[
        "onset_posterior"
    ]

    no_commission = payload[
        "no_commission_posterior"
    ]

    t_onset = payload[
        "t_onset"
    ]

    detected = payload[
        "detected"
    ]

    known_fraud = (
        t_onset >= 0
    )

    normal = (
        ~known_fraud
    )

    detected_fraud = (
        known_fraud
        & detected
    )

    censored_fraud = (
        known_fraud
        & ~detected
    )

    time_steps = np.arange(
        posterior.shape[1],
        dtype=np.float64,
    )

    onset_mass = (
        posterior.sum(
            axis=1
        )
    )

    # --------------------------------------------------------
    # Mode among onset states
    # --------------------------------------------------------

    posterior_mode = (
        posterior.argmax(
            axis=1
        )
    )

    # --------------------------------------------------------
    # Mean conditional on commission
    # --------------------------------------------------------

    posterior_mean = np.full(
        len(
            t_onset
        ),
        np.nan,
        dtype=np.float64,
    )

    valid_mass = (
        onset_mass > 0
    )

    posterior_mean[
        valid_mass
    ] = (
        posterior[
            valid_mass
        ]
        * time_steps[None, :]
    ).sum(
        axis=1
    ) / onset_mass[
        valid_mass
    ]

    # --------------------------------------------------------
    # True t_f probability
    # --------------------------------------------------------

    true_tf_probability = np.full(
        len(
            t_onset
        ),
        np.nan,
        dtype=np.float64,
    )

    conditional_true_tf_probability = np.full(
        len(
            t_onset
        ),
        np.nan,
        dtype=np.float64,
    )

    expected_abs_error = np.full(
        len(
            t_onset
        ),
        np.nan,
        dtype=np.float64,
    )

    known_indices = np.where(
        known_fraud
    )[0]

    for index in known_indices:

        tf = int(
            t_onset[
                index
            ]
        )

        true_tf_probability[
            index
        ] = posterior[
            index,
            tf,
        ]

        if (
            onset_mass[
                index
            ] > 0
        ):

            conditional = (
                posterior[
                    index
                ]
                / onset_mass[
                    index
                ]
            )

            conditional_true_tf_probability[
                index
            ] = conditional[
                tf
            ]

            expected_abs_error[
                index
            ] = np.sum(
                conditional
                * np.abs(
                    time_steps
                    - tf
                )
            )

    # --------------------------------------------------------
    # Actual posterior sampling
    # conditional on commission
    # --------------------------------------------------------

    torch.manual_seed(
        DIAGNOSTIC_SAMPLE_SEED
    )

    sampled_tf_mae = np.full(
        len(
            t_onset
        ),
        np.nan,
        dtype=np.float64,
    )

    sampled_tf_mean = np.full(
        len(
            t_onset
        ),
        np.nan,
        dtype=np.float64,
    )

    for index in known_indices:

        mass = onset_mass[
            index
        ]

        if mass <= 0:
            continue

        conditional = (
            posterior[
                index
            ]
            / mass
        )

        samples = torch.multinomial(
            torch.as_tensor(
                conditional,
                dtype=torch.float32,
            ),
            DIAGNOSTIC_SAMPLES,
            replacement=True,
        ).numpy()

        sampled_tf_mean[
            index
        ] = float(
            samples.mean()
        )

        sampled_tf_mae[
            index
        ] = float(
            np.abs(
                samples
                - t_onset[
                    index
                ]
            ).mean()
        )

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    mode_error = (
        posterior_mode
        - t_onset
    )

    mean_error = (
        posterior_mean
        - t_onset
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    metrics = {
        "n_known_fraud":
            int(
                known_fraud.sum()
            ),

        "n_normal":
            int(
                normal.sum()
            ),

        "n_detected_fraud":
            int(
                detected_fraud.sum()
            ),

        "n_censored_fraud":
            int(
                censored_fraud.sum()
            ),

        # ----------------------------------------------------
        # Mode
        # ----------------------------------------------------

        "posterior_mode_mae":
            safe_mean(
                np.abs(
                    posterior_mode[
                        known_fraud
                    ]
                    - t_onset[
                        known_fraud
                    ]
                )
            ),

        "posterior_mode_signed_error":
            safe_mean(
                mode_error[
                    known_fraud
                ]
            ),

        "posterior_mode_before_tf_rate":
            safe_mean(
                (
                    posterior_mode[
                        known_fraud
                    ]
                    < t_onset[
                        known_fraud
                    ]
                ).astype(
                    float
                )
            ),

        "posterior_mode_exact_tf_rate":
            safe_mean(
                (
                    posterior_mode[
                        known_fraud
                    ]
                    == t_onset[
                        known_fraud
                    ]
                ).astype(
                    float
                )
            ),

        "posterior_mode_after_tf_rate":
            safe_mean(
                (
                    posterior_mode[
                        known_fraud
                    ]
                    > t_onset[
                        known_fraud
                    ]
                ).astype(
                    float
                )
            ),

        # ----------------------------------------------------
        # Posterior mean
        # ----------------------------------------------------

        "posterior_mean_mae":
            safe_mean(
                np.abs(
                    posterior_mean[
                        known_fraud
                    ]
                    - t_onset[
                        known_fraud
                    ]
                )
            ),

        "posterior_mean_signed_error":
            safe_mean(
                mean_error[
                    known_fraud
                ]
            ),

        "detected_fraud_mean_mae":
            safe_mean(
                np.abs(
                    posterior_mean[
                        detected_fraud
                    ]
                    - t_onset[
                        detected_fraud
                    ]
                )
            ),

        "censored_fraud_mean_mae":
            safe_mean(
                np.abs(
                    posterior_mean[
                        censored_fraud
                    ]
                    - t_onset[
                        censored_fraud
                    ]
                )
            ),

        # ----------------------------------------------------
        # Posterior quality
        # ----------------------------------------------------

        "posterior_expected_abs_error":
            safe_mean(
                expected_abs_error[
                    known_fraud
                ]
            ),

        "mean_true_tf_probability_unconditional":
            safe_mean(
                true_tf_probability[
                    known_fraud
                ]
            ),

        "mean_true_tf_probability_given_commission":
            safe_mean(
                conditional_true_tf_probability[
                    known_fraud
                ]
            ),

        "onset_nll_unconditional":
            safe_mean(
                -np.log(
                    np.clip(
                        true_tf_probability[
                            known_fraud
                        ],
                        1e-12,
                        1.0,
                    )
                )
            ),

        # ----------------------------------------------------
        # Sampling diagnostic
        # ----------------------------------------------------

        "sampled_tf_mae_given_commission":
            safe_mean(
                sampled_tf_mae[
                    known_fraud
                ]
            ),

        "sampled_tf_mean_error":
            safe_mean(
                sampled_tf_mean[
                    known_fraud
                ]
                - t_onset[
                    known_fraud
                ]
            ),

        # ----------------------------------------------------
        # No-commission posterior
        # ----------------------------------------------------

        "normal_no_commission_posterior":
            safe_mean(
                no_commission[
                    normal
                ]
            ),

        "censored_fraud_no_commission_posterior":
            safe_mean(
                no_commission[
                    censored_fraud
                ]
            ),
    }

    return {
        "metrics":
            metrics,

        "posterior_mode":
            posterior_mode,

        "posterior_mean":
            posterior_mean,

        "true_tf_probability":
            true_tf_probability,

        "conditional_true_tf_probability":
            conditional_true_tf_probability,

        "expected_abs_error":
            expected_abs_error,

        "sampled_tf_mae":
            sampled_tf_mae,

        "sampled_tf_mean":
            sampled_tf_mean,
    }


# ============================================================
# Synthetic occurrence classification diagnostics
#
# Again:
# true t_onset is used ONLY for evaluation.
# ============================================================


def select_threshold(
    labels,
    scores,
):
    candidates = np.linspace(
        0.05,
        0.95,
        181,
    )

    f1_values = []

    for threshold in candidates:

        pred = (
            scores
            >= threshold
        )

        f1_values.append(
            f1_score(
                labels,
                pred,
                zero_division=0,
            )
        )

    return float(
        candidates[
            int(
                np.argmax(
                    f1_values
                )
            )
        ]
    )


def occurrence_metrics(
    labels,
    scores,
    threshold,
):
    prediction = (
        scores
        >= threshold
    )

    return {
        "occurrence_auroc":
            safe_auc(
                roc_auc_score,
                labels,
                scores,
            ),

        "occurrence_auprc":
            safe_auc(
                average_precision_score,
                labels,
                scores,
            ),

        "occurrence_accuracy":
            float(
                accuracy_score(
                    labels,
                    prediction,
                )
            ),

        "occurrence_precision":
            float(
                precision_score(
                    labels,
                    prediction,
                    zero_division=0,
                )
            ),

        "occurrence_recall":
            float(
                recall_score(
                    labels,
                    prediction,
                    zero_division=0,
                )
            ),

        "occurrence_f1":
            float(
                f1_score(
                    labels,
                    prediction,
                    zero_division=0,
                )
            ),
    }


# ============================================================
# Test entity IDs
# ============================================================


def get_test_entity_ids(
    dataset,
    test_loader,
):
    arrays = (
        dataset.arrays
    )

    if "entity_id" in arrays:

        all_ids = np.asarray(
            arrays[
                "entity_id"
            ]
        )

    else:

        all_ids = np.arange(
            len(
                arrays["x"]
            )
        )

    subset = (
        test_loader.dataset
    )

    if hasattr(
        subset,
        "indices",
    ):

        return all_ids[
            np.asarray(
                subset.indices
            )
        ]

    return all_ids[
        : len(
            subset
        )
    ]


# ============================================================
# Train one latent inference variant
# ============================================================


def train_variant(
    inference,
    dataset,
    train_loader,
    val_loader,
    test_loader,
):
    seed_everything(
        SEED
    )

    print()
    print(
        "=" * 80
    )
    print(
        f"Training latent decoupled model: {inference}"
    )
    print(
        "=" * 80
    )

    model = (
        LatentDecoupledJointModel(
            input_dim=
                dataset.arrays[
                    "x"
                ].shape[-1],

            hidden_dim=
                HIDDEN_DIM,
        )
        .to(
            DEVICE
        )
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # NO t_f based checkpoint selection.
    #
    # We select by validation observed-data marginal NLL.
    # --------------------------------------------------------

    best_state = None

    best_val_nll = (
        float("inf")
    )

    best_epoch = -1

    stale = 0

    history = []

    for epoch in range(
        MAX_EPOCHS
    ):

        train_payload = run_epoch(
            model,
            train_loader,
            inference,
            optimizer=optimizer,
        )

        val_payload = run_epoch(
            model,
            val_loader,
            inference,
            optimizer=None,
        )

        # ----------------------------------------------------
        # t_f metrics printed only as diagnostics.
        # They DO NOT select the checkpoint.
        # ----------------------------------------------------

        val_posterior = (
            evaluate_posterior(
                val_payload
            )
        )

        val_mean_mae = (
            val_posterior[
                "metrics"
            ][
                "posterior_mean_mae"
            ]
        )

        val_mode_mae = (
            val_posterior[
                "metrics"
            ][
                "posterior_mode_mae"
            ]
        )

        current_val_nll = (
            val_payload[
                "exact_nll"
            ]
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_payload[
                        "training_loss"
                    ],

                "train_exact_nll":
                    train_payload[
                        "exact_nll"
                    ],

                "val_exact_nll":
                    current_val_nll,

                # Diagnostic only:
                "val_posterior_mean_mae":
                    val_mean_mae,

                "val_posterior_mode_mae":
                    val_mode_mae,
            }
        )

        print(
            f"epoch={epoch:02d} "
            f"train_loss="
            f"{train_payload['training_loss']:.4f} "
            f"train_nll="
            f"{train_payload['exact_nll']:.4f} "
            f"val_nll="
            f"{current_val_nll:.4f} "
            f"diag_mean_mae="
            f"{val_mean_mae:.4f} "
            f"diag_mode_mae="
            f"{val_mode_mae:.4f}"
        )

        if (
            current_val_nll
            < best_val_nll
        ):

            best_val_nll = (
                current_val_nll
            )

            best_epoch = (
                epoch
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            stale = 0

        else:

            stale += 1

            if stale >= PATIENCE:

                print(
                    "Early stopping."
                )

                break

    if best_state is None:

        raise RuntimeError(
            "No valid checkpoint obtained."
        )

    model.load_state_dict(
        best_state
    )

    print()
    print(
        f"Best {inference} checkpoint: "
        f"epoch={best_epoch}, "
        f"val_exact_nll="
        f"{best_val_nll:.4f}"
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    run_dir = (
        RUN_ROOT
        / inference
        / f"seed_{SEED}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        run_dir
        / "model.pt",
    )

    (
        run_dir
        / "history.json"
    ).write_text(
        json.dumps(
            history,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Test
    # ========================================================

    test_payload = run_epoch(
        model,
        test_loader,
        inference,
        optimizer=None,
    )

    posterior_analysis = (
        evaluate_posterior(
            test_payload
        )
    )

    posterior_metrics = (
        posterior_analysis[
            "metrics"
        ]
    )

    # --------------------------------------------------------
    # Synthetic fraud occurrence metrics.
    #
    # Hidden t_onset is evaluation-only.
    # --------------------------------------------------------

    test_occurrence_label = (
        test_payload[
            "t_onset"
        ] >= 0
    ).astype(
        int
    )

    val_payload = run_epoch(
        model,
        val_loader,
        inference,
        optimizer=None,
    )

    val_occurrence_label = (
        val_payload[
            "t_onset"
        ] >= 0
    ).astype(
        int
    )

    threshold = select_threshold(
        val_occurrence_label,
        val_payload[
            "occurrence_probability"
        ],
    )

    classification_metrics = (
        occurrence_metrics(
            test_occurrence_label,
            test_payload[
                "occurrence_probability"
            ],
            threshold,
        )
    )

    result = {
        "inference":
            inference,

        "seed":
            SEED,

        "best_epoch":
            best_epoch,

        "best_val_exact_nll":
            best_val_nll,

        "test_exact_nll":
            test_payload[
                "exact_nll"
            ],

        "occurrence_threshold":
            threshold,

        **classification_metrics,

        **posterior_metrics,
    }

    print()
    print(
        "=" * 80
    )
    print(
        f"TEST RESULT: {inference}"
    )
    print(
        "=" * 80
    )

    for key, value in (
        result.items()
    ):

        print(
            f"{key}: {value}"
        )

    # ========================================================
    # Entity diagnostics
    # ========================================================

    entity_ids = (
        get_test_entity_ids(
            dataset,
            test_loader,
        )
    )

    true_tf = (
        test_payload[
            "t_onset"
        ]
    )

    detected = (
        test_payload[
            "detected"
        ]
    )

    fraud = (
        true_tf >= 0
    )

    group = np.full(
        len(
            true_tf
        ),
        "normal",
        dtype=object,
    )

    group[
        fraud
        & detected
    ] = "detected_fraud"

    group[
        fraud
        & ~detected
    ] = "censored_fraud"

    frame = pd.DataFrame(
        {
            "entity_id":
                entity_ids,

            "group":
                group,

            "true_tf":
                true_tf,

            "observed_td":
                test_payload[
                    "t_detect"
                ],

            "detected":
                detected.astype(int),

            "occurrence_probability":
                test_payload[
                    "occurrence_probability"
                ],

            "no_commission_posterior":
                test_payload[
                    "no_commission_posterior"
                ],

            "posterior_mode_tf":
                posterior_analysis[
                    "posterior_mode"
                ],

            "posterior_mean_tf":
                posterior_analysis[
                    "posterior_mean"
                ],

            "true_tf_probability":
                posterior_analysis[
                    "true_tf_probability"
                ],

            "true_tf_probability_given_commission":
                posterior_analysis[
                    "conditional_true_tf_probability"
                ],

            "expected_abs_error":
                posterior_analysis[
                    "expected_abs_error"
                ],

            "sampled_tf_mae":
                posterior_analysis[
                    "sampled_tf_mae"
                ],

            "sampled_tf_mean":
                posterior_analysis[
                    "sampled_tf_mean"
                ],
        }
    )

    variant_dir = (
        OUTPUT_DIR
        / inference
    )

    variant_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        variant_dir
        / "entity_diagnostics.csv",
        index=False,
    )

    np.savez_compressed(
        variant_dir
        / "posterior_data.npz",

        entity_id=
            entity_ids,

        onset_posterior=
            test_payload[
                "onset_posterior"
            ],

        latent_posterior=
            test_payload[
                "latent_posterior"
            ],

        no_commission_posterior=
            test_payload[
                "no_commission_posterior"
            ],

        occurrence_probability=
            test_payload[
                "occurrence_probability"
            ],

        t_onset=
            test_payload[
                "t_onset"
            ],

        t_detect=
            test_payload[
                "t_detect"
            ],

        detected=
            test_payload[
                "detected"
            ],
    )

    (
        variant_dir
        / "summary.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return result


# ============================================================
# Main
# ============================================================


def main():

    seed_everything(
        SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUN_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = load_npz(
        DATA_PATH
    )

    (
        train_loader,
        val_loader,
        test_loader,
    ) = make_loaders(
        dataset,
        BATCH_SIZE,
        SPLIT,
        SEED,
    )

    print(
        "Split sizes:",
        len(
            train_loader.dataset
        ),
        len(
            val_loader.dataset
        ),
        len(
            test_loader.dataset
        ),
    )

    # --------------------------------------------------------
    # Verify that true tf is NOT part of the latent model loss.
    # --------------------------------------------------------

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "  Training supervision uses detected and t_detect only."
    )
    print(
        "  t_onset is used only for post-hoc synthetic diagnostics."
    )

    all_results = {}

    for inference in (
        "exact",
        "stochastic_em",
    ):

        result = train_variant(
            inference,
            dataset,
            train_loader,
            val_loader,
            test_loader,
        )

        all_results[
            inference
        ] = result

    # ========================================================
    # Overall summary
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

    print()
    print(
        "=" * 80
    )
    print(
        "FINAL LATENT DECOUPLED COMPARISON"
    )
    print(
        "=" * 80
    )

    for inference in (
        "exact",
        "stochastic_em",
    ):

        result = all_results[
            inference
        ]

        print()
        print(
            inference
        )

        print(
            "  occurrence AUROC:",
            result[
                "occurrence_auroc"
            ],
        )

        print(
            "  occurrence AUPRC:",
            result[
                "occurrence_auprc"
            ],
        )

        print(
            "  posterior mean MAE:",
            result[
                "posterior_mean_mae"
            ],
        )

        print(
            "  posterior mode MAE:",
            result[
                "posterior_mode_mae"
            ],
        )

        print(
            "  sampled tf MAE:",
            result[
                "sampled_tf_mae_given_commission"
            ],
        )

        print(
            "  expected abs error:",
            result[
                "posterior_expected_abs_error"
            ],
        )

        print(
            "  P(true tf | commission):",
            result[
                "mean_true_tf_probability_given_commission"
            ],
        )

        print(
            "  mode before tf:",
            result[
                "posterior_mode_before_tf_rate"
            ],
        )

        print(
            "  normal no-commission posterior:",
            result[
                "normal_no_commission_posterior"
            ],
        )

        print(
            "  censored fraud no-commission posterior:",
            result[
                "censored_fraud_no_commission_posterior"
            ],
        )

    # ========================================================
    # Previous reference values
    # ========================================================

    print()
    print(
        "=" * 80
    )
    print(
        "REFERENCE RESULTS"
    )
    print(
        "=" * 80
    )

    print(
        "Original Joint-Sampling:"
    )
    print(
        "  posterior mean MAE = 4.6302"
    )
    print(
        "  mode MAE = 8.9111"
    )
    print(
        "  sampled tf MAE = 5.5213"
    )
    print(
        "  normal no-commission = 0.0080"
    )

    print()
    print(
        "Decoupled Oracle:"
    )
    print(
        "  posterior mean MAE = 1.8354"
    )
    print(
        "  mode MAE = 2.1852"
    )
    print(
        "  occurrence AUROC = 0.99997"
    )

    print()
    print(
        "Remember:"
    )
    print(
        "  Decoupled Oracle uses true tf during training."
    )
    print(
        "  Latent Decoupled does NOT use true tf during training."
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