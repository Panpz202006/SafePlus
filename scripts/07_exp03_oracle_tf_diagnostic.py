from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from safeplus.data import load_npz, make_loaders
from safeplus.evaluation.metrics import evaluate_predictions
from safeplus.models.common import SequenceEncoder, time_mask
from safeplus.utils import seed_everything


# ============================================================
# Experiment 03C-v2
# Oracle true-tf recovery diagnostic
#
# IMPORTANT:
# This is NOT the final SafePlus model.
#
# Ground-truth t_f is deliberately used during training so that
# we can answer one diagnostic question:
#
# If the true fraud-onset time were available, how accurately
# could the current encoder + hazard formulation recover t_f
# from the synthetic behavioral sequence?
#
# The checkpoint is ALSO selected using validation true-t_f
# posterior-mean MAE.
#
# Therefore this is an ORACLE diagnostic only.
# It must never be reported as a deployable method.
# ============================================================


# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path(
    r"data/processed/synthetic_demo.npz"
)

RUN_DIR = Path(
    r"runs/synthetic/exp03/oracle-tf-mean-selection/seed_0"
)

REPORT_DIR = Path(
    r"reports/experiment03/oracle_tf_mean_selection"
)

PREVIOUS_JOINT_SUMMARY = Path(
    r"reports/experiment03/posterior_diagnostics/summary.json"
)

SEED = 0
DEVICE = "cpu"

HIDDEN_DIM = 32
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

MAX_EPOCHS = 40
PATIENCE = 7

SPLIT = [0.7, 0.1, 0.2]


# ============================================================
# Oracle onset model
# ============================================================


class OracleOnsetModel(nn.Module):
    """
    Oracle commission-onset model.

    The architecture uses the same causal sequence-encoder family
    as SafePlus, but true t_f is directly used as training
    supervision.

    For a fraudulent entity whose true onset is k:

        P(t_f = k | x)
        =
        prod_{j < k} (1 - h_f(j)) * h_f(k)

    For a normal entity:

        P(t_f > T | x)
        =
        prod_{j <= T} (1 - h_f(j))

    The model does NOT use t_detect.

    This model is only an oracle diagnostic.
    """

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

        self.commission_head = nn.Linear(
            hidden_dim,
            1,
        )

    def forward(
        self,
        x,
        lengths,
    ):
        # ----------------------------------------------------
        # Causal hidden states
        # ----------------------------------------------------

        hidden = self.encoder(
            x,
            lengths,
        )

        logits = self.commission_head(
            hidden
        ).squeeze(-1)

        mask = time_mask(
            lengths,
            x.shape[1],
        )

        # ----------------------------------------------------
        # Discrete hazard
        # ----------------------------------------------------

        # log h_f(t)
        log_h = F.logsigmoid(
            logits
        )

        # log(1 - h_f(t))
        log_s = F.logsigmoid(
            -logits
        )

        # Padding must not contribute to survival.
        log_s_valid = log_s.masked_fill(
            ~mask,
            0.0,
        )

        # ----------------------------------------------------
        # Survival through each time step
        # ----------------------------------------------------

        cumulative_log_survival = (
            log_s_valid.cumsum(dim=1)
        )

        # Survival strictly before time t.
        before = torch.cat(
            [
                logits.new_zeros(
                    (len(logits), 1)
                ),
                cumulative_log_survival[:, :-1],
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # P(t_f = t | x)
        # ----------------------------------------------------

        event_log_mass = (
            before + log_h
        )

        event_log_mass = (
            event_log_mass.masked_fill(
                ~mask,
                -torch.inf,
            )
        )

        event_mass = (
            event_log_mass.exp()
        )

        # ----------------------------------------------------
        # P(t_f > T | x)
        # ----------------------------------------------------

        no_commission_log_prob = (
            log_s_valid.sum(dim=1)
        )

        no_commission_prob = (
            no_commission_log_prob.exp()
        )

        # ----------------------------------------------------
        # Cumulative online fraud risk
        # ----------------------------------------------------

        risk = -torch.expm1(
            cumulative_log_survival
        )

        commission_hazard = (
            logits.sigmoid().masked_fill(
                ~mask,
                0.0,
            )
        )

        return {
            "logits": logits,
            "commission_hazard": commission_hazard,
            "event_log_mass": event_log_mass,
            "event_mass": event_mass,
            "no_commission_log_prob": no_commission_log_prob,
            "no_commission_prob": no_commission_prob,
            "risk": risk,
        }

    def loss(
        self,
        batch,
    ):
        out = self(
            batch["x"],
            batch["lengths"],
        )

        t_onset = batch[
            "t_onset"
        ]

        known_fraud = (
            t_onset >= 0
        )

        # ----------------------------------------------------
        # Normal entities:
        #
        # likelihood = P(t_f > T | x)
        # ----------------------------------------------------

        log_likelihood = (
            out[
                "no_commission_log_prob"
            ].clone()
        )

        # ----------------------------------------------------
        # Fraud entities:
        #
        # likelihood = P(t_f = true_t_f | x)
        # ----------------------------------------------------

        fraud_indices = torch.where(
            known_fraud
        )[0]

        if len(fraud_indices) > 0:

            true_tf = t_onset[
                fraud_indices
            ]

            log_likelihood[
                fraud_indices
            ] = out[
                "event_log_mass"
            ][
                fraud_indices,
                true_tf,
            ]

        loss = (
            -log_likelihood.mean()
        )

        return loss, out


# ============================================================
# Utility functions
# ============================================================


def move_batch(
    batch,
):
    return {
        key: value.to(DEVICE)
        for key, value in batch.items()
    }


def safe_mean(
    values,
):
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


# ============================================================
# Run one epoch
# ============================================================


def run_epoch(
    model,
    loader,
    optimizer=None,
):
    training = (
        optimizer is not None
    )

    model.train(
        training
    )

    total_loss = 0.0
    total_count = 0

    all_risk = []
    all_event_mass = []
    all_no_commission = []

    all_detected = []
    all_t_detect = []
    all_t_onset = []
    all_lengths = []

    for raw_batch in loader:

        batch = move_batch(
            raw_batch
        )

        with torch.set_grad_enabled(
            training
        ):

            loss, out = model.loss(
                batch
            )

            if training:

                optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    5.0,
                )

                optimizer.step()

        batch_size = len(
            batch["x"]
        )

        total_loss += (
            float(
                loss.detach()
            )
            * batch_size
        )

        total_count += (
            batch_size
        )

        # ----------------------------------------------------
        # Risk
        # ----------------------------------------------------

        risk = (
            out["risk"]
            .detach()
            .clone()
        )

        steps = torch.arange(
            risk.shape[1],
            device=risk.device,
        )[None, :]

        last_valid_risk = risk.gather(
            1,
            (
                batch["lengths"] - 1
            )[:, None],
        )

        risk = torch.where(
            steps
            < batch["lengths"][:, None],
            risk,
            last_valid_risk,
        )

        all_risk.append(
            risk.cpu().numpy()
        )

        # ----------------------------------------------------
        # Onset probability distribution
        # ----------------------------------------------------

        all_event_mass.append(
            out[
                "event_mass"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        all_no_commission.append(
            out[
                "no_commission_prob"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        all_detected.append(
            batch[
                "detected"
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        all_t_detect.append(
            batch[
                "t_detect"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        all_t_onset.append(
            batch[
                "t_onset"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        all_lengths.append(
            batch[
                "lengths"
            ]
            .detach()
            .cpu()
            .numpy()
        )

    return {
        "loss":
            total_loss
            / total_count,

        "risk":
            np.concatenate(
                all_risk
            ),

        "event_mass":
            np.concatenate(
                all_event_mass
            ),

        "no_commission":
            np.concatenate(
                all_no_commission
            ),

        "detected":
            np.concatenate(
                all_detected
            ),

        "t_detect":
            np.concatenate(
                all_t_detect
            ),

        "t_onset":
            np.concatenate(
                all_t_onset
            ),

        "lengths":
            np.concatenate(
                all_lengths
            ),
    }


# ============================================================
# Onset metrics
# ============================================================


def onset_metrics(
    payload,
):
    event_mass = payload[
        "event_mass"
    ]

    no_commission = payload[
        "no_commission"
    ]

    t_onset = payload[
        "t_onset"
    ]

    detected = payload[
        "detected"
    ].astype(bool)

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

    # --------------------------------------------------------
    # Posterior mode
    # --------------------------------------------------------

    posterior_mode = (
        event_mass.argmax(
            axis=1
        )
    )

    # --------------------------------------------------------
    # Total probability mass assigned to commission
    # inside the observation window
    # --------------------------------------------------------

    onset_mass = (
        event_mass.sum(
            axis=1
        )
    )

    time_steps = np.arange(
        event_mass.shape[1],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Posterior mean conditional on commission
    # --------------------------------------------------------

    posterior_mean = np.full(
        len(t_onset),
        np.nan,
        dtype=np.float64,
    )

    valid_mass = (
        onset_mass > 0
    )

    posterior_mean[
        valid_mass
    ] = (
        event_mass[
            valid_mass
        ]
        * time_steps[None, :]
    ).sum(
        axis=1
    ) / onset_mass[
        valid_mass
    ]

    # --------------------------------------------------------
    # Probability assigned to the true onset
    # --------------------------------------------------------

    true_tf_probability = np.full(
        len(t_onset),
        np.nan,
        dtype=np.float64,
    )

    known_indices = np.where(
        known_fraud
    )[0]

    true_tf_probability[
        known_indices
    ] = event_mass[
        known_indices,
        t_onset[
            known_indices
        ],
    ]

    # --------------------------------------------------------
    # Expected absolute temporal error
    # under posterior conditioned on commission
    # --------------------------------------------------------

    expected_abs_error = np.full(
        len(t_onset),
        np.nan,
        dtype=np.float64,
    )

    for index in known_indices:

        mass = onset_mass[
            index
        ]

        if mass <= 0:
            continue

        conditional_posterior = (
            event_mass[
                index
            ]
            / mass
        )

        expected_abs_error[
            index
        ] = np.sum(
            conditional_posterior
            * np.abs(
                time_steps
                - t_onset[
                    index
                ]
            )
        )

    # --------------------------------------------------------
    # Signed errors
    # --------------------------------------------------------

    posterior_mean_signed_error = (
        posterior_mean
        - t_onset
    )

    posterior_mode_signed_error = (
        posterior_mode
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
        # Mode estimates
        # ----------------------------------------------------

        "mode_mae_all_known_fraud":
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

        "mode_mae_detected_fraud":
            safe_mean(
                np.abs(
                    posterior_mode[
                        detected_fraud
                    ]
                    - t_onset[
                        detected_fraud
                    ]
                )
            ),

        "mode_mae_censored_fraud":
            safe_mean(
                np.abs(
                    posterior_mode[
                        censored_fraud
                    ]
                    - t_onset[
                        censored_fraud
                    ]
                )
            ),

        "mode_signed_error":
            safe_mean(
                posterior_mode_signed_error[
                    known_fraud
                ]
            ),

        # ----------------------------------------------------
        # Posterior mean estimates
        # ----------------------------------------------------

        "posterior_mean_mae_all_known_fraud":
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

        "posterior_mean_mae_detected_fraud":
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

        "posterior_mean_mae_censored_fraud":
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

        "posterior_mean_signed_error":
            safe_mean(
                posterior_mean_signed_error[
                    known_fraud
                ]
            ),

        # ----------------------------------------------------
        # Full posterior quality
        # ----------------------------------------------------

        "posterior_expected_abs_error":
            safe_mean(
                expected_abs_error[
                    known_fraud
                ]
            ),

        "mean_true_tf_probability":
            safe_mean(
                true_tf_probability[
                    known_fraud
                ]
            ),

        "onset_nll_all_known_fraud":
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
        # No-commission state
        # ----------------------------------------------------

        "mean_no_commission_probability_normal":
            safe_mean(
                no_commission[
                    normal
                ]
            ),

        "mean_no_commission_probability_fraud":
            safe_mean(
                no_commission[
                    known_fraud
                ]
            ),

        # ----------------------------------------------------
        # Early/on-time/late mode counts
        # ----------------------------------------------------

        "mode_before_true_tf_rate":
            safe_mean(
                (
                    posterior_mode[
                        known_fraud
                    ]
                    < t_onset[
                        known_fraud
                    ]
                ).astype(
                    np.float64
                )
            ),

        "mode_exact_true_tf_rate":
            safe_mean(
                (
                    posterior_mode[
                        known_fraud
                    ]
                    == t_onset[
                        known_fraud
                    ]
                ).astype(
                    np.float64
                )
            ),

        "mode_after_true_tf_rate":
            safe_mean(
                (
                    posterior_mode[
                        known_fraud
                    ]
                    > t_onset[
                        known_fraud
                    ]
                ).astype(
                    np.float64
                )
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

        "expected_abs_error":
            expected_abs_error,
    }


# ============================================================
# Threshold selection
# ============================================================


def select_threshold(
    labels,
    risk,
):
    candidates = np.linspace(
        0.05,
        0.95,
        181,
    )

    f1_scores = []

    for threshold in candidates:

        metrics = evaluate_predictions(
            labels,
            risk,
            threshold,
        )

        f1_scores.append(
            metrics["f1"]
        )

    best_index = int(
        np.argmax(
            f1_scores
        )
    )

    return float(
        candidates[
            best_index
        ]
    )


# ============================================================
# Recover entity IDs in held-out test order
# ============================================================


def get_test_entity_ids(
    dataset,
    test_loader,
):
    arrays = (
        dataset.arrays
    )

    if "entity_id" in arrays:

        all_entity_ids = np.asarray(
            arrays[
                "entity_id"
            ]
        )

    else:

        all_entity_ids = np.arange(
            len(
                arrays["x"]
            )
        )

    test_dataset = (
        test_loader.dataset
    )

    if hasattr(
        test_dataset,
        "indices",
    ):

        indices = np.asarray(
            test_dataset.indices
        )

        return all_entity_ids[
            indices
        ]

    return all_entity_ids[
        : len(
            test_dataset
        )
    ]


# ============================================================
# Main
# ============================================================


def main():

    seed_everything(
        SEED
    )

    RUN_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

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
    # Model
    # --------------------------------------------------------

    model = OracleOnsetModel(
        input_dim=
            dataset.arrays[
                "x"
            ].shape[-1],

        hidden_dim=
            HIDDEN_DIM,
    ).to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ========================================================
    # Oracle checkpoint selection
    #
    # IMPORTANT:
    #
    # We select the checkpoint using validation
    # posterior-mean onset MAE.
    #
    # This uses true t_f on validation and therefore is
    # NOT allowed for the real latent model.
    # ========================================================

    best_state = None

    best_val_mean_mae = (
        float("inf")
    )

    best_epoch = -1

    stale = 0

    history = []

    print()

    print(
        "Training Oracle true-tf model..."
    )

    print(
        "Checkpoint selection: "
        "validation posterior-mean onset MAE"
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        MAX_EPOCHS
    ):

        train_payload = run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
        )

        val_payload = run_epoch(
            model,
            val_loader,
            optimizer=None,
        )

        val_analysis = onset_metrics(
            val_payload
        )

        val_metrics = (
            val_analysis[
                "metrics"
            ]
        )

        current_val_mean_mae = (
            val_metrics[
                "posterior_mean_mae_all_known_fraud"
            ]
        )

        current_val_mode_mae = (
            val_metrics[
                "mode_mae_all_known_fraud"
            ]
        )

        history_row = {
            "epoch":
                epoch,

            "train_loss":
                train_payload[
                    "loss"
                ],

            "val_loss":
                val_payload[
                    "loss"
                ],

            "val_mode_mae":
                current_val_mode_mae,

            "val_mean_mae":
                current_val_mean_mae,

            "val_expected_abs_error":
                val_metrics[
                    "posterior_expected_abs_error"
                ],

            "val_true_tf_probability":
                val_metrics[
                    "mean_true_tf_probability"
                ],

            "val_normal_no_commission":
                val_metrics[
                    "mean_no_commission_probability_normal"
                ],
        }

        history.append(
            history_row
        )

        print(
            f"epoch={epoch:02d} "
            f"train_loss="
            f"{train_payload['loss']:.4f} "
            f"val_loss="
            f"{val_payload['loss']:.4f} "
            f"val_mode_mae="
            f"{current_val_mode_mae:.4f} "
            f"val_mean_mae="
            f"{current_val_mean_mae:.4f}"
        )

        # ----------------------------------------------------
        # Save best Oracle checkpoint using mean-onset MAE
        # ----------------------------------------------------

        if (
            current_val_mean_mae
            < best_val_mean_mae
        ):

            best_val_mean_mae = (
                current_val_mean_mae
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
            "No valid Oracle checkpoint was obtained."
        )

    # --------------------------------------------------------
    # Restore best Oracle checkpoint
    # --------------------------------------------------------

    print()

    print(
        f"Best oracle checkpoint: "
        f"epoch={best_epoch}, "
        f"val_mean_mae="
        f"{best_val_mean_mae:.4f}"
    )

    model.load_state_dict(
        best_state
    )

    # --------------------------------------------------------
    # Save model and training history
    # --------------------------------------------------------

    torch.save(
        model.state_dict(),
        RUN_DIR / "model.pt",
    )

    (
        RUN_DIR
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
    # Validation threshold
    # ========================================================

    val_payload = run_epoch(
        model,
        val_loader,
        optimizer=None,
    )

    # Here the Oracle binary label means:
    #
    # fraud occurred somewhere inside the observation window.
    val_labels = (
        val_payload[
            "t_onset"
        ] >= 0
    ).astype(
        int
    )

    threshold = select_threshold(
        val_labels,
        val_payload[
            "risk"
        ],
    )

    # ========================================================
    # Held-out test evaluation
    # ========================================================

    test_payload = run_epoch(
        model,
        test_loader,
        optimizer=None,
    )

    test_labels = (
        test_payload[
            "t_onset"
        ] >= 0
    ).astype(
        int
    )

    classification_metrics = (
        evaluate_predictions(
            test_labels,
            test_payload[
                "risk"
            ],
            threshold,
        )
    )

    test_analysis = onset_metrics(
        test_payload
    )

    test_onset_metrics = (
        test_analysis[
            "metrics"
        ]
    )

    # ========================================================
    # Final summary
    # ========================================================

    summary = {
        "diagnostic":
            (
                "Oracle true-tf supervised "
                "onset recovery"
            ),

        "warning":
            (
                "Ground-truth t_f is used "
                "during training and validation "
                "checkpoint selection. "
                "This is a diagnostic oracle, "
                "not a deployable method."
            ),

        "checkpoint_selection":
            (
                "validation posterior-mean "
                "onset MAE"
            ),

        "best_epoch":
            int(
                best_epoch
            ),

        "best_val_mean_mae":
            float(
                best_val_mean_mae
            ),

        "seed":
            SEED,

        "hidden_dim":
            HIDDEN_DIM,

        "threshold":
            threshold,

        **classification_metrics,

        **test_onset_metrics,
    }

    print()

    print(
        "=" * 80
    )

    print(
        "ORACLE TRUE-tf DIAGNOSTIC RESULT"
    )

    print(
        "=" * 80
    )

    for key, value in (
        summary.items()
    ):

        print(
            f"{key}: {value}"
        )

    (
        REPORT_DIR
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Entity-level diagnostics
    # ========================================================

    entity_ids = (
        get_test_entity_ids(
            dataset,
            test_loader,
        )
    )

    t_onset = test_payload[
        "t_onset"
    ]

    detected = test_payload[
        "detected"
    ].astype(
        bool
    )

    known_fraud = (
        t_onset >= 0
    )

    group = np.full(
        len(
            t_onset
        ),
        "normal",
        dtype=object,
    )

    group[
        known_fraud
        & detected
    ] = "detected_fraud"

    group[
        known_fraud
        & ~detected
    ] = "censored_fraud"

    posterior_mode = (
        test_analysis[
            "posterior_mode"
        ]
    )

    posterior_mean = (
        test_analysis[
            "posterior_mean"
        ]
    )

    true_tf_probability = (
        test_analysis[
            "true_tf_probability"
        ]
    )

    expected_abs_error = (
        test_analysis[
            "expected_abs_error"
        ]
    )

    posterior_mean_signed_error = (
        posterior_mean
        - t_onset
    )

    posterior_mode_signed_error = (
        posterior_mode
        - t_onset
    )

    entity_frame = pd.DataFrame(
        {
            "entity_id":
                entity_ids,

            "group":
                group,

            "true_tf":
                t_onset,

            "observed_td":
                test_payload[
                    "t_detect"
                ],

            "detected":
                test_payload[
                    "detected"
                ],

            "posterior_mode_tf":
                posterior_mode,

            "posterior_mode_signed_error":
                posterior_mode_signed_error,

            "posterior_mean_tf":
                posterior_mean,

            "posterior_mean_signed_error":
                posterior_mean_signed_error,

            "true_tf_probability":
                true_tf_probability,

            "expected_abs_error":
                expected_abs_error,

            "no_commission_probability":
                test_payload[
                    "no_commission"
                ],
        }
    )

    entity_frame.to_csv(
        REPORT_DIR
        / "entity_diagnostics.csv",
        index=False,
    )

    # ========================================================
    # Save full posterior data
    # ========================================================

    np.savez_compressed(
        REPORT_DIR
        / "oracle_posterior_data.npz",

        entity_id=
            entity_ids,

        event_mass=
            test_payload[
                "event_mass"
            ],

        no_commission_probability=
            test_payload[
                "no_commission"
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

        posterior_mode=
            posterior_mode,

        posterior_mean=
            posterior_mean,
    )

    # ========================================================
    # Compare with current latent Joint models
    # ========================================================

    if PREVIOUS_JOINT_SUMMARY.exists():

        previous_summary = (
            json.loads(
                PREVIOUS_JOINT_SUMMARY.read_text(
                    encoding="utf-8"
                )
            )
        )

        print()

        print(
            "=" * 80
        )

        print(
            "CURRENT LATENT vs ORACLE-v2"
        )

        print(
            "=" * 80
        )

        for model_name in (
            "joint_exact",
            "joint_sampling",
        ):

            if model_name not in previous_summary:
                continue

            old = previous_summary[
                model_name
            ]

            print()

            print(
                model_name
            )

            print(
                "  latent mode MAE:",
                old.get(
                    "posterior_mode_mae_all_known_fraud"
                ),
            )

            print(
                "  latent posterior-mean MAE:",
                old.get(
                    "posterior_mean_mae_all_known_fraud"
                ),
            )

            print(
                "  latent expected abs error:",
                old.get(
                    "posterior_expected_abs_error_all_known_fraud"
                ),
            )

            print(
                "  latent true-tf probability:",
                old.get(
                    "mean_true_tf_probability"
                ),
            )

            print(
                "  latent normal no-commission:",
                old.get(
                    "mean_no_commission_posterior_normal"
                ),
            )

        print()

        print(
            "oracle-v2 best epoch:",
            best_epoch,
        )

        print(
            "oracle-v2 best validation mean MAE:",
            best_val_mean_mae,
        )

        print(
            "oracle-v2 test mode MAE:",
            test_onset_metrics[
                "mode_mae_all_known_fraud"
            ],
        )

        print(
            "oracle-v2 test posterior-mean MAE:",
            test_onset_metrics[
                "posterior_mean_mae_all_known_fraud"
            ],
        )

        print(
            "oracle-v2 test expected abs error:",
            test_onset_metrics[
                "posterior_expected_abs_error"
            ],
        )

        print(
            "oracle-v2 test mean true-tf probability:",
            test_onset_metrics[
                "mean_true_tf_probability"
            ],
        )

        print(
            "oracle-v2 normal no-commission:",
            test_onset_metrics[
                "mean_no_commission_probability_normal"
            ],
        )

    # ========================================================
    # Save configuration
    # ========================================================

    config_payload = {
        "seed":
            SEED,

        "device":
            DEVICE,

        "hidden_dim":
            HIDDEN_DIM,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "max_epochs":
            MAX_EPOCHS,

        "patience":
            PATIENCE,

        "split":
            SPLIT,

        "checkpoint_selection":
            (
                "validation posterior-mean "
                "onset MAE"
            ),

        "oracle_warning":
            (
                "True t_f is used during "
                "training and validation "
                "checkpoint selection. "
                "Diagnostic use only."
            ),
    }

    (
        RUN_DIR
        / "config.json"
    ).write_text(
        json.dumps(
            config_payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # Finish
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "Saved outputs"
    )

    print(
        "=" * 80
    )

    print(
        RUN_DIR
        / "model.pt"
    )

    print(
        RUN_DIR
        / "history.json"
    )

    print(
        RUN_DIR
        / "config.json"
    )

    print(
        REPORT_DIR
        / "summary.json"
    )

    print(
        REPORT_DIR
        / "entity_diagnostics.csv"
    )

    print(
        REPORT_DIR
        / "oracle_posterior_data.npz"
    )


if __name__ == "__main__":
    main()