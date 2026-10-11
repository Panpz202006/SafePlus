from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from safeplus.data import load_npz, make_loaders
from safeplus.models.common import SequenceEncoder, time_mask
from safeplus.utils import seed_everything


# ============================================================
# Experiment 03E
# Decoupled occurrence-timing diagnostic
#
# Goal:
# Separate two questions that are currently entangled inside
# the commission hazard:
#
#   1. Does fraud occur within the observation window?
#   2. If fraud occurs, at which time step does it begin?
#
# IMPORTANT:
# True t_f is used to supervise the timing head.
# Therefore this is an ORACLE DIAGNOSTIC, not the final method.
#
# The purpose is to test whether separating occurrence and
# timing removes the early-onset collapse observed in the
# single-hazard formulation.
# ============================================================


DATA_PATH = Path(
    r"data/processed/synthetic_demo.npz"
)

OUTPUT_DIR = Path(
    r"reports/experiment03/decoupled_occurrence_timing"
)

RUN_DIR = Path(
    r"runs/synthetic/exp03/decoupled-occurrence-timing/seed_0"
)

SEED = 0
DEVICE = "cpu"

HIDDEN_DIM = 32
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

MAX_EPOCHS = 40
PATIENCE = 7

SPLIT = [0.7, 0.1, 0.2]

# Equal importance for:
#   occurrence classification
#   onset timing
OCCURRENCE_WEIGHT = 0.5
TIMING_WEIGHT = 0.5


# ============================================================
# Model
# ============================================================


class DecoupledOccurrenceTimingModel(nn.Module):
    """
    One shared causal GRU encoder, but two separate prediction heads.

    Occurrence head:
        P(fraud occurs within horizon | x)

    Timing head:
        P(t_f = t | fraud occurs, x)

    Timing is represented as a categorical distribution over
    valid sequence positions using a masked softmax.

    Unlike the previous hazard event mass:

        h(t) * prod_{j<t}(1-h(j))

    this timing distribution does NOT contain a survival product.

    Therefore this diagnostic tests whether the previous early-time
    preference is caused by coupling timing to survival.
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

        # Sequence-level fraud occurrence.
        self.occurrence_head = nn.Linear(
            hidden_dim,
            1,
        )

        # Per-time-step onset score.
        self.timing_head = nn.Linear(
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

        batch_size, seq_len, _ = hidden.shape

        # ----------------------------------------------------
        # Occurrence head
        #
        # Use the last valid causal hidden state as a summary
        # of the observed sequence.
        # ----------------------------------------------------

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

        occurrence_probability = (
            occurrence_logit.sigmoid()
        )

        # ----------------------------------------------------
        # Timing head
        # ----------------------------------------------------

        timing_logits = (
            self.timing_head(
                hidden
            ).squeeze(-1)
        )

        valid_mask = time_mask(
            lengths,
            seq_len,
        )

        timing_logits = timing_logits.masked_fill(
            ~valid_mask,
            -torch.inf,
        )

        timing_posterior = F.softmax(
            timing_logits,
            dim=1,
        )

        return {
            "occurrence_logit":
                occurrence_logit,

            "occurrence_probability":
                occurrence_probability,

            "timing_logits":
                timing_logits,

            "timing_posterior":
                timing_posterior,
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

        fraud = (
            t_onset >= 0
        )

        fraud_label = fraud.float()

        # ----------------------------------------------------
        # Task 1:
        # fraud occurrence
        # ----------------------------------------------------

        occurrence_loss = (
            F.binary_cross_entropy_with_logits(
                out[
                    "occurrence_logit"
                ],
                fraud_label,
            )
        )

        # ----------------------------------------------------
        # Task 2:
        # onset timing
        #
        # Only fraud users have a true onset.
        # ----------------------------------------------------

        fraud_indices = torch.where(
            fraud
        )[0]

        if len(fraud_indices) > 0:

            true_tf = t_onset[
                fraud_indices
            ]

            timing_loss = F.cross_entropy(
                out[
                    "timing_logits"
                ][
                    fraud_indices
                ],
                true_tf,
            )

        else:

            timing_loss = (
                occurrence_loss
                * 0.0
            )

        total_loss = (
            OCCURRENCE_WEIGHT
            * occurrence_loss
            +
            TIMING_WEIGHT
            * timing_loss
        )

        out.update(
            occurrence_loss=
                occurrence_loss,

            timing_loss=
                timing_loss,
        )

        return (
            total_loss,
            out,
        )


# ============================================================
# Utilities
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


def safe_auc(
    function,
    labels,
    score,
):
    if len(
        np.unique(labels)
    ) < 2:
        return float("nan")

    return float(
        function(
            labels,
            score,
        )
    )


# ============================================================
# Epoch runner
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
    total_occurrence_loss = 0.0
    total_timing_loss = 0.0
    total_count = 0

    occurrence_probabilities = []
    timing_posteriors = []

    t_onsets = []
    t_detects = []
    detected_all = []
    lengths_all = []

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

        n = len(
            batch["x"]
        )

        total_loss += (
            float(
                loss.detach()
            )
            * n
        )

        total_occurrence_loss += (
            float(
                out[
                    "occurrence_loss"
                ].detach()
            )
            * n
        )

        total_timing_loss += (
            float(
                out[
                    "timing_loss"
                ].detach()
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

        timing_posteriors.append(
            out[
                "timing_posterior"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        t_onsets.append(
            batch[
                "t_onset"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        t_detects.append(
            batch[
                "t_detect"
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

        lengths_all.append(
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

        "occurrence_loss":
            total_occurrence_loss
            / total_count,

        "timing_loss":
            total_timing_loss
            / total_count,

        "occurrence_probability":
            np.concatenate(
                occurrence_probabilities
            ),

        "timing_posterior":
            np.concatenate(
                timing_posteriors
            ),

        "t_onset":
            np.concatenate(
                t_onsets
            ),

        "t_detect":
            np.concatenate(
                t_detects
            ),

        "detected":
            np.concatenate(
                detected_all
            ),

        "lengths":
            np.concatenate(
                lengths_all
            ),
    }


# ============================================================
# Timing metrics
# ============================================================


def evaluate_timing(
    payload,
):
    posterior = payload[
        "timing_posterior"
    ]

    t_onset = payload[
        "t_onset"
    ]

    detected = payload[
        "detected"
    ]

    fraud = (
        t_onset >= 0
    )

    detected_fraud = (
        fraud
        & detected
    )

    censored_fraud = (
        fraud
        & ~detected
    )

    time_steps = np.arange(
        posterior.shape[1],
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Posterior mode
    # --------------------------------------------------------

    mode = posterior.argmax(
        axis=1
    )

    # --------------------------------------------------------
    # Posterior mean
    # --------------------------------------------------------

    posterior_mean = (
        posterior
        * time_steps[None, :]
    ).sum(
        axis=1
    )

    # --------------------------------------------------------
    # Probability assigned to true tf
    # --------------------------------------------------------

    true_probability = np.full(
        len(t_onset),
        np.nan,
        dtype=np.float64,
    )

    expected_abs_error = np.full(
        len(t_onset),
        np.nan,
        dtype=np.float64,
    )

    fraud_indices = np.where(
        fraud
    )[0]

    for index in fraud_indices:

        tf = t_onset[
            index
        ]

        true_probability[
            index
        ] = posterior[
            index,
            tf,
        ]

        expected_abs_error[
            index
        ] = np.sum(
            posterior[
                index
            ]
            * np.abs(
                time_steps
                - tf
            )
        )

    mode_error = (
        mode
        - t_onset
    )

    mean_error = (
        posterior_mean
        - t_onset
    )

    metrics = {
        "n_fraud":
            int(
                fraud.sum()
            ),

        "mode_mae":
            safe_mean(
                np.abs(
                    mode[
                        fraud
                    ]
                    - t_onset[
                        fraud
                    ]
                )
            ),

        "posterior_mean_mae":
            safe_mean(
                np.abs(
                    posterior_mean[
                        fraud
                    ]
                    - t_onset[
                        fraud
                    ]
                )
            ),

        "expected_abs_error":
            safe_mean(
                expected_abs_error[
                    fraud
                ]
            ),

        "mean_true_tf_probability":
            safe_mean(
                true_probability[
                    fraud
                ]
            ),

        "onset_nll":
            safe_mean(
                -np.log(
                    np.clip(
                        true_probability[
                            fraud
                        ],
                        1e-12,
                        1.0,
                    )
                )
            ),

        "mode_signed_error":
            safe_mean(
                mode_error[
                    fraud
                ]
            ),

        "posterior_mean_signed_error":
            safe_mean(
                mean_error[
                    fraud
                ]
            ),

        "mode_before_tf_rate":
            safe_mean(
                (
                    mode[
                        fraud
                    ]
                    < t_onset[
                        fraud
                    ]
                ).astype(
                    float
                )
            ),

        "mode_exact_tf_rate":
            safe_mean(
                (
                    mode[
                        fraud
                    ]
                    == t_onset[
                        fraud
                    ]
                ).astype(
                    float
                )
            ),

        "mode_after_tf_rate":
            safe_mean(
                (
                    mode[
                        fraud
                    ]
                    > t_onset[
                        fraud
                    ]
                ).astype(
                    float
                )
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
    }

    return {
        "metrics":
            metrics,

        "mode":
            mode,

        "posterior_mean":
            posterior_mean,

        "true_probability":
            true_probability,

        "expected_abs_error":
            expected_abs_error,
    }


# ============================================================
# Occurrence metrics
# ============================================================


def select_occurrence_threshold(
    labels,
    probabilities,
):
    candidates = np.linspace(
        0.05,
        0.95,
        181,
    )

    scores = []

    for threshold in candidates:

        prediction = (
            probabilities
            >= threshold
        )

        score = f1_score(
            labels,
            prediction,
            zero_division=0,
        )

        scores.append(
            score
        )

    return float(
        candidates[
            int(
                np.argmax(
                    scores
                )
            )
        ]
    )


def evaluate_occurrence(
    labels,
    probabilities,
    threshold,
):
    prediction = (
        probabilities
        >= threshold
    )

    return {
        "occurrence_auroc":
            safe_auc(
                roc_auc_score,
                labels,
                probabilities,
            ),

        "occurrence_auprc":
            safe_auc(
                average_precision_score,
                labels,
                probabilities,
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

        indices = np.asarray(
            subset.indices
        )

        return all_ids[
            indices
        ]

    return all_ids[
        : len(
            subset
        )
    ]


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

    RUN_DIR.mkdir(
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
    # Model
    # --------------------------------------------------------

    model = (
        DecoupledOccurrenceTimingModel(
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
    # Oracle checkpoint selection:
    # validation timing posterior-mean MAE
    # --------------------------------------------------------

    best_state = None

    best_epoch = -1

    best_val_mean_mae = (
        float("inf")
    )

    stale = 0

    history = []

    print()
    print(
        "Training decoupled occurrence/timing diagnostic..."
    )

    # --------------------------------------------------------
    # Training
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

        val_timing = evaluate_timing(
            val_payload
        )

        val_mean_mae = (
            val_timing[
                "metrics"
            ][
                "posterior_mean_mae"
            ]
        )

        row = {
            "epoch":
                epoch,

            "train_loss":
                train_payload[
                    "loss"
                ],

            "train_occurrence_loss":
                train_payload[
                    "occurrence_loss"
                ],

            "train_timing_loss":
                train_payload[
                    "timing_loss"
                ],

            "val_loss":
                val_payload[
                    "loss"
                ],

            "val_occurrence_loss":
                val_payload[
                    "occurrence_loss"
                ],

            "val_timing_loss":
                val_payload[
                    "timing_loss"
                ],

            "val_timing_mode_mae":
                val_timing[
                    "metrics"
                ][
                    "mode_mae"
                ],

            "val_timing_mean_mae":
                val_mean_mae,

            "val_true_tf_probability":
                val_timing[
                    "metrics"
                ][
                    "mean_true_tf_probability"
                ],
        }

        history.append(
            row
        )

        print(
            f"epoch={epoch:02d} "
            f"train_loss="
            f"{row['train_loss']:.4f} "
            f"val_loss="
            f"{row['val_loss']:.4f} "
            f"val_mode_mae="
            f"{row['val_timing_mode_mae']:.4f} "
            f"val_mean_mae="
            f"{row['val_timing_mean_mae']:.4f}"
        )

        if (
            val_mean_mae
            < best_val_mean_mae
        ):

            best_val_mean_mae = (
                val_mean_mae
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

    # --------------------------------------------------------
    # Restore best checkpoint
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    print()
    print(
        f"Best checkpoint: "
        f"epoch={best_epoch}, "
        f"val_timing_mean_mae="
        f"{best_val_mean_mae:.4f}"
    )

    torch.save(
        model.state_dict(),
        RUN_DIR
        / "model.pt",
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
    # Validation occurrence threshold
    # ========================================================

    val_payload = run_epoch(
        model,
        val_loader,
        optimizer=None,
    )

    val_occurrence_labels = (
        val_payload[
            "t_onset"
        ] >= 0
    ).astype(
        int
    )

    occurrence_threshold = (
        select_occurrence_threshold(
            val_occurrence_labels,
            val_payload[
                "occurrence_probability"
            ],
        )
    )

    # ========================================================
    # Held-out test
    # ========================================================

    test_payload = run_epoch(
        model,
        test_loader,
        optimizer=None,
    )

    test_occurrence_labels = (
        test_payload[
            "t_onset"
        ] >= 0
    ).astype(
        int
    )

    occurrence_metrics = (
        evaluate_occurrence(
            test_occurrence_labels,
            test_payload[
                "occurrence_probability"
            ],
            occurrence_threshold,
        )
    )

    timing_analysis = (
        evaluate_timing(
            test_payload
        )
    )

    timing_metrics = (
        timing_analysis[
            "metrics"
        ]
    )

    # ========================================================
    # Final summary
    # ========================================================

    summary = {
        "diagnostic":
            (
                "Decoupled occurrence-timing "
                "oracle diagnostic"
            ),

        "warning":
            (
                "True t_f supervises the timing head "
                "and validation onset MAE selects the "
                "checkpoint. Diagnostic only."
            ),

        "seed":
            SEED,

        "hidden_dim":
            HIDDEN_DIM,

        "occurrence_weight":
            OCCURRENCE_WEIGHT,

        "timing_weight":
            TIMING_WEIGHT,

        "best_epoch":
            best_epoch,

        "best_val_timing_mean_mae":
            best_val_mean_mae,

        "occurrence_threshold":
            occurrence_threshold,

        **occurrence_metrics,

        **timing_metrics,
    }

    print()
    print(
        "=" * 80
    )
    print(
        "DECOUPLED OCCURRENCE-TIMING RESULT"
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
        OUTPUT_DIR
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

    true_tf = (
        test_payload[
            "t_onset"
        ]
    )

    fraud = (
        true_tf >= 0
    )

    detected = (
        test_payload[
            "detected"
        ]
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

    posterior_mode = (
        timing_analysis[
            "mode"
        ]
    )

    posterior_mean = (
        timing_analysis[
            "posterior_mean"
        ]
    )

    entity_frame = pd.DataFrame(
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

            "occurrence_probability":
                test_payload[
                    "occurrence_probability"
                ],

            "timing_mode_tf":
                posterior_mode,

            "timing_mean_tf":
                posterior_mean,

            "timing_mean_signed_error":
                posterior_mean
                - true_tf,

            "true_tf_probability":
                timing_analysis[
                    "true_probability"
                ],

            "expected_abs_error":
                timing_analysis[
                    "expected_abs_error"
                ],
        }
    )

    entity_frame.to_csv(
        OUTPUT_DIR
        / "entity_diagnostics.csv",
        index=False,
    )

    np.savez_compressed(
        OUTPUT_DIR
        / "posterior_data.npz",

        entity_id=
            entity_ids,

        timing_posterior=
            test_payload[
                "timing_posterior"
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

    # ========================================================
    # Compare against previous diagnostics
    # ========================================================

    print()
    print(
        "=" * 80
    )
    print(
        "KEY COMPARISON"
    )
    print(
        "=" * 80
    )

    print(
        "Current decoupled timing posterior-mean MAE:",
        timing_metrics[
            "posterior_mean_mae"
        ],
    )

    print(
        "Current decoupled timing mode MAE:",
        timing_metrics[
            "mode_mae"
        ],
    )

    print(
        "Current decoupled P(true tf):",
        timing_metrics[
            "mean_true_tf_probability"
        ],
    )

    print(
        "Current decoupled mode-before-tf rate:",
        timing_metrics[
            "mode_before_tf_rate"
        ],
    )

    print(
        "Current occurrence AUROC:",
        occurrence_metrics[
            "occurrence_auroc"
        ],
    )

    print(
        "Current occurrence AUPRC:",
        occurrence_metrics[
            "occurrence_auprc"
        ],
    )

    # --------------------------------------------------------
    # Known previous values printed for convenient comparison.
    # --------------------------------------------------------

    print()
    print(
        "Reference:"
    )

    print(
        "  Joint-Sampling posterior-mean MAE = 4.6302"
    )

    print(
        "  Joint-Sampling mode MAE = 8.9111"
    )

    print(
        "  Fraud-only posterior-mean MAE = 2.7004"
    )

    print(
        "  Fraud-only mode MAE = 4.4593"
    )

    print(
        "  Balanced posterior-mean MAE = 2.9834"
    )

    print(
        "  Balanced mode MAE = 8.9852"
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
        / "entity_diagnostics.csv"
    )

    print(
        OUTPUT_DIR
        / "posterior_data.npz"
    )


if __name__ == "__main__":
    main()