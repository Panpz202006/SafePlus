from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from safeplus.data import load_npz, make_loaders
from safeplus.models.common import SequenceEncoder, time_mask
from safeplus.utils import seed_everything


DATA_PATH = Path(
    r"data/processed/synthetic_demo.npz"
)

OUTPUT_DIR = Path(
    r"reports/experiment03/oracle_objective_diagnostic"
)

SEED = 0
DEVICE = "cpu"

HIDDEN_DIM = 32
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

MAX_EPOCHS = 40
PATIENCE = 7

SPLIT = [0.7, 0.1, 0.2]


class OracleOnsetModel(nn.Module):

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

        log_h = F.logsigmoid(
            logits
        )

        log_s = F.logsigmoid(
            -logits
        )

        log_s_valid = log_s.masked_fill(
            ~mask,
            0.0,
        )

        cumulative_log_survival = (
            log_s_valid.cumsum(dim=1)
        )

        before = torch.cat(
            [
                logits.new_zeros(
                    (len(logits), 1)
                ),
                cumulative_log_survival[:, :-1],
            ],
            dim=1,
        )

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

        no_commission_log_prob = (
            log_s_valid.sum(dim=1)
        )

        no_commission_prob = (
            no_commission_log_prob.exp()
        )

        return {
            "event_log_mass":
                event_log_mass,

            "event_mass":
                event_mass,

            "no_commission_log_prob":
                no_commission_log_prob,

            "no_commission_prob":
                no_commission_prob,
        }


def move_batch(batch):

    return {
        key: value.to(DEVICE)
        for key, value in batch.items()
    }


def compute_loss(
    model,
    batch,
    objective,
):
    out = model(
        batch["x"],
        batch["lengths"],
    )

    tf = batch[
        "t_onset"
    ]

    fraud = (
        tf >= 0
    )

    normal = (
        ~fraud
    )

    fraud_indices = torch.where(
        fraud
    )[0]

    normal_indices = torch.where(
        normal
    )[0]

    fraud_loss = None
    normal_loss = None

    if len(fraud_indices) > 0:

        true_tf = tf[
            fraud_indices
        ]

        fraud_log_prob = out[
            "event_log_mass"
        ][
            fraud_indices,
            true_tf,
        ]

        fraud_loss = (
            -fraud_log_prob.mean()
        )

    if len(normal_indices) > 0:

        normal_log_prob = out[
            "no_commission_log_prob"
        ][
            normal_indices
        ]

        normal_loss = (
            -normal_log_prob.mean()
        )

    if objective == "fraud_only":

        if fraud_loss is None:
            return None, out

        loss = fraud_loss

    elif objective == "balanced":

        losses = []

        if fraud_loss is not None:
            losses.append(
                fraud_loss
            )

        if normal_loss is not None:
            losses.append(
                normal_loss
            )

        loss = torch.stack(
            losses
        ).mean()

    else:

        raise ValueError(
            f"Unknown objective: {objective}"
        )

    return loss, out


def run_epoch(
    model,
    loader,
    objective,
    optimizer=None,
):
    training = (
        optimizer is not None
    )

    model.train(
        training
    )

    total_loss = 0.0
    loss_batches = 0

    event_masses = []
    no_commissions = []

    onsets = []
    detected = []

    for raw_batch in loader:

        batch = move_batch(
            raw_batch
        )

        with torch.set_grad_enabled(
            training
        ):

            loss, out = compute_loss(
                model,
                batch,
                objective,
            )

            if (
                training
                and loss is not None
            ):

                optimizer.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    5.0,
                )

                optimizer.step()

        if loss is not None:

            total_loss += float(
                loss.detach()
            )

            loss_batches += 1

        event_masses.append(
            out[
                "event_mass"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        no_commissions.append(
            out[
                "no_commission_prob"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        onsets.append(
            batch[
                "t_onset"
            ]
            .detach()
            .cpu()
            .numpy()
        )

        detected.append(
            batch[
                "detected"
            ]
            .detach()
            .cpu()
            .numpy()
            .astype(bool)
        )

    return {
        "loss":
            total_loss
            / max(
                1,
                loss_batches,
            ),

        "event_mass":
            np.concatenate(
                event_masses
            ),

        "no_commission":
            np.concatenate(
                no_commissions
            ),

        "t_onset":
            np.concatenate(
                onsets
            ),

        "detected":
            np.concatenate(
                detected
            ),
    }


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


def evaluate_onset(payload):

    event_mass = payload[
        "event_mass"
    ]

    tf = payload[
        "t_onset"
    ]

    no_commission = payload[
        "no_commission"
    ]

    known = (
        tf >= 0
    )

    normal = (
        ~known
    )

    detected = payload[
        "detected"
    ]

    detected_fraud = (
        known
        & detected
    )

    censored_fraud = (
        known
        & ~detected
    )

    times = np.arange(
        event_mass.shape[1],
        dtype=np.float64,
    )

    mode = event_mass.argmax(
        axis=1
    )

    onset_mass = event_mass.sum(
        axis=1
    )

    posterior_mean = np.full(
        len(tf),
        np.nan,
        dtype=np.float64,
    )

    valid = (
        onset_mass > 0
    )

    posterior_mean[
        valid
    ] = (
        event_mass[
            valid
        ]
        * times[None, :]
    ).sum(
        axis=1
    ) / onset_mass[
        valid
    ]

    true_probability = np.full(
        len(tf),
        np.nan,
        dtype=np.float64,
    )

    expected_abs_error = np.full(
        len(tf),
        np.nan,
        dtype=np.float64,
    )

    known_indices = np.where(
        known
    )[0]

    for i in known_indices:

        true_probability[
            i
        ] = event_mass[
            i,
            tf[i],
        ]

        if onset_mass[i] > 0:

            conditional = (
                event_mass[i]
                / onset_mass[i]
            )

            expected_abs_error[
                i
            ] = np.sum(
                conditional
                * np.abs(
                    times
                    - tf[i]
                )
            )

    signed_mode_error = (
        mode - tf
    )

    signed_mean_error = (
        posterior_mean - tf
    )

    return {
        "n_fraud":
            int(
                known.sum()
            ),

        "n_normal":
            int(
                normal.sum()
            ),

        "mode_mae":
            safe_mean(
                np.abs(
                    mode[
                        known
                    ]
                    - tf[
                        known
                    ]
                )
            ),

        "posterior_mean_mae":
            safe_mean(
                np.abs(
                    posterior_mean[
                        known
                    ]
                    - tf[
                        known
                    ]
                )
            ),

        "expected_abs_error":
            safe_mean(
                expected_abs_error[
                    known
                ]
            ),

        "mean_true_tf_probability":
            safe_mean(
                true_probability[
                    known
                ]
            ),

        "onset_nll":
            safe_mean(
                -np.log(
                    np.clip(
                        true_probability[
                            known
                        ],
                        1e-12,
                        1.0,
                    )
                )
            ),

        "mode_signed_error":
            safe_mean(
                signed_mode_error[
                    known
                ]
            ),

        "mean_signed_error":
            safe_mean(
                signed_mean_error[
                    known
                ]
            ),

        "mode_before_tf_rate":
            safe_mean(
                (
                    mode[
                        known
                    ]
                    < tf[
                        known
                    ]
                ).astype(
                    float
                )
            ),

        "mode_exact_tf_rate":
            safe_mean(
                (
                    mode[
                        known
                    ]
                    == tf[
                        known
                    ]
                ).astype(
                    float
                )
            ),

        "mode_after_tf_rate":
            safe_mean(
                (
                    mode[
                        known
                    ]
                    > tf[
                        known
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
                    - tf[
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
                    - tf[
                        censored_fraud
                    ]
                )
            ),

        "normal_no_commission_probability":
            safe_mean(
                no_commission[
                    normal
                ]
            ),
    }


def train_one_objective(
    objective,
    dataset,
    train_loader,
    val_loader,
    test_loader,
):

    seed_everything(
        SEED
    )

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

    best_state = None

    best_epoch = -1

    best_val_mean_mae = (
        float("inf")
    )

    stale = 0

    history = []

    print()
    print(
        "=" * 80
    )

    print(
        f"Training objective: {objective}"
    )

    print(
        "=" * 80
    )

    for epoch in range(
        MAX_EPOCHS
    ):

        train_payload = run_epoch(
            model,
            train_loader,
            objective,
            optimizer=optimizer,
        )

        val_payload = run_epoch(
            model,
            val_loader,
            objective,
            optimizer=None,
        )

        val_metrics = evaluate_onset(
            val_payload
        )

        current_mae = (
            val_metrics[
                "posterior_mean_mae"
            ]
        )

        history.append(
            {
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
                    val_metrics[
                        "mode_mae"
                    ],

                "val_mean_mae":
                    current_mae,

                "val_expected_abs_error":
                    val_metrics[
                        "expected_abs_error"
                    ],

                "val_true_tf_probability":
                    val_metrics[
                        "mean_true_tf_probability"
                    ],
            }
        )

        print(
            f"epoch={epoch:02d} "
            f"train_loss="
            f"{train_payload['loss']:.4f} "
            f"val_loss="
            f"{val_payload['loss']:.4f} "
            f"val_mode_mae="
            f"{val_metrics['mode_mae']:.4f} "
            f"val_mean_mae="
            f"{current_mae:.4f}"
        )

        if (
            current_mae
            < best_val_mean_mae
        ):

            best_val_mean_mae = (
                current_mae
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
            "No checkpoint obtained."
        )

    model.load_state_dict(
        best_state
    )

    test_payload = run_epoch(
        model,
        test_loader,
        objective,
        optimizer=None,
    )

    test_metrics = evaluate_onset(
        test_payload
    )

    result = {
        "objective":
            objective,

        "best_epoch":
            best_epoch,

        "best_val_posterior_mean_mae":
            best_val_mean_mae,

        **test_metrics,
    }

    print()

    print(
        f"BEST CHECKPOINT: "
        f"epoch={best_epoch}, "
        f"val_mean_mae="
        f"{best_val_mean_mae:.4f}"
    )

    print()

    print(
        f"TEST RESULT: {objective}"
    )

    for key, value in (
        result.items()
    ):

        print(
            f"{key}: {value}"
        )

    return (
        result,
        history,
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_everything(
        SEED
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

    all_results = {}

    all_histories = {}

    for objective in (
        "fraud_only",
        "balanced",
    ):

        result, history = (
            train_one_objective(
                objective,
                dataset,
                train_loader,
                val_loader,
                test_loader,
            )
        )

        all_results[
            objective
        ] = result

        all_histories[
            objective
        ] = history

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

    (
        OUTPUT_DIR
        / "history.json"
    ).write_text(
        json.dumps(
            all_histories,
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
        "FINAL COMPARISON"
    )

    print(
        "=" * 80
    )

    for objective in (
        "fraud_only",
        "balanced",
    ):

        result = all_results[
            objective
        ]

        print()
        print(
            objective
        )

        print(
            "  posterior mean MAE:",
            result[
                "posterior_mean_mae"
            ],
        )

        print(
            "  mode MAE:",
            result[
                "mode_mae"
            ],
        )

        print(
            "  expected abs error:",
            result[
                "expected_abs_error"
            ],
        )

        print(
            "  P(true tf):",
            result[
                "mean_true_tf_probability"
            ],
        )

        print(
            "  mode before tf:",
            result[
                "mode_before_tf_rate"
            ],
        )

        print(
            "  normal no-commission:",
            result[
                "normal_no_commission_probability"
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
        / "history.json"
    )


if __name__ == "__main__":
    main()