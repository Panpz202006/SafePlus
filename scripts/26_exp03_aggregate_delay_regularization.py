from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Experiment 03S
# Aggregate Delay Regularization
#
# 中文：
# 总体 detection-delay 分布正则化。
#
#
# Motivation:
#
# Experiment 03R showed:
#
#   observed NLL becomes better
#
# but:
#
#   true t_f recovery becomes worse
#   predicted delay collapses toward 1.
#
#
# This experiment adds:
#
#   L_total
#
#   =
#
#   L_observed
#
#   +
#
#   lambda * L_delay
#
#
# where L_delay prevents the AGGREGATE predicted delay
# distribution from collapsing to very short delays.
#
#
# IMPORTANT:
#
# No individual true t_f is used for training.
#
# True t_f is used ONLY for post-hoc synthetic evaluation.
#
#
# This is still a diagnostic intervention because the target
# global delay distribution comes from the synthetic generator.
# It is NOT the final real-data method.
# ============================================================


BASE_SCRIPT = Path(
    r"scripts/25_exp03_objective_drift_audit.py"
)

DATA_PATH = Path(
    r"data/processed/synthetic_same_delay_consistent.npz"
)

OUTPUT_DIR = Path(
    r"reports/experiment03/"
    r"aggregate_delay_regularization"
)


SEED = 0

HIDDEN_DIM = 32

BATCH_SIZE = 64

LEARNING_RATE = 0.001

MAX_EPOCHS = 40

PATIENCE = 7

DELAY_MEAN = 4.0


# ============================================================
# Regularization strengths
# ============================================================


LAMBDA_VALUES = [
    0.0,
    0.1,
    0.5,
    1.0,
]


# ============================================================
# Load Experiment 03R utilities
# ============================================================


if not BASE_SCRIPT.exists():

    raise FileNotFoundError(
        BASE_SCRIPT
    )


spec = importlib.util.spec_from_file_location(
    "exp03_objective_drift",
    BASE_SCRIPT,
)

if (
    spec is None
    or spec.loader is None
):

    raise RuntimeError(
        "Could not load Experiment 03R script."
    )


base = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    base
)


# Reuse strict-causal Timing architecture
TimingModel = base.TimingModel

load_data = base.load_data

split_indices = base.split_indices

build_delay_log_prior = (
    base.build_delay_log_prior
)

observed_nll = (
    base.observed_nll
)

evaluate = (
    base.evaluate
)

set_seed = (
    base.set_seed
)


# ============================================================
# Dataset
#
# IMPORTANT:
#
# We select users using detected == 1 only.
#
# true_tf is carried ONLY so that evaluate() can calculate
# synthetic diagnostic metrics after training.
#
# true_tf is NEVER used inside the training loss.
# ============================================================


class DetectedDataset(
    Dataset
):

    def __init__(
        self,
        x,
        true_tf,
        detected,
        td,
        indices,
    ):

        keep = [
            int(idx)
            for idx in indices
            if detected[idx] == 1
        ]

        self.indices = np.asarray(
            keep,
            dtype=np.int64,
        )

        self.x = x

        self.true_tf = true_tf

        self.td = td

    def __len__(
        self,
    ):

        return len(
            self.indices
        )

    def __getitem__(
        self,
        position,
    ):

        idx = int(
            self.indices[
                position
            ]
        )

        return {
            "entity_id":
                torch.tensor(
                    idx,
                    dtype=torch.long,
                ),

            "x":
                torch.from_numpy(
                    self.x[idx]
                ),

            "true_tf":
                torch.tensor(
                    self.true_tf[idx],
                    dtype=torch.long,
                ),

            "td":
                torch.tensor(
                    self.td[idx],
                    dtype=torch.long,
                ),
        }


# ============================================================
# Convert Timing q(tf) into q(delay)
#
#
# For a detected user:
#
#     delay = td - tf
#
#
# Example:
#
#     td = 12
#
#     tf = 11 -> delay=1
#     tf = 10 -> delay=2
#     tf =  9 -> delay=3
#     ...
#
#
# Input:
#
#     q shape = [B, T]
#
#
# Output:
#
#     q_delay shape = [B, T+1]
#
#
# index d stores probability of delay=d.
# ============================================================


def timing_to_delay_distribution(
    q,
    td,
):

    batch_size = (
        q.shape[0]
    )

    seq_len = (
        q.shape[1]
    )

    device = (
        q.device
    )

    steps = torch.arange(
        seq_len,
        device=device,
    )[None, :]

    delays = (
        td[:, None]
        - steps
    )

    valid = (
        delays >= 1
    )

    safe_delay = delays.clamp(
        min=0,
        max=seq_len,
    )

    q_valid = (
        q
        * valid.float()
    )

    delay_distribution = torch.zeros(
        batch_size,
        seq_len + 1,
        dtype=q.dtype,
        device=device,
    )

    delay_distribution.scatter_add_(
        dim=1,
        index=safe_delay,
        src=q_valid,
    )

    # Numerical normalization
    total = delay_distribution.sum(
        dim=1,
        keepdim=True,
    )

    delay_distribution = (
        delay_distribution
        / total.clamp_min(
            1e-12
        )
    )

    return delay_distribution


# ============================================================
# Build per-user legal target delay distribution
#
#
# Global Oracle prior:
#
#     P(delay=d)
#
#
# But different users have different td.
#
# Example:
#
#     td=3
#
# legal delays:
#
#     1,2,3
#
#
# Therefore we condition the global prior on:
#
#     delay <= td
#
#
# This avoids forcing impossible delay values on a user.
# ============================================================


def target_delay_distribution(
    td,
    delay_log_prior,
    seq_len,
):

    device = (
        td.device
    )

    delay_axis = torch.arange(
        seq_len + 1,
        device=device,
    )[None, :]

    prior_probability = torch.exp(
        delay_log_prior[
            :seq_len + 1
        ]
    )[None, :]

    legal = (
        (delay_axis >= 1)
        &
        (
            delay_axis
            <= td[:, None]
        )
    )

    target = (
        prior_probability
        * legal.float()
    )

    target = (
        target
        / target.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(
            1e-12
        )
    )

    return target


# ============================================================
# Aggregate delay regularization
#
#
# Predicted aggregate distribution:
#
#     average q_delay over current batch
#
#
# Target aggregate distribution:
#
#     average legal oracle prior over current batch
#
#
# KL:
#
#     KL(target || prediction)
#
#
# Why target || prediction?
#
# If prediction collapses around delay=1,
# but the target still has probability mass at 2,3,4,...,
# this direction penalizes missing probability mass.
# ============================================================


def aggregate_delay_regularization(
    q,
    td,
    delay_log_prior,
):

    seq_len = (
        q.shape[1]
    )

    predicted_user = (
        timing_to_delay_distribution(
            q,
            td,
        )
    )

    target_user = (
        target_delay_distribution(
            td,
            delay_log_prior,
            seq_len,
        )
    )

    predicted = predicted_user.mean(
        dim=0
    )

    target = target_user.mean(
        dim=0
    )

    # Ignore delay=0
    predicted = predicted[
        1:
    ]

    target = target[
        1:
    ]

    predicted = (
        predicted
        / predicted.sum().clamp_min(
            1e-12
        )
    )

    target = (
        target
        / target.sum().clamp_min(
            1e-12
        )
    )

    eps = 1e-10

    kl = torch.sum(
        target
        * (
            torch.log(
                target + eps
            )
            -
            torch.log(
                predicted + eps
            )
        )
    )

    return (
        kl,
        predicted,
        target,
    )


# ============================================================
# Validation loss
#
# IMPORTANT:
#
# Checkpoint selection does NOT use true t_f.
#
# It only uses:
#
#     observed NLL
#
#     +
#
#     lambda * delay regularization
#
# ============================================================


@torch.no_grad()
def validation_objective(
    model,
    loader,
    device,
    delay_log_prior,
    lambda_delay,
):

    model.eval()

    total_observed = 0.0

    total_reg = 0.0

    total_n = 0

    for batch in loader:

        x = batch[
            "x"
        ].to(
            device
        )

        td = batch[
            "td"
        ].to(
            device
        )

        log_q, q = model(
            x,
            td,
        )

        obs = observed_nll(
            log_q,
            td,
            delay_log_prior,
        )

        reg, _, _ = (
            aggregate_delay_regularization(
                q,
                td,
                delay_log_prior,
            )
        )

        batch_size = len(
            td
        )

        total_observed += float(
            obs.item()
        ) * batch_size

        total_reg += float(
            reg.item()
        ) * batch_size

        total_n += batch_size

    observed = (
        total_observed
        / total_n
    )

    regularization = (
        total_reg
        / total_n
    )

    objective = (
        observed
        +
        lambda_delay
        * regularization
    )

    return {
        "observed_nll":
            observed,

        "delay_regularization":
            regularization,

        "total_objective":
            objective,
    }


# ============================================================
# Aggregate test statistics
# ============================================================


@torch.no_grad()
def aggregate_delay_stats(
    model,
    loader,
    device,
    delay_log_prior,
):

    model.eval()

    predicted_sum = None

    target_sum = None

    total_n = 0

    for batch in loader:

        x = batch[
            "x"
        ].to(
            device
        )

        td = batch[
            "td"
        ].to(
            device
        )

        _, q = model(
            x,
            td,
        )

        seq_len = q.shape[
            1
        ]

        predicted_user = (
            timing_to_delay_distribution(
                q,
                td,
            )
        )

        target_user = (
            target_delay_distribution(
                td,
                delay_log_prior,
                seq_len,
            )
        )

        if predicted_sum is None:

            predicted_sum = (
                predicted_user.sum(
                    dim=0
                )
            )

            target_sum = (
                target_user.sum(
                    dim=0
                )
            )

        else:

            predicted_sum += (
                predicted_user.sum(
                    dim=0
                )
            )

            target_sum += (
                target_user.sum(
                    dim=0
                )
            )

        total_n += len(
            td
        )

    predicted = (
        predicted_sum
        / total_n
    )

    target = (
        target_sum
        / total_n
    )

    delay_axis = torch.arange(
        len(
            predicted
        ),
        device=device,
        dtype=torch.float32,
    )

    predicted_mean_delay = float(
        (
            predicted
            * delay_axis
        ).sum().item()
    )

    target_mean_delay = float(
        (
            target
            * delay_axis
        ).sum().item()
    )

    predicted_short_rate = float(
        predicted[
            1:3
        ].sum().item()
    )

    target_short_rate = float(
        target[
            1:3
        ].sum().item()
    )

    eps = 1e-10

    pred_nonzero = predicted[
        1:
    ]

    target_nonzero = target[
        1:
    ]

    kl = float(
        torch.sum(
            target_nonzero
            * (
                torch.log(
                    target_nonzero + eps
                )
                -
                torch.log(
                    pred_nonzero + eps
                )
            )
        ).item()
    )

    return {
        "aggregate_predicted_mean_delay":
            predicted_mean_delay,

        "aggregate_target_mean_delay":
            target_mean_delay,

        "aggregate_predicted_delay_le_2":
            predicted_short_rate,

        "aggregate_target_delay_le_2":
            target_short_rate,

        "aggregate_delay_kl":
            kl,
    }


# ============================================================
# Train one lambda
# ============================================================


def train_one_lambda(
    lambda_delay,
    train_loader,
    val_loader,
    test_loader,
    input_dim,
    device,
    delay_log_prior,
):

    # Same initialization across lambdas
    set_seed(
        SEED
    )

    model = TimingModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_state = None

    best_val_objective = float(
        "inf"
    )

    best_epoch = -1

    patience_count = 0

    history = []

    print()

    print(
        "=" * 80
    )

    print(
        f"LAMBDA = {lambda_delay}"
    )

    print(
        "=" * 80
    )

    for epoch in range(
        MAX_EPOCHS
    ):

        model.train()

        running_obs = 0.0

        running_reg = 0.0

        running_total = 0.0

        n_examples = 0

        for batch in train_loader:

            x = batch[
                "x"
            ].to(
                device
            )

            td = batch[
                "td"
            ].to(
                device
            )

            # IMPORTANT:
            # true_tf is NEVER read in training.

            optimizer.zero_grad()

            log_q, q = model(
                x,
                td,
            )

            obs_loss = observed_nll(
                log_q,
                td,
                delay_log_prior,
            )

            reg_loss, _, _ = (
                aggregate_delay_regularization(
                    q,
                    td,
                    delay_log_prior,
                )
            )

            total_loss = (
                obs_loss
                +
                lambda_delay
                * reg_loss
            )

            total_loss.backward()

            optimizer.step()

            batch_size = len(
                td
            )

            running_obs += float(
                obs_loss.item()
            ) * batch_size

            running_reg += float(
                reg_loss.item()
            ) * batch_size

            running_total += float(
                total_loss.item()
            ) * batch_size

            n_examples += batch_size

        val = validation_objective(
            model,
            val_loader,
            device,
            delay_log_prior,
            lambda_delay,
        )

        # ----------------------------------------------------
        # Diagnostic metrics:
        #
        # true_tf is used ONLY here for synthetic monitoring.
        #
        # It does NOT select the checkpoint.
        # ----------------------------------------------------

        val_diag = evaluate(
            model,
            val_loader,
            device,
            delay_log_prior,
        )

        row = {
            "lambda":
                lambda_delay,

            "epoch":
                epoch,

            "train_observed_nll":
                running_obs
                / n_examples,

            "train_delay_regularization":
                running_reg
                / n_examples,

            "train_total_loss":
                running_total
                / n_examples,

            "val_observed_nll":
                val[
                    "observed_nll"
                ],

            "val_delay_regularization":
                val[
                    "delay_regularization"
                ],

            "val_total_objective":
                val[
                    "total_objective"
                ],

            "diag_tf_mae":
                val_diag[
                    "posterior_mean_tf_mae"
                ],

            "diag_delay":
                val_diag[
                    "posterior_mean_delay"
                ],
        }

        history.append(
            row
        )

        print(
            f"epoch={epoch:02d} "
            f"train_obs="
            f"{row['train_observed_nll']:.4f} "
            f"train_reg="
            f"{row['train_delay_regularization']:.4f} "
            f"val_obs="
            f"{row['val_observed_nll']:.4f} "
            f"val_reg="
            f"{row['val_delay_regularization']:.4f} "
            f"val_obj="
            f"{row['val_total_objective']:.4f} "
            f"diag_tf_mae="
            f"{row['diag_tf_mae']:.4f} "
            f"diag_delay="
            f"{row['diag_delay']:.4f}"
        )

        # ----------------------------------------------------
        # Checkpoint selection:
        #
        # ONLY observable training objective.
        # ----------------------------------------------------

        if (
            val[
                "total_objective"
            ]
            <
            best_val_objective
        ):

            best_val_objective = (
                val[
                    "total_objective"
                ]
            )

            best_epoch = (
                epoch
            )

            best_state = {
                k:
                    v.detach()
                    .cpu()
                    .clone()

                for k, v in (
                    model.state_dict()
                    .items()
                )
            }

            patience_count = 0

        else:

            patience_count += 1

            if (
                patience_count
                >= PATIENCE
            ):

                print(
                    "Early stopping."
                )

                break

    if best_state is None:

        raise RuntimeError(
            "No checkpoint selected."
        )

    model.load_state_dict(
        best_state
    )

    model.to(
        device
    )

    # ========================================================
    # TEST
    # ========================================================

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        delay_log_prior,
    )

    aggregate_metrics = (
        aggregate_delay_stats(
            model,
            test_loader,
            device,
            delay_log_prior,
        )
    )

    result = {
        "lambda":
            lambda_delay,

        "best_epoch":
            best_epoch,

        "best_val_objective":
            best_val_objective,

        **test_metrics,

        **aggregate_metrics,
    }

    print()

    print(
        "TEST RESULT"
    )

    print(
        "  lambda:",
        lambda_delay,
    )

    print(
        "  best epoch:",
        best_epoch,
    )

    print(
        "  observed NLL:",
        result[
            "observed_nll"
        ],
    )

    print(
        "  raw Timing tf MAE:",
        result[
            "raw_timing_mean_tf_mae"
        ],
    )

    print(
        "  posterior tf MAE:",
        result[
            "posterior_mean_tf_mae"
        ],
    )

    print(
        "  posterior mean delay:",
        result[
            "posterior_mean_delay"
        ],
    )

    print(
        "  true delay mean:",
        result[
            "true_delay_mean"
        ],
    )

    print(
        "  aggregate predicted delay:",
        result[
            "aggregate_predicted_mean_delay"
        ],
    )

    print(
        "  aggregate target delay:",
        result[
            "aggregate_target_mean_delay"
        ],
    )

    print(
        "  aggregate delay KL:",
        result[
            "aggregate_delay_kl"
        ],
    )

    lambda_name = str(
        lambda_delay
    ).replace(
        ".",
        "p",
    )

    pd.DataFrame(
        history
    ).to_csv(
        OUTPUT_DIR
        / (
            f"lambda_{lambda_name}"
            f"_history.csv"
        ),
        index=False,
    )

    torch.save(
        model.state_dict(),
        OUTPUT_DIR
        / (
            f"lambda_{lambda_name}"
            f"_best.pt"
        ),
    )

    return result


# ============================================================
# Main
# ============================================================


def main():

    set_seed(
        SEED
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "=" * 80
    )

    print(
        "EXPERIMENT 03S:"
    )

    print(
        "AGGREGATE DELAY REGULARIZATION"
    )

    print(
        "=" * 80
    )

    print(
        "Device:",
        device,
    )

    print()

    print(
        "核心问题："
    )

    print(
        "Observed likelihood 会把 Timing 推向"
    )

    print(
        "short-delay solution。"
    )

    print()

    print(
        "本实验加入总体 delay distribution 约束："
    )

    print(
        "L = L_observed + lambda * L_delay"
    )

    print()

    print(
        "训练和 checkpoint selection"
    )

    print(
        "均不使用真实 t_f。"
    )

    print(
        "=" * 80
    )

    # ========================================================
    # Data
    # ========================================================

    (
        x,
        true_tf,
        detected,
        td,
    ) = load_data()

    (
        train_idx,
        val_idx,
        test_idx,
    ) = split_indices(
        len(
            x
        ),
        seed=SEED,
    )

    train_dataset = DetectedDataset(
        x,
        true_tf,
        detected,
        td,
        train_idx,
    )

    val_dataset = DetectedDataset(
        x,
        true_tf,
        detected,
        td,
        val_idx,
    )

    test_dataset = DetectedDataset(
        x,
        true_tf,
        detected,
        td,
        test_idx,
    )

    print()

    print(
        "Detected-user split sizes:"
    )

    print(
        "  train:",
        len(
            train_dataset
        ),
    )

    print(
        "  val:",
        len(
            val_dataset
        ),
    )

    print(
        "  test:",
        len(
            test_dataset
        ),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    input_dim = (
        x.shape[-1]
    )

    seq_len = (
        x.shape[1]
    )

    delay_log_prior = (
        build_delay_log_prior(
            max_delay=seq_len,
            device=device,
        )
    )

    # ========================================================
    # Sweep lambda
    # ========================================================

    results = []

    for lambda_delay in (
        LAMBDA_VALUES
    ):

        result = train_one_lambda(
            lambda_delay,
            train_loader,
            val_loader,
            test_loader,
            input_dim,
            device,
            delay_log_prior,
        )

        results.append(
            result
        )

    summary = pd.DataFrame(
        results
    )

    summary.to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False,
    )

    # ========================================================
    # Compact comparison
    # ========================================================

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

    columns = [
        "lambda",
        "observed_nll",
        "raw_timing_mean_tf_mae",
        "posterior_mean_tf_mae",
        "posterior_mean_delay",
        "true_delay_mean",
        "aggregate_predicted_mean_delay",
        "aggregate_delay_kl",
    ]

    print(
        summary[
            columns
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "HOW TO READ:"
    )

    print()

    print(
        "Good signal:"
    )

    print(
        "  lambda > 0"
    )

    print(
        "  predicted delay moves away from 1"
    )

    print(
        "  tf MAE decreases"
    )

    print(
        "  observed NLL does not explode"
    )

    print()

    print(
        "If delay distribution becomes correct"
    )

    print(
        "but tf MAE does NOT improve,"
    )

    print(
        "global aggregate constraint is insufficient."
    )

    print()

    print(
        "Then we need user-conditioned /"
    )

    print(
        "behavior-grounded delay information."
    )

    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT_DIR
        / "summary.csv"
    )


if __name__ == "__main__":
    main()