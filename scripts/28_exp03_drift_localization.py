from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


# ============================================================
# Experiment 03U
# Drift Localization
#
# 中文：
# observed likelihood 导致的 short-delay drift
# 到底主要发生在哪里？
#
#
# Starting point:
#   Experiment 03R Oracle checkpoint
#
# Same starting weights for all groups.
#
#
# Four groups:
#
# A. freeze encoder + freeze timing head
#
# B. freeze encoder + train timing head
#
# C. train encoder + freeze timing head
#
# D. train encoder + train timing head
#
#
# Goal:
#
# distinguish:
#
#   representation drift
#
# vs
#
#   timing-head / objective drift
# ============================================================


BASE_SCRIPT = Path(
    r"scripts/25_exp03_objective_drift_audit.py"
)

CHECKPOINT = Path(
    r"reports/experiment03/"
    r"objective_drift_audit/"
    r"oracle_starting_point.pt"
)

OUTPUT_DIR = Path(
    r"reports/experiment03/"
    r"drift_localization"
)


SEED = 0

BATCH_SIZE = 64

LEARNING_RATE = 0.001

DRIFT_EPOCHS = 15


# ============================================================
# Load Experiment 03R utilities
# ============================================================


if not BASE_SCRIPT.exists():

    raise FileNotFoundError(
        BASE_SCRIPT
    )


if not CHECKPOINT.exists():

    raise FileNotFoundError(
        CHECKPOINT
    )


spec = importlib.util.spec_from_file_location(
    "exp03r",
    BASE_SCRIPT,
)

if spec is None or spec.loader is None:

    raise RuntimeError(
        "Could not load Experiment 03R."
    )


base = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    base
)


# ============================================================
# Configurations
# ============================================================


CONFIGS = [
    {
        "name": "freeze_all",
        "train_encoder": False,
        "train_head": False,
    },

    {
        "name": "head_only",
        "train_encoder": False,
        "train_head": True,
    },

    {
        "name": "encoder_only",
        "train_encoder": True,
        "train_head": False,
    },

    {
        "name": "train_both",
        "train_encoder": True,
        "train_head": True,
    },
]


# ============================================================
# Freeze / unfreeze
# ============================================================


def configure_trainable(
    model,
    train_encoder,
    train_head,
):

    for parameter in model.parameters():

        parameter.requires_grad = False

    if train_encoder:

        for parameter in model.encoder.parameters():

            parameter.requires_grad = True

    if train_head:

        for parameter in model.timing_head.parameters():

            parameter.requires_grad = True


# ============================================================
# Run one configuration
# ============================================================


def run_configuration(
    config,
    input_dim,
    train_loader,
    val_loader,
    test_loader,
    device,
    delay_log_prior,
):

    base.set_seed(
        SEED
    )

    model = base.TimingModel(
        input_dim=input_dim,
        hidden_dim=base.HIDDEN_DIM,
    ).to(
        device
    )

    state = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(
        state
    )

    configure_trainable(
        model,
        config["train_encoder"],
        config["train_head"],
    )

    # ========================================================
    # Starting point
    # ========================================================

    before = base.evaluate(
        model,
        test_loader,
        device,
        delay_log_prior,
    )

    print()

    print(
        "=" * 80
    )

    print(
        config["name"]
    )

    print(
        "=" * 80
    )

    print(
        "BEFORE:"
    )

    print(
        "  observed NLL:",
        before["observed_nll"],
    )

    print(
        "  raw tf MAE:",
        before[
            "raw_timing_mean_tf_mae"
        ],
    )

    print(
        "  posterior tf MAE:",
        before[
            "posterior_mean_tf_mae"
        ],
    )

    print(
        "  posterior mean delay:",
        before[
            "posterior_mean_delay"
        ],
    )

    # ========================================================
    # Frozen reference
    # ========================================================

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    history = []

    if len(trainable_parameters) == 0:

        print()

        print(
            "All parameters frozen."
        )

        after = before

    else:

        optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=LEARNING_RATE,
        )

        for epoch in range(
            DRIFT_EPOCHS
        ):

            model.train()

            train_loss = 0.0

            train_n = 0

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
                #
                # true_tf is NOT used here.

                optimizer.zero_grad()

                log_q, _ = model(
                    x,
                    td,
                )

                loss = base.observed_nll(
                    log_q,
                    td,
                    delay_log_prior,
                )

                loss.backward()

                optimizer.step()

                batch_size = len(
                    td
                )

                train_loss += float(
                    loss.item()
                ) * batch_size

                train_n += batch_size

            val = base.evaluate(
                model,
                val_loader,
                device,
                delay_log_prior,
            )

            row = {
                "config":
                    config["name"],

                "epoch":
                    epoch,

                "train_nll":
                    train_loss
                    / train_n,

                **val,
            }

            history.append(
                row
            )

            print(
                f"epoch={epoch:02d} "
                f"val_nll="
                f"{val['observed_nll']:.4f} "
                f"raw_tf_mae="
                f"{val['raw_timing_mean_tf_mae']:.4f} "
                f"post_tf_mae="
                f"{val['posterior_mean_tf_mae']:.4f} "
                f"post_delay="
                f"{val['posterior_mean_delay']:.4f}"
            )

        after = base.evaluate(
            model,
            test_loader,
            device,
            delay_log_prior,
        )

    # ========================================================
    # Comparison
    # ========================================================

    result = {
        "config":
            config["name"],

        "train_encoder":
            config[
                "train_encoder"
            ],

        "train_head":
            config[
                "train_head"
            ],

        "before_nll":
            before[
                "observed_nll"
            ],

        "after_nll":
            after[
                "observed_nll"
            ],

        "before_raw_tf_mae":
            before[
                "raw_timing_mean_tf_mae"
            ],

        "after_raw_tf_mae":
            after[
                "raw_timing_mean_tf_mae"
            ],

        "before_posterior_tf_mae":
            before[
                "posterior_mean_tf_mae"
            ],

        "after_posterior_tf_mae":
            after[
                "posterior_mean_tf_mae"
            ],

        "before_delay":
            before[
                "posterior_mean_delay"
            ],

        "after_delay":
            after[
                "posterior_mean_delay"
            ],
    }

    print()

    print(
        "AFTER:"
    )

    print(
        "  observed NLL:",
        after[
            "observed_nll"
        ],
    )

    print(
        "  raw tf MAE:",
        after[
            "raw_timing_mean_tf_mae"
        ],
    )

    print(
        "  posterior tf MAE:",
        after[
            "posterior_mean_tf_mae"
        ],
    )

    print(
        "  posterior mean delay:",
        after[
            "posterior_mean_delay"
        ],
    )

    if history:

        pd.DataFrame(
            history
        ).to_csv(
            OUTPUT_DIR
            / (
                config["name"]
                + "_history.csv"
            ),
            index=False,
        )

    return result


# ============================================================
# Main
# ============================================================


def main():

    base.set_seed(
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
        "EXPERIMENT 03U:"
    )

    print(
        "WHERE DOES THE DRIFT HAPPEN?"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Same Oracle starting checkpoint."
    )

    print()

    print(
        "Only observed likelihood is optimized."
    )

    print(
        "No true t_f is used during drift training."
    )

    print()

    # ========================================================
    # Data
    # ========================================================

    (
        x,
        true_tf,
        detected,
        td,
    ) = base.load_data()

    (
        train_idx,
        val_idx,
        test_idx,
    ) = base.split_indices(
        len(x),
        seed=SEED,
    )

    train_dataset = (
        base.DetectedFraudDataset(
            x,
            true_tf,
            detected,
            td,
            train_idx,
        )
    )

    val_dataset = (
        base.DetectedFraudDataset(
            x,
            true_tf,
            detected,
            td,
            val_idx,
        )
    )

    test_dataset = (
        base.DetectedFraudDataset(
            x,
            true_tf,
            detected,
            td,
            test_idx,
        )
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

    input_dim = x.shape[
        -1
    ]

    seq_len = x.shape[
        1
    ]

    delay_log_prior = (
        base.build_delay_log_prior(
            max_delay=seq_len,
            device=device,
        )
    )

    # ========================================================
    # Four groups
    # ========================================================

    results = []

    for config in CONFIGS:

        result = run_configuration(
            config,
            input_dim,
            train_loader,
            val_loader,
            test_loader,
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
        "config",
        "after_nll",
        "after_raw_tf_mae",
        "after_posterior_tf_mae",
        "after_delay",
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
        "Interpretation:"
    )

    print()

    print(
        "If HEAD-ONLY already collapses:"
    )

    print(
        "  Timing head / objective is sufficient "
        "to cause the shortcut."
    )

    print()

    print(
        "If ENCODER-ONLY collapses strongly:"
    )

    print(
        "  representation itself is being destroyed."
    )

    print()

    print(
        "If only TRAIN-BOTH collapses:"
    )

    print(
        "  the failure mainly comes from "
        "encoder-head interaction."
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