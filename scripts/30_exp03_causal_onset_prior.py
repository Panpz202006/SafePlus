from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


# ============================================================
# Experiment 03W
# Causal Onset Prior
#
# 中文：
# 因果式 onset prior 诊断。
#
#
# Previous endpoint-conditioned Timing:
#
#     candidate tf=k
#
#     uses:
#
#         h_k
#         AND
#         h_td
#
# This gives Timing direct access to the detection endpoint.
#
#
# Current experiment:
#
#     candidate tf=k
#
#     ONLY uses:
#
#         x_0 ... x_k
#
# through a causal hazard:
#
#     h_f(k)
#
#
# Onset probability:
#
# P(tf=k | x)
#
# =
#
# h_f(k)
# *
# product_{j<k}(1-h_f(j))
#
#
# Then detection time enters separately through:
#
#     P(delay = td-k)
#
#
# Training NEVER uses true tf.
#
# true tf is used only for synthetic diagnostics.
# ============================================================


SSL_SCRIPT = Path(
    r"scripts/29_exp03_self_supervised_representation.py"
)

OUTPUT_DIR = Path(
    r"reports/experiment03/"
    r"causal_onset_prior"
)

OLD_RESULT = Path(
    r"reports/experiment03/"
    r"self_supervised_representation/"
    r"latent_summary.csv"
)


SEED = 0

HIDDEN_DIM = 32

BATCH_SIZE = 64

LEARNING_RATE = 0.001

MAX_EPOCHS = 30

PATIENCE = 7


# ============================================================
# Load 03V
# ============================================================


if not SSL_SCRIPT.exists():

    raise FileNotFoundError(
        SSL_SCRIPT
    )


spec = importlib.util.spec_from_file_location(
    "exp03v",
    SSL_SCRIPT,
)

if spec is None or spec.loader is None:

    raise RuntimeError(
        "Could not load Experiment 03V."
    )


sslmod = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    sslmod
)


base = sslmod.base


# ============================================================
# Causal onset model
#
# IMPORTANT:
#
# No endpoint representation h_td is supplied
# to the onset scorer.
# ============================================================


class CausalOnsetModel(
    nn.Module
):

    def __init__(
        self,
        input_dim,
        hidden_dim,
    ):

        super().__init__()

        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # ----------------------------------------------------
        # Only candidate-local causal representation h_k.
        # ----------------------------------------------------

        self.onset_head = nn.Sequential(
            nn.Linear(
                hidden_dim,
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
        td,
    ):

        hidden, _ = self.encoder(
            x
        )

        logits = (
            self.onset_head(
                hidden
            )
            .squeeze(-1)
        )

        # ====================================================
        # Discrete commission hazard
        # ====================================================

        log_hazard = F.logsigmoid(
            logits
        )

        log_survival = F.logsigmoid(
            -logits
        )

        # ----------------------------------------------------
        # cumulative survival BEFORE each t
        #
        # t=0:
        #     survival before event = 1
        #
        # t=1:
        #     survive step 0
        #
        # ...
        # ----------------------------------------------------

        cumulative_survival = (
            torch.cumsum(
                log_survival,
                dim=1,
            )
        )

        survival_before = torch.cat(
            [
                torch.zeros_like(
                    cumulative_survival[
                        :,
                        :1,
                    ]
                ),

                cumulative_survival[
                    :,
                    :-1,
                ],
            ],
            dim=1,
        )

        # ====================================================
        # Log onset mass
        #
        # log P(tf=t | x)
        #
        # =
        #
        # log h_t
        #
        # +
        #
        # sum_{j<t} log(1-h_j)
        # ====================================================

        log_onset_mass = (
            log_hazard
            +
            survival_before
        )

        seq_len = (
            x.shape[1]
        )

        steps = torch.arange(
            seq_len,
            device=x.device,
        )[None, :]

        # ----------------------------------------------------
        # For detected users:
        #
        # tf < td
        #
        # delay >= 1
        # ----------------------------------------------------

        valid = (
            steps
            < td[:, None]
        )

        log_onset_mass = (
            log_onset_mass.masked_fill(
                ~valid,
                -torch.inf,
            )
        )

        # ====================================================
        # Conditional Timing distribution among legal
        # candidate onset times.
        #
        # NOTE:
        #
        # td only determines which candidate times are legal.
        # It does NOT enter the onset score itself.
        # ====================================================

        log_q = (
            log_onset_mass
            -
            torch.logsumexp(
                log_onset_mass,
                dim=1,
                keepdim=True,
            )
        )

        q = log_q.exp()

        return (
            log_q,
            q,
        )


# ============================================================
# Copy encoder state
# ============================================================


def clone_state(
    module,
):

    return {
        key:
            value.detach()
            .cpu()
            .clone()

        for key, value
        in module.state_dict().items()
    }


# ============================================================
# Train one configuration
# ============================================================


def run_one(
    name,
    encoder_state,
    train_encoder,
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

    model = CausalOnsetModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
    ).to(
        device
    )

    if encoder_state is not None:

        model.encoder.load_state_dict(
            encoder_state
        )

    if not train_encoder:

        for parameter in (
            model.encoder.parameters()
        ):

            parameter.requires_grad = False

    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.Adam(
        trainable,
        lr=LEARNING_RATE,
    )

    best_state = None

    best_val_nll = float(
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
        name
    )

    print(
        "=" * 80
    )

    for epoch in range(
        MAX_EPOCHS
    ):

        model.train()

        train_sum = 0.0

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

            # ------------------------------------------------
            # true_tf is NEVER used in training.
            # ------------------------------------------------

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

            train_sum += float(
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
            "name":
                name,

            "epoch":
                epoch,

            "train_nll":
                train_sum
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
            f"delay="
            f"{val['posterior_mean_delay']:.4f}"
        )

        # ----------------------------------------------------
        # Realistic checkpoint selection:
        #
        # only observed NLL.
        #
        # true tf does NOT select checkpoint.
        # ----------------------------------------------------

        if (
            val[
                "observed_nll"
            ]
            <
            best_val_nll
        ):

            best_val_nll = (
                val[
                    "observed_nll"
                ]
            )

            best_epoch = (
                epoch
            )

            best_state = clone_state(
                model
            )

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
            "No checkpoint."
        )

    model.load_state_dict(
        best_state
    )

    test = base.evaluate(
        model,
        test_loader,
        device,
        delay_log_prior,
    )

    result = {
        "name":
            name,

        "train_encoder":
            train_encoder,

        "best_epoch":
            best_epoch,

        "observed_nll":
            test[
                "observed_nll"
            ],

        "raw_tf_mae":
            test[
                "raw_timing_mean_tf_mae"
            ],

        "posterior_tf_mae":
            test[
                "posterior_mean_tf_mae"
            ],

        "posterior_delay":
            test[
                "posterior_mean_delay"
            ],

        "true_delay":
            test[
                "true_delay_mean"
            ],
    }

    pd.DataFrame(
        history
    ).to_csv(
        OUTPUT_DIR
        / f"{name}_history.csv",
        index=False,
    )

    print()

    print(
        "TEST:"
    )

    for key, value in (
        result.items()
    ):

        print(
            key,
            ":",
            value,
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
        "EXPERIMENT 03W:"
    )

    print(
        "CAUSAL ONSET PRIOR"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Key change:"
    )

    print(
        "Timing scorer no longer receives h_td."
    )

    print(
        "Candidate tf=k only uses x_0 ... x_k."
    )

    print()

    print(
        "td enters ONLY through:"
    )

    print(
        "  1. legal boundary tf < td"
    )

    print(
        "  2. detection-delay likelihood"
    )

    print()

    print(
        "Training uses NO true tf."
    )

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
        len(
            x
        ),
        seed=SEED,
    )

    input_dim = (
        x.shape[-1]
    )

    seq_len = (
        x.shape[1]
    )

    # ========================================================
    # Self-supervised Encoder
    #
    # Re-run the same 03V SSL pretraining.
    # It uses ONLY x.
    # ========================================================

    ssl_encoder_state, ssl_mse = (
        sslmod.pretrain_ssl(
            x,
            train_idx,
            val_idx,
            device,
        )
    )

    print()

    print(
        "SSL validation MSE:",
        ssl_mse,
    )

    # ========================================================
    # Random Encoder reference
    # ========================================================

    base.set_seed(
        SEED
    )

    random_model = CausalOnsetModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
    )

    random_encoder_state = clone_state(
        random_model.encoder
    )

    # ========================================================
    # Detected fraud loaders
    # ========================================================

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

    delay_log_prior = (
        base.build_delay_log_prior(
            max_delay=seq_len,
            device=device,
        )
    )

    # ========================================================
    # Three configurations
    # ========================================================

    results = []

    results.append(
        run_one(
            name="causal_random_frozen",
            encoder_state=random_encoder_state,
            train_encoder=False,
            input_dim=input_dim,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            delay_log_prior=delay_log_prior,
        )
    )

    results.append(
        run_one(
            name="causal_ssl_frozen",
            encoder_state=ssl_encoder_state,
            train_encoder=False,
            input_dim=input_dim,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            delay_log_prior=delay_log_prior,
        )
    )

    results.append(
        run_one(
            name="causal_ssl_trainable",
            encoder_state=ssl_encoder_state,
            train_encoder=True,
            input_dim=input_dim,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            delay_log_prior=delay_log_prior,
        )
    )

    new_summary = pd.DataFrame(
        results
    )

    new_summary.to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False,
    )

    # ========================================================
    # Old endpoint-conditioned references from 03V
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "CAUSAL TIMING RESULT"
    )

    print(
        "=" * 80
    )

    print(
        new_summary.to_string(
            index=False
        )
    )

    if OLD_RESULT.exists():

        old = pd.read_csv(
            OLD_RESULT
        )

        print()

        print(
            "=" * 80
        )

        print(
            "03V ENDPOINT-CONDITIONED REFERENCE"
        )

        print(
            "=" * 80
        )

        print(
            old.to_string(
                index=False
            )
        )

    print()

    print(
        "=" * 80
    )

    print(
        "HOW TO READ"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "If causal_ssl_frozen is clearly better "
        "than 03V ssl_frozen:"
    )

    print(
        "  endpoint conditioning was an important shortcut."
    )

    print()

    print(
        "If causal variants still collapse toward delay=1:"
    )

    print(
        "  removing h_td is not sufficient."
    )

    print(
        "  The observed td likelihood itself is still "
        "insufficient to identify user-level tf."
    )

    print()

    print(
        "Then the next method must introduce an "
        "explicit behavior-likelihood / onset-evidence term."
    )


if __name__ == "__main__":
    main()