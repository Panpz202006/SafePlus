from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Experiment 03V
# Self-Supervised Representation Preservation
#
# 中文：
# 自监督行为表示 + 表示保护诊断
#
#
# Core questions:
#
# 1. 不使用 true t_f / t_d，
#    能否从行为序列 x 中预训练出有 onset 信息的 Encoder？
#
# 2. 如果这个 Encoder 有用，
#    冻结它以后做 latent Timing，
#    能否减轻 observed likelihood 导致的 representation drift？
#
#
# Self-supervised task:
#
#     x_0 ... x_t
#
#          ↓
#
#        GRU
#
#          ↓
#
#     predict x_{t+1}
#
#
# No fraud label is used.
# No true t_f is used.
# No t_d is used during SSL pretraining.
# ============================================================


BASE_SCRIPT = Path(
    r"scripts/25_exp03_objective_drift_audit.py"
)

OUTPUT_DIR = Path(
    r"reports/experiment03/"
    r"self_supervised_representation"
)


SEED = 0

HIDDEN_DIM = 32

BATCH_SIZE = 64

LEARNING_RATE = 0.001

SSL_EPOCHS = 25

PROBE_EPOCHS = 20

LATENT_EPOCHS = 20

PATIENCE = 6


# ============================================================
# Load 03R utilities
# ============================================================


if not BASE_SCRIPT.exists():

    raise FileNotFoundError(
        BASE_SCRIPT
    )


spec = importlib.util.spec_from_file_location(
    "exp03r",
    BASE_SCRIPT,
)

if (
    spec is None
    or spec.loader is None
):

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
# Dataset for self-supervised pretraining
#
# No labels are used.
# ============================================================


class SequenceDataset(
    Dataset
):

    def __init__(
        self,
        x,
        indices,
    ):

        self.x = x

        self.indices = np.asarray(
            indices,
            dtype=np.int64,
        )

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

        return torch.from_numpy(
            self.x[
                idx
            ]
        )


# ============================================================
# Self-supervised next-step predictor
#
# Encoder:
#
#     x_0 ... x_t -> h_t
#
# Predictor:
#
#     h_t -> predicted x_{t+1}
#
#
# Because raw noise is difficult to predict,
# the encoder is encouraged to capture persistent
# temporal structure / gradual behavior change.
# ============================================================


class NextStepPredictor(
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

        self.predictor = nn.Linear(
            hidden_dim,
            input_dim,
        )

    def forward(
        self,
        x,
    ):

        hidden, _ = self.encoder(
            x
        )

        # h_t predicts x_{t+1}
        prediction = self.predictor(
            hidden[
                :,
                :-1,
                :,
            ]
        )

        target = x[
            :,
            1:,
            :
        ]

        return (
            prediction,
            target,
        )


# ============================================================
# Copy Encoder state
# ============================================================


def clone_state_dict(
    module,
):

    return {
        key:
            value.detach()
            .cpu()
            .clone()

        for key, value in (
            module.state_dict()
            .items()
        )
    }


# ============================================================
# Stage A
# Self-supervised pretraining
# ============================================================


def pretrain_ssl(
    x,
    train_idx,
    val_idx,
    device,
):

    input_dim = x.shape[
        -1
    ]

    model = NextStepPredictor(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
    ).to(
        device
    )

    train_loader = DataLoader(
        SequenceDataset(
            x,
            train_idx,
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        SequenceDataset(
            x,
            val_idx,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    loss_fn = nn.MSELoss()

    best_state = None

    best_val_loss = float(
        "inf"
    )

    patience_count = 0

    history = []

    print()

    print(
        "=" * 80
    )

    print(
        "STAGE A: SELF-SUPERVISED PRETRAINING"
    )

    print(
        "=" * 80
    )

    print(
        "No t_f."
    )

    print(
        "No t_d."
    )

    print(
        "Only x is used."
    )

    print()

    for epoch in range(
        SSL_EPOCHS
    ):

        model.train()

        train_sum = 0.0

        train_n = 0

        for batch_x in train_loader:

            batch_x = batch_x.to(
                device
            )

            optimizer.zero_grad()

            prediction, target = model(
                batch_x
            )

            loss = loss_fn(
                prediction,
                target,
            )

            loss.backward()

            optimizer.step()

            batch_size = len(
                batch_x
            )

            train_sum += float(
                loss.item()
            ) * batch_size

            train_n += batch_size

        model.eval()

        val_sum = 0.0

        val_n = 0

        with torch.no_grad():

            for batch_x in val_loader:

                batch_x = batch_x.to(
                    device
                )

                prediction, target = model(
                    batch_x
                )

                loss = loss_fn(
                    prediction,
                    target,
                )

                batch_size = len(
                    batch_x
                )

                val_sum += float(
                    loss.item()
                ) * batch_size

                val_n += batch_size

        train_loss = (
            train_sum
            / train_n
        )

        val_loss = (
            val_sum
            / val_n
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_mse":
                    train_loss,

                "val_mse":
                    val_loss,
            }
        )

        print(
            f"epoch={epoch:02d} "
            f"train_mse={train_loss:.4f} "
            f"val_mse={val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = (
                val_loss
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_count = 0

        else:

            patience_count += 1

            if patience_count >= PATIENCE:

                print(
                    "SSL early stopping."
                )

                break

    if best_state is None:

        raise RuntimeError(
            "SSL pretraining failed."
        )

    model.load_state_dict(
        best_state
    )

    pd.DataFrame(
        history
    ).to_csv(
        OUTPUT_DIR
        / "ssl_history.csv",
        index=False,
    )

    encoder_state = clone_state_dict(
        model.encoder
    )

    return (
        encoder_state,
        best_val_loss,
    )


# ============================================================
# Build a Timing model with supplied encoder
# ============================================================


def build_timing_model(
    input_dim,
    encoder_state,
    device,
):

    model = base.TimingModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
    ).to(
        device
    )

    if encoder_state is not None:

        model.encoder.load_state_dict(
            encoder_state
        )

    return model


# ============================================================
# Diagnostic Oracle Probe
#
# IMPORTANT:
#
# true t_f is used here ONLY to measure whether
# the frozen representation contains onset information.
#
# This is NOT the final latent method.
# ============================================================


def oracle_probe(
    name,
    encoder_state,
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

    model = build_timing_model(
        input_dim,
        encoder_state,
        device,
    )

    # Freeze Encoder.
    for parameter in (
        model.encoder.parameters()
    ):

        parameter.requires_grad = False

    optimizer = torch.optim.Adam(
        model.timing_head.parameters(),
        lr=LEARNING_RATE,
    )

    best_state = None

    best_val_mae = float(
        "inf"
    )

    patience_count = 0

    print()

    print(
        "=" * 80
    )

    print(
        f"ORACLE PROBE: {name}"
    )

    print(
        "=" * 80
    )

    for epoch in range(
        PROBE_EPOCHS
    ):

        model.train()

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

            true_tf = batch[
                "true_tf"
            ].to(
                device
            )

            optimizer.zero_grad()

            log_q, _ = model(
                x,
                td,
            )

            loss = base.oracle_timing_loss(
                log_q,
                true_tf,
            )

            loss.backward()

            optimizer.step()

        val = base.evaluate(
            model,
            val_loader,
            device,
            delay_log_prior,
        )

        val_mae = val[
            "raw_timing_mean_tf_mae"
        ]

        print(
            f"epoch={epoch:02d} "
            f"val_raw_tf_mae="
            f"{val_mae:.4f}"
        )

        if val_mae < best_val_mae:

            best_val_mae = (
                val_mae
            )

            best_state = clone_state_dict(
                model
            )

            patience_count = 0

        else:

            patience_count += 1

            if patience_count >= PATIENCE:

                break

    model.load_state_dict(
        best_state
    )

    test = base.evaluate(
        model,
        test_loader,
        device,
        delay_log_prior,
    )

    print()

    print(
        "TEST RAW TIMING MAE:",
        test[
            "raw_timing_mean_tf_mae"
        ],
    )

    return test[
        "raw_timing_mean_tf_mae"
    ]


# ============================================================
# Latent observed-likelihood training
#
# NO true t_f in the loss.
# ============================================================


def latent_train(
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

    model = build_timing_model(
        input_dim,
        encoder_state,
        device,
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

    patience_count = 0

    history = []

    print()

    print(
        "=" * 80
    )

    print(
        f"LATENT TRAINING: {name}"
    )

    print(
        "=" * 80
    )

    for epoch in range(
        LATENT_EPOCHS
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

            # true_tf is NOT used.

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

        history.append(
            {
                "epoch":
                    epoch,

                "train_nll":
                    train_sum
                    / train_n,

                **val,
            }
        )

        print(
            f"epoch={epoch:02d} "
            f"val_nll="
            f"{val['observed_nll']:.4f} "
            f"diag_raw_tf_mae="
            f"{val['raw_timing_mean_tf_mae']:.4f} "
            f"diag_post_tf_mae="
            f"{val['posterior_mean_tf_mae']:.4f} "
            f"delay="
            f"{val['posterior_mean_delay']:.4f}"
        )

        # Realistic checkpoint selection:
        # NO true t_f.
        if val[
            "observed_nll"
        ] < best_val_nll:

            best_val_nll = val[
                "observed_nll"
            ]

            best_state = clone_state_dict(
                model
            )

            patience_count = 0

        else:

            patience_count += 1

            if patience_count >= PATIENCE:

                break

    model.load_state_dict(
        best_state
    )

    test = base.evaluate(
        model,
        test_loader,
        device,
        delay_log_prior,
    )

    pd.DataFrame(
        history
    ).to_csv(
        OUTPUT_DIR
        / f"{name}_history.csv",
        index=False,
    )

    result = {
        "name":
            name,

        "train_encoder":
            train_encoder,

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
        "EXPERIMENT 03V:"
    )

    print(
        "SELF-SUPERVISED REPRESENTATION"
    )

    print(
        "=" * 80
    )

    print(
        "Device:",
        device,
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

    input_dim = x.shape[
        -1
    ]

    seq_len = x.shape[
        1
    ]

    # ========================================================
    # SSL pretraining
    # ========================================================

    ssl_encoder_state, ssl_val_mse = (
        pretrain_ssl(
            x,
            train_idx,
            val_idx,
            device,
        )
    )

    print()

    print(
        "Best SSL validation MSE:",
        ssl_val_mse,
    )

    # ========================================================
    # Build detected-fraud loaders
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
    # Same random Encoder reference
    # ========================================================

    base.set_seed(
        SEED
    )

    random_model = base.TimingModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
    )

    random_encoder_state = clone_state_dict(
        random_model.encoder
    )

    # ========================================================
    # Diagnostic Oracle probes
    # ========================================================

    random_probe_mae = oracle_probe(
        "random_frozen_encoder",
        random_encoder_state,
        input_dim,
        train_loader,
        val_loader,
        test_loader,
        device,
        delay_log_prior,
    )

    ssl_probe_mae = oracle_probe(
        "ssl_frozen_encoder",
        ssl_encoder_state,
        input_dim,
        train_loader,
        val_loader,
        test_loader,
        device,
        delay_log_prior,
    )

    # ========================================================
    # Latent observed-likelihood comparisons
    # ========================================================

    results = []

    results.append(
        latent_train(
            name="random_frozen",
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
        latent_train(
            name="ssl_frozen",
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
        latent_train(
            name="ssl_trainable",
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

    summary = pd.DataFrame(
        results
    )

    summary.to_csv(
        OUTPUT_DIR
        / "latent_summary.csv",
        index=False,
    )

    probe_summary = pd.DataFrame(
        [
            {
                "encoder":
                    "random",

                "oracle_probe_tf_mae":
                    random_probe_mae,
            },

            {
                "encoder":
                    "self_supervised",

                "oracle_probe_tf_mae":
                    ssl_probe_mae,
            },
        ]
    )

    probe_summary.to_csv(
        OUTPUT_DIR
        / "probe_summary.csv",
        index=False,
    )

    # ========================================================
    # Final
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "ORACLE PROBE COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        probe_summary.to_string(
            index=False
        )
    )

    print()

    print(
        "=" * 80
    )

    print(
        "LATENT COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()

    print(
        "Interpretation:"
    )

    print()

    print(
        "1. If SSL probe < random probe:"
    )

    print(
        "   self-supervised encoder contains more onset information."
    )

    print()

    print(
        "2. If SSL frozen > random frozen:"
    )

    print(
        "   preserving behavior representation helps latent inference."
    )

    print()

    print(
        "3. If SSL trainable becomes worse than SSL frozen:"
    )

    print(
        "   observed likelihood destroys useful representation."
    )


if __name__ == "__main__":
    main()