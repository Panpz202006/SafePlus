from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader


# ============================================================
# Experiment 03X
# Causal Local-Score Timing
#
# 中文：
# 因果局部打分 Timing。
#
#
# 03W:
#
#   P(tf=k)
#     =
#   h(k) * product_{j<k}(1-h(j))
#
# Problem:
#
#   candidate times are coupled through survival product.
#
#
# 03X:
#
#   score_k = s(h_k)
#
#   q(tf=k | x)
#     =
#   softmax(score_k)
#
#
# IMPORTANT:
#
# - score_k only sees x_0 ... x_k
# - NO h_td
# - NO survival product
# - td only defines legal candidates k < td
# - true tf is NEVER used in latent training
# ============================================================


BASE_SCRIPT = Path(
    r"scripts/30_exp03_causal_onset_prior.py"
)

OUTPUT_DIR = Path(
    r"reports/experiment03/"
    r"causal_local_softmax"
)

REFERENCE_03W = Path(
    r"reports/experiment03/"
    r"causal_onset_prior/"
    r"summary.csv"
)

REFERENCE_03V = Path(
    r"reports/experiment03/"
    r"self_supervised_representation/"
    r"latent_summary.csv"
)


SEED = 0
HIDDEN_DIM = 32
BATCH_SIZE = 64


# ============================================================
# Load Experiment 03W utilities
# ============================================================


if not BASE_SCRIPT.exists():

    raise FileNotFoundError(
        BASE_SCRIPT
    )


spec = importlib.util.spec_from_file_location(
    "exp03w",
    BASE_SCRIPT,
)

if spec is None or spec.loader is None:

    raise RuntimeError(
        "Could not load Experiment 03W."
    )


previous = importlib.util.module_from_spec(
    spec
)

spec.loader.exec_module(
    previous
)


base = previous.base

sslmod = previous.sslmod


# ============================================================
# Causal local-score Timing model
# ============================================================


class CausalLocalTimingModel(
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

        # ----------------------------------------------------
        # Causal GRU:
        #
        # h_k only sees x_0 ... x_k.
        # ----------------------------------------------------

        hidden, _ = self.encoder(
            x
        )

        # ----------------------------------------------------
        # Each candidate k gets its OWN local score.
        #
        # No h_td.
        # No survival product.
        # ----------------------------------------------------

        logits = (
            self.onset_head(
                hidden
            )
            .squeeze(-1)
        )

        seq_len = (
            logits.shape[1]
        )

        steps = torch.arange(
            seq_len,
            device=x.device,
        )[None, :]

        # ----------------------------------------------------
        # Minimum-delay consistency:
        #
        # tf < td
        # ----------------------------------------------------

        valid = (
            steps
            < td[:, None]
        )

        logits = (
            logits.masked_fill(
                ~valid,
                -torch.inf,
            )
        )

        # ----------------------------------------------------
        # Conditional distribution among legal candidate tf.
        # ----------------------------------------------------

        log_q = F.log_softmax(
            logits,
            dim=1,
        )

        q = log_q.exp()

        return (
            log_q,
            q,
        )


# ============================================================
# Replace 03W model class
# ============================================================


previous.CausalOnsetModel = (
    CausalLocalTimingModel
)

previous.OUTPUT_DIR = (
    OUTPUT_DIR
)


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
        "EXPERIMENT 03X:"
    )

    print(
        "CAUSAL LOCAL-SCORE TIMING"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Key design:"
    )

    print(
        "  candidate tf=k uses ONLY h_k"
    )

    print(
        "  h_k only sees x_0 ... x_k"
    )

    print(
        "  NO h_td"
    )

    print(
        "  NO hazard survival product"
    )

    print(
        "  td only defines tf < td"
    )

    print()

    print(
        "Latent training uses NO true tf."
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
    # Same SSL Encoder as 03V / 03W
    #
    # SSL uses only x.
    # ========================================================

    (
        ssl_encoder_state,
        ssl_mse,
    ) = sslmod.pretrain_ssl(
        x,
        train_idx,
        val_idx,
        device,
    )

    print()

    print(
        "SSL validation MSE:",
        ssl_mse,
    )

    # ========================================================
    # Random encoder
    # ========================================================

    base.set_seed(
        SEED
    )

    random_model = CausalLocalTimingModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
    )

    random_encoder_state = (
        previous.clone_state(
            random_model.encoder
        )
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
    # Same three configurations as 03W
    # ========================================================

    results = []

    results.append(

        previous.run_one(

            name="local_random_frozen",

            encoder_state=
                random_encoder_state,

            train_encoder=False,

            input_dim=input_dim,

            train_loader=train_loader,

            val_loader=val_loader,

            test_loader=test_loader,

            device=device,

            delay_log_prior=
                delay_log_prior,
        )
    )

    results.append(

        previous.run_one(

            name="local_ssl_frozen",

            encoder_state=
                ssl_encoder_state,

            train_encoder=False,

            input_dim=input_dim,

            train_loader=train_loader,

            val_loader=val_loader,

            test_loader=test_loader,

            device=device,

            delay_log_prior=
                delay_log_prior,
        )
    )

    results.append(

        previous.run_one(

            name="local_ssl_trainable",

            encoder_state=
                ssl_encoder_state,

            train_encoder=True,

            input_dim=input_dim,

            train_loader=train_loader,

            val_loader=val_loader,

            test_loader=test_loader,

            device=device,

            delay_log_prior=
                delay_log_prior,
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
    # Comparison
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "03X LOCAL-SCORE RESULT"
    )

    print(
        "=" * 80
    )

    print(
        new_summary.to_string(
            index=False
        )
    )

    if REFERENCE_03W.exists():

        old_w = pd.read_csv(
            REFERENCE_03W
        )

        print()

        print(
            "=" * 80
        )

        print(
            "03W HAZARD REFERENCE"
        )

        print(
            "=" * 80
        )

        print(
            old_w.to_string(
                index=False
            )
        )

    if REFERENCE_03V.exists():

        old_v = pd.read_csv(
            REFERENCE_03V
        )

        print()

        print(
            "=" * 80
        )

        print(
            "03V ENDPOINT REFERENCE"
        )

        print(
            "=" * 80
        )

        print(
            old_v.to_string(
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
        "If local_ssl_frozen improves over"
    )

    print(
        "causal_ssl_frozen:"
    )

    print(
        "  survival-product timing coupling"
    )

    print(
        "  was hurting onset localization."
    )

    print()

    print(
        "If local_ssl_trainable keeps better"
    )

    print(
        "  tf MAE without collapsing delay:"
    )

    print(
        "  causal local timing is a better"
    )

    print(
        "  parameterization."
    )

    print()

    print(
        "If results remain around ~2.3-2.5:"
    )

    print(
        "  stop changing Timing architecture."
    )

    print(
        "  observed td likelihood alone is"
    )

    print(
        "  insufficient."
    )

    print()

    print(
        "  Next step should explicitly model"
    )

    print(
        "  behavior likelihood p(x | tf)."
    )


if __name__ == "__main__":

    main()