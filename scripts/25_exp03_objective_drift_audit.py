from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Experiment 03R
# Objective Drift Audit
#
# 中文：
# 训练目标漂移诊断。
#
# 核心问题：
#
# 一个已经通过 true t_f 学会一定 onset 定位能力的
# Timing model，
#
# 当 true t_f supervision 被移除，
# 只继续优化 observed t_d likelihood 时，
#
# 它会不会逐渐：
#
#   1. observed NLL 变好
#   2. true t_f MAE 变差
#   3. predicted detection delay 变短
#
# 如果成立：
#
#   说明 observed likelihood 本身存在
#   short-delay optimization bias。
#
#
# IMPORTANT:
#
# 本实验只使用 detected fraud 用户，
# 目的是隔离 Timing/objective。
#
# 不研究 occurrence。
# 不研究 censored branch。
# 不作为最终模型性能。
# ============================================================


DATA_PATH = Path(
    r"data/processed/synthetic_same_delay_consistent.npz"
)

OUTPUT_DIR = Path(
    r"reports/experiment03/objective_drift_audit"
)


SEED = 0

HIDDEN_DIM = 32

BATCH_SIZE = 64

LEARNING_RATE = 0.001

ORACLE_MAX_EPOCHS = 20

ORACLE_PATIENCE = 7

DRIFT_EPOCHS = 25

DELAY_MEAN = 4.0


# ============================================================
# Reproducibility
# ============================================================


def set_seed(seed: int):

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


# ============================================================
# Load synthetic data
# ============================================================


def load_data():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            DATA_PATH
        )

    data = np.load(
        DATA_PATH,
        allow_pickle=True,
    )

    print(
        "Dataset keys:",
        list(data.keys()),
    )

    x = data["x"].astype(
        np.float32
    )

    # --------------------------------------------------------
    # true t_f
    # --------------------------------------------------------

    if "onset" in data:

        true_tf = data[
            "onset"
        ].astype(
            np.int64
        )

    elif "t_onset" in data:

        true_tf = data[
            "t_onset"
        ].astype(
            np.int64
        )

    else:

        raise KeyError(
            "Cannot find onset / t_onset."
        )

    # --------------------------------------------------------
    # detected label
    # --------------------------------------------------------

    if "detected" not in data:

        raise KeyError(
            "Cannot find detected."
        )

    detected = data[
        "detected"
    ].astype(
        np.int64
    )

    # --------------------------------------------------------
    # observed t_d
    # --------------------------------------------------------

    if "t_detect" in data:

        td = data[
            "t_detect"
        ].astype(
            np.int64
        )

    elif "td" in data:

        td = data[
            "td"
        ].astype(
            np.int64
        )

    else:

        raise KeyError(
            "Cannot find t_detect / td."
        )

    return (
        x,
        true_tf,
        detected,
        td,
    )


# ============================================================
# Same split logic
# ============================================================


def split_indices(
    n: int,
    seed: int = 0,
):

    rng = np.random.default_rng(
        seed
    )

    indices = rng.permutation(
        n
    )

    n_train = int(
        0.7 * n
    )

    n_val = int(
        0.1 * n
    )

    train_idx = indices[
        :n_train
    ]

    val_idx = indices[
        n_train:
        n_train + n_val
    ]

    test_idx = indices[
        n_train + n_val:
    ]

    return (
        train_idx,
        val_idx,
        test_idx,
    )


# ============================================================
# Dataset
# ============================================================


class DetectedFraudDataset(
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

        # ----------------------------------------------------
        # Only:
        #
        #   known fraud
        #   AND detected
        #
        # This isolates Timing.
        # ----------------------------------------------------

        keep = []

        for idx in indices:

            if (
                true_tf[idx] >= 0
                and detected[idx] == 1
            ):

                keep.append(
                    int(idx)
                )

        self.indices = np.asarray(
            keep,
            dtype=np.int64,
        )

        self.x = x

        self.true_tf = true_tf

        self.td = td

    def __len__(self):

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
# Timing model
#
# Strict causal:
#
# GRU hidden h_t only sees x_0 ... x_t.
#
# Candidate tf=k is scored using:
#
#   h_k
#   +
#   h_td
#
# Therefore it may use evidence observed up to td,
# but never anything after td.
# ============================================================


class TimingModel(
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

        self.timing_head = nn.Sequential(
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
        td,
    ):

        hidden, _ = self.encoder(
            x
        )

        batch_size = (
            hidden.shape[0]
        )

        seq_len = (
            hidden.shape[1]
        )

        device = (
            hidden.device
        )

        batch_indices = torch.arange(
            batch_size,
            device=device,
        )

        # ----------------------------------------------------
        # h_td summarizes x_0 ... x_td
        # ----------------------------------------------------

        endpoint_hidden = hidden[
            batch_indices,
            td,
        ]

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

        pair = torch.cat(
            [
                hidden,
                endpoint_expanded,
            ],
            dim=-1,
        )

        logits = (
            self.timing_head(
                pair
            )
            .squeeze(-1)
        )

        # ----------------------------------------------------
        # Minimum-delay consistency:
        #
        # candidate tf must satisfy:
        #
        #     tf < td
        #
        # So delay >= 1.
        # ----------------------------------------------------

        steps = torch.arange(
            seq_len,
            device=device,
        )[None, :]

        valid = (
            steps
            < td[:, None]
        )

        logits = logits.masked_fill(
            ~valid,
            -torch.inf,
        )

        log_q = torch.log_softmax(
            logits,
            dim=1,
        )

        q = log_q.exp()

        return (
            log_q,
            q,
        )


# ============================================================
# Oracle delay prior
#
# Synthetic:
#
#   E ~ Exponential(scale=4)
#   delay = max(1, int(E))
# ============================================================


def build_delay_log_prior(
    max_delay: int,
    device,
):

    probabilities = np.zeros(
        max_delay + 1,
        dtype=np.float64,
    )

    for delay in range(
        1,
        max_delay + 1,
    ):

        if delay == 1:

            p = (
                1.0
                - math.exp(
                    -2.0
                    / DELAY_MEAN
                )
            )

        else:

            p = (
                math.exp(
                    -delay
                    / DELAY_MEAN
                )
                -
                math.exp(
                    -(delay + 1)
                    / DELAY_MEAN
                )
            )

        probabilities[
            delay
        ] = max(
            p,
            1e-30,
        )

    log_prior = np.full(
        max_delay + 1,
        -np.inf,
        dtype=np.float32,
    )

    log_prior[
        1:
    ] = np.log(
        probabilities[
            1:
        ]
    )

    return torch.tensor(
        log_prior,
        dtype=torch.float32,
        device=device,
    )


# ============================================================
# Stage A:
# Oracle Timing loss
#
# Uses true t_f.
#
# ONLY diagnostic pretraining.
# ============================================================


def oracle_timing_loss(
    log_q,
    true_tf,
):

    batch_indices = torch.arange(
        len(
            true_tf
        ),
        device=true_tf.device,
    )

    return (
        -log_q[
            batch_indices,
            true_tf,
        ].mean()
    )


# ============================================================
# Stage B:
# Observed-data likelihood
#
# true t_f is NOT used.
#
#
# P(td | x)
#
# =
#
# sum_k
#
#   q(tf=k | x, td)
#
#   *
#
#   P(delay = td-k)
#
#
# This is the objective whose drift we want to inspect.
# ============================================================


def observed_nll(
    log_q,
    td,
    delay_log_prior,
):

    batch_size = (
        log_q.shape[0]
    )

    seq_len = (
        log_q.shape[1]
    )

    device = (
        log_q.device
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
        min=1,
        max=len(
            delay_log_prior
        ) - 1,
    )

    log_delay = delay_log_prior[
        safe_delay
    ]

    log_delay = log_delay.masked_fill(
        ~valid,
        -torch.inf,
    )

    log_likelihood = torch.logsumexp(
        log_q
        + log_delay,
        dim=1,
    )

    return (
        -log_likelihood.mean()
    )


# ============================================================
# Evaluate
#
# We calculate TWO distributions:
#
# 1. raw Timing q(tf | x, td)
#
# 2. observed posterior:
#
#       q(tf|x,td) * P(delay)
#
#    normalized over legal tf.
#
#
# This lets us see whether Timing itself drifts.
# ============================================================


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    delay_log_prior,
):

    model.eval()

    all_true_tf = []

    all_td = []

    all_raw_mean_tf = []

    all_raw_mode_tf = []

    all_post_mean_tf = []

    all_post_mode_tf = []

    total_observed_nll = 0.0

    total_n = 0

    for batch in loader:

        x = batch[
            "x"
        ].to(
            device
        )

        true_tf = batch[
            "true_tf"
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

        batch_size = (
            len(
                true_tf
            )
        )

        seq_len = (
            q.shape[1]
        )

        # ----------------------------------------------------
        # Raw Timing
        # ----------------------------------------------------

        steps = torch.arange(
            seq_len,
            device=device,
            dtype=torch.float32,
        )[None, :]

        raw_mean_tf = (
            q
            * steps
        ).sum(
            dim=1
        )

        raw_mode_tf = q.argmax(
            dim=1
        )

        # ----------------------------------------------------
        # Observed posterior
        # ----------------------------------------------------

        int_steps = torch.arange(
            seq_len,
            device=device,
        )[None, :]

        delays = (
            td[:, None]
            - int_steps
        )

        valid = (
            delays >= 1
        )

        safe_delay = delays.clamp(
            min=1,
            max=len(
                delay_log_prior
            ) - 1,
        )

        log_delay = delay_log_prior[
            safe_delay
        ]

        log_delay = (
            log_delay
            .masked_fill(
                ~valid,
                -torch.inf,
            )
        )

        log_joint = (
            log_q
            + log_delay
        )

        log_norm = torch.logsumexp(
            log_joint,
            dim=1,
            keepdim=True,
        )

        posterior = (
            log_joint
            - log_norm
        ).exp()

        post_mean_tf = (
            posterior
            * steps
        ).sum(
            dim=1
        )

        post_mode_tf = posterior.argmax(
            dim=1
        )

        # ----------------------------------------------------
        # Observed NLL
        # ----------------------------------------------------

        nll = (
            -log_norm.squeeze(
                1
            )
        )

        total_observed_nll += float(
            nll.sum().item()
        )

        total_n += batch_size

        all_true_tf.append(
            true_tf.cpu().numpy()
        )

        all_td.append(
            td.cpu().numpy()
        )

        all_raw_mean_tf.append(
            raw_mean_tf.cpu().numpy()
        )

        all_raw_mode_tf.append(
            raw_mode_tf.cpu().numpy()
        )

        all_post_mean_tf.append(
            post_mean_tf.cpu().numpy()
        )

        all_post_mode_tf.append(
            post_mode_tf.cpu().numpy()
        )

    true_tf = np.concatenate(
        all_true_tf
    )

    td = np.concatenate(
        all_td
    )

    raw_mean_tf = np.concatenate(
        all_raw_mean_tf
    )

    raw_mode_tf = np.concatenate(
        all_raw_mode_tf
    )

    post_mean_tf = np.concatenate(
        all_post_mean_tf
    )

    post_mode_tf = np.concatenate(
        all_post_mode_tf
    )

    metrics = {
        "observed_nll":
            total_observed_nll
            / total_n,

        # ----------------------------------------------------
        # Raw Timing q
        # ----------------------------------------------------

        "raw_timing_mean_tf_mae":
            float(
                np.mean(
                    np.abs(
                        raw_mean_tf
                        - true_tf
                    )
                )
            ),

        "raw_timing_mode_tf_mae":
            float(
                np.mean(
                    np.abs(
                        raw_mode_tf
                        - true_tf
                    )
                )
            ),

        "raw_timing_mean_delay":
            float(
                np.mean(
                    td
                    - raw_mean_tf
                )
            ),

        # ----------------------------------------------------
        # Posterior after delay prior
        # ----------------------------------------------------

        "posterior_mean_tf_mae":
            float(
                np.mean(
                    np.abs(
                        post_mean_tf
                        - true_tf
                    )
                )
            ),

        "posterior_mode_tf_mae":
            float(
                np.mean(
                    np.abs(
                        post_mode_tf
                        - true_tf
                    )
                )
            ),

        "posterior_mean_delay":
            float(
                np.mean(
                    td
                    - post_mean_tf
                )
            ),

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        "true_delay_mean":
            float(
                np.mean(
                    td
                    - true_tf
                )
            ),
    }

    return metrics


# ============================================================
# Stage A:
# Oracle pretrain
# ============================================================


def oracle_pretrain(
    model,
    train_loader,
    val_loader,
    device,
    delay_log_prior,
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_state = None

    best_val_mae = float(
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
        "STAGE A: ORACLE TIMING PRETRAIN"
    )

    print(
        "=" * 80
    )

    print(
        "这一阶段使用 true t_f，"
    )

    print(
        "只为了得到一个原本会定位 onset 的 Timing 模型。"
    )

    print()

    for epoch in range(
        ORACLE_MAX_EPOCHS
    ):

        model.train()

        running_loss = 0.0

        n_examples = 0

        for batch in train_loader:

            x = batch[
                "x"
            ].to(
                device
            )

            true_tf = batch[
                "true_tf"
            ].to(
                device
            )

            td = batch[
                "td"
            ].to(
                device
            )

            optimizer.zero_grad()

            log_q, _ = model(
                x,
                td,
            )

            loss = oracle_timing_loss(
                log_q,
                true_tf,
            )

            loss.backward()

            optimizer.step()

            running_loss += float(
                loss.item()
            ) * len(
                true_tf
            )

            n_examples += len(
                true_tf
            )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            delay_log_prior,
        )

        # ----------------------------------------------------
        # For Oracle pretrain,
        # select checkpoint using RAW timing mean tf MAE.
        #
        # This is allowed because this is explicitly
        # an Oracle diagnostic stage.
        # ----------------------------------------------------

        val_mae = val_metrics[
            "raw_timing_mean_tf_mae"
        ]

        row = {
            "stage":
                "oracle_pretrain",

            "epoch":
                epoch,

            "train_oracle_loss":
                running_loss
                / n_examples,

            **val_metrics,
        }

        history.append(
            row
        )

        print(
            f"epoch={epoch:02d} "
            f"oracle_loss="
            f"{row['train_oracle_loss']:.4f} "
            f"raw_tf_mae="
            f"{val_metrics['raw_timing_mean_tf_mae']:.4f} "
            f"post_tf_mae="
            f"{val_metrics['posterior_mean_tf_mae']:.4f} "
            f"obs_nll="
            f"{val_metrics['observed_nll']:.4f} "
            f"post_delay="
            f"{val_metrics['posterior_mean_delay']:.4f}"
        )

        if val_mae < best_val_mae:

            best_val_mae = (
                val_mae
            )

            best_epoch = (
                epoch
            )

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_count = 0

        else:

            patience_count += 1

            if (
                patience_count
                >= ORACLE_PATIENCE
            ):

                print(
                    "Oracle early stopping."
                )

                break

    if best_state is None:

        raise RuntimeError(
            "Oracle pretraining failed."
        )

    model.load_state_dict(
        best_state
    )

    print()

    print(
        "Best Oracle checkpoint:"
    )

    print(
        "  epoch =",
        best_epoch,
    )

    print(
        "  val raw timing mean tf MAE =",
        best_val_mae,
    )

    return (
        history,
        best_epoch,
    )


# ============================================================
# Stage B:
# Remove true tf supervision and optimize observed likelihood
# ============================================================


def objective_drift(
    model,
    train_loader,
    val_loader,
    device,
    delay_log_prior,
):

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # New optimizer.
    #
    # Same model weights,
    # but Oracle supervision is now completely removed.
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = []

    print()

    print(
        "=" * 80
    )

    print(
        "STAGE B: OBSERVED-LIKELIHOOD DRIFT"
    )

    print(
        "=" * 80
    )

    print(
        "从这一刻开始，训练完全不使用 true t_f。"
    )

    print(
        "只优化 observed t_d likelihood。"
    )

    print()

    # --------------------------------------------------------
    # Epoch -1:
    # BEFORE observed-likelihood training
    # --------------------------------------------------------

    initial = evaluate(
        model,
        val_loader,
        device,
        delay_log_prior,
    )

    history.append(
        {
            "stage":
                "before_drift",

            "epoch":
                -1,

            "train_observed_nll":
                np.nan,

            **initial,
        }
    )

    print(
        "BEFORE DRIFT:"
    )

    print(
        f"  observed NLL = "
        f"{initial['observed_nll']:.4f}"
    )

    print(
        f"  raw timing tf MAE = "
        f"{initial['raw_timing_mean_tf_mae']:.4f}"
    )

    print(
        f"  posterior tf MAE = "
        f"{initial['posterior_mean_tf_mae']:.4f}"
    )

    print(
        f"  posterior mean delay = "
        f"{initial['posterior_mean_delay']:.4f}"
    )

    print(
        f"  true delay mean = "
        f"{initial['true_delay_mean']:.4f}"
    )

    print()

    for epoch in range(
        DRIFT_EPOCHS
    ):

        model.train()

        running_loss = 0.0

        n_examples = 0

        for batch in train_loader:

            x = batch[
                "x"
            ].to(
                device
            )

            # ---------------------------------------------
            # true_tf is deliberately NOT read here.
            # ---------------------------------------------

            td = batch[
                "td"
            ].to(
                device
            )

            optimizer.zero_grad()

            log_q, _ = model(
                x,
                td,
            )

            loss = observed_nll(
                log_q,
                td,
                delay_log_prior,
            )

            loss.backward()

            optimizer.step()

            running_loss += float(
                loss.item()
            ) * len(
                td
            )

            n_examples += len(
                td
            )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            delay_log_prior,
        )

        row = {
            "stage":
                "observed_drift",

            "epoch":
                epoch,

            "train_observed_nll":
                running_loss
                / n_examples,

            **val_metrics,
        }

        history.append(
            row
        )

        print(
            f"epoch={epoch:02d} "
            f"train_nll="
            f"{row['train_observed_nll']:.4f} "
            f"val_nll="
            f"{val_metrics['observed_nll']:.4f} "
            f"raw_tf_mae="
            f"{val_metrics['raw_timing_mean_tf_mae']:.4f} "
            f"post_tf_mae="
            f"{val_metrics['posterior_mean_tf_mae']:.4f} "
            f"post_delay="
            f"{val_metrics['posterior_mean_delay']:.4f}"
        )

    return history


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
        "EXPERIMENT 03R: OBJECTIVE DRIFT AUDIT"
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
        "Stage A:"
    )

    print(
        "  用 true t_f 先训练一个会定位 onset 的 Timing。"
    )

    print()

    print(
        "Stage B:"
    )

    print(
        "  删除 true t_f supervision。"
    )

    print(
        "  只优化 observed t_d likelihood。"
    )

    print()

    print(
        "观察："
    )

    print(
        "  NLL 是否下降；"
    )

    print(
        "  t_f MAE 是否上升；"
    )

    print(
        "  predicted delay 是否缩短。"
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

    train_dataset = DetectedFraudDataset(
        x,
        true_tf,
        detected,
        td,
        train_idx,
    )

    val_dataset = DetectedFraudDataset(
        x,
        true_tf,
        detected,
        td,
        val_idx,
    )

    test_dataset = DetectedFraudDataset(
        x,
        true_tf,
        detected,
        td,
        test_idx,
    )

    print()

    print(
        "Detected-fraud split sizes:"
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

    # ========================================================
    # Model
    # ========================================================

    input_dim = (
        x.shape[-1]
    )

    model = TimingModel(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
    ).to(
        device
    )

    seq_len = (
        x.shape[1]
    )

    delay_log_prior = build_delay_log_prior(
        max_delay=seq_len,
        device=device,
    )

    # ========================================================
    # Stage A
    # ========================================================

    (
        oracle_history,
        oracle_best_epoch,
    ) = oracle_pretrain(
        model,
        train_loader,
        val_loader,
        device,
        delay_log_prior,
    )

    # --------------------------------------------------------
    # Evaluate best Oracle Timing on TEST
    # before drift.
    # --------------------------------------------------------

    test_before = evaluate(
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
        "TEST BEFORE DRIFT"
    )

    print(
        "=" * 80
    )

    for key, value in (
        test_before.items()
    ):

        print(
            f"{key}: {value}"
        )

    # Save Oracle starting point
    torch.save(
        model.state_dict(),
        OUTPUT_DIR
        / "oracle_starting_point.pt",
    )

    # ========================================================
    # Stage B
    # ========================================================

    drift_history = objective_drift(
        model,
        train_loader,
        val_loader,
        device,
        delay_log_prior,
    )

    # --------------------------------------------------------
    # Final test after drift
    # --------------------------------------------------------

    test_after = evaluate(
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
        "TEST AFTER DRIFT"
    )

    print(
        "=" * 80
    )

    for key, value in (
        test_after.items()
    ):

        print(
            f"{key}: {value}"
        )

    # ========================================================
    # Direct comparison
    # ========================================================

    print()

    print(
        "=" * 80
    )

    print(
        "BEFORE vs AFTER OBSERVED-LIKELIHOOD TRAINING"
    )

    print(
        "=" * 80
    )

    print(
        "Observed NLL:"
    )

    print(
        " ",
        test_before[
            "observed_nll"
        ],
        "->",
        test_after[
            "observed_nll"
        ],
    )

    print()

    print(
        "Raw Timing tf MAE:"
    )

    print(
        " ",
        test_before[
            "raw_timing_mean_tf_mae"
        ],
        "->",
        test_after[
            "raw_timing_mean_tf_mae"
        ],
    )

    print()

    print(
        "Posterior tf MAE:"
    )

    print(
        " ",
        test_before[
            "posterior_mean_tf_mae"
        ],
        "->",
        test_after[
            "posterior_mean_tf_mae"
        ],
    )

    print()

    print(
        "Posterior mean delay:"
    )

    print(
        " ",
        test_before[
            "posterior_mean_delay"
        ],
        "->",
        test_after[
            "posterior_mean_delay"
        ],
    )

    print()

    print(
        "True delay mean:"
    )

    print(
        " ",
        test_before[
            "true_delay_mean"
        ],
    )

    # ========================================================
    # Save histories
    # ========================================================

    oracle_df = pd.DataFrame(
        oracle_history
    )

    drift_df = pd.DataFrame(
        drift_history
    )

    oracle_df.to_csv(
        OUTPUT_DIR
        / "oracle_pretrain_history.csv",
        index=False,
    )

    drift_df.to_csv(
        OUTPUT_DIR
        / "objective_drift_history.csv",
        index=False,
    )

    comparison = pd.DataFrame(
        [
            {
                "metric":
                    "observed_nll",

                "before":
                    test_before[
                        "observed_nll"
                    ],

                "after":
                    test_after[
                        "observed_nll"
                    ],
            },

            {
                "metric":
                    "raw_timing_mean_tf_mae",

                "before":
                    test_before[
                        "raw_timing_mean_tf_mae"
                    ],

                "after":
                    test_after[
                        "raw_timing_mean_tf_mae"
                    ],
            },

            {
                "metric":
                    "posterior_mean_tf_mae",

                "before":
                    test_before[
                        "posterior_mean_tf_mae"
                    ],

                "after":
                    test_after[
                        "posterior_mean_tf_mae"
                    ],
            },

            {
                "metric":
                    "posterior_mean_delay",

                "before":
                    test_before[
                        "posterior_mean_delay"
                    ],

                "after":
                    test_after[
                        "posterior_mean_delay"
                    ],
            },
        ]
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "before_after_comparison.csv",
        index=False,
    )

    torch.save(
        model.state_dict(),
        OUTPUT_DIR
        / "after_observed_likelihood.pt",
    )

    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT_DIR
        / "objective_drift_history.csv"
    )

    print(
        OUTPUT_DIR
        / "before_after_comparison.csv"
    )


if __name__ == "__main__":
    main()