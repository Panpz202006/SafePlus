from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Experiment 04D
# Confidence Validity Audit
#
# 目的：
# 检查哪些 confidence feature（置信度特征）
# 真的能够判断：
#
#   当前用户更应该相信 Behavior Evidence
#   还是 Delay Evidence
#
# IMPORTANT:
# 1. Behavior model 只在 TRAIN 上训练。
# 2. Global scale 只从 TRAIN 估计。
# 3. Confidence validity 只在 VALIDATION 上诊断。
# 4. TEST 在本实验中完全不使用。
# ============================================================


BASE_04A = Path(
    r"scripts/34_exp04_learned_behavior_evidence.py"
)

BASE_03Y = Path(
    r"scripts/32_exp03_behavior_likelihood_identification.py"
)

OUTPUT_DIR = Path(
    r"reports/experiment04/confidence_validity_audit"
)

SEED = 0


# ============================================================
# Dynamic import
# ============================================================


def load_module(path, module_name):

    if not path.exists():
        raise FileNotFoundError(path)

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load {path}"
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


# ============================================================
# Basic helpers
# ============================================================


def posterior_median(
    candidates,
    posterior,
):

    cumulative = np.cumsum(
        posterior
    )

    index = int(
        np.searchsorted(
            cumulative,
            0.5,
        )
    )

    index = min(
        index,
        len(candidates) - 1,
    )

    return int(
        candidates[index]
    )


def normalized_entropy(
    posterior,
):

    posterior = np.asarray(
        posterior,
        dtype=np.float64,
    )

    k = len(posterior)

    if k <= 1:
        return 0.0

    entropy = float(
        -np.sum(
            posterior
            * np.log(
                np.maximum(
                    posterior,
                    1e-30,
                )
            )
        )
    )

    return float(
        entropy
        / math.log(k)
    )


def spearman_correlation(
    x,
    y,
):
    """
    Spearman Correlation（斯皮尔曼相关系数）

    不关心两个变量是否是线性关系，
    主要看：
        一个变量变大时，
        另一个变量是否总体也跟着变大或变小。
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    rank_x = (
        pd.Series(x)
        .rank(method="average")
        .to_numpy()
    )

    rank_y = (
        pd.Series(y)
        .rank(method="average")
        .to_numpy()
    )

    if (
        np.std(rank_x) < 1e-12
        or np.std(rank_y) < 1e-12
    ):
        return np.nan

    return float(
        np.corrcoef(
            rank_x,
            rank_y,
        )[0, 1]
    )


# ============================================================
# Behavior candidate scores
# ============================================================


def behavior_candidate_scores(
    x_user,
    observed_td,
    behavior_params,
    exp04a,
):

    d = int(observed_td)

    observed_x = x_user[
        : d + 1
    ]

    candidates = np.arange(
        0,
        d,
        dtype=np.int64,
    )

    scores = []

    for k in candidates:

        score = (
            exp04a.candidate_behavior_log_likelihood(
                observed_x=observed_x,
                candidate_tf=int(k),
                params=behavior_params,
            )
        )

        scores.append(score)

    return (
        candidates,
        np.asarray(
            scores,
            dtype=np.float64,
        ),
    )


# ============================================================
# TRAIN-only global behavior scale
# ============================================================


def estimate_global_behavior_scale(
    x,
    fraud_label,
    detected,
    td,
    train_idx,
    behavior_params,
    exp04a,
):

    score_stds = []

    for idx in train_idx:

        if fraud_label[idx] != 1:
            continue

        if detected[idx] != 1:
            continue

        (
            candidates,
            scores,
        ) = behavior_candidate_scores(
            x_user=x[idx],
            observed_td=td[idx],
            behavior_params=behavior_params,
            exp04a=exp04a,
        )

        if len(candidates) <= 1:
            continue

        std = float(
            np.std(scores)
        )

        if std > 1e-8:
            score_stds.append(std)

    if not score_stds:
        raise RuntimeError(
            "Could not estimate global behavior scale."
        )

    return float(
        np.median(
            np.asarray(
                score_stds,
                dtype=np.float64,
            )
        )
    )


# ============================================================
# Delay posterior
# ============================================================


def build_delay_posterior(
    candidates,
    observed_td,
    exp03y,
):

    d = int(observed_td)

    delay_scores = []

    for k in candidates:

        delay = (
            d - int(k)
        )

        probability = max(
            exp03y.delay_probability(
                delay
            ),
            1e-30,
        )

        delay_scores.append(
            math.log(probability)
        )

    delay_scores = np.asarray(
        delay_scores,
        dtype=np.float64,
    )

    return exp04a.stable_softmax(
        delay_scores
    )


# ============================================================
# Main
# ============================================================


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "EXPERIMENT 04D: "
        "CONFIDENCE VALIDITY AUDIT"
    )
    print("=" * 80)

    print()
    print(
        "Confidence = 置信度；"
        "模型觉得 Behavior Evidence 有多可靠"
    )

    print(
        "Entropy = 熵；"
        "后验越分散，熵越高"
    )

    print(
        "Margin = 第一名概率 - 第二名概率"
    )

    print(
        "Spearman Correlation = "
        "斯皮尔曼相关系数"
    )

    print(
        "Behavior Advantage = "
        "|Delay error| - |Behavior error|"
    )

    print(
        "Behavior Advantage > 0 "
        "表示 Behavior 比 Delay 更准"
    )

    print()

    # ========================================================
    # Load modules
    # ========================================================

    exp04a = load_module(
        BASE_04A,
        "exp04a",
    )

    exp03y = load_module(
        BASE_03Y,
        "exp03y",
    )

    # ========================================================
    # Load dataset
    # ========================================================

    data = np.load(
        exp04a.DATA_PATH,
        allow_pickle=True,
    )

    x = data[
        "x"
    ].astype(
        np.float64
    )

    detected = data[
        "detected"
    ].astype(
        np.int64
    )

    td = data[
        "t_detect"
    ].astype(
        np.int64
    )

    true_tf = data[
        "t_onset"
    ].astype(
        np.int64
    )

    fraud_label = (
        exp04a.get_entity_fraud_label(
            data
        )
    )

    if "entity_id" in data:

        entity_id = data[
            "entity_id"
        ].astype(
            np.int64
        )

    else:

        entity_id = np.arange(
            len(x)
        )

    (
        train_idx,
        val_idx,
        test_idx,
    ) = exp04a.split_indices(
        len(x),
        seed=SEED,
    )

    print(
        f"Split: "
        f"train={len(train_idx)} "
        f"val={len(val_idx)} "
        f"test={len(test_idx)}"
    )

    print(
        "IMPORTANT: "
        "04D only analyzes VALIDATION."
    )

    print(
        "TEST is not used."
    )

    # ========================================================
    # Train Behavior model on TRAIN
    # ========================================================

    (
        behavior_params,
        em_history,
    ) = exp04a.train_emission_em(
        x=x,
        fraud_label=fraud_label,
        detected=detected,
        td=td,
        train_idx=train_idx,
    )

    # ========================================================
    # Global scale from TRAIN
    # ========================================================

    global_scale = (
        estimate_global_behavior_scale(
            x=x,
            fraud_label=fraud_label,
            detected=detected,
            td=td,
            train_idx=train_idx,
            behavior_params=behavior_params,
            exp04a=exp04a,
        )
    )

    print()
    print(
        "Global behavior scale from TRAIN =",
        f"{global_scale:.6f}",
    )

    # ========================================================
    # VALIDATION user-level audit
    # ========================================================

    rows = []

    for idx in val_idx:

        if detected[idx] != 1:
            continue

        d = int(
            td[idx]
        )

        tf = int(
            true_tf[idx]
        )

        (
            candidates,
            raw_scores,
        ) = behavior_candidate_scores(
            x_user=x[idx],
            observed_td=d,
            behavior_params=behavior_params,
            exp04a=exp04a,
        )

        # ----------------------------------------------------
        # Global-scaled Behavior posterior
        # ----------------------------------------------------

        scaled_scores = (
            raw_scores
            - raw_scores.mean()
        ) / global_scale

        behavior_posterior = (
            exp04a.stable_softmax(
                scaled_scores
            )
        )

        # ----------------------------------------------------
        # Delay posterior
        # ----------------------------------------------------

        delay_scores = []

        for k in candidates:

            delay = (
                d - int(k)
            )

            p = max(
                exp03y.delay_probability(
                    delay
                ),
                1e-30,
            )

            delay_scores.append(
                math.log(p)
            )

        delay_posterior = (
            exp04a.stable_softmax(
                np.asarray(
                    delay_scores,
                    dtype=np.float64,
                )
            )
        )

        # ----------------------------------------------------
        # Behavior / Delay prediction
        # ----------------------------------------------------

        behavior_tf = (
            posterior_median(
                candidates,
                behavior_posterior,
            )
        )

        delay_tf = (
            posterior_median(
                candidates,
                delay_posterior,
            )
        )

        behavior_error = abs(
            behavior_tf - tf
        )

        delay_error = abs(
            delay_tf - tf
        )

        behavior_advantage = (
            delay_error
            - behavior_error
        )

        # ----------------------------------------------------
        # Confidence features
        # ----------------------------------------------------

        entropy = normalized_entropy(
            behavior_posterior
        )

        entropy_confidence = (
            1.0 - entropy
        )

        sorted_probs = np.sort(
            behavior_posterior
        )

        top1_probability = float(
            sorted_probs[-1]
        )

        if len(
            sorted_probs
        ) >= 2:

            margin = float(
                sorted_probs[-1]
                - sorted_probs[-2]
            )

        else:

            margin = 1.0

        posterior_mean = float(
            np.sum(
                behavior_posterior
                * candidates
            )
        )

        posterior_variance = float(
            np.sum(
                behavior_posterior
                * (
                    candidates
                    - posterior_mean
                ) ** 2
            )
        )

        posterior_std = math.sqrt(
            max(
                posterior_variance,
                0.0,
            )
        )

        negative_posterior_std = (
            -posterior_std
        )

        score_spread = float(
            (
                np.max(raw_scores)
                - np.min(raw_scores)
            )
            / global_scale
        )

        score_std = float(
            np.std(raw_scores)
            / global_scale
        )

        # top score gap
        sorted_scores = np.sort(
            scaled_scores
        )

        if len(
            sorted_scores
        ) >= 2:

            score_margin = float(
                sorted_scores[-1]
                - sorted_scores[-2]
            )

        else:

            score_margin = 0.0

        rows.append(
            {
                "index":
                    int(idx),

                "entity_id":
                    int(
                        entity_id[idx]
                    ),

                "true_tf":
                    tf,

                "observed_td":
                    d,

                "true_delay":
                    d - tf,

                "behavior_tf":
                    behavior_tf,

                "delay_tf":
                    delay_tf,

                "behavior_error":
                    behavior_error,

                "delay_error":
                    delay_error,

                "behavior_advantage":
                    behavior_advantage,

                "behavior_wins":
                    int(
                        behavior_error
                        < delay_error
                    ),

                "delay_wins":
                    int(
                        delay_error
                        < behavior_error
                    ),

                "tie":
                    int(
                        delay_error
                        == behavior_error
                    ),

                "entropy":
                    entropy,

                "entropy_confidence":
                    entropy_confidence,

                "top1_probability":
                    top1_probability,

                "probability_margin":
                    margin,

                "negative_posterior_std":
                    negative_posterior_std,

                "score_spread":
                    score_spread,

                "score_std":
                    score_std,

                "score_margin":
                    score_margin,
            }
        )

    audit_df = pd.DataFrame(
        rows
    )

    print()
    print("=" * 80)
    print(
        "VALIDATION USER-LEVEL SUMMARY"
    )
    print("=" * 80)

    print(
        "n users =",
        len(audit_df),
    )

    print(
        "Behavior wins =",
        int(
            audit_df[
                "behavior_wins"
            ].sum()
        ),
    )

    print(
        "Delay wins =",
        int(
            audit_df[
                "delay_wins"
            ].sum()
        ),
    )

    print(
        "Ties =",
        int(
            audit_df[
                "tie"
            ].sum()
        ),
    )

    print(
        "Mean Behavior Advantage =",
        float(
            audit_df[
                "behavior_advantage"
            ].mean()
        ),
    )

    # ========================================================
    # Correlation analysis
    # ========================================================

    feature_names = [
        "entropy_confidence",
        "top1_probability",
        "probability_margin",
        "negative_posterior_std",
        "score_spread",
        "score_std",
        "score_margin",
    ]

    summary_rows = []

    target = audit_df[
        "behavior_advantage"
    ].to_numpy()

    for feature in feature_names:

        values = audit_df[
            feature
        ].to_numpy()

        correlation = (
            spearman_correlation(
                values,
                target,
            )
        )

        summary_rows.append(
            {
                "feature":
                    feature,

                "spearman_with_behavior_advantage":
                    correlation,

                "absolute_spearman":
                    (
                        abs(correlation)
                        if not np.isnan(
                            correlation
                        )
                        else np.nan
                    ),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df = (
        summary_df
        .sort_values(
            "absolute_spearman",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    print()
    print("=" * 80)
    print(
        "CONFIDENCE FEATURE SUMMARY "
        "(VALIDATION ONLY)"
    )
    print("=" * 80)

    print(
        summary_df.to_string(
            index=False
        )
    )

    best_row = (
        summary_df.iloc[0]
    )

    print()
    print(
        "Best absolute Spearman feature:"
    )

    print(
        "  feature =",
        best_row[
            "feature"
        ],
    )

    print(
        "  correlation =",
        float(
            best_row[
                "spearman_with_behavior_advantage"
            ]
        ),
    )

    print()
    print(
        "Interpretation:"
    )

    print(
        "Positive correlation means:"
    )

    print(
        "  larger confidence tends to coincide "
        "with larger Behavior advantage."
    )

    print(
        "Correlation near 0 means:"
    )

    print(
        "  this confidence signal does not tell us "
        "when Behavior should be trusted more."
    )

    # ========================================================
    # Save
    # ========================================================

    em_history.to_csv(
        OUTPUT_DIR
        / "em_history.csv",
        index=False,
    )

    audit_df.to_csv(
        OUTPUT_DIR
        / "validation_user_confidence.csv",
        index=False,
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "confidence_feature_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()