from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Experiment 04A
# Learned Behavior Evidence via Latent Two-State Emission Model
#
# Goal:
#   Learn onset evidence from behavior x itself.
#
# Constraints:
#   1) true tf is NEVER used for training.
#   2) td is NOT a likelihood term or model feature.
#   3) td is used only as a strict-causal observation boundary.
#   4) fraud entities have one latent change point:
#          t < tf  -> normal state
#          t >= tf -> fraud state
#   5) tf is inferred by EM.
# ============================================================

DATA_PATH = Path(r"data/processed/synthetic_same_delay_consistent.npz")
REFERENCE_03Y = Path(r"scripts/32_exp03_behavior_likelihood_identification.py")
OUTPUT_DIR = Path(r"reports/experiment04/learned_behavior_evidence")

SEED = 0
MAX_EM_ITERS = 50
EM_TOL = 1e-5
VAR_FLOOR = 0.05 ** 2
INIT_FRAUD_TAIL_FRACTION = 0.25


def split_indices(n, seed=0):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_train = int(0.7 * n)
    n_val = int(0.1 * n)
    return (
        indices[:n_train],
        indices[n_train:n_train + n_val],
        indices[n_train + n_val:],
    )


def stable_softmax(scores):
    scores = np.asarray(scores, dtype=np.float64)
    scores = scores - np.max(scores)
    probs = np.exp(scores)
    total = float(probs.sum())
    if total <= 0:
        return np.ones_like(probs) / len(probs)
    return probs / total


def logsumexp(values):
    values = np.asarray(values, dtype=np.float64)
    m = float(np.max(values))
    return m + math.log(float(np.exp(values - m).sum()))


def posterior_median(candidates, posterior):
    cumulative = np.cumsum(posterior)
    index = int(np.searchsorted(cumulative, 0.5))
    index = min(index, len(candidates) - 1)
    return int(candidates[index])


def delay_bucket(delay):
    if delay <= 2:
        return "delay_1_2"
    if delay <= 5:
        return "delay_3_5"
    return "delay_6_plus"


def get_entity_fraud_label(data):
    for key in ['is_fraud', 'fraud', 'label', 'y']:
        if key in data:
            print('Entity fraud label source:', key)
            return data[key].astype(np.int64)

    if 't_onset' in data:
        onset = data['t_onset'].astype(np.int64)

        if np.any(np.less(onset, 0)):
            print('WARNING: explicit is_fraud is missing.')
            print('Recovering ONLY the binary entity label from t_onset validity.')
            print('Negative t_onset = normal; non-negative t_onset = fraud.')
            print('The numeric tf value is NOT used for training.')
            return np.greater_equal(onset, 0).astype(np.int64)

    raise KeyError('No usable entity-level fraud label was found.')


def observation_end(detected_value, td_value, sequence_length):
    """
    Strict-causal observation boundary.

    detected == 1:
        observe x[0:td+1]

    detected == 0:
        observe the full available window

    td is used ONLY as the observation boundary.
    """
    if int(detected_value) == 1:
        return min(int(td_value), sequence_length - 1)
    return sequence_length - 1


def candidate_onsets(detected_value, obs_end):
    """
    detected fraud:
        tf < td, so candidates are 0..obs_end-1

    undetected/censored fraud:
        tf may be anywhere in observed window
    """
    if int(detected_value) == 1:
        last = max(obs_end, 1)
        return np.arange(0, last, dtype=np.int64)

    return np.arange(0, obs_end + 1, dtype=np.int64)


def gaussian_log_prob(observations, mean, variance):
    """
    Diagonal Gaussian emission.
    Returns one log-probability per time step.
    """
    variance = np.maximum(variance, VAR_FLOOR)
    residual = observations - mean

    return -0.5 * (
        np.sum(np.log(2.0 * math.pi * variance))
        + np.sum(residual ** 2 / variance, axis=-1)
    )


def initialize_emissions(
    x,
    fraud_label,
    detected,
    td,
    train_idx,
    lengths=None,
):
    """
    Normal state initialization:
        all observed points from TRAIN normal entities.

    Fraud state initialization:
        tail of observed prefixes from TRAIN fraud entities.

    No true tf is used.
    """
    sequence_length = x.shape[1]

    normal_points = []
    fraud_tail_points = []

    for idx in train_idx:
        end = observation_end(
            detected[idx],
            td[idx],
            sequence_length if lengths is None else int(lengths[idx]),
        )

        observed = x[idx, :end + 1, :]

        if fraud_label[idx] == 0:
            normal_points.append(observed)
        else:
            tail_length = max(
                1,
                int(math.ceil(
                    len(observed) * INIT_FRAUD_TAIL_FRACTION
                )),
            )
            fraud_tail_points.append(observed[-tail_length:])

    if not normal_points:
        raise RuntimeError("No normal TRAIN entities found.")

    if not fraud_tail_points:
        raise RuntimeError("No fraud TRAIN entities found.")

    normal_points = np.concatenate(normal_points, axis=0)
    fraud_tail_points = np.concatenate(fraud_tail_points, axis=0)

    normal_mean = np.mean(normal_points, axis=0)
    normal_var = np.maximum(
        np.var(normal_points, axis=0),
        VAR_FLOOR,
    )

    fraud_mean = np.mean(fraud_tail_points, axis=0)
    fraud_var = np.maximum(
        np.var(fraud_tail_points, axis=0),
        VAR_FLOOR,
    )

    return {
        "normal_mean": normal_mean,
        "normal_var": normal_var,
        "fraud_mean": fraud_mean,
        "fraud_var": fraud_var,
        "n_normal_points": len(normal_points),
        "n_fraud_tail_points": len(fraud_tail_points),
    }


def candidate_behavior_log_likelihood(
    observed_x,
    candidate_tf,
    params,
):
    """
    If tf = k:
        t < k  -> normal emission
        t >= k -> fraud emission
    """
    normal_lp = gaussian_log_prob(
        observed_x,
        params["normal_mean"],
        params["normal_var"],
    )

    fraud_lp = gaussian_log_prob(
        observed_x,
        params["fraud_mean"],
        params["fraud_var"],
    )

    k = int(candidate_tf)

    total = 0.0

    if k > 0:
        total += float(normal_lp[:k].sum())

    total += float(fraud_lp[k:].sum())

    return total


def onset_posterior_from_behavior(
    observed_x,
    candidates,
    params,
):
    """
    Uniform onset prior:
        posterior(tf=k | x) proportional to p(x | tf=k)
    """
    log_prior = -math.log(len(candidates))

    scores = np.asarray(
        [
            candidate_behavior_log_likelihood(
                observed_x,
                int(k),
                params,
            ) + log_prior
            for k in candidates
        ],
        dtype=np.float64,
    )

    return stable_softmax(scores), scores


def train_emission_em(
    x,
    fraud_label,
    detected,
    td,
    train_idx,
    lengths=None,
):
    sequence_length = x.shape[1]
    feature_dim = x.shape[2]

    params = initialize_emissions(
        x,
        fraud_label,
        detected,
        td,
        train_idx,
        lengths=lengths,
    )

    print("\nInitial emission parameters:")
    print("  normal mean:", np.round(params["normal_mean"], 3))
    print("  fraud mean:", np.round(params["fraud_mean"], 3))
    print("  normal points:", params["n_normal_points"])
    print("  fraud-tail init points:", params["n_fraud_tail_points"])

    history = []
    previous_objective = None

    for iteration in range(1, MAX_EM_ITERS + 1):
        normal_weight_sum = 0.0
        fraud_weight_sum = 0.0

        normal_sum = np.zeros(feature_dim, dtype=np.float64)
        fraud_sum = np.zeros(feature_dim, dtype=np.float64)

        normal_sq_sum = np.zeros(feature_dim, dtype=np.float64)
        fraud_sq_sum = np.zeros(feature_dim, dtype=np.float64)

        total_log_likelihood = 0.0
        posterior_entropies = []

        n_train_normal = 0
        n_train_fraud = 0

        for idx in train_idx:
            end = observation_end(
                detected[idx],
                td[idx],
                sequence_length if lengths is None else int(lengths[idx]),
            )

            observed = x[idx, :end + 1, :]

            if fraud_label[idx] == 0:
                n_train_normal += 1

                normal_lp = gaussian_log_prob(
                    observed,
                    params["normal_mean"],
                    params["normal_var"],
                )

                total_log_likelihood += float(normal_lp.sum())

                weights = np.ones(len(observed), dtype=np.float64)

                normal_weight_sum += float(weights.sum())
                normal_sum += np.sum(
                    observed * weights[:, None],
                    axis=0,
                )
                normal_sq_sum += np.sum(
                    observed ** 2 * weights[:, None],
                    axis=0,
                )
                continue

            n_train_fraud += 1

            candidates = candidate_onsets(
                detected[idx],
                end,
            )

            posterior, scores = onset_posterior_from_behavior(
                observed,
                candidates,
                params,
            )

            total_log_likelihood += logsumexp(scores)

            entropy = -np.sum(
                posterior
                * np.log(np.maximum(posterior, 1e-30))
            )
            posterior_entropies.append(float(entropy))

            # P(fraud at time t) = P(tf <= t)
            fraud_state_prob = np.zeros(
                len(observed),
                dtype=np.float64,
            )

            posterior_map = {
                int(k): float(p)
                for k, p in zip(candidates, posterior)
            }

            cumulative = 0.0

            for t in range(len(observed)):
                cumulative += posterior_map.get(t, 0.0)
                fraud_state_prob[t] = cumulative

            fraud_state_prob = np.clip(
                fraud_state_prob,
                0.0,
                1.0,
            )

            normal_state_prob = 1.0 - fraud_state_prob

            normal_weight_sum += float(normal_state_prob.sum())
            fraud_weight_sum += float(fraud_state_prob.sum())

            normal_sum += np.sum(
                observed * normal_state_prob[:, None],
                axis=0,
            )
            fraud_sum += np.sum(
                observed * fraud_state_prob[:, None],
                axis=0,
            )

            normal_sq_sum += np.sum(
                observed ** 2 * normal_state_prob[:, None],
                axis=0,
            )
            fraud_sq_sum += np.sum(
                observed ** 2 * fraud_state_prob[:, None],
                axis=0,
            )

        if normal_weight_sum <= 0:
            raise RuntimeError(
                "Normal emission received zero weight."
            )

        if fraud_weight_sum <= 0:
            raise RuntimeError(
                "Fraud emission received zero weight."
            )

        new_normal_mean = normal_sum / normal_weight_sum
        new_fraud_mean = fraud_sum / fraud_weight_sum

        new_normal_var = (
            normal_sq_sum / normal_weight_sum
            - new_normal_mean ** 2
        )
        new_fraud_var = (
            fraud_sq_sum / fraud_weight_sum
            - new_fraud_mean ** 2
        )

        params["normal_mean"] = new_normal_mean
        params["normal_var"] = np.maximum(
            new_normal_var,
            VAR_FLOOR,
        )

        params["fraud_mean"] = new_fraud_mean
        params["fraud_var"] = np.maximum(
            new_fraud_var,
            VAR_FLOOR,
        )

        mean_entropy = (
            float(np.mean(posterior_entropies))
            if posterior_entropies
            else 0.0
        )

        history.append(
            {
                "iteration": iteration,
                "train_log_likelihood": float(total_log_likelihood),
                "mean_fraud_posterior_entropy": mean_entropy,
                "normal_weight_sum": normal_weight_sum,
                "fraud_weight_sum": fraud_weight_sum,
                "n_train_normal": n_train_normal,
                "n_train_fraud": n_train_fraud,
            }
        )

        print(
            f"EM {iteration:02d} | "
            f"logL={total_log_likelihood:.6f} | "
            f"entropy={mean_entropy:.6f} | "
            f"normal_w={normal_weight_sum:.1f} | "
            f"fraud_w={fraud_weight_sum:.1f}"
        )

        if previous_objective is not None:
            improvement = total_log_likelihood - previous_objective

            if abs(improvement) < EM_TOL:
                print("EM converged.")
                break

        previous_objective = total_log_likelihood

    return params, pd.DataFrame(history)


def infer_user(
    x_user,
    detected_value,
    td_value,
    params,
):
    sequence_length = len(x_user)

    end = observation_end(
        detected_value,
        td_value,
        sequence_length,
    )

    observed = x_user[:end + 1]

    candidates = candidate_onsets(
        detected_value,
        end,
    )

    posterior, scores = onset_posterior_from_behavior(
        observed,
        candidates,
        params,
    )

    mean_tf = float(
        np.sum(posterior * candidates)
    )

    median_tf = posterior_median(
        candidates,
        posterior,
    )

    mode_tf = int(
        candidates[np.argmax(posterior)]
    )

    return {
        "candidates": candidates,
        "posterior": posterior,
        "scores": scores,
        "mean_tf": mean_tf,
        "median_tf": median_tf,
        "mode_tf": mode_tf,
        "obs_end": end,
    }


def summarize_estimator(
    df,
    prediction_column,
):
    error = (
        df[prediction_column].to_numpy()
        - df["true_tf"].to_numpy()
    )

    absolute_error = np.abs(error)

    return {
        "estimator": prediction_column,
        "n": len(df),
        "mae": float(absolute_error.mean()),
        "signed_error": float(error.mean()),
        "within_1": float((absolute_error <= 1).mean()),
        "within_2": float((absolute_error <= 2).mean()),
    }


def load_exp03y_reference():
    if not REFERENCE_03Y.exists():
        return None

    spec = importlib.util.spec_from_file_location(
        "exp03y_reference",
        REFERENCE_03Y,
    )

    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def handcrafted_behavior_only_predictions(
    exp03y,
    x,
    detected,
    td,
    train_idx,
    test_idx,
):
    """
    Reconstruct Experiment 03Y behavior-only posterior
    on detected fraud users.

    No delay evidence is used here.
    """
    anchors = exp03y.estimate_behavior_anchors(
        x=x,
        detected=detected,
        td=td,
        train_idx=train_idx,
    )

    rows = []

    for idx in test_idx:
        if detected[idx] != 1:
            continue

        d = int(td[idx])

        candidates = np.arange(
            0,
            d,
            dtype=np.int64,
        )

        scores = []

        for k in candidates:
            ll = exp03y.behavior_log_likelihood(
                observed_x=x[idx],
                candidate_tf=int(k),
                observed_td=d,
                normal_center=anchors["normal_center"],
                fraud_center=anchors["fraud_center"],
                scale=anchors["scale"],
            )
            scores.append(ll)

        scores = np.asarray(scores, dtype=np.float64)

        std = float(np.std(scores))

        if std < 1e-8:
            normalized = np.zeros_like(scores)
        else:
            normalized = (
                scores - scores.mean()
            ) / std

        posterior = exp03y.softmax(
            normalized
        )

        rows.append(
            {
                "index": int(idx),
                "handcrafted_mean_tf": float(
                    np.sum(posterior * candidates)
                ),
                "handcrafted_median_tf": posterior_median(
                    candidates,
                    posterior,
                ),
                "handcrafted_mode_tf": int(
                    candidates[np.argmax(posterior)]
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("EXPERIMENT 04A:")
    print("LEARNED BEHAVIOR EVIDENCE")
    print("=" * 80)
    print()
    print("Question:")
    print(
        "Can behavior x alone learn useful evidence "
        "for latent fraud-onset tf?"
    )
    print()
    print("IMPORTANT:")
    print("  true tf is NEVER used for training.")
    print("  td is NOT a likelihood term.")
    print(
        "  td is used only as a strict-causal "
        "observation boundary."
    )

    data = np.load(
        DATA_PATH,
        allow_pickle=True,
    )

    print("\nDataset keys:")
    print(list(data.keys()))

    x = data["x"].astype(np.float64)
    detected = data["detected"].astype(np.int64)
    td = data["t_detect"].astype(np.int64)
    true_tf = data["t_onset"].astype(np.int64)

    fraud_label = get_entity_fraud_label(data)

    if "entity_id" in data:
        entity_id = data["entity_id"].astype(np.int64)
    else:
        entity_id = np.arange(len(x))

    train_idx, val_idx, test_idx = split_indices(
        len(x),
        seed=SEED,
    )

    print("\nSplit sizes:")
    print("  train:", len(train_idx))
    print("  val:", len(val_idx))
    print("  test:", len(test_idx))

    print("\nTRAIN entity labels:")
    print(
        "  normal:",
        int(np.sum(fraud_label[train_idx] == 0)),
    )
    print(
        "  fraud:",
        int(np.sum(fraud_label[train_idx] == 1)),
    )

    params, history_df = train_emission_em(
        x=x,
        fraud_label=fraud_label,
        detected=detected,
        td=td,
        train_idx=train_idx,
    )

    print("\n" + "=" * 80)
    print("FINAL LEARNED EMISSIONS")
    print("=" * 80)
    print("Normal mean:")
    print(np.round(params["normal_mean"], 4))
    print("\nFraud mean:")
    print(np.round(params["fraud_mean"], 4))
    print("\nFraud - Normal:")
    print(
        np.round(
            params["fraud_mean"]
            - params["normal_mean"],
            4,
        )
    )

    rows = []

    for idx in test_idx:
        if fraud_label[idx] != 1:
            continue

        prediction = infer_user(
            x_user=x[idx],
            detected_value=detected[idx],
            td_value=td[idx],
            params=params,
        )

        tf = int(true_tf[idx])
        d = int(td[idx])

        row = {
            "index": int(idx),
            "entity_id": int(entity_id[idx]),
            "detected": int(detected[idx]),
            "true_tf": tf,
            "observed_td": d,
            "obs_end": int(prediction["obs_end"]),
            "learned_mean_tf": prediction["mean_tf"],
            "learned_median_tf": prediction["median_tf"],
            "learned_mode_tf": prediction["mode_tf"],
        }

        if detected[idx] == 1:
            true_delay = d - tf
            row["true_delay"] = true_delay
            row["delay_bucket"] = delay_bucket(true_delay)
        else:
            row["true_delay"] = np.nan
            row["delay_bucket"] = "censored"

        rows.append(row)

    entity_df = pd.DataFrame(rows)

    detected_test_df = entity_df[
        entity_df["detected"] == 1
    ].copy()

    estimator_rows = []

    for prediction_column in [
        "learned_mean_tf",
        "learned_median_tf",
        "learned_mode_tf",
    ]:
        metrics = summarize_estimator(
            detected_test_df,
            prediction_column,
        )
        metrics["method"] = "learned_behavior"
        estimator_rows.append(metrics)

    estimator_summary = pd.DataFrame(
        estimator_rows
    )

    print("\n" + "=" * 80)
    print("DETECTED TEST FRAUD:")
    print("LEARNED BEHAVIOR-ONLY tf RECOVERY")
    print("=" * 80)
    print(
        estimator_summary.to_string(
            index=False
        )
    )

    bucket_rows = []

    for bucket in [
        "delay_1_2",
        "delay_3_5",
        "delay_6_plus",
    ]:
        sub = detected_test_df[
            detected_test_df["delay_bucket"] == bucket
        ].copy()

        for prediction_column in [
            "learned_mean_tf",
            "learned_median_tf",
            "learned_mode_tf",
        ]:
            if len(sub) == 0:
                continue

            error = np.abs(
                sub[prediction_column].to_numpy()
                - sub["true_tf"].to_numpy()
            )

            bucket_rows.append(
                {
                    "delay_bucket": bucket,
                    "estimator": prediction_column,
                    "n_users": len(sub),
                    "mae": float(error.mean()),
                }
            )

    bucket_summary = pd.DataFrame(bucket_rows)

    print("\n" + "=" * 80)
    print("BY TRUE DELAY")
    print("=" * 80)
    print(bucket_summary.to_string(index=False))

    exp03y = load_exp03y_reference()
    comparison_summary = None

    if exp03y is not None:
        handcrafted_df = handcrafted_behavior_only_predictions(
            exp03y=exp03y,
            x=x,
            detected=detected,
            td=td,
            train_idx=train_idx,
            test_idx=test_idx,
        )

        detected_test_df = detected_test_df.merge(
            handcrafted_df,
            on="index",
            how="left",
        )

        comparison_rows = []

        for method_name, column in [
            ("handcrafted_behavior_mean", "handcrafted_mean_tf"),
            ("learned_behavior_mean", "learned_mean_tf"),
            ("handcrafted_behavior_median", "handcrafted_median_tf"),
            ("learned_behavior_median", "learned_median_tf"),
            ("handcrafted_behavior_mode", "handcrafted_mode_tf"),
            ("learned_behavior_mode", "learned_mode_tf"),
        ]:
            valid = detected_test_df[
                [column, "true_tf"]
            ].dropna()

            error = (
                valid[column].to_numpy()
                - valid["true_tf"].to_numpy()
            )

            comparison_rows.append(
                {
                    "method": method_name,
                    "n": len(valid),
                    "mae": float(np.mean(np.abs(error))),
                    "signed_error": float(np.mean(error)),
                }
            )

        comparison_summary = pd.DataFrame(
            comparison_rows
        )

        print("\n" + "=" * 80)
        print("HANDCRAFTED 03Y vs LEARNED 04A")
        print("=" * 80)
        print(
            comparison_summary.to_string(
                index=False
            )
        )

    else:
        print("\nWARNING:")
        print(
            "Could not load Experiment 03Y reference."
        )
        print(
            "Learned 04A results are still valid, "
            "but handcrafted comparison is skipped."
        )

    history_df.to_csv(
        OUTPUT_DIR / "em_history.csv",
        index=False,
    )

    entity_df.to_csv(
        OUTPUT_DIR / "all_fraud_entity_diagnostics.csv",
        index=False,
    )

    detected_test_df.to_csv(
        OUTPUT_DIR / "detected_fraud_entity_diagnostics.csv",
        index=False,
    )

    estimator_summary.to_csv(
        OUTPUT_DIR / "estimator_summary.csv",
        index=False,
    )

    bucket_summary.to_csv(
        OUTPUT_DIR / "delay_bucket_summary.csv",
        index=False,
    )

    if comparison_summary is not None:
        comparison_summary.to_csv(
            OUTPUT_DIR / "handcrafted_vs_learned.csv",
            index=False,
        )

    np.savez(
        OUTPUT_DIR / "learned_emission_params.npz",
        normal_mean=params["normal_mean"],
        normal_var=params["normal_var"],
        fraud_mean=params["fraud_mean"],
        fraud_var=params["fraud_var"],
    )

    print("\n" + "=" * 80)
    print("HOW TO INTERPRET 04A")
    print("=" * 80)
    print()
    print("Strong positive signal if:")
    print(
        "1. learned behavior-only MAE is better "
        "than handcrafted behavior-only;"
    )
    print(
        "2. learned posterior does not show a "
        "strong systematic left/right bias;"
    )
    print(
        "3. long-delay users also improve;"
    )
    print(
        "4. EM converges without collapsing the "
        "fraud emission onto the normal emission."
    )
    print()
    print("If 04A works:")
    print(
        "  proceed to 04B = Learned Behavior + Delay."
    )
    print()
    print("If 04A fails:")
    print(
        "  inspect emission collapse / posterior bias "
        "before introducing td likelihood."
    )


if __name__ == "__main__":
    main()
