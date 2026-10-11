from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from safeplus.data import load_npz, make_loaders
from safeplus.models import build_model
from safeplus.utils import load_config, seed_everything


# ============================================================
# Experiment 03 posterior / sampling diagnostic
# ============================================================

EXACT_RUN = Path(r"runs/synthetic/exp03/safeplus-exact/seed_0")
SEM_RUN = Path(r"runs/synthetic/exp03/safeplus-sem/seed_0")

OUTPUT_DIR = Path(r"reports/experiment03/posterior_diagnostics")

DEVICE = "cpu"

# Number of posterior samples drawn for each known-fraud entity.
N_SAMPLES = 1000

# Reproducible sampling.
SAMPLING_SEED = 20261008

# Representative entity used in previous experiments.
TARGET_ENTITY_ID = 1067


def _to_device(batch, device):
    return {
        key: value.to(device)
        for key, value in batch.items()
    }


def _get_test_entity_ids(dataset, test_loader):
    """
    Recover entity IDs in exactly the same order as the held-out test loader.
    """
    arrays = dataset.arrays

    if "entity_id" in arrays:
        all_ids = np.asarray(arrays["entity_id"])
    else:
        all_ids = np.arange(len(arrays["x"]))

    test_dataset = test_loader.dataset

    if hasattr(test_dataset, "indices"):
        indices = np.asarray(test_dataset.indices)
        return all_ids[indices]

    # Fallback only if the loader does not use torch.utils.data.Subset.
    return all_ids[: len(test_dataset)]


def load_posterior_run(run_dir: Path, model_label: str):
    print("=" * 80)
    print(f"Loading {model_label}")
    print(f"Run directory: {run_dir}")
    print("=" * 80)

    cfg = load_config(run_dir / "config.yaml")
    seed_everything(cfg["seed"])

    dataset = load_npz(cfg["data"]["path"])

    _, _, test_loader = make_loaders(
        dataset,
        cfg["training"]["batch_size"],
        cfg["data"]["split"],
        cfg["seed"],
    )

    entity_ids = _get_test_entity_ids(dataset, test_loader)

    model_options = dict(cfg["model"])
    model_name = model_options.pop("name")

    model = build_model(
        model_name,
        dataset.arrays["x"].shape[-1],
        **model_options,
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            run_dir / "model.pt",
            map_location=DEVICE,
            weights_only=True,
        )
    )

    model.eval()

    onset_posteriors = []
    latent_posteriors = []
    no_commission_probs = []

    detected_all = []
    t_detect_all = []
    t_onset_all = []
    lengths_all = []
    risks_all = []

    with torch.inference_mode():
        for raw_batch in test_loader:
            batch = _to_device(raw_batch, DEVICE)

            _, out = model.loss(batch)

            if "onset_posterior" not in out:
                raise RuntimeError(
                    f"{model_label} did not produce onset_posterior."
                )

            onset_posteriors.append(
                out["onset_posterior"].detach().cpu().numpy()
            )

            latent_posteriors.append(
                out["latent_posterior"].detach().cpu().numpy()
            )

            no_commission_probs.append(
                out["no_commission_posterior"].detach().cpu().numpy()
            )

            detected_all.append(
                batch["detected"].detach().cpu().numpy().astype(bool)
            )

            t_detect_all.append(
                batch["t_detect"].detach().cpu().numpy()
            )

            t_onset_all.append(
                batch["t_onset"].detach().cpu().numpy()
            )

            lengths_all.append(
                batch["lengths"].detach().cpu().numpy()
            )

            risks_all.append(
                out["risk"].detach().cpu().numpy()
            )

    onset_posterior = np.concatenate(onset_posteriors)
    latent_posterior = np.concatenate(latent_posteriors)
    no_commission = np.concatenate(no_commission_probs)

    detected = np.concatenate(detected_all)
    t_detect = np.concatenate(t_detect_all)
    t_onset = np.concatenate(t_onset_all)
    lengths = np.concatenate(lengths_all)
    risk = np.concatenate(risks_all)

    if len(entity_ids) != len(t_onset):
        raise RuntimeError(
            f"Entity-ID alignment failed: "
            f"{len(entity_ids)} ids vs {len(t_onset)} test rows."
        )

    n, length = onset_posterior.shape

    # --------------------------------------------------------
    # Posterior point estimates
    # --------------------------------------------------------

    # Same estimator currently used by metrics.py.
    posterior_mode = onset_posterior.argmax(axis=1)

    # Probability that commission occurred within the observed horizon.
    onset_mass = onset_posterior.sum(axis=1)

    steps = np.arange(length, dtype=np.float64)

    # Posterior mean conditional on commission having occurred.
    posterior_mean_cond = np.full(n, np.nan, dtype=np.float64)

    valid_mass = onset_mass > 0

    posterior_mean_cond[valid_mass] = (
        onset_posterior[valid_mass] * steps[None, :]
    ).sum(axis=1) / onset_mass[valid_mass]

    known_fraud = t_onset >= 0
    detected_fraud = known_fraud & detected
    censored_fraud = known_fraud & (~detected)
    normal = ~known_fraud

    # --------------------------------------------------------
    # Probability placed on the true tf
    # --------------------------------------------------------

    true_tf_prob = np.full(n, np.nan, dtype=np.float64)

    known_idx = np.where(known_fraud)[0]

    true_tf_prob[known_idx] = onset_posterior[
        known_idx,
        t_onset[known_idx],
    ]

    # --------------------------------------------------------
    # Deterministic expected absolute error under posterior
    # conditional on commission.
    # --------------------------------------------------------

    expected_abs_error_cond = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    for i in known_idx:
        mass = onset_mass[i]

        if mass <= 0:
            continue

        conditional = onset_posterior[i] / mass

        expected_abs_error_cond[i] = np.sum(
            conditional
            * np.abs(
                np.arange(length)
                - t_onset[i]
            )
        )

    # --------------------------------------------------------
    # Actual posterior sampling
    # --------------------------------------------------------

    torch.manual_seed(SAMPLING_SEED)

    latent_tensor = torch.as_tensor(
        latent_posterior,
        dtype=torch.float32,
    )

    # Full latent posterior:
    # columns 0 ... L-1 = commission at that time
    # column L = no commission through horizon.
    full_samples = torch.multinomial(
        latent_tensor,
        N_SAMPLES,
        replacement=True,
    ).numpy()

    no_commission_state = length

    sampled_no_commission_rate = (
        full_samples == no_commission_state
    ).mean(axis=1)

    # --------------------------------------------------------
    # Sampling tf conditional on commission.
    #
    # This is useful for answering:
    # "Given that fraud really occurred, where does the posterior
    # locate tf?"
    # --------------------------------------------------------

    sampled_tf_mean_cond = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    sampled_tf_mae_cond = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    conditional_samples = {}

    for i in known_idx:
        probs = onset_posterior[i].copy()

        mass = probs.sum()

        if mass <= 0:
            continue

        probs = probs / mass

        probs_tensor = torch.as_tensor(
            probs,
            dtype=torch.float32,
        )

        samples = torch.multinomial(
            probs_tensor,
            N_SAMPLES,
            replacement=True,
        ).numpy()

        conditional_samples[i] = samples

        sampled_tf_mean_cond[i] = samples.mean()

        sampled_tf_mae_cond[i] = np.abs(
            samples - t_onset[i]
        ).mean()

    # --------------------------------------------------------
    # Entity-level table
    # --------------------------------------------------------

    group = np.full(n, "normal", dtype=object)
    group[detected_fraud] = "detected_fraud"
    group[censored_fraud] = "censored_fraud"

    frame = pd.DataFrame(
        {
            "entity_id": entity_ids,
            "group": group,
            "detected": detected.astype(int),
            "length": lengths,
            "true_tf": t_onset,
            "observed_td": t_detect,
            "posterior_mode_tf": posterior_mode,
            "posterior_mean_tf_given_commission": posterior_mean_cond,
            "posterior_onset_mass": onset_mass,
            "no_commission_posterior": no_commission,
            "true_tf_posterior_probability": true_tf_prob,
            "expected_abs_error_given_commission": expected_abs_error_cond,
            "sampled_tf_mean_given_commission": sampled_tf_mean_cond,
            "sampled_tf_mae_given_commission": sampled_tf_mae_cond,
            "sampled_no_commission_rate": sampled_no_commission_rate,
        }
    )

    # --------------------------------------------------------
    # Aggregate diagnostics
    # --------------------------------------------------------

    def mean_or_nan(values):
        values = np.asarray(values)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            return float("nan")

        return float(values.mean())

    summary = {
        "model": model_label,
        "n_test": int(n),
        "n_normal": int(normal.sum()),
        "n_detected_fraud": int(detected_fraud.sum()),
        "n_censored_fraud": int(censored_fraud.sum()),

        # Same point-estimate MAE as metrics.py.
        "posterior_mode_mae_all_known_fraud": mean_or_nan(
            np.abs(
                posterior_mode[known_fraud]
                - t_onset[known_fraud]
            )
        ),

        "posterior_mode_mae_detected_fraud": mean_or_nan(
            np.abs(
                posterior_mode[detected_fraud]
                - t_onset[detected_fraud]
            )
        ),

        "posterior_mode_mae_censored_fraud": mean_or_nan(
            np.abs(
                posterior_mode[censored_fraud]
                - t_onset[censored_fraud]
            )
        ),

        "posterior_mean_mae_all_known_fraud": mean_or_nan(
            np.abs(
                posterior_mean_cond[known_fraud]
                - t_onset[known_fraud]
            )
        ),

        "posterior_expected_abs_error_all_known_fraud": mean_or_nan(
            expected_abs_error_cond[known_fraud]
        ),

        "sampled_tf_mae_given_commission_all_known_fraud": mean_or_nan(
            sampled_tf_mae_cond[known_fraud]
        ),

        "mean_true_tf_probability": mean_or_nan(
            true_tf_prob[known_fraud]
        ),

        "onset_nll_all_known_fraud": mean_or_nan(
            -np.log(
                np.clip(
                    true_tf_prob[known_fraud],
                    1e-12,
                    1.0,
                )
            )
        ),

        "mean_no_commission_posterior_normal": mean_or_nan(
            no_commission[normal]
        ),

        "mean_no_commission_posterior_censored_fraud": mean_or_nan(
            no_commission[censored_fraud]
        ),

        "full_posterior_sample_no_commission_rate_censored_fraud": mean_or_nan(
            sampled_no_commission_rate[censored_fraud]
        ),
    }

    return {
        "summary": summary,
        "frame": frame,
        "entity_ids": entity_ids,
        "onset_posterior": onset_posterior,
        "latent_posterior": latent_posterior,
        "no_commission": no_commission,
        "detected": detected,
        "t_detect": t_detect,
        "t_onset": t_onset,
        "lengths": lengths,
        "risk": risk,
        "conditional_samples": conditional_samples,
    }


def print_summary(result):
    print()
    print("=" * 80)
    print(result["summary"]["model"])
    print("=" * 80)

    for key, value in result["summary"].items():
        print(f"{key}: {value}")


def print_entity(result, entity_id):
    ids = np.asarray(result["entity_ids"])

    matches = np.where(
        ids.astype(str) == str(entity_id)
    )[0]

    if len(matches) == 0:
        return False

    i = int(matches[0])

    tf = int(result["t_onset"][i])
    td = int(result["t_detect"][i])
    detected = bool(result["detected"][i])

    posterior = result["onset_posterior"][i]
    no_commission = float(result["no_commission"][i])

    print()
    print("=" * 80)
    print(
        f'{result["summary"]["model"]} | '
        f'Entity {entity_id}'
    )
    print("=" * 80)

    print(f"true tf              = {tf}")
    print(f"observed td          = {td}")
    print(f"detected             = {detected}")
    print(
        f"posterior mode tf    = "
        f"{int(posterior.argmax())}"
    )

    mass = posterior.sum()

    if mass > 0:
        mean_tf = (
            posterior
            * np.arange(len(posterior))
        ).sum() / mass
        print(f"posterior mean tf    = {mean_tf:.4f}")

    print(
        f"no-commission prob   = "
        f"{no_commission:.6f}"
    )

    if tf >= 0:
        print(
            f"P(true tf)           = "
            f"{posterior[tf]:.6f}"
        )

    print()
    print("Posterior over tf:")

    for t, p in enumerate(posterior):
        if p > 0:
            marker = ""

            if t == tf:
                marker += " <-- TRUE tf"

            if t == td:
                marker += " <-- td"

            print(
                f"t={t:2d}  "
                f"p={p:.6f}"
                f"{marker}"
            )

    if i in result["conditional_samples"]:
        samples = result["conditional_samples"][i]

        values, counts = np.unique(
            samples,
            return_counts=True,
        )

        print()
        print(
            f"Conditional sampling distribution "
            f"({N_SAMPLES} samples):"
        )

        for value, count in zip(values, counts):
            marker = ""

            if value == tf:
                marker = " <-- TRUE tf"

            print(
                f"tf={int(value):2d}: "
                f"{int(count):4d} samples "
                f"({count / N_SAMPLES:.3f})"
                f"{marker}"
            )

    return True


def plot_entity_comparison(
    exact,
    sem,
    entity_id,
    output_path,
):
    exact_ids = np.asarray(exact["entity_ids"])
    sem_ids = np.asarray(sem["entity_ids"])

    exact_match = np.where(
        exact_ids.astype(str) == str(entity_id)
    )[0]

    sem_match = np.where(
        sem_ids.astype(str) == str(entity_id)
    )[0]

    if len(exact_match) == 0 or len(sem_match) == 0:
        print(
            f"Entity {entity_id} not present in test split; "
            f"posterior figure skipped."
        )
        return

    ie = int(exact_match[0])
    isem = int(sem_match[0])

    tf = int(exact["t_onset"][ie])
    td = int(exact["t_detect"][ie])

    p_exact = exact["onset_posterior"][ie]
    p_sem = sem["onset_posterior"][isem]

    times = np.arange(len(p_exact))

    plt.figure(figsize=(9, 5))

    plt.plot(
        times,
        p_exact,
        marker="o",
        label="Joint-Exact",
    )

    plt.plot(
        times,
        p_sem,
        marker="s",
        label="Joint-Sampling",
    )

    if tf >= 0:
        plt.axvline(
            tf,
            linestyle="--",
            label=f"True tf = {tf}",
        )

    plt.axvline(
        td,
        linestyle=":",
        label=f"Observed td = {td}",
    )

    plt.xlabel("Time step")
    plt.ylabel(r"$p(t_f=t \mid t_d,x)$")
    plt.title(
        f"Latent fraud-onset posterior: entity {entity_id}"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_result(result, prefix):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result["frame"].to_csv(
        OUTPUT_DIR / f"{prefix}_entity_diagnostics.csv",
        index=False,
    )

    np.savez_compressed(
        OUTPUT_DIR / f"{prefix}_posterior_data.npz",
        entity_id=result["entity_ids"],
        onset_posterior=result["onset_posterior"],
        latent_posterior=result["latent_posterior"],
        no_commission_posterior=result["no_commission"],
        detected=result["detected"],
        t_detect=result["t_detect"],
        t_onset=result["t_onset"],
        lengths=result["lengths"],
    )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    exact = load_posterior_run(
        EXACT_RUN,
        "Joint-Exact",
    )

    sem = load_posterior_run(
        SEM_RUN,
        "Joint-Sampling",
    )

    print_summary(exact)
    print_summary(sem)

    save_result(
        exact,
        "joint_exact",
    )

    save_result(
        sem,
        "joint_sampling",
    )

    combined_summary = {
        "joint_exact": exact["summary"],
        "joint_sampling": sem["summary"],
    }

    (
        OUTPUT_DIR / "summary.json"
    ).write_text(
        json.dumps(
            combined_summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Representative entity
    # --------------------------------------------------------

    target = TARGET_ENTITY_ID

    exact_found = print_entity(
        exact,
        target,
    )

    sem_found = print_entity(
        sem,
        target,
    )

    # If entity 1067 is unexpectedly not in the test split,
    # automatically use the first detected fraud entity.
    if not (exact_found and sem_found):
        candidates = np.where(
            (exact["t_onset"] >= 0)
            & exact["detected"]
        )[0]

        if len(candidates) == 0:
            raise RuntimeError(
                "No detected fraud entity found in test set."
            )

        fallback_index = int(candidates[0])
        target = exact["entity_ids"][fallback_index]

        print()
        print(
            f"Entity {TARGET_ENTITY_ID} not in test split. "
            f"Using entity {target} instead."
        )

        print_entity(
            exact,
            target,
        )

        print_entity(
            sem,
            target,
        )

    plot_entity_comparison(
        exact,
        sem,
        target,
        OUTPUT_DIR
        / f"entity_{target}_posterior_comparison.png",
    )

    print()
    print("=" * 80)
    print("Saved outputs")
    print("=" * 80)

    print(
        OUTPUT_DIR
        / "joint_exact_entity_diagnostics.csv"
    )

    print(
        OUTPUT_DIR
        / "joint_sampling_entity_diagnostics.csv"
    )

    print(
        OUTPUT_DIR
        / "joint_exact_posterior_data.npz"
    )

    print(
        OUTPUT_DIR
        / "joint_sampling_posterior_data.npz"
    )

    print(
        OUTPUT_DIR
        / "summary.json"
    )

    print(
        OUTPUT_DIR
        / f"entity_{target}_posterior_comparison.png"
    )


if __name__ == "__main__":
    main()