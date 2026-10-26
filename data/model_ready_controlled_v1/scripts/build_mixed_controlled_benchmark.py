from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stratified_split(status: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    split = np.empty(len(status), dtype="<U5")
    for value in np.unique(status):
        indices = np.flatnonzero(status == value)
        indices = rng.permutation(indices)
        train_end = len(indices) * 70 // 100
        val_end = train_end + len(indices) * 10 // 100
        split[indices[:train_end]] = "train"
        split[indices[train_end:val_end]] = "val"
        split[indices[val_end:]] = "test"
    return split


def build(
    output: Path,
    count: int,
    fraud_rate: float,
    censor_rate: float,
    seed: int,
    length: int,
    features: int,
    min_prefix: int,
    delay_max: int,
    geometric_p: float,
) -> dict:
    if count <= 0:
        raise ValueError("count must be positive")
    if not 0 < fraud_rate < 1:
        raise ValueError("fraud_rate must be in (0, 1)")
    if not 0 <= censor_rate < 1:
        raise ValueError("censor_rate must be in [0, 1)")
    if length <= min_prefix + delay_max:
        raise ValueError("length is too short for prefix and delay")
    if not 0 < geometric_p <= 1:
        raise ValueError("geometric_p must be in (0, 1]")

    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    fraud_count = int(round(count * fraud_rate))
    censored_fraud_count = int(round(fraud_count * censor_rate))
    detected_fraud_count = fraud_count - censored_fraud_count
    normal_count = count - fraud_count

    status = np.array(
        ["normal"] * normal_count
        + ["fraud_detected"] * detected_fraud_count
        + ["fraud_censored"] * censored_fraud_count,
        dtype="<U16",
    )
    status = status[rng.permutation(count)]
    true_fraud = status != "normal"
    detected = status == "fraud_detected"

    x = rng.normal(0.0, 1.0, size=(count, length, features)).astype(np.float32)
    lengths = np.full(count, length, dtype=np.int64)
    t_onset = np.full(count, -1, dtype=np.int64)
    t_detect = np.full(count, length - 1, dtype=np.int64)
    behavior_mode = np.full(count, "none", dtype="<U12")
    signal_strength = np.zeros(count, dtype=np.float32)
    delay_steps = np.full(count, -1, dtype=np.int64)

    fraud_indices = np.flatnonzero(true_fraud)
    latest_onset = length - delay_max - 1
    t_onset[fraud_indices] = rng.integers(
        min_prefix, latest_onset + 1, size=len(fraud_indices), dtype=np.int64
    )
    behavior_mode[fraud_indices] = np.where(
        rng.random(len(fraud_indices)) < 0.5, "aligned", "gradual"
    )
    signal_strength[fraud_indices] = rng.uniform(
        0.35, 1.0, size=len(fraud_indices)
    ).astype(np.float32)

    signal = rng.normal(1.0, 0.10, size=features).astype(np.float32)
    entity_scale = rng.uniform(0.85, 1.15, size=count).astype(np.float32)
    for i in fraud_indices:
        remaining = length - t_onset[i]
        if behavior_mode[i] == "aligned":
            multiplier = np.ones((remaining, 1), dtype=np.float32)
        else:
            multiplier = np.linspace(0.15, 1.0, remaining, dtype=np.float32)[:, None]
        x[i, t_onset[i] :, :] += (
            entity_scale[i] * signal_strength[i] * multiplier * signal
        )

    detected_indices = np.flatnonzero(detected)
    sampled_delay = np.clip(
        rng.geometric(geometric_p, size=len(detected_indices)), 1, delay_max
    ).astype(np.int64)
    delay_steps[detected_indices] = sampled_delay
    t_detect[detected_indices] = t_onset[detected_indices] + sampled_delay

    split = stratified_split(status, rng)
    entity_id = np.arange(1, count + 1, dtype=np.int64)
    public_id = np.array([f"mix_v1_{i:05d}" for i in entity_id])

    np.savez_compressed(
        output / "model_input.npz",
        x=x,
        lengths=lengths,
        detected=detected.astype(np.int64),
        t_detect=t_detect,
        t_onset=t_onset,
        entity_id=entity_id,
    )

    index_rows: list[dict] = []
    timeline_rows: list[dict] = []
    for i in range(count):
        index_rows.append(
            {
                "entity_id": public_id[i],
                "model_entity_id": int(entity_id[i]),
                "split": split[i],
                "sequence_length": length,
                "feature_count": features,
                "status_evaluation_only": status[i],
                "true_fraud_evaluation_only": int(true_fraud[i]),
                "detected_observation": int(detected[i]),
                "tf_index_evaluation_only": int(t_onset[i]),
                "tf_source": "controlled_injection" if true_fraud[i] else "not_applicable",
                "td_or_censor_index": int(t_detect[i]),
                "td_source": "controlled_delay" if detected[i] else "observation_horizon",
                "delay_steps": int(delay_steps[i]),
                "behavior_mode": behavior_mode[i],
                "signal_strength": float(signal_strength[i]),
                "sequence_valid": 1,
            }
        )
        for t in range(length):
            row = {
                "entity_id": public_id[i],
                "model_entity_id": int(entity_id[i]),
                "split": split[i],
                "time_index": t,
                "status_evaluation_only": status[i],
                "is_tf_step_evaluation_only": int(true_fraud[i] and t == t_onset[i]),
                "is_td_step_evaluation_only": int(detected[i] and t == t_detect[i]),
            }
            row.update({f"x_{j}": float(x[i, t, j]) for j in range(features)})
            timeline_rows.append(row)

    write_csv(output / "sequence_index.csv", list(index_rows[0]), index_rows)
    write_csv(output / "timeline_evaluation_only.csv", list(timeline_rows[0]), timeline_rows)

    status_counts = {value: int((status == value).sum()) for value in np.unique(status)}
    split_status_counts = {
        part: {value: int(((split == part) & (status == value)).sum()) for value in np.unique(status)}
        for part in ("train", "val", "test")
    }
    checks = {
        "entity_count": count,
        "unique_entity_count": int(len(np.unique(entity_id))),
        "timeline_row_count": int(count * length),
        "array_shape": list(x.shape),
        "status_counts": status_counts,
        "split_status_counts": split_status_counts,
        "true_fraud_prevalence": float(true_fraud.mean()),
        "censoring_rate_among_true_fraud": float((status[true_fraud] == "fraud_censored").mean()),
        "all_finite_features": bool(np.isfinite(x).all()),
        "normal_has_no_tf": bool((t_onset[~true_fraud] == -1).all()),
        "fraud_has_valid_tf": bool((t_onset[true_fraud] >= min_prefix).all()),
        "detected_has_tf_before_td": bool((t_onset[detected] < t_detect[detected]).all()),
        "undetected_uses_horizon": bool((t_detect[~detected] == length - 1).all()),
        "all_td_within_horizon": bool((t_detect < lengths).all()),
        "all_required_checks_pass": False,
    }
    checks["all_required_checks_pass"] = bool(
        checks["unique_entity_count"] == count
        and checks["all_finite_features"]
        and checks["normal_has_no_tf"]
        and checks["fraud_has_valid_tf"]
        and checks["detected_has_tf_before_td"]
        and checks["undetected_uses_horizon"]
        and checks["all_td_within_horizon"]
    )
    (output / "validation_report.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "dataset_name": "SafePlus mixed controlled benchmark v1",
        "provenance": "fully controlled synthetic sequences; no real platform timestamps",
        "seed": seed,
        "entity_count": count,
        "sequence_length": length,
        "feature_count": features,
        "class_protocol": {
            "true_fraud_rate": fraud_rate,
            "censoring_rate_among_true_fraud": censor_rate,
            "status_counts": status_counts,
        },
        "behavior_protocol": {
            "modes": ["aligned", "gradual"],
            "assignment": "prespecified random 50/50 among true fraud entities",
            "signal_strength": "uniform per fraud entity in [0.35, 1.0]",
        },
        "delay_protocol": {
            "distribution": "truncated geometric",
            "p": geometric_p,
            "range_steps": [1, delay_max],
            "undetected_endpoint": length - 1,
        },
        "model_contract": {
            "model_input": "model_input.npz",
            "required_keys": ["x", "lengths", "detected", "t_detect", "t_onset", "entity_id"],
            "feature_leakage_rule": "status, true fraud, tf, td phase and split are excluded from x",
            "t_onset": "evaluation only; -1 for normal entities",
            "detected": "observed event indicator, not identical to true fraud under censoring",
        },
        "scope_limit": "Controlled benchmark for pipeline and sensitivity experiments, not real-world evidence.",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = f"""# SafePlus mixed controlled benchmark v1

This dataset contains {count} independent controlled synthetic account sequences.
It includes normal controls, detected fraud sequences, and censored fraud
sequences. It is designed to exercise discrimination, censoring, delayed
detection, and latent-onset evaluation in one model-ready package.

## Composition

- normal controls: {normal_count}
- detected fraud: {detected_fraud_count}
- censored fraud: {censored_fraud_count}
- sequence shape: {count} x {length} x {features}
- split: stratified entity-level 70/10/20

## Important label distinction

`detected` is the observed event indicator used by the current model contract.
It is not the same as true fraud when a fraud sequence is censored. The true
fraud status and injected onset are stored only in evaluation metadata and are
not included in `x`.

## Provenance and limitation

All features, onsets, delays, and censoring decisions are generated under the
prespecified protocol in `manifest.json`. They are not Wikipedia, Twitter, or
other real-platform observations and must not be reported as manually verified
events. This benchmark supports controlled method checks, not real-world claims.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    hashed = [
        "model_input.npz",
        "sequence_index.csv",
        "timeline_evaluation_only.csv",
        "validation_report.json",
        "manifest.json",
        "README.md",
    ]
    (output / "checksums_sha256.json").write_text(
        json.dumps({name: sha256(output / name) for name in hashed}, indent=2),
        encoding="utf-8",
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--fraud-rate", type=float, default=0.35)
    parser.add_argument("--censor-rate", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--length", type=int, default=24)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--min-prefix", type=int, default=4)
    parser.add_argument("--delay-max", type=int, default=8)
    parser.add_argument("--geometric-p", type=float, default=0.25)
    args = parser.parse_args()
    report = build(
        args.output,
        args.count,
        args.fraud_rate,
        args.censor_rate,
        args.seed,
        args.length,
        args.features,
        args.min_prefix,
        args.delay_max,
        args.geometric_p,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
