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


def build_batch(
    output_dir: Path,
    batch_id: str,
    entity_prefix: str,
    entity_id_offset: int,
    count: int,
    seed: int,
    length: int,
    features: int,
    delay_steps: int,
    delay_mode: str,
    delay_max: int,
    geometric_p: float,
    behavior_mode: str,
    signal_strength: float,
    min_prefix: int,
) -> dict:
    if count <= 0:
        raise ValueError("count must be positive")
    if min_prefix < 2:
        raise ValueError("min_prefix must be at least 2")
    if delay_mode not in {"fixed", "truncated_geometric"}:
        raise ValueError("delay_mode must be fixed or truncated_geometric")
    if behavior_mode not in {"aligned_step", "gradual_ramp"}:
        raise ValueError("behavior_mode must be aligned_step or gradual_ramp")
    if not 0 < geometric_p <= 1:
        raise ValueError("geometric_p must be in (0, 1]")
    if signal_strength <= 0:
        raise ValueError("signal_strength must be positive")
    policy_max_delay = delay_steps if delay_mode == "fixed" else delay_max
    if length <= min_prefix + policy_max_delay:
        raise ValueError("length is too short for the requested prefix and delay")

    output_dir.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(count, length, features)).astype(np.float32)
    lengths = np.full(count, length, dtype=np.int64)
    latest_onset = length - policy_max_delay - 1
    t_onset = rng.integers(min_prefix, latest_onset + 1, size=count, dtype=np.int64)
    if delay_mode == "fixed":
        entity_delays = np.full(count, delay_steps, dtype=np.int64)
        delay_policy_id = f"fixed_delay_{delay_steps}_v1"
        delay_policy_description = f"fixed delay of {delay_steps} steps"
    else:
        entity_delays = np.clip(
            rng.geometric(geometric_p, size=count), 1, delay_max
        ).astype(np.int64)
        delay_policy_id = f"truncated_geometric_p{geometric_p:g}_max{delay_max}_v1"
        delay_policy_description = (
            f"geometric(p={geometric_p:g}) delay clipped to 1..{delay_max} steps"
        )
    t_detect = t_onset + entity_delays
    detected = np.ones(count, dtype=np.int64)
    public_entity_id = np.array(
        [f"{entity_prefix}_{i:04d}" for i in range(1, count + 1)],
        dtype=f"<U{len(entity_prefix) + 5}",
    )
    model_entity_id = np.arange(
        1 + entity_id_offset, count + 1 + entity_id_offset, dtype=np.int64
    )

    signal = (
        signal_strength * rng.normal(1.0, 0.10, size=features)
    ).astype(np.float32)
    entity_scale = rng.uniform(0.85, 1.15, size=count).astype(np.float32)
    for i in range(count):
        if behavior_mode == "aligned_step":
            x[i, t_onset[i] :, :] += entity_scale[i] * signal
        else:
            ramp = np.linspace(
                0.15, 1.0, length - t_onset[i], dtype=np.float32
            )[:, None]
            x[i, t_onset[i] :, :] += entity_scale[i] * ramp * signal

    behavior_description = (
        "additive feature shift begins at full strength exactly at tf"
        if behavior_mode == "aligned_step"
        else "additive feature shift begins weakly at tf and increases gradually"
    )
    alignment_condition = (
        "change_starts_exactly_at_tf"
        if behavior_mode == "aligned_step"
        else "gradual_signal_begins_at_tf"
    )

    order = rng.permutation(count)
    split = np.empty(count, dtype="<U5")
    split[order[: int(count * 0.70)]] = "train"
    split[order[int(count * 0.70) : int(count * 0.80)]] = "val"
    split[order[int(count * 0.80) :]] = "test"

    npz_path = output_dir / "model_input.npz"
    np.savez_compressed(
        npz_path,
        x=x,
        lengths=lengths,
        detected=detected,
        t_detect=t_detect,
        t_onset=t_onset,
        entity_id=model_entity_id,
    )

    index_rows: list[dict] = []
    timeline_rows: list[dict] = []
    for i in range(count):
        index_rows.append(
            {
                "batch_id": batch_id,
                "entity_id": public_entity_id[i],
                "model_entity_id": int(model_entity_id[i]),
                "split": split[i],
                "sequence_length": int(lengths[i]),
                "feature_count": features,
                "normal_prefix_length": int(t_onset[i]),
                "tf_index": int(t_onset[i]),
                "tf_source": "controlled_injection",
                "tf_type": "known_synthetic_action_time",
                "td_index": int(t_detect[i]),
                "td_source": "controlled_delay",
                "delay_steps": int(entity_delays[i]),
                "delay_policy_id": delay_policy_id,
                "censored": 0,
                "behavior_change_start_index": int(t_onset[i]),
                "alignment_condition": alignment_condition,
                "sequence_valid": 1,
                "validation_notes": "normal prefix present; tf<td; td within horizon",
            }
        )
        for t in range(length):
            phase = "normal_prefix" if t < t_onset[i] else "post_tf"
            if t == t_detect[i]:
                phase = "detection_step"
            row = {
                "entity_id": public_entity_id[i],
                "model_entity_id": int(model_entity_id[i]),
                "split": split[i],
                "time_index": t,
                "phase_evaluation_only": phase,
                "is_tf_step_evaluation_only": int(t == t_onset[i]),
                "is_td_step_evaluation_only": int(t == t_detect[i]),
            }
            row.update({f"x_{j}": float(x[i, t, j]) for j in range(features)})
            timeline_rows.append(row)

    index_path = output_dir / "sequence_index.csv"
    write_csv(index_path, list(index_rows[0].keys()), index_rows)
    timeline_path = output_dir / "timeline_evaluation_only.csv"
    write_csv(timeline_path, list(timeline_rows[0].keys()), timeline_rows)

    checks = {
        "entity_count": int(count),
        "unique_entity_count": int(len(np.unique(model_entity_id))),
        "timeline_row_count": int(count * length),
        "all_finite_features": bool(np.isfinite(x).all()),
        "all_have_normal_prefix": bool((t_onset >= min_prefix).all()),
        "all_tf_before_td": bool((t_onset < t_detect).all()),
        "all_td_within_horizon": bool((t_detect < lengths).all()),
        "all_delay_within_policy": bool(
            ((t_detect - t_onset) >= 1).all()
            and ((t_detect - t_onset) <= policy_max_delay).all()
            and (
                delay_mode != "fixed"
                or ((t_detect - t_onset) == delay_steps).all()
            )
        ),
        "all_detected": bool((detected == 1).all()),
        "split_counts": {
            "train": int((split == "train").sum()),
            "val": int((split == "val").sum()),
            "test": int((split == "test").sum()),
        },
        "tf_min": int(t_onset.min()),
        "tf_max": int(t_onset.max()),
        "td_min": int(t_detect.min()),
        "td_max": int(t_detect.max()),
        "delay_min": int(entity_delays.min()),
        "delay_max": int(entity_delays.max()),
        "delay_mean": float(entity_delays.mean()),
        "pre_tf_feature_mean": float(
            np.concatenate([x[i, : t_onset[i], :].reshape(-1) for i in range(count)]).mean()
        ),
        "post_tf_feature_mean": float(
            np.concatenate([x[i, t_onset[i] :, :].reshape(-1) for i in range(count)]).mean()
        ),
    }
    checks["all_required_checks_pass"] = bool(
        checks["unique_entity_count"] == count
        and checks["all_finite_features"]
        and checks["all_have_normal_prefix"]
        and checks["all_tf_before_td"]
        and checks["all_td_within_horizon"]
        and checks["all_delay_within_policy"]
        and checks["all_detected"]
    )

    validation_path = output_dir / "validation_report.json"
    validation_path.write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "dataset_name": "SafePlus controlled onset " + batch_id,
        "batch_id": batch_id,
        "created_for": "latent-onset and delayed-detection controlled experiment",
        "generation": {
            "generator": "generate_model_ready_batch.py",
            "seed": seed,
            "entity_count": count,
            "sequence_length": length,
            "feature_count": features,
            "normal_prefix_min_steps": min_prefix,
            "behavior_mode": behavior_mode,
            "behavior_change": behavior_description,
            "signal_strength": signal_strength,
            "tf_source": "controlled_injection; not a real-world timestamp",
            "td_source": "controlled delay; not a platform observation",
            "delay_policy": {
                "id": delay_policy_id,
                "mode": delay_mode,
                "description": delay_policy_description,
                "fixed_steps": delay_steps if delay_mode == "fixed" else None,
                "maximum_steps": policy_max_delay,
                "geometric_p": geometric_p if delay_mode == "truncated_geometric" else None,
            },
            "censoring_rate": 0.0,
        },
        "model_contract": {
            "required_npz_keys": [
                "x",
                "lengths",
                "detected",
                "t_detect",
                "t_onset",
                "entity_id",
            ],
            "t_onset_use": "evaluation only; must not enter model features",
            "timeline_phase_columns": "evaluation only; must not enter model features",
        },
        "scope_limit": (
            "This positive-only batch is suitable for controlled onset-recovery checks. "
            "It is not sufficient by itself for AUPRC, AUROC, precision, recall, or F1; "
            "those require negative or censored controls."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = f"""# SafePlus controlled batch

This batch contains {count} positive sequences. Every sequence has:

- a normal prefix of at least {min_prefix} steps;
- a known controlled onset t_onset;
- an experimental detection endpoint generated after t_onset;
- chronological length {length} with {features} model features per step.

## Label provenance

t_onset is a controlled injected action time. t_detect is generated by the
prespecified policy {delay_policy_id}: {delay_policy_description}. Neither value is a real platform
timestamp. The source is recorded in sequence_index.csv and manifest.json.

## Files

- model_input.npz: canonical SafePlus model arrays.
- sequence_index.csv: one row per sequence with tf/td provenance and checks.
- timeline_evaluation_only.csv: step-level features and evaluation-only phases.
- validation_report.json: deterministic validation results.
- manifest.json: protocol and model-contract metadata.

## Purpose

This condition uses behavior mode {behavior_mode}: {behavior_description}. The
delay mechanism is declared above. Later 100-sequence batches should independently vary
change alignment, delay distribution, signal strength, and censoring. Samples
must not be selected according to model scores.

## Limitation

The batch contains no negative controls. Use it for controlled onset recovery,
not as a standalone discrimination benchmark.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    hashes = {
        p.name: sha256(p)
        for p in [
            npz_path,
            index_path,
            timeline_path,
            validation_path,
            manifest_path,
        ]
    }
    (output_dir / "checksums_sha256.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-id", default="batch_001_aligned_fixed_delay_100")
    parser.add_argument("--entity-prefix", default="ctrl_b001")
    parser.add_argument("--entity-id-offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--length", type=int, default=24)
    parser.add_argument("--features", type=int, default=8)
    parser.add_argument("--delay-steps", type=int, default=4)
    parser.add_argument(
        "--delay-mode",
        choices=["fixed", "truncated_geometric"],
        default="fixed",
    )
    parser.add_argument("--delay-max", type=int, default=8)
    parser.add_argument("--geometric-p", type=float, default=0.25)
    parser.add_argument(
        "--behavior-mode",
        choices=["aligned_step", "gradual_ramp"],
        default="aligned_step",
    )
    parser.add_argument("--signal-strength", type=float, default=1.0)
    parser.add_argument("--min-prefix", type=int, default=4)
    args = parser.parse_args()
    checks = build_batch(
        output_dir=args.output,
        batch_id=args.batch_id,
        entity_prefix=args.entity_prefix,
        entity_id_offset=args.entity_id_offset,
        count=args.count,
        seed=args.seed,
        length=args.length,
        features=args.features,
        delay_steps=args.delay_steps,
        delay_mode=args.delay_mode,
        delay_max=args.delay_max,
        geometric_p=args.geometric_p,
        behavior_mode=args.behavior_mode,
        signal_strength=args.signal_strength,
        min_prefix=args.min_prefix,
    )
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
