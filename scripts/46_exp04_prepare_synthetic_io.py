"""Convert the existing SafePlus synthetic dataset to separate I/O Contract v1.0 files.

THIS IS FOR SYNTHETIC COMPATIBILITY ONLY; NEVER USE IT FOR REAL DATA.
No model fitting is performed. Labels are physically separated from model inputs.

Example:
  python scripts/46_exp04_prepare_synthetic_io.py --source data/processed/synthetic_same_delay_consistent.npz --out runs/io_contract_v1/synthetic_same_delay
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def convert(source: Path, out: Path, seed: int = 0, force: bool = False):
    out.mkdir(parents=True, exist_ok=True)
    expected = ["inputs.npz", "split.csv", "train_labels.csv", "gold_onset.csv",
                "metadata.json", "delay_prior.json"]
    if not force and any((out / name).exists() for name in expected):
        raise FileExistsError("Output already exists. Choose a new --out or explicitly use --force")

    with np.load(source, allow_pickle=False) as data:
        required = ["x", "detected", "t_detect", "t_onset"]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"Synthetic source missing keys: {missing}")
        x = data["x"].astype(np.float64, copy=True)
        detected = data["detected"].astype(np.int64)
        td = data["t_detect"].astype(np.int64)
        tf = data["t_onset"].astype(np.int64)
        if "lengths" in data:
            lengths = data["lengths"].astype(np.int64)
        else:
            lengths = np.full(len(x), x.shape[1], dtype=np.int64)
        if "entity_id" in data:
            eid = data["entity_id"]
        else:
            eid = np.arange(len(x), dtype=np.int64)
        if "is_fraud" in data:
            fraud = data["is_fraud"].astype(np.int64)
            label_source = "source_is_fraud"
        else:
            # Explicit, constrained SYNTHETIC-ONLY historical recovery.
            # Never apply this to a real dataset. Gold tf stays outside inputs.npz.
            fraud = (tf >= 0).astype(np.int64)
            label_source = "SYNTHETIC_ONLY_fraud_derived_from_injected_tf"
            print("WARNING: SYNTHETIC ONLY: deriving entity fraud labels from injected tf")

    if x.ndim != 3:
        raise ValueError("x must be [N,L,D]")
    n, l, d = x.shape
    if not (len(detected) == len(td) == len(tf) == len(eid) == len(lengths) == n):
        raise ValueError("Source arrays have inconsistent N")
    if not np.all(lengths == l):
        raise ValueError("Current frozen EM not length-masked: legacy conversion accepts fixed-length only")
    if not np.all(np.isin(detected, [0, 1])):
        raise ValueError("invalid detected labels")
    if not np.all(np.isin(fraud, [0, 1])):
        raise ValueError("invalid fraud labels")

    # For nondetected, sentinel is the observation horizon, never a literal td.
    td = td.copy()
    td[detected == 0] = lengths[detected == 0]
    if np.any((detected == 1) & ((td < 1) | (td >= lengths))):
        raise ValueError("Detected events must have 1 <= t_detect < lengths")

    # Prevent after-detection features from being present in model-visible input.
    for i in np.flatnonzero(detected == 1):
        x[i, td[i] + 1:, :] = 0.0
    if not np.all(np.isfinite(x)):
        raise ValueError("NaN/Inf in synthetic features")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_train = int(0.7 * n)
    n_val = int(0.1 * n)
    train, val, test = indices[:n_train], indices[n_train:n_train + n_val], indices[n_train + n_val:]
    splits = [None] * n
    for label, subset in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        for i in subset:
            splits[int(i)] = label

    # Save model-visible x and observed td only; no label or true onset.
    np.savez_compressed(out / "inputs.npz",
                        entity_id=eid, x=x, lengths=lengths,
                        detected=detected, t_detect=td)
    write_csv(out / "split.csv", ["entity_id", "split"],
              [[str(eid[i]), splits[i]] for i in range(n)])
    write_csv(out / "train_labels.csv", ["entity_id", "is_fraud"],
              [[str(eid[i]), int(fraud[i])] for i in train])
    write_csv(out / "gold_onset.csv",
              ["entity_id", "is_fraud", "onset_label_type", "t_onset",
               "t_onset_min", "t_onset_max", "source_event_id", "evidence_ids",
               "annotation_version"],
              [[str(eid[i]), int(fraud[i]), "EXACT" if fraud[i] == 1 else "NOT_FRAUD",
                int(tf[i]) if fraud[i] == 1 else -1,
                "", "", "synthetic_generator", "synthetic_injected_truth", "synthetic_v1"]
               for i in range(n)])

    meta = {
        "schema_version": "1.0",
        "dataset_id": "synthetic_same_delay_consistent",
        "dataset_kind": "synthetic",
        "time_unit": "synthetic_step",
        "time_origin": "index_0_is_first_observed_step",
        "time_index_base": 0,
        "feature_names": [f"feature_{j}" for j in range(d)],
        "t_detect_definition": "simulated_operational_detection_time",
        "detected_bin_features": "pre_detection_only",
        "post_detection_policy": "masked_zero",
        "split_method": f"entity_random_seed_{seed}_70_10_20",
        "preprocessing": "none; post-detection covariates zero-masked",
        "data_source": str(source),
        "license": "project_generated_synthetic",
        "annotation_guideline_version": "synthetic_injected_exact",
        "fraud_label_source": label_source,
        "observed_horizon": l,
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    prior = {
        "mode": "synthetic_known",
        "definition": "Delta=t_detect-t_onset, positive integer; simulator delay rule",
        "source": "known_project_synthetic_generator; do not estimate from test gold",
        "parameters": {"generator_delay_family": "max(1,int(Exponential(scale=4)))", "scale": 4.0},
        "version": "exp04_original_synthetic_prior",
    }
    (out / "delay_prior.json").write_text(json.dumps(prior, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"PREPARED: {out}")
    print(f"N={n} L={l} D={d}; TRAIN={len(train)} VAL={len(val)} TEST={len(test)}")
    print("inputs.npz has x/lengths/detected/t_detect/entity_id ONLY")
    print("Gold tf and TRAIN fraud labels have been stored in separate files")
    return out


def main():
    parser = argparse.ArgumentParser(description="SYNTHETIC ONLY: prepare I/O v1 bundle")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    convert(args.source, args.out, args.seed, args.force)


if __name__ == "__main__":
    main()
