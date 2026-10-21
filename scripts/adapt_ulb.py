from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PCA_FEATURES = [f"V{index}" for index in range(1, 29)]
FEATURE_NAMES = ["time_robust", *PCA_FEATURES, "log1p_amount_robust"]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _median_iqr(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = float(q75 - q25)
    return median, scale if scale > 0 else 1.0


def adapt_ulb(source: Path, output: Path, train_ratio: float = 0.7, valid_ratio: float = 0.1):
    frame = pd.read_csv(source)
    required = {"Time", "Amount", "Class", *PCA_FEATURES}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing ULB columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("ULB input contains missing values")
    raw_rows = len(frame)
    raw_fraud = int(frame["Class"].sum())
    duplicate_rows = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().sort_values("Time", kind="stable").reset_index(drop=True)
    frame["clean_row_id"] = np.arange(len(frame), dtype=np.int64)

    cut1_position = int(len(frame) * train_ratio)
    cut2_position = int(len(frame) * (train_ratio + valid_ratio))
    cut1_time = float(frame.iloc[cut1_position]["Time"])
    cut2_time = float(frame.iloc[cut2_position]["Time"])
    split_masks = {
        "train": frame["Time"] < cut1_time,
        "valid": frame["Time"].ge(cut1_time) & frame["Time"].lt(cut2_time),
        "test": frame["Time"].ge(cut2_time),
    }
    if any(not mask.any() for mask in split_masks.values()):
        raise ValueError("chronological split produced an empty partition")

    train = frame.loc[split_masks["train"]]
    time_center, time_scale = _median_iqr(train["Time"].to_numpy(np.float64))
    log_amount = np.log1p(frame["Amount"].to_numpy(np.float64))
    amount_center, amount_scale = _median_iqr(log_amount[split_masks["train"].to_numpy()])

    x = np.empty((len(frame), len(FEATURE_NAMES)), dtype=np.float32)
    x[:, 0] = (frame["Time"].to_numpy(np.float64) - time_center) / time_scale
    x[:, 1:29] = frame[PCA_FEATURES].to_numpy(np.float32)
    x[:, 29] = (log_amount - amount_center) / amount_scale
    y = frame["Class"].to_numpy(np.int8)
    output.mkdir(parents=True, exist_ok=True)
    split_stats = {}
    for name, mask in split_masks.items():
        indices = np.flatnonzero(mask.to_numpy())
        np.savez_compressed(
            output / f"{name}.npz",
            x=x[indices],
            y=y[indices],
            time=frame.iloc[indices]["Time"].to_numpy(np.float64),
            clean_row_id=frame.iloc[indices]["clean_row_id"].to_numpy(np.int64),
        )
        split_stats[name] = {
            "transactions": len(indices),
            "fraud": int(y[indices].sum()),
            "fraud_rate": float(y[indices].mean()),
            "time_min": float(frame.iloc[indices]["Time"].min()),
            "time_max": float(frame.iloc[indices]["Time"].max()),
        }

    manifest = {
        "dataset": "ULB Credit Card Fraud Detection",
        "source": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        "license": "DbCL-1.0",
        "source_file": {"bytes": source.stat().st_size, "sha256": digest(source)},
        "raw": {
            "transactions": raw_rows,
            "fraud": raw_fraud,
            "exact_duplicate_rows_removed": duplicate_rows,
        },
        "clean": {
            "transactions": len(frame),
            "fraud": int(y.sum()),
            "fraud_rate": float(y.mean()),
            "time_min": float(frame["Time"].min()),
            "time_max": float(frame["Time"].max()),
        },
        "construction": {
            "task": "transaction-level binary classification",
            "split": "chronological 70/10/20 target with equal timestamps kept together",
            "features": FEATURE_NAMES,
            "scaler_fit": "training partition only",
            "time_center": time_center,
            "time_scale_iqr": time_scale,
            "log1p_amount_center": amount_center,
            "log1p_amount_scale_iqr": amount_scale,
            "customer_sequence_available": False,
            "onset_available": False,
        },
        "splits": split_stats,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Package ULB as chronological transactions.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = adapt_ulb(args.source, args.output)
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
