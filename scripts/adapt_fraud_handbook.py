from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "log1p_tx_count",
    "log1p_amount_sum",
    "log1p_amount_mean",
    "log1p_amount_max",
    "log1p_amount_std",
    "log1p_unique_terminals",
    "unique_terminal_share",
    "mean_hour_sin",
    "mean_hour_cos",
]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _split_groups(groups: np.ndarray, labels: np.ndarray, ratios, seed: int):
    rng = np.random.default_rng(seed)
    result = [[], [], []]
    for label in np.unique(labels):
        selected = groups[labels == label].copy()
        rng.shuffle(selected)
        cut1 = int(len(selected) * ratios[0])
        cut2 = cut1 + int(len(selected) * ratios[1])
        for destination, values in zip(
            result, (selected[:cut1], selected[cut1:cut2], selected[cut2:])
        ):
            destination.extend(values.tolist())
    return [set(values) for values in result]


def build_arrays(
    source: Path,
    window_days: int = 30,
    delay_days: int = 7,
    censor_rate: float = 0.2,
    seed: int = 20261005,
):
    files = sorted(source.glob("*.pkl"))
    if not files:
        raise ValueError("no daily pickle files found")
    frame = pd.concat([pd.read_pickle(path) for path in files], ignore_index=True)
    required = {
        "TX_DATETIME",
        "CUSTOMER_ID",
        "TERMINAL_ID",
        "TX_AMOUNT",
        "TX_TIME_DAYS",
        "TX_FRAUD",
        "TX_FRAUD_SCENARIO",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing handbook columns: {sorted(missing)}")
    frame["window"] = (frame["TX_TIME_DAYS"] // window_days).astype(np.int16)
    frame["day"] = (frame["TX_TIME_DAYS"] % window_days).astype(np.int16)
    hour = frame["TX_DATETIME"].dt.hour + frame["TX_DATETIME"].dt.minute / 60
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    grouped = (
        frame.groupby(["CUSTOMER_ID", "window", "day"], sort=True)
        .agg(
            tx_count=("TX_AMOUNT", "size"),
            amount_sum=("TX_AMOUNT", "sum"),
            amount_mean=("TX_AMOUNT", "mean"),
            amount_max=("TX_AMOUNT", "max"),
            amount_std=("TX_AMOUNT", "std"),
            unique_terminals=("TERMINAL_ID", "nunique"),
            mean_hour_sin=("hour_sin", "mean"),
            mean_hour_cos=("hour_cos", "mean"),
        )
        .reset_index()
    )
    grouped["amount_std"] = grouped["amount_std"].fillna(0)
    grouped["unique_terminal_share"] = grouped["unique_terminals"] / grouped["tx_count"]

    windows = (
        frame[["CUSTOMER_ID", "window"]]
        .drop_duplicates()
        .sort_values(["CUSTOMER_ID", "window"])
        .reset_index(drop=True)
    )
    window_index = pd.MultiIndex.from_frame(windows)
    lookup = pd.Series(np.arange(len(windows)), index=window_index)
    active_index = pd.MultiIndex.from_frame(grouped[["CUSTOMER_ID", "window"]])
    rows = lookup.loc[active_index].to_numpy(dtype=np.int64)
    days = grouped["day"].to_numpy(dtype=np.int64)
    x = np.zeros((len(windows), window_days, len(FEATURE_NAMES)), dtype=np.float32)
    log_columns = [
        "tx_count",
        "amount_sum",
        "amount_mean",
        "amount_max",
        "amount_std",
        "unique_terminals",
    ]
    x[rows, days, :6] = np.log1p(grouped[log_columns].to_numpy(dtype=np.float32))
    x[rows, days, 6:] = grouped[
        ["unique_terminal_share", "mean_hour_sin", "mean_hour_cos"]
    ].to_numpy(dtype=np.float32)

    onset_arrays = {}
    for scenario in (0, 1, 2, 3):
        fraud = frame["TX_FRAUD"].eq(1)
        if scenario:
            fraud &= frame["TX_FRAUD_SCENARIO"].eq(scenario)
        lookup_onset = frame.loc[fraud].groupby(["CUSTOMER_ID", "window"])["day"].min()
        name = "t_onset" if scenario == 0 else f"t_onset_scenario_{scenario}"
        onset_arrays[name] = lookup_onset.reindex(window_index, fill_value=-1).to_numpy(
            dtype=np.int64
        )

    t_onset = onset_arrays["t_onset"]
    candidate_detect = t_onset + delay_days
    detectable = (t_onset >= 0) & (candidate_detect < window_days)
    rng = np.random.default_rng(seed)
    simulated_censored = np.zeros(len(windows), dtype=bool)
    candidates = np.flatnonzero(detectable)
    simulated_censored[candidates] = rng.random(len(candidates)) < censor_rate
    detected = detectable & ~simulated_censored
    t_detect = np.where(detected, candidate_detect, -1).astype(np.int64)

    customers = np.asarray(sorted(frame["CUSTOMER_ID"].unique()))
    customer_to_group = {customer: index for index, customer in enumerate(customers)}
    group_id = windows["CUSTOMER_ID"].map(customer_to_group).to_numpy(dtype=np.int64)
    arrays = {
        "x": x,
        "lengths": np.full(len(windows), window_days, dtype=np.int64),
        "detected": detected,
        "t_detect": t_detect,
        "entity_id": np.arange(len(windows), dtype=np.int64),
        "group_id": group_id,
        **onset_arrays,
    }
    positive_customers = set(frame.loc[frame["TX_FRAUD"] == 1, "CUSTOMER_ID"])
    customer_labels = np.asarray(
        [customer in positive_customers for customer in customers], dtype=np.int8
    )
    metadata = {
        "daily_files": len(files),
        "transactions": len(frame),
        "customers": len(customers),
        "terminals": int(frame["TERMINAL_ID"].nunique()),
        "windows": len(windows),
        "positive_windows": int((t_onset >= 0).sum()),
        "positive_window_rate": float((t_onset >= 0).mean()),
        "observed_detections": int(detected.sum()),
        "simulated_censored_positive_windows": int(simulated_censored.sum()),
        "naturally_censored_positive_windows": int(((t_onset >= 0) & ~detectable).sum()),
        "feature_names": FEATURE_NAMES,
        "scenario_positive_windows": {
            str(scenario): int((onset_arrays[f"t_onset_scenario_{scenario}"] >= 0).sum())
            for scenario in (1, 2, 3)
        },
    }
    return arrays, metadata, customers, customer_labels, files


def adapt_fraud_handbook(
    source: Path,
    output: Path,
    window_days: int = 30,
    delay_days: int = 7,
    censor_rate: float = 0.2,
    seed: int = 20261005,
    ratios=(0.7, 0.1, 0.2),
):
    arrays, metadata, customers, labels, files = build_arrays(
        source, window_days, delay_days, censor_rate, seed
    )
    customer_parts = _split_groups(customers, labels, ratios, seed)
    group_customers = {index: customer for index, customer in enumerate(customers)}
    output.mkdir(parents=True, exist_ok=True)
    split_stats = {}
    for name, selected_customers in zip(("train", "valid", "test"), customer_parts):
        selected_groups = {
            index for index, customer in group_customers.items() if customer in selected_customers
        }
        indices = np.asarray(
            [index for index, group in enumerate(arrays["group_id"]) if group in selected_groups],
            dtype=np.int64,
        )
        np.savez_compressed(
            output / f"{name}.npz", **{key: value[indices] for key, value in arrays.items()}
        )
        onset = arrays["t_onset"][indices]
        split_stats[name] = {
            "customers": len(selected_customers),
            "windows": len(indices),
            "positive_windows": int((onset >= 0).sum()),
            "positive_window_rate": float((onset >= 0).mean()),
            "observed_detections": int(arrays["detected"][indices].sum()),
        }

    download_manifest = source / "download_manifest.json"
    manifest = {
        "dataset": "Fraud Detection Handbook simulated transactions",
        "source": "https://github.com/Fraud-Detection-Handbook/simulated-data-raw",
        "source_commit": "6e67dbd0a3bfe0d7ec33abc4bce5f37cd4ff0d6a",
        "source_file_count": len(files),
        "source_total_bytes": sum(path.stat().st_size for path in files),
        "source_manifest_sha256": digest(download_manifest) if download_manifest.exists() else None,
        "construction": {
            "entity": "non-overlapping customer-level 30-day episode window",
            "window_days": window_days,
            "t_onset": "first fraud-labelled day in a window; evaluation only",
            "scenario_onsets": "first labelled day for each of scenarios 1, 2 and 3",
            "t_detect": f"t_onset + {delay_days} days when observed, otherwise -1",
            "simulated_positive_censor_rate": censor_rate,
            "split_unit": "customer",
            "split": list(ratios),
            "seed": seed,
        },
        **metadata,
        "splits": split_stats,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Adapt Fraud Detection Handbook daily files.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--delay-days", type=int, default=7)
    parser.add_argument("--censor-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    manifest = adapt_fraud_handbook(
        args.source,
        args.output,
        args.window_days,
        args.delay_days,
        args.censor_rate,
        args.seed,
    )
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
