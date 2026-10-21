from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_FEATURES = [
    "log1p_tx_count",
    "log1p_amount_sum",
    "log1p_amount_mean",
    "log1p_amount_max",
    "log1p_unique_merchants",
    "log1p_distance_mean_km",
    "log1p_distance_max_km",
]


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    value = np.sin(delta_lat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(value))


def _stratified_cards(cards: np.ndarray, labels: np.ndarray, ratios, seed: int):
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must sum to one")
    rng = np.random.default_rng(seed)
    result = [[], [], []]
    for label in np.unique(labels):
        selected = cards[labels == label].copy()
        rng.shuffle(selected)
        cut1 = int(len(selected) * ratios[0])
        cut2 = cut1 + int(len(selected) * ratios[1])
        for destination, values in zip(
            result, (selected[:cut1], selected[cut1:cut2], selected[cut2:])
        ):
            destination.extend(values.tolist())
    return [set(values) for values in result]


def build_arrays(
    train_csv: Path,
    test_csv: Path,
    window_days: int = 30,
    delay_days: int = 2,
    censor_rate: float = 0.2,
    seed: int = 20261005,
):
    if window_days < 2:
        raise ValueError("window_days must be at least two")
    if delay_days < 0:
        raise ValueError("delay_days must be non-negative")
    if not 0 <= censor_rate <= 1:
        raise ValueError("censor_rate must be between zero and one")

    columns = [
        "trans_date_trans_time",
        "cc_num",
        "merchant",
        "category",
        "amt",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "is_fraud",
    ]
    frames = [
        pd.read_csv(path, usecols=columns, dtype={"cc_num": "string"})
        for path in (train_csv, test_csv)
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(
        frame["trans_date_trans_time"], format="%Y-%m-%d %H:%M:%S", errors="raise"
    )
    frame["is_fraud"] = pd.to_numeric(frame["is_fraud"], errors="raise").astype(np.int8)
    if not set(frame["is_fraud"].unique()).issubset({0, 1}):
        raise ValueError("is_fraud must be binary")

    start = frame["timestamp"].min().floor("D")
    elapsed_days = (frame["timestamp"] - start).dt.days
    frame["window"] = (elapsed_days // window_days).astype(np.int16)
    frame["day"] = (elapsed_days % window_days).astype(np.int16)
    frame["distance_km"] = _haversine_km(
        frame["lat"].to_numpy(),
        frame["long"].to_numpy(),
        frame["merch_lat"].to_numpy(),
        frame["merch_long"].to_numpy(),
    )

    categories = sorted(frame["category"].unique())
    grouped = frame.groupby(["cc_num", "window", "day"], sort=True).agg(
        tx_count=("amt", "size"),
        amount_sum=("amt", "sum"),
        amount_mean=("amt", "mean"),
        amount_max=("amt", "max"),
        unique_merchants=("merchant", "nunique"),
        distance_mean_km=("distance_km", "mean"),
        distance_max_km=("distance_km", "max"),
    )
    category_counts = pd.crosstab(
        [frame["cc_num"], frame["window"], frame["day"]], frame["category"]
    ).reindex(columns=categories, fill_value=0)
    category_shares = category_counts.div(category_counts.sum(axis=1), axis=0)
    active = grouped.join(category_shares, how="left").reset_index()

    windows = (
        frame[["cc_num", "window"]]
        .drop_duplicates()
        .sort_values(["cc_num", "window"])
        .reset_index(drop=True)
    )
    window_key = pd.MultiIndex.from_frame(windows[["cc_num", "window"]])
    key_to_index = pd.Series(np.arange(len(windows)), index=window_key)
    active_key = pd.MultiIndex.from_frame(active[["cc_num", "window"]])
    row_index = key_to_index.loc[active_key].to_numpy(dtype=np.int64)
    day_index = active["day"].to_numpy(dtype=np.int64)

    x = np.zeros(
        (len(windows), window_days, len(BASE_FEATURES) + len(categories)), dtype=np.float32
    )
    numeric_columns = [
        "tx_count",
        "amount_sum",
        "amount_mean",
        "amount_max",
        "unique_merchants",
        "distance_mean_km",
        "distance_max_km",
    ]
    x[row_index, day_index, : len(BASE_FEATURES)] = np.log1p(
        active[numeric_columns].to_numpy(dtype=np.float32)
    )
    x[row_index, day_index, len(BASE_FEATURES) :] = active[categories].to_numpy(dtype=np.float32)

    fraud_rows = frame.loc[frame["is_fraud"] == 1]
    onset_lookup = fraud_rows.groupby(["cc_num", "window"])["day"].min()
    t_onset = onset_lookup.reindex(window_key, fill_value=-1).to_numpy(dtype=np.int64)
    candidate_detect = t_onset + delay_days
    detectable = (t_onset >= 0) & (candidate_detect < window_days)
    rng = np.random.default_rng(seed)
    simulated_censored = np.zeros(len(windows), dtype=bool)
    positive_indices = np.flatnonzero(detectable)
    simulated_censored[positive_indices] = rng.random(len(positive_indices)) < censor_rate
    detected = detectable & ~simulated_censored
    t_detect = np.where(detected, candidate_detect, -1).astype(np.int64)

    cards = np.asarray(sorted(frame["cc_num"].unique()))
    card_to_group = {card: index for index, card in enumerate(cards)}
    group_id = windows["cc_num"].map(card_to_group).to_numpy(dtype=np.int64)
    arrays = {
        "x": x,
        "lengths": np.full(len(windows), window_days, dtype=np.int64),
        "detected": detected,
        "t_detect": t_detect,
        "t_onset": t_onset,
        "entity_id": np.arange(len(windows), dtype=np.int64),
        "group_id": group_id,
    }
    positive_cards = set(fraud_rows["cc_num"].unique())
    card_labels = np.asarray([card in positive_cards for card in cards], dtype=np.int8)
    metadata = {
        "transactions": len(frame),
        "cards": len(cards),
        "windows": len(windows),
        "positive_windows": int((t_onset >= 0).sum()),
        "positive_window_rate": float((t_onset >= 0).mean()),
        "observed_detections": int(detected.sum()),
        "simulated_censored_positive_windows": int(simulated_censored.sum()),
        "naturally_censored_positive_windows": int(((t_onset >= 0) & ~detectable).sum()),
        "feature_names": BASE_FEATURES + [f"category_share:{name}" for name in categories],
        "categories": categories,
        "start": str(start),
        "end": str(frame["timestamp"].max()),
    }
    return arrays, metadata, cards, card_labels


def adapt_sparkov(
    train_csv: Path,
    test_csv: Path,
    output: Path,
    window_days: int = 30,
    delay_days: int = 2,
    censor_rate: float = 0.2,
    seed: int = 20261005,
    ratios=(0.7, 0.1, 0.2),
):
    arrays, metadata, cards, card_labels = build_arrays(
        train_csv, test_csv, window_days, delay_days, censor_rate, seed
    )
    card_parts = _stratified_cards(cards, card_labels, ratios, seed)
    group_cards = {index: card for index, card in enumerate(cards)}
    output.mkdir(parents=True, exist_ok=True)
    split_stats = {}
    for name, selected_cards in zip(("train", "valid", "test"), card_parts):
        selected_groups = {index for index, card in group_cards.items() if card in selected_cards}
        indices = np.asarray(
            [index for index, group in enumerate(arrays["group_id"]) if group in selected_groups],
            dtype=np.int64,
        )
        np.savez_compressed(
            output / f"{name}.npz", **{key: value[indices] for key, value in arrays.items()}
        )
        onset = arrays["t_onset"][indices]
        split_stats[name] = {
            "cards": len(selected_cards),
            "windows": len(indices),
            "positive_windows": int((onset >= 0).sum()),
            "positive_window_rate": float((onset >= 0).mean()),
            "observed_detections": int(arrays["detected"][indices].sum()),
        }

    manifest = {
        "dataset": "Sparkov credit-card transactions",
        "source": "https://www.kaggle.com/datasets/kartik2112/fraud-detection",
        "generator": "https://github.com/namebrandon/Sparkov_Data_Generation",
        "license": "CC0-1.0 as declared on the dataset page",
        "source_files": {
            train_csv.name: file_digest(train_csv),
            test_csv.name: file_digest(test_csv),
        },
        "construction": {
            "entity": "non-overlapping card-level episode window",
            "window_days": window_days,
            "time_source": "trans_date_trans_time; unix_time excluded due seven-year inconsistency",
            "t_onset": "first fraud-labelled day within a window; evaluation only",
            "t_detect": f"t_onset + {delay_days} days when observed, otherwise -1",
            "simulated_positive_censor_rate": censor_rate,
            "split": list(ratios),
            "split_unit": "credit card",
            "seed": seed,
            "excluded_fields": [
                "cc_num",
                "first",
                "last",
                "street",
                "city",
                "state",
                "zip",
                "job",
                "dob",
                "trans_num",
                "unix_time",
                "is_fraud",
            ],
        },
        **metadata,
        "splits": split_stats,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Adapt Sparkov transactions to episode sequences.")
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--delay-days", type=int, default=2)
    parser.add_argument("--censor-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    manifest = adapt_sparkov(
        args.train_csv,
        args.test_csv,
        args.output,
        args.window_days,
        args.delay_days,
        args.censor_rate,
        args.seed,
    )
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
