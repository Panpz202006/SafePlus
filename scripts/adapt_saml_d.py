from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PAYMENT_TYPES = [
    "ACH",
    "Cash Deposit",
    "Cash Withdrawal",
    "Cheque",
    "Credit card",
    "Cross-border",
    "Debit card",
]
SUSPICIOUS_TYPES = [
    "Behavioural_Change_1",
    "Behavioural_Change_2",
    "Bipartite",
    "Cash_Withdrawal",
    "Cycle",
    "Deposit-Send",
    "Fan_In",
    "Fan_Out",
    "Gather-Scatter",
    "Layered_Fan_In",
    "Layered_Fan_Out",
    "Over-Invoicing",
    "Scatter-Gather",
    "Single_large",
    "Smurfing",
    "Stacked Bipartite",
    "Structuring",
]
FEATURE_NAMES = [
    "log1p_out_count",
    "log1p_out_amount_sum",
    "log1p_out_amount_max",
    "log1p_in_count",
    "log1p_in_amount_sum",
    "log1p_in_amount_max",
    "out_cross_location_share",
    "out_currency_mismatch_share",
    "out_self_transfer_share",
    "in_cross_location_share",
    "in_currency_mismatch_share",
] + [f"out_payment_share:{name}" for name in PAYMENT_TYPES] + [
    "out_mean_hour_sin",
    "out_mean_hour_cos",
]


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _split(values: np.ndarray, ratios, rng: np.random.Generator):
    values = values.copy()
    rng.shuffle(values)
    cut1 = int(len(values) * ratios[0])
    cut2 = cut1 + int(len(values) * ratios[1])
    return values[:cut1], values[cut1:cut2], values[cut2:]


def _date_bounds(source: Path, chunksize: int):
    minimum = None
    maximum = None
    for chunk in pd.read_csv(source, usecols=["Date"], chunksize=chunksize):
        dates = pd.to_datetime(chunk["Date"], format="%Y-%m-%d")
        current_min, current_max = dates.min(), dates.max()
        minimum = current_min if minimum is None else min(minimum, current_min)
        maximum = current_max if maximum is None else max(maximum, current_max)
    return minimum, maximum


def discover_cohort(
    source: Path,
    window_days: int,
    negative_ratio: int,
    ratios,
    seed: int,
    chunksize: int,
):
    start, end = _date_bounds(source, chunksize)
    n_windows = (end - start).days // window_days + 1
    all_key_parts = []
    positive_key_parts = []
    columns = ["Date", "Sender_account", "Receiver_account", "Is_laundering"]
    for chunk in pd.read_csv(source, usecols=columns, chunksize=chunksize):
        days = (pd.to_datetime(chunk["Date"]) - start).dt.days.to_numpy(np.int64)
        windows = days // window_days
        sender = chunk["Sender_account"].to_numpy(np.int64)
        receiver = chunk["Receiver_account"].to_numpy(np.int64)
        sender_keys = sender * n_windows + windows
        receiver_keys = receiver * n_windows + windows
        all_key_parts.extend((sender_keys, receiver_keys))
        suspicious = chunk["Is_laundering"].to_numpy(dtype=bool)
        positive_key_parts.extend((sender_keys[suspicious], receiver_keys[suspicious]))

    active_keys, activity = np.unique(np.concatenate(all_key_parts), return_counts=True)
    positive_keys = np.unique(np.concatenate(positive_key_parts))
    positive_accounts = np.unique(positive_keys // n_windows)
    active_accounts = active_keys // n_windows
    never_positive = ~np.isin(active_accounts, positive_accounts)
    control_keys = active_keys[never_positive]
    control_activity = activity[never_positive]
    control_accounts = control_keys // n_windows

    rng = np.random.default_rng(seed)
    positive_account_parts = _split(positive_accounts, ratios, rng)
    negative_accounts = np.unique(control_accounts)
    negative_account_parts = _split(negative_accounts, ratios, rng)
    split_keys = []
    sampling_weights: dict[int, float] = {}
    split_stats = []
    for positive_part, negative_part in zip(positive_account_parts, negative_account_parts):
        part_positive = positive_keys[
            np.isin(positive_keys // n_windows, positive_part)
        ]
        part_control_mask = np.isin(control_accounts, negative_part)
        candidates = control_keys[part_control_mask]
        candidate_activity = control_activity[part_control_mask]
        positive_activity = activity[np.searchsorted(active_keys, part_positive)]
        positive_strata = (part_positive % n_windows) * 64 + np.floor(
            np.log2(positive_activity)
        ).astype(np.int64)
        candidate_strata = (candidates % n_windows) * 64 + np.floor(
            np.log2(candidate_activity)
        ).astype(np.int64)
        selected_controls = []
        for stratum, positive_count in zip(
            *np.unique(positive_strata, return_counts=True)
        ):
            available = np.flatnonzero(candidate_strata == stratum)
            take = min(len(available), int(positive_count) * negative_ratio)
            chosen = rng.choice(available, size=take, replace=False)
            selected = candidates[chosen]
            selected_controls.append(selected)
            weight = len(available) / take if take else 1.0
            sampling_weights.update({int(key): weight for key in selected})
        selected_controls = np.concatenate(selected_controls)
        sampling_weights.update({int(key): 1.0 for key in part_positive})
        selected = np.sort(np.concatenate([part_positive, selected_controls]))
        split_keys.append(selected)
        split_stats.append(
            {
                "positive_windows": len(part_positive),
                "control_windows": len(selected_controls),
                "accounts": len(np.unique(selected // n_windows)),
            }
        )
    return {
        "start": start,
        "end": end,
        "n_windows": n_windows,
        "split_keys": split_keys,
        "sampling_weights": sampling_weights,
        "active_windows": len(active_keys),
        "positive_windows_population": len(positive_keys),
        "active_accounts": len(np.unique(active_accounts)),
        "positive_accounts": len(positive_accounts),
        "split_stats": split_stats,
    }


def _combine(parts: list[pd.DataFrame]) -> pd.DataFrame:
    frame = pd.concat(parts, ignore_index=True)
    aggregation = {column: "sum" for column in frame.columns if column not in {"key", "day"}}
    if "amount_max" in aggregation:
        aggregation["amount_max"] = "max"
    return frame.groupby(["key", "day"], as_index=False, sort=True).agg(aggregation)


def build_arrays(
    source: Path,
    window_days: int = 30,
    delay_days: int = 7,
    censor_rate: float = 0.2,
    negative_ratio: int = 10,
    ratios=(0.7, 0.1, 0.2),
    seed: int = 20261005,
    chunksize: int = 500_000,
):
    cohort = discover_cohort(
        source, window_days, negative_ratio, ratios, seed, chunksize
    )
    selected_keys = np.sort(np.concatenate(cohort["split_keys"]))
    key_to_row = {int(key): index for index, key in enumerate(selected_keys)}
    selected_set = set(key_to_row)
    outgoing_parts, incoming_parts, payment_parts, suspicious_parts = [], [], [], []
    for chunk in pd.read_csv(source, chunksize=chunksize):
        timestamps = pd.to_datetime(chunk["Date"] + " " + chunk["Time"])
        absolute_days = (timestamps.dt.normalize() - cohort["start"]).dt.days.to_numpy(np.int64)
        windows = absolute_days // window_days
        days = absolute_days % window_days
        sender = chunk["Sender_account"].to_numpy(np.int64)
        receiver = chunk["Receiver_account"].to_numpy(np.int64)
        sender_keys = sender * cohort["n_windows"] + windows
        receiver_keys = receiver * cohort["n_windows"] + windows
        cross_location = chunk["Sender_bank_location"].ne(
            chunk["Receiver_bank_location"]
        ).to_numpy(np.int8)
        currency_mismatch = chunk["Payment_currency"].ne(
            chunk["Received_currency"]
        ).to_numpy(np.int8)
        self_transfer = (sender == receiver).astype(np.int8)
        hours = timestamps.dt.hour.to_numpy() + timestamps.dt.minute.to_numpy() / 60
        hour_sin = np.sin(2 * np.pi * hours / 24)
        hour_cos = np.cos(2 * np.pi * hours / 24)

        sender_mask = np.fromiter(
            (int(key) in selected_set for key in sender_keys), bool, len(sender_keys)
        )
        receiver_mask = np.fromiter(
            (int(key) in selected_set for key in receiver_keys), bool, len(receiver_keys)
        )
        out = pd.DataFrame(
            {
                "key": sender_keys[sender_mask],
                "day": days[sender_mask],
                "amount": chunk.loc[sender_mask, "Amount"].to_numpy(),
                "cross_location": cross_location[sender_mask],
                "currency_mismatch": currency_mismatch[sender_mask],
                "self_transfer": self_transfer[sender_mask],
                "hour_sin": hour_sin[sender_mask],
                "hour_cos": hour_cos[sender_mask],
                "payment": chunk.loc[sender_mask, "Payment_type"].to_numpy(),
            }
        )
        incoming = pd.DataFrame(
            {
                "key": receiver_keys[receiver_mask],
                "day": days[receiver_mask],
                "amount": chunk.loc[receiver_mask, "Amount"].to_numpy(),
                "cross_location": cross_location[receiver_mask],
                "currency_mismatch": currency_mismatch[receiver_mask],
            }
        )
        if not out.empty:
            outgoing_parts.append(
                out.groupby(["key", "day"], as_index=False).agg(
                    count=("amount", "size"),
                    amount_sum=("amount", "sum"),
                    amount_max=("amount", "max"),
                    cross_location=("cross_location", "sum"),
                    currency_mismatch=("currency_mismatch", "sum"),
                    self_transfer=("self_transfer", "sum"),
                    hour_sin=("hour_sin", "sum"),
                    hour_cos=("hour_cos", "sum"),
                )
            )
            payment = pd.crosstab([out["key"], out["day"]], out["payment"])
            payment_parts.append(payment.reindex(columns=PAYMENT_TYPES, fill_value=0).reset_index())
        if not incoming.empty:
            incoming_parts.append(
                incoming.groupby(["key", "day"], as_index=False).agg(
                    count=("amount", "size"),
                    amount_sum=("amount", "sum"),
                    amount_max=("amount", "max"),
                    cross_location=("cross_location", "sum"),
                    currency_mismatch=("currency_mismatch", "sum"),
                )
            )

        suspicious = chunk["Is_laundering"].to_numpy(dtype=bool)
        for keys, selected in ((sender_keys, sender_mask), (receiver_keys, receiver_mask)):
            mask = suspicious & selected
            if mask.any():
                suspicious_parts.append(
                    pd.DataFrame(
                        {
                            "key": keys[mask],
                            "day": days[mask],
                            "typology": chunk.loc[mask, "Laundering_type"].to_numpy(),
                        }
                    )
                )

    outgoing, incoming, payments = map(
        _combine, (outgoing_parts, incoming_parts, payment_parts)
    )
    x = np.zeros((len(selected_keys), window_days, len(FEATURE_NAMES)), np.float32)
    out_rows = outgoing["key"].map(key_to_row).to_numpy(np.int64)
    out_days = outgoing["day"].to_numpy(np.int64)
    out_count = outgoing["count"].to_numpy(np.float32)
    x[out_rows, out_days, :3] = np.log1p(
        outgoing[["count", "amount_sum", "amount_max"]].to_numpy(np.float32)
    )
    x[out_rows, out_days, 6:9] = outgoing[
        ["cross_location", "currency_mismatch", "self_transfer"]
    ].to_numpy(np.float32) / out_count[:, None]
    x[out_rows, out_days, 18:20] = outgoing[["hour_sin", "hour_cos"]].to_numpy(
        np.float32
    ) / out_count[:, None]
    in_rows = incoming["key"].map(key_to_row).to_numpy(np.int64)
    in_days = incoming["day"].to_numpy(np.int64)
    in_count = incoming["count"].to_numpy(np.float32)
    x[in_rows, in_days, 3:6] = np.log1p(
        incoming[["count", "amount_sum", "amount_max"]].to_numpy(np.float32)
    )
    x[in_rows, in_days, 9:11] = incoming[
        ["cross_location", "currency_mismatch"]
    ].to_numpy(np.float32) / in_count[:, None]
    payment_rows = payments["key"].map(key_to_row).to_numpy(np.int64)
    payment_days = payments["day"].to_numpy(np.int64)
    payment_values = payments[PAYMENT_TYPES].to_numpy(np.float32)
    x[payment_rows, payment_days, 11:18] = payment_values / payment_values.sum(
        axis=1, keepdims=True
    )

    suspicious = pd.concat(suspicious_parts, ignore_index=True)
    index = pd.Index(selected_keys)
    onset = suspicious.groupby("key")["day"].min()
    t_onset = onset.reindex(index, fill_value=-1).to_numpy(np.int64)
    typology_onset = np.full((len(selected_keys), len(SUSPICIOUS_TYPES)), -1, np.int64)
    for column, name in enumerate(SUSPICIOUS_TYPES):
        values = suspicious.loc[suspicious["typology"].eq(name)].groupby("key")["day"].min()
        typology_onset[:, column] = values.reindex(index, fill_value=-1).to_numpy(np.int64)

    candidate_detect = t_onset + delay_days
    detectable = (t_onset >= 0) & (candidate_detect < window_days)
    rng = np.random.default_rng(seed)
    simulated_censored = np.zeros(len(selected_keys), dtype=bool)
    candidates = np.flatnonzero(detectable)
    simulated_censored[candidates] = rng.random(len(candidates)) < censor_rate
    detected = detectable & ~simulated_censored
    accounts = selected_keys // cohort["n_windows"]
    unique_accounts = np.unique(accounts)
    account_to_group = {int(account): i for i, account in enumerate(unique_accounts)}
    arrays = {
        "x": x,
        "lengths": np.full(len(selected_keys), window_days, np.int64),
        "detected": detected,
        "t_detect": np.where(detected, candidate_detect, -1).astype(np.int64),
        "t_onset": t_onset,
        "t_onset_by_typology": typology_onset,
        "entity_id": np.arange(len(selected_keys), dtype=np.int64),
        "group_id": np.asarray([account_to_group[int(a)] for a in accounts], np.int64),
        "sampling_weight": np.asarray(
            [cohort["sampling_weights"][int(key)] for key in selected_keys], np.float32
        ),
    }
    metadata = {
        "date_min": str(cohort["start"].date()),
        "date_max": str(cohort["end"].date()),
        "population_active_accounts": cohort["active_accounts"],
        "population_positive_accounts": cohort["positive_accounts"],
        "population_active_windows": cohort["active_windows"],
        "population_positive_windows": cohort["positive_windows_population"],
        "cohort_windows": len(selected_keys),
        "cohort_positive_windows": int((t_onset >= 0).sum()),
        "feature_names": FEATURE_NAMES,
        "typology_names": SUSPICIOUS_TYPES,
        "observed_detections": int(detected.sum()),
        "simulated_censored_positive_windows": int(simulated_censored.sum()),
        "naturally_censored_positive_windows": int(((t_onset >= 0) & ~detectable).sum()),
    }
    return arrays, metadata, cohort, selected_keys


def adapt_saml_d(source: Path, output: Path, **kwargs):
    arrays, metadata, cohort, selected_keys = build_arrays(source, **kwargs)
    output.mkdir(parents=True, exist_ok=True)
    split_stats = {}
    for name, keys in zip(("train", "valid", "test"), cohort["split_keys"]):
        indices = np.searchsorted(selected_keys, keys)
        np.savez_compressed(
            output / f"{name}.npz", **{key: value[indices] for key, value in arrays.items()}
        )
        split_stats[name] = {
            "windows": len(indices),
            "accounts": len(np.unique(keys // cohort["n_windows"])),
            "positive_windows": int((arrays["t_onset"][indices] >= 0).sum()),
            "observed_detections": int(arrays["detected"][indices].sum()),
        }
    manifest = {
        "dataset": "SAML-D current Kaggle release",
        "source": "https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml",
        "license": "CC-BY-NC-SA-4.0",
        "source_file": {"bytes": source.stat().st_size, "sha256": digest(source)},
        "construction": {
            "entity": "account in a non-overlapping calendar window",
            "window_days": kwargs.get("window_days", 30),
            "negative_sampling": "matched within split by calendar window and floor(log2 activity)",
            "negative_ratio": kwargs.get("negative_ratio", 10),
            "split_unit": "account before negative sampling",
            "t_onset": "first laundering-labelled transaction involving the account in the window",
            "t_detect": f"t_onset + {kwargs.get('delay_days', 7)} days with simulated censoring",
            "seed": kwargs.get("seed", 20261005),
        },
        **metadata,
        "splits": split_stats,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Adapt SAML-D into matched account windows.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--delay-days", type=int, default=7)
    parser.add_argument("--censor-rate", type=float, default=0.2)
    parser.add_argument("--negative-ratio", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    manifest = adapt_saml_d(
        args.source,
        args.output,
        window_days=args.window_days,
        delay_days=args.delay_days,
        censor_rate=args.censor_rate,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
