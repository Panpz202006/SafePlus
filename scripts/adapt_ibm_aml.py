from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
FEATURE_NAMES = [
    "log1p_out_count",
    "log1p_out_amount_sum",
    "log1p_out_amount_max",
    "log1p_in_count",
    "log1p_in_amount_sum",
    "log1p_in_amount_max",
    "out_cross_bank_share",
    "out_currency_mismatch_share",
    "out_self_transfer_share",
    "in_cross_bank_share",
    "in_currency_mismatch_share",
] + [f"out_format_share:{name}" for name in FORMATS]


def _duplicate_hash_candidates(transactions: Path, chunksize: int) -> set[int]:
    """Find repeated row hashes without retaining the full CSV in memory."""
    hashes = []
    for chunk in pd.read_csv(transactions, chunksize=chunksize):
        hashes.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy(np.uint64))
    values = np.concatenate(hashes)
    duplicated = pd.Series(values).duplicated(keep=False).to_numpy()
    return {int(value) for value in values[duplicated]}


def _drop_exact_duplicates(
    chunk: pd.DataFrame,
    duplicate_hashes: set[int],
    seen_rows: set[tuple[Any, ...]],
) -> tuple[pd.DataFrame, int]:
    if not duplicate_hashes:
        return chunk, 0
    hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy(np.uint64)
    keep = np.ones(len(chunk), dtype=bool)
    for position in np.flatnonzero(np.isin(hashes, list(duplicate_hashes))):
        row = tuple(chunk.iloc[position].tolist())
        if row in seen_rows:
            keep[position] = False
        else:
            seen_rows.add(row)
    removed = int((~keep).sum())
    return chunk.iloc[keep].copy(), removed


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _combine_groups(parts: list[pd.DataFrame]) -> pd.DataFrame:
    frame = pd.concat(parts, ignore_index=True)
    aggregation = {column: "sum" for column in frame.columns if column not in {"entity", "day"}}
    for column in ("amount_max",):
        if column in aggregation:
            aggregation[column] = "max"
    return frame.groupby(["entity", "day"], sort=True, as_index=False).agg(aggregation)


def _split_groups(groups: np.ndarray, labels: np.ndarray, ratios, seed: int):
    rng = np.random.default_rng(seed)
    result = [[], [], []]
    for label in np.unique(labels):
        selected = groups[labels == label].copy()
        rng.shuffle(selected)
        cut1 = int(len(selected) * ratios[0])
        cut2 = cut1 + int(len(selected) * ratios[1])
        for destination, values in zip(result, (selected[:cut1], selected[cut1:cut2], selected[cut2:])):
            destination.extend(values.tolist())
    return [set(values) for values in result]


def build_arrays(
    transactions: Path,
    end_date: str = "2022-09-10",
    delay_days: int = 1,
    censor_rate: float = 0.2,
    seed: int = 20261005,
    chunksize: int = 500_000,
):
    cutoff = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    start = pd.Timestamp("2022-09-01")
    horizon = (cutoff - start).days
    if horizon < 2:
        raise ValueError("end_date must provide at least two days")

    outgoing_parts = []
    incoming_parts = []
    format_parts = []
    fraud_onsets = []
    all_accounts: set[str] = set()
    retained_rows = 0
    retained_laundering = 0
    duplicate_rows_removed = 0
    duplicate_hashes = _duplicate_hash_candidates(transactions, chunksize)
    seen_duplicate_rows: set[tuple[Any, ...]] = set()

    dtypes = {
        "From Bank": "string",
        "Account": "string",
        "To Bank": "string",
        "Account.1": "string",
    }
    for chunk in pd.read_csv(transactions, chunksize=chunksize, dtype=dtypes):
        chunk, removed = _drop_exact_duplicates(
            chunk, duplicate_hashes, seen_duplicate_rows
        )
        duplicate_rows_removed += removed
        timestamp = pd.to_datetime(chunk["Timestamp"], format="%Y/%m/%d %H:%M", errors="raise")
        chunk = chunk.loc[timestamp < cutoff].copy()
        timestamp = timestamp.loc[chunk.index]
        if chunk.empty:
            continue
        chunk["day"] = (timestamp - start).dt.days.astype(np.int16)
        chunk["src"] = chunk["From Bank"] + "|" + chunk["Account"]
        chunk["dst"] = chunk["To Bank"] + "|" + chunk["Account.1"]
        chunk["cross_bank"] = chunk["From Bank"].ne(chunk["To Bank"]).astype(np.int8)
        chunk["currency_mismatch"] = chunk["Payment Currency"].ne(
            chunk["Receiving Currency"]
        ).astype(np.int8)
        chunk["self_transfer"] = chunk["src"].eq(chunk["dst"]).astype(np.int8)
        all_accounts.update(chunk["src"].unique())
        all_accounts.update(chunk["dst"].unique())
        retained_rows += len(chunk)
        retained_laundering += int(chunk["Is Laundering"].sum())

        outgoing = chunk.groupby(["src", "day"], as_index=False).agg(
            count=("Amount Paid", "size"),
            amount_sum=("Amount Paid", "sum"),
            amount_max=("Amount Paid", "max"),
            cross_bank=("cross_bank", "sum"),
            currency_mismatch=("currency_mismatch", "sum"),
            self_transfer=("self_transfer", "sum"),
        ).rename(columns={"src": "entity"})
        incoming = chunk.groupby(["dst", "day"], as_index=False).agg(
            count=("Amount Received", "size"),
            amount_sum=("Amount Received", "sum"),
            amount_max=("Amount Received", "max"),
            cross_bank=("cross_bank", "sum"),
            currency_mismatch=("currency_mismatch", "sum"),
        ).rename(columns={"dst": "entity"})
        outgoing_parts.append(outgoing)
        incoming_parts.append(incoming)

        formats = pd.crosstab([chunk["src"], chunk["day"]], chunk["Payment Format"])
        formats = formats.reindex(columns=FORMATS, fill_value=0).reset_index()
        formats = formats.rename(columns={"src": "entity"})
        format_parts.append(formats)

        fraud = chunk.loc[chunk["Is Laundering"] == 1, ["src", "dst", "day"]]
        fraud_onsets.append(
            pd.concat(
                [
                    fraud[["src", "day"]].rename(columns={"src": "entity"}),
                    fraud[["dst", "day"]].rename(columns={"dst": "entity"}),
                ],
                ignore_index=True,
            )
        )

    outgoing = _combine_groups(outgoing_parts)
    incoming = _combine_groups(incoming_parts)
    format_counts = _combine_groups(format_parts)
    accounts = np.asarray(sorted(all_accounts))
    account_to_index = {account: index for index, account in enumerate(accounts)}
    x = np.zeros((len(accounts), horizon, len(FEATURE_NAMES)), dtype=np.float32)

    out_rows = outgoing["entity"].map(account_to_index).to_numpy(dtype=np.int64)
    out_days = outgoing["day"].to_numpy(dtype=np.int64)
    out_count = outgoing["count"].to_numpy(dtype=np.float32)
    x[out_rows, out_days, 0:3] = np.log1p(
        outgoing[["count", "amount_sum", "amount_max"]].to_numpy(dtype=np.float32)
    )
    x[out_rows, out_days, 6] = outgoing["cross_bank"].to_numpy(dtype=np.float32) / out_count
    x[out_rows, out_days, 7] = outgoing["currency_mismatch"].to_numpy(dtype=np.float32) / out_count
    x[out_rows, out_days, 8] = outgoing["self_transfer"].to_numpy(dtype=np.float32) / out_count

    in_rows = incoming["entity"].map(account_to_index).to_numpy(dtype=np.int64)
    in_days = incoming["day"].to_numpy(dtype=np.int64)
    in_count = incoming["count"].to_numpy(dtype=np.float32)
    x[in_rows, in_days, 3:6] = np.log1p(
        incoming[["count", "amount_sum", "amount_max"]].to_numpy(dtype=np.float32)
    )
    x[in_rows, in_days, 9] = incoming["cross_bank"].to_numpy(dtype=np.float32) / in_count
    x[in_rows, in_days, 10] = incoming["currency_mismatch"].to_numpy(dtype=np.float32) / in_count

    format_rows = format_counts["entity"].map(account_to_index).to_numpy(dtype=np.int64)
    format_days = format_counts["day"].to_numpy(dtype=np.int64)
    format_values = format_counts[FORMATS].to_numpy(dtype=np.float32)
    format_totals = format_values.sum(axis=1, keepdims=True)
    x[format_rows, format_days, 11:] = format_values / format_totals

    fraud_frame = pd.concat(fraud_onsets, ignore_index=True)
    onset_lookup = fraud_frame.groupby("entity")["day"].min()
    t_onset = onset_lookup.reindex(accounts, fill_value=-1).to_numpy(dtype=np.int64)
    candidate_detect = t_onset + delay_days
    detectable = (t_onset >= 0) & (candidate_detect < horizon)
    rng = np.random.default_rng(seed)
    simulated_censored = np.zeros(len(accounts), dtype=bool)
    candidates = np.flatnonzero(detectable)
    simulated_censored[candidates] = rng.random(len(candidates)) < censor_rate
    detected = detectable & ~simulated_censored
    t_detect = np.where(detected, candidate_detect, -1).astype(np.int64)

    arrays = {
        "x": x,
        "lengths": np.full(len(accounts), horizon, dtype=np.int64),
        "detected": detected,
        "t_detect": t_detect,
        "t_onset": t_onset,
        "entity_id": np.arange(len(accounts), dtype=np.int64),
        "group_id": np.arange(len(accounts), dtype=np.int64),
    }
    metadata = {
        "retained_transactions": retained_rows,
        "retained_laundering_transactions": retained_laundering,
        "retained_laundering_rate": retained_laundering / retained_rows,
        "exact_duplicate_rows_removed": duplicate_rows_removed,
        "accounts": len(accounts),
        "positive_accounts": int((t_onset >= 0).sum()),
        "positive_account_rate": float((t_onset >= 0).mean()),
        "horizon_days": horizon,
        "feature_names": FEATURE_NAMES,
        "observed_detections": int(detected.sum()),
        "simulated_censored_positive_accounts": int(simulated_censored.sum()),
        "naturally_censored_positive_accounts": int(((t_onset >= 0) & ~detectable).sum()),
    }
    return arrays, metadata, accounts


def adapt_ibm_aml(
    transactions: Path,
    patterns: Path,
    output: Path,
    end_date: str = "2022-09-10",
    delay_days: int = 1,
    censor_rate: float = 0.2,
    seed: int = 20261005,
    ratios=(0.7, 0.1, 0.2),
):
    arrays, metadata, accounts = build_arrays(
        transactions, end_date, delay_days, censor_rate, seed
    )
    labels = (arrays["t_onset"] >= 0).astype(np.int8)
    group_parts = _split_groups(accounts, labels, ratios, seed)
    output.mkdir(parents=True, exist_ok=True)
    split_stats = {}
    for name, selected_accounts in zip(("train", "valid", "test"), group_parts):
        indices = np.asarray(
            [index for index, account in enumerate(accounts) if account in selected_accounts],
            dtype=np.int64,
        )
        np.savez_compressed(
            output / f"{name}.npz", **{key: value[indices] for key, value in arrays.items()}
        )
        onset = arrays["t_onset"][indices]
        split_stats[name] = {
            "accounts": len(indices),
            "positive_accounts": int((onset >= 0).sum()),
            "positive_rate": float((onset >= 0).mean()),
            "observed_detections": int(arrays["detected"][indices].sum()),
        }

    manifest = {
        "dataset": "IBM AML HI-Small",
        "source": "https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml",
        "paper": "https://arxiv.org/abs/2306.16424",
        "license": "CDLA-Sharing-1.0",
        "source_files": {
            transactions.name: digest(transactions),
            patterns.name: digest(patterns),
        },
        "construction": {
            "entity": "bank-and-account pair",
            "time_axis": "daily, 2022-09-01 through 2022-09-10",
            "tail_policy": "exclude 2022-09-11 through 2022-09-18 from the primary experiment",
            "t_onset": "first laundering-labelled transaction involving the account; evaluation only",
            "t_detect": f"t_onset + {delay_days} day when observed, otherwise -1",
            "simulated_positive_censor_rate": censor_rate,
            "split_unit": "account",
            "split": list(ratios),
            "seed": seed,
        },
        **metadata,
        "splits": split_stats,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Adapt IBM AML HI-Small account sequences.")
    parser.add_argument("--transactions", type=Path, required=True)
    parser.add_argument("--patterns", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--end-date", default="2022-09-10")
    parser.add_argument("--delay-days", type=int, default=1)
    parser.add_argument("--censor-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()
    manifest = adapt_ibm_aml(
        args.transactions,
        args.patterns,
        args.output,
        args.end_date,
        args.delay_days,
        args.censor_rate,
        args.seed,
    )
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
