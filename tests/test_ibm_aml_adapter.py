from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from safeplus.data.dataset import load_npz

SCRIPT = Path(__file__).parents[1] / "scripts" / "adapt_ibm_aml.py"
SPEC = importlib.util.spec_from_file_location("adapt_ibm_aml", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_ibm_aml_primary_period_and_account_split(tmp_path):
    rows = []
    for index in range(20):
        rows.extend(
            [
                {
                    "Timestamp": "2022/09/01 01:00",
                    "From Bank": f"B{index}",
                    "Account": f"A{index}",
                    "To Bank": "TARGET",
                    "Account.1": f"T{index}",
                    "Amount Received": 10.0,
                    "Receiving Currency": "US Dollar",
                    "Amount Paid": 10.0,
                    "Payment Currency": "US Dollar",
                    "Payment Format": "ACH",
                    "Is Laundering": int(index % 2 == 0),
                },
                {
                    "Timestamp": "2022/09/12 01:00",
                    "From Bank": f"B{index}",
                    "Account": f"A{index}",
                    "To Bank": "TARGET",
                    "Account.1": f"T{index}",
                    "Amount Received": 20.0,
                    "Receiving Currency": "US Dollar",
                    "Amount Paid": 20.0,
                    "Payment Currency": "US Dollar",
                    "Payment Format": "Wire",
                    "Is Laundering": 1,
                },
            ]
        )
    transactions = tmp_path / "HI-Small_Trans.csv"
    patterns = tmp_path / "HI-Small_Patterns.txt"
    pd.DataFrame(rows).to_csv(transactions, index=False)
    patterns.write_text("test\n", encoding="utf-8")
    output = tmp_path / "out"
    manifest = MODULE.adapt_ibm_aml(
        transactions, patterns, output, delay_days=1, censor_rate=0, seed=2
    )

    groups = []
    for name in ("train", "valid", "test"):
        with np.load(output / f"{name}.npz", allow_pickle=False) as content:
            groups.append(set(content["group_id"].tolist()))
        load_npz(output / f"{name}.npz")
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert manifest["retained_transactions"] == 20
    assert manifest["exact_duplicate_rows_removed"] == 0
    assert manifest["horizon_days"] == 10


def test_ibm_aml_label_is_not_a_feature(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "Timestamp": "2022/09/02 01:00",
                "From Bank": "B1",
                "Account": "A1",
                "To Bank": "B2",
                "Account.1": "A2",
                "Amount Received": 5.0,
                "Receiving Currency": "Euro",
                "Amount Paid": 6.0,
                "Payment Currency": "US Dollar",
                "Payment Format": "Cash",
                "Is Laundering": 1,
            }
        ]
    )
    transactions = tmp_path / "transactions.csv"
    frame.to_csv(transactions, index=False)
    arrays, metadata, _ = MODULE.build_arrays(
        transactions, delay_days=1, censor_rate=0, seed=1, chunksize=1
    )
    assert set(arrays["t_onset"].tolist()) == {1}
    assert "Is Laundering" not in metadata["feature_names"]
    assert arrays["x"].shape[-1] == 18


def test_ibm_aml_removes_exact_duplicates_across_chunks(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "Timestamp": "2022/09/02 01:00",
                "From Bank": "B1",
                "Account": "A1",
                "To Bank": "B2",
                "Account.1": "A2",
                "Amount Received": 5.0,
                "Receiving Currency": "Euro",
                "Amount Paid": 6.0,
                "Payment Currency": "US Dollar",
                "Payment Format": "Cash",
                "Is Laundering": 0,
            },
            {
                "Timestamp": "2022/09/03 01:00",
                "From Bank": "B3",
                "Account": "A3",
                "To Bank": "B4",
                "Account.1": "A4",
                "Amount Received": 7.0,
                "Receiving Currency": "Euro",
                "Amount Paid": 7.0,
                "Payment Currency": "Euro",
                "Payment Format": "ACH",
                "Is Laundering": 0,
            },
        ]
    )
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    transactions = tmp_path / "transactions.csv"
    frame.to_csv(transactions, index=False)

    _, metadata, _ = MODULE.build_arrays(transactions, chunksize=2)

    assert metadata["exact_duplicate_rows_removed"] == 1
    assert metadata["retained_transactions"] == 2
