from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parents[1] / "scripts" / "adapt_ulb.py"
SPEC = importlib.util.spec_from_file_location("adapt_ulb", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_ulb_deduplicates_and_uses_chronological_splits(tmp_path):
    rows = []
    for index in range(100):
        row = {"Time": float(index // 2), "Amount": float(index + 1), "Class": index % 17 == 0}
        row.update({f"V{number}": float(index + number) for number in range(1, 29)})
        rows.append(row)
    rows.append(rows[0].copy())
    source = tmp_path / "creditcard.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    output = tmp_path / "output"

    manifest = MODULE.adapt_ulb(source, output)

    arrays = {name: np.load(output / f"{name}.npz") for name in ("train", "valid", "test")}
    assert sum(len(value["y"]) for value in arrays.values()) == 100
    assert arrays["train"]["time"].max() < arrays["valid"]["time"].min()
    assert arrays["valid"]["time"].max() < arrays["test"]["time"].min()
    assert all(value["x"].shape[1] == 30 for value in arrays.values())
    assert manifest["raw"]["exact_duplicate_rows_removed"] == 1
    assert manifest["construction"]["customer_sequence_available"] is False
