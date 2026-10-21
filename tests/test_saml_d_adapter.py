from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from safeplus.data.dataset import load_npz

SCRIPT = Path(__file__).parents[1] / "scripts" / "adapt_saml_d.py"
SPEC = importlib.util.spec_from_file_location("adapt_saml_d", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _frame():
    rows = []
    for index in range(60):
        suspicious = index < 20
        rows.append(
            {
                "Time": "10:30:00",
                "Date": "2022-10-07",
                "Sender_account": 1000 + index,
                "Receiver_account": 2000 + index,
                "Amount": 10.0 + index,
                "Payment_currency": "UK pounds",
                "Received_currency": "UK pounds",
                "Sender_bank_location": "UK",
                "Receiver_bank_location": "UK",
                "Payment_type": "ACH",
                "Is_laundering": int(suspicious),
                "Laundering_type": "Structuring" if suspicious else "Normal_Fan_Out",
            }
        )
    return pd.DataFrame(rows)


def test_saml_d_matched_cohort_and_account_split(tmp_path):
    source = tmp_path / "SAML-D.csv"
    _frame().to_csv(source, index=False)
    output = tmp_path / "output"
    manifest = MODULE.adapt_saml_d(
        source,
        output,
        window_days=30,
        delay_days=7,
        censor_rate=0,
        negative_ratio=1,
        seed=3,
        chunksize=17,
    )

    groups = []
    for name in ("train", "valid", "test"):
        dataset = load_npz(output / f"{name}.npz")
        groups.append(set(dataset.arrays["group_id"].tolist()))
        assert dataset.arrays["x"].shape[1:] == (30, 20)
        assert np.isfinite(dataset.arrays["x"]).all()
        assert (dataset.arrays["sampling_weight"] >= 1).all()
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert manifest["cohort_positive_windows"] == 40
    assert "Is_laundering" not in manifest["feature_names"]
