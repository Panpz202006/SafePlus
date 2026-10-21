from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from safeplus.data.dataset import load_npz

SCRIPT = Path(__file__).parents[1] / "scripts" / "adapt_fraud_handbook.py"
SPEC = importlib.util.spec_from_file_location("adapt_fraud_handbook", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_handbook_adapter_group_isolation_and_scenarios(tmp_path):
    rows = []
    transaction_id = 0
    for customer in range(20):
        for day in (0, 3, 31, 35):
            fraud = int(customer % 2 == 0 and day in (3, 35))
            rows.append(
                {
                    "TRANSACTION_ID": transaction_id,
                    "TX_DATETIME": pd.Timestamp("2018-04-01") + pd.Timedelta(days=day, hours=5),
                    "CUSTOMER_ID": customer,
                    "TERMINAL_ID": customer % 4,
                    "TX_AMOUNT": 10.0 + day,
                    "TX_TIME_SECONDS": day * 86400 + 18000,
                    "TX_TIME_DAYS": day,
                    "TX_FRAUD": fraud,
                    "TX_FRAUD_SCENARIO": 3 if fraud else 0,
                }
            )
            transaction_id += 1
    frame = pd.DataFrame(rows)
    for day, part in frame.groupby(frame.TX_TIME_DAYS // 30):
        part.to_pickle(tmp_path / f"2018-0{day + 4}-01.pkl")
    output = tmp_path / "out"
    manifest = MODULE.adapt_fraud_handbook(
        tmp_path, output, window_days=30, delay_days=2, censor_rate=0, seed=3
    )

    groups = []
    for name in ("train", "valid", "test"):
        with np.load(output / f"{name}.npz", allow_pickle=False) as content:
            groups.append(set(content["group_id"].tolist()))
            onset = content["t_onset"]
            scenario = content["t_onset_scenario_3"]
            assert np.array_equal(onset, scenario)
        load_npz(output / f"{name}.npz")
    assert groups[0].isdisjoint(groups[1])
    assert groups[0].isdisjoint(groups[2])
    assert groups[1].isdisjoint(groups[2])
    assert manifest["windows"] == 40


def test_handbook_null_day_and_detection(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "TRANSACTION_ID": 0,
                "TX_DATETIME": pd.Timestamp("2018-04-01 05:00:00"),
                "CUSTOMER_ID": 1,
                "TERMINAL_ID": 2,
                "TX_AMOUNT": 10.0,
                "TX_TIME_SECONDS": 18000,
                "TX_TIME_DAYS": 0,
                "TX_FRAUD": 0,
                "TX_FRAUD_SCENARIO": 0,
            },
            {
                "TRANSACTION_ID": 1,
                "TX_DATETIME": pd.Timestamp("2018-04-03 05:00:00"),
                "CUSTOMER_ID": 1,
                "TERMINAL_ID": 3,
                "TX_AMOUNT": 50.0,
                "TX_TIME_SECONDS": 2 * 86400 + 18000,
                "TX_TIME_DAYS": 2,
                "TX_FRAUD": 1,
                "TX_FRAUD_SCENARIO": 2,
            },
        ]
    )
    frame.to_pickle(tmp_path / "2018-04-01.pkl")
    arrays, metadata, _, _, _ = MODULE.build_arrays(
        tmp_path, window_days=30, delay_days=2, censor_rate=0, seed=1
    )
    assert arrays["t_onset"].tolist() == [2]
    assert arrays["t_detect"].tolist() == [4]
    assert arrays["t_onset_scenario_2"].tolist() == [2]
    assert np.all(arrays["x"][0, 1] == 0)
    assert arrays["x"].shape[-1] == len(metadata["feature_names"])
