from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from safeplus.data.dataset import load_npz

SCRIPT = Path(__file__).parents[1] / "scripts" / "adapt_sparkov.py"
SPEC = importlib.util.spec_from_file_location("adapt_sparkov", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _row(card, day, fraud=0, category="food_dining"):
    timestamp = pd.Timestamp("2019-01-01") + pd.Timedelta(days=day, hours=3)
    return {
        "trans_date_trans_time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "cc_num": card,
        "merchant": "shop",
        "category": category,
        "amt": 10.0 + day,
        "lat": 40.0,
        "long": -70.0,
        "merch_lat": 40.1,
        "merch_long": -70.1,
        "is_fraud": fraud,
    }


def test_sparkov_group_split_and_time_labels(tmp_path):
    rows = []
    for card_index in range(20):
        for day in (0, 2, 31, 35):
            rows.append(_row(str(card_index), day, int(card_index % 2 == 0 and day in (2, 35))))
    frame = pd.DataFrame(rows)
    train_csv = tmp_path / "fraudTrain.csv"
    test_csv = tmp_path / "fraudTest.csv"
    frame.iloc[:40].to_csv(train_csv, index=False)
    frame.iloc[40:].to_csv(test_csv, index=False)
    output = tmp_path / "out"

    manifest = MODULE.adapt_sparkov(
        train_csv, test_csv, output, window_days=30, delay_days=2, censor_rate=0, seed=4
    )
    group_sets = []
    for name in ("train", "valid", "test"):
        with np.load(output / f"{name}.npz", allow_pickle=False) as content:
            group_sets.append(set(content["group_id"].tolist()))
            onset = content["t_onset"]
            detected = content["detected"]
            assert np.all(content["t_detect"][detected] == onset[detected] + 2)
        load_npz(output / f"{name}.npz")

    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])
    assert manifest["windows"] == 40
    assert "is_fraud" not in manifest["feature_names"]


def test_sparkov_null_days_are_zero(tmp_path):
    frame = pd.DataFrame([_row("1", 0), _row("1", 2, fraud=1)])
    train_csv = tmp_path / "fraudTrain.csv"
    test_csv = tmp_path / "fraudTest.csv"
    frame.iloc[:1].to_csv(train_csv, index=False)
    frame.iloc[1:].to_csv(test_csv, index=False)
    arrays, metadata, _, _ = MODULE.build_arrays(
        train_csv, test_csv, window_days=30, delay_days=1, censor_rate=0, seed=1
    )

    assert arrays["t_onset"].tolist() == [2]
    assert arrays["t_detect"].tolist() == [3]
    assert np.all(arrays["x"][0, 1] == 0)
    assert arrays["x"].shape[-1] == len(metadata["feature_names"])
