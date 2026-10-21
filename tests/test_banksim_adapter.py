from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from safeplus.data.dataset import load_npz

SCRIPT = Path(__file__).parents[1] / "scripts" / "adapt_banksim.py"
SPEC = importlib.util.spec_from_file_location("adapt_banksim", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_banksim_adapter_is_customer_disjoint_and_label_free(tmp_path):
    rows = []
    for customer_index in range(20):
        for step in range(6):
            rows.append(
                {
                    "step": step,
                    "customer": f"'C{customer_index}'",
                    "age": "'2'",
                    "gender": "'F'",
                    "zipcodeOri": "'Z1'",
                    "merchant": f"'M{step % 2}'",
                    "zipMerchant": "'Z1'",
                    "category": "'travel'" if step % 2 else "'food'",
                    "amount": float(customer_index + step + 1),
                    "fraud": int(customer_index % 2 == 0 and step >= 2),
                }
            )
    source = tmp_path / "banksim.csv"
    pd.DataFrame(rows).to_csv(source, index=False)
    output = tmp_path / "adapted"

    manifest = MODULE.adapt_banksim(source, output, delay=1, censor_rate=0.0, seed=7)
    splits = {}
    for name in ("train", "valid", "test"):
        with np.load(output / f"{name}.npz", allow_pickle=False) as content:
            splits[name] = {key: content[key] for key in content.files}
        load_npz(output / f"{name}.npz")

    id_sets = [set(split["entity_id"].tolist()) for split in splits.values()]
    assert id_sets[0].isdisjoint(id_sets[1])
    assert id_sets[0].isdisjoint(id_sets[2])
    assert id_sets[1].isdisjoint(id_sets[2])
    assert sum(map(len, id_sets)) == 20
    assert "fraud" not in manifest["feature_names"]
    assert manifest["feature_names"] == [
        *MODULE.BASE_FEATURES,
        "category_share:food",
        "category_share:travel",
    ]
    assert all(split["x"].shape[1:] == (6, 7) for split in splits.values())


def test_banksim_detection_time_and_null_steps(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "step": 0,
                "customer": "'A'",
                "merchant": "'M1'",
                "category": "'food'",
                "amount": 3,
                "fraud": 0,
            },
            {
                "step": 2,
                "customer": "'A'",
                "merchant": "'M2'",
                "category": "'food'",
                "amount": 5,
                "fraud": 1,
            },
            {
                "step": 0,
                "customer": "'B'",
                "merchant": "'M1'",
                "category": "'food'",
                "amount": 2,
                "fraud": 0,
            },
            {
                "step": 5,
                "customer": "'B'",
                "merchant": "'M1'",
                "category": "'food'",
                "amount": 4,
                "fraud": 0,
            },
        ]
    )
    source = tmp_path / "banksim.csv"
    frame.to_csv(source, index=False)
    arrays, _ = MODULE.build_arrays(source, delay=2, censor_rate=0.0, seed=1)

    assert arrays["t_onset"].tolist() == [2, -1]
    assert arrays["t_detect"].tolist() == [4, -1]
    assert arrays["detected"].tolist() == [True, False]
    assert np.all(arrays["x"][0, 1] == 0)
    assert np.all(arrays["x"][1, 1:5] == 0)
