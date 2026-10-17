from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(
    r"data/processed/synthetic_same_delay_consistent.npz"
)

BASE_04A = Path(
    r"scripts/34_exp04_learned_behavior_evidence.py"
)

BASE_03Y = Path(
    r"scripts/32_exp03_behavior_likelihood_identification.py"
)

BASE_04F = Path(
    r"scripts/40_exp04_convex_fusion_audit.py"
)

OUTPUT_DIR = Path(
    r"reports/experiment04/final_pipeline_smoke"
)


FINAL_W = 2.0 / 3.0
FINAL_T = 1.0 / 3.0

SPLIT_SEED = 0


def load_module(
    path,
    name,
):

    spec = (
        importlib.util.spec_from_file_location(
            name,
            path,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):

        raise RuntimeError(
            f"Could not load {path}"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)

    print(
        "EXPERIMENT 04 FINAL FROZEN PIPELINE"
    )

    print("=" * 80)

    print(
        f"w = {FINAL_W}"
    )

    print(
        f"T = {FINAL_T}"
    )

    print(
        "Parameters are frozen."
    )

    # ========================================================
    # Load modules
    # ========================================================

    exp04a = load_module(
        BASE_04A,
        "exp04a",
    )

    exp03y = load_module(
        BASE_03Y,
        "exp03y",
    )

    exp04f = load_module(
        BASE_04F,
        "exp04f",
    )

    # ========================================================
    # Load data
    # ========================================================

    data = np.load(
        DATA_PATH,
        allow_pickle=True,
    )

    x = data[
        "x"
    ].astype(
        np.float64
    )

    detected = data[
        "detected"
    ].astype(
        np.int64
    )

    td = data[
        "t_detect"
    ].astype(
        np.int64
    )

    true_tf = data[
        "t_onset"
    ].astype(
        np.int64
    )

    if "entity_id" in data:

        entity_id = data[
            "entity_id"
        ].astype(
            np.int64
        )

    else:

        entity_id = np.arange(
            len(x),
            dtype=np.int64,
        )

    if "is_fraud" in data:

        fraud_label = data[
            "is_fraud"
        ].astype(
            np.int64
        )

    else:

        print(
            "Legacy synthetic file: "
            "recovering only binary is_fraud "
            "from t_onset validity."
        )

        fraud_label = (
            true_tf >= 0
        ).astype(
            np.int64
        )

    # ========================================================
    # Split
    # ========================================================

    (
        train_idx,
        val_idx,
        test_idx,
    ) = exp04a.split_indices(
        len(x),
        seed=SPLIT_SEED,
    )

    print(
        f"split: "
        f"train={len(train_idx)} "
        f"val={len(val_idx)} "
        f"test={len(test_idx)}"
    )

    # ========================================================
    # Train Behavior Evidence
    # ========================================================

    (
        behavior_params,
        em_history,
    ) = exp04a.train_emission_em(
        x=x,
        fraud_label=fraud_label,
        detected=detected,
        td=td,
        train_idx=train_idx,
    )

    # ========================================================
    # TRAIN-only global calibration
    # ========================================================

    global_scale = (
        exp04f.estimate_global_behavior_scale(
            x=x,
            fraud_label=fraud_label,
            detected=detected,
            td=td,
            train_idx=train_idx,
            behavior_params=behavior_params,
            exp04a=exp04a,
        )
    )

    # ========================================================
    # Frozen final fusion
    # ========================================================

    (
        summary,
        pred_df,
    ) = exp04f.evaluate_config(
        split_name=
            "final_pipeline_smoke",
        indices=test_idx,
        x=x,
        detected=detected,
        td=td,
        true_tf=true_tf,
        entity_id=entity_id,
        behavior_params=
            behavior_params,
        global_scale=
            global_scale,
        exp04a=exp04a,
        exp03y=exp03y,
        mode="convex",
        w=FINAL_W,
        temperature=FINAL_T,
    )

    # ========================================================
    # Results
    # ========================================================

    print()

    print("=" * 80)

    print(
        "TEST METRICS"
    )

    print("=" * 80)

    print(
        f"n = {summary['n']}"
    )

    print(
        f"MAE = "
        f"{summary['mae']:.6f}"
    )

    print(
        f"Signed Error = "
        f"{summary['signed_error']:.6f}"
    )

    print(
        f"Within-1 = "
        f"{summary['within_1']:.6f}"
    )

    print(
        f"Within-2 = "
        f"{summary['within_2']:.6f}"
    )

    # ========================================================
    # Save
    # ========================================================

    pred_df.to_csv(
        OUTPUT_DIR
        / "test_predictions.csv",
        index=False,
    )

    em_history.to_csv(
        OUTPUT_DIR
        / "em_history.csv",
        index=False,
    )

    pd.DataFrame(
        [
            summary
        ]
    ).to_csv(
        OUTPUT_DIR
        / "summary.csv",
        index=False,
    )

    print()

    print(
        "Saved to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()