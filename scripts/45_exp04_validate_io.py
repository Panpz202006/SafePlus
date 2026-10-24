"""SafePlus Experiment 04 I/O contract validator (v1.0).

Run from repository root:
  python scripts/45_exp04_validate_io.py --bundle data/contract_v1/synthetic_same_delay

No model training is performed here. Only checks the data/label/config contract.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REQUIRED_INPUT_KEYS = {"entity_id", "x", "lengths", "detected", "t_detect"}
REQUIRED_META = {
    "schema_version", "dataset_id", "dataset_kind", "time_unit",
    "time_origin", "time_index_base", "feature_names",
    "t_detect_definition", "detected_bin_features", "post_detection_policy",
    "split_method", "preprocessing", "data_source", "license",
    "annotation_guideline_version",
}


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"{path}: CSV header is missing")
        absent = required - set(reader.fieldnames)
        if absent:
            raise ValueError(f"{path}: missing columns {sorted(absent)}")
        return list(reader)


def check_unique(rows: list[dict[str, str]], label: str, errors: list[str]):
    ids = [r["entity_id"] for r in rows]
    if len(ids) != len(set(ids)):
        errors.append(f"{label}: duplicate entity_id")


def validate_bundle(bundle: Path, *, check_gold: bool = True) -> tuple[list[str], list[str], dict]:
    errors, warnings = [], []
    stats: dict = {"bundle": str(bundle)}
    for filename in ["inputs.npz", "split.csv", "train_labels.csv",
                     "metadata.json", "delay_prior.json"]:
        if not (bundle / filename).is_file():
            errors.append(f"Missing required file: {filename}")
    if errors:
        return errors, warnings, stats

    try:
        with np.load(bundle / "inputs.npz", allow_pickle=False) as loaded:
            keys = set(loaded.files)
            missing = REQUIRED_INPUT_KEYS - keys
            extra = keys - REQUIRED_INPUT_KEYS
            if missing:
                errors.append(f"inputs.npz missing {sorted(missing)}")
            if extra:
                errors.append(f"inputs.npz has forbidden extra keys {sorted(extra)}; "
                              "labels / t_onset must be separate")
            if errors:
                return errors, warnings, stats
            x = np.asarray(loaded["x"])
            lengths = np.asarray(loaded["lengths"])
            detected = np.asarray(loaded["detected"])
            td = np.asarray(loaded["t_detect"])
            raw_ids = np.asarray(loaded["entity_id"])
        meta = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
        prior = json.loads((bundle / "delay_prior.json").read_text(encoding="utf-8"))
        splits = read_csv(bundle / "split.csv", {"entity_id", "split"})
        train = read_csv(bundle / "train_labels.csv", {"entity_id", "is_fraud"})
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as ex:
        errors.append(f"Could not parse bundle: {ex}")
        return errors, warnings, stats

    if x.ndim != 3 or not np.issubdtype(x.dtype, np.number):
        errors.append("x must be numeric [N,L,D]")
        return errors, warnings, stats
    n, l, d = x.shape
    stats.update({"N": n, "L": l, "D": d})
    if n < 1 or l < 2 or d < 1:
        errors.append("x dimensions are too small")
    for name, arr in [("lengths", lengths), ("detected", detected),
                      ("t_detect", td), ("entity_id", raw_ids)]:
        if arr.ndim != 1 or len(arr) != n:
            errors.append(f"{name} must be a 1-D array of length N")
    if errors:
        return errors, warnings, stats
    if raw_ids.dtype.kind not in "iuUS":
        errors.append("entity_id must be integer or Unicode/string, not object/pickle")
    ids = list(map(str, raw_ids.tolist()))
    if len(set(ids)) != n:
        errors.append("inputs.npz: entity_id must be unique")
    for name, arr in [("lengths", lengths), ("detected", detected), ("t_detect", td)]:
        if arr.dtype.kind not in "iu":
            errors.append(f"{name} must be integer dtype")
    if errors:
        return errors, warnings, stats

    if not np.all((lengths >= 1) & (lengths <= l)):
        errors.append("lengths must satisfy 1 <= lengths <= L")
    length_aware = (meta.get("schema_version") == "1.1"
                    and meta.get("variable_length_policy") == "slice_before_fit_and_predict")
    if np.any(lengths != l) and not length_aware:
        errors.append("FROZEN_ENGINE_UNSUPPORTED: variable lengths are not yet handled "
                      "end-to-end by Exp04 EM; do not silently run this bundle")
    if not np.all(np.isin(detected, [0, 1])):
        errors.append("detected must contain only 0 or 1")
    valid_det = (detected == 1) & (td >= 1) & (td < lengths)
    valid_cen = (detected == 0) & (td == lengths)
    if not np.all(valid_det | valid_cen):
        errors.append("Invalid t_detect: detected requires 1 <= d < lengths; "
                      "undetected requires t_detect == lengths (censor marker)")
    if not np.all(np.isfinite(x)):
        errors.append("x contains NaN/Inf; convert raw missing data using TRAIN-only rules")
    if length_aware:
        for i, actual_length in enumerate(lengths):
            if np.any(x[i, int(actual_length):, :] != 0):
                errors.append(f"entity_id={ids[i]}: storage padding must be zero")
    for i in np.flatnonzero(detected == 1):
        if td[i] + 1 < l and not np.all(x[i, int(td[i])+1:, :] == 0):
            errors.append(f"entity_id={ids[i]}: x after detected bin must be masked to zero")
            if len(errors) > 20:
                break

    absent_meta = REQUIRED_META - set(meta)
    if absent_meta:
        errors.append(f"metadata.json missing {sorted(absent_meta)}")
    if meta.get("schema_version") not in ("1.0", "1.1"):
        errors.append("metadata.schema_version must equal '1.0' or '1.1'")
    if meta.get("schema_version") == "1.1" and not length_aware:
        errors.append("schema 1.1 requires variable_length_policy=slice_before_fit_and_predict")
    if meta.get("dataset_kind") not in ("synthetic", "real"):
        errors.append("metadata.dataset_kind must be synthetic or real")
    if meta.get("time_index_base") != 0:
        errors.append("time_index_base must be 0")
    if not isinstance(meta.get("feature_names"), list) or len(meta["feature_names"]) != d:
        errors.append("feature_names must contain exactly D features in x order")
    if meta.get("detected_bin_features") != "pre_detection_only":
        errors.append("x[t_detect] must be explicitly declared 'pre_detection_only'")
    if meta.get("post_detection_policy") != "masked_zero":
        errors.append("v1 requires post_detection_policy='masked_zero'")
    if not str(meta.get("t_detect_definition", "")).strip():
        errors.append("t_detect_definition cannot be blank")

    check_unique(splits, "split.csv", errors)
    check_unique(train, "train_labels.csv", errors)
    idset = set(ids)
    splits_by_id = {r["entity_id"]: r["split"] for r in splits}
    if set(splits_by_id) != idset:
        errors.append("split.csv must contain exactly one row for each inputs.npz entity")
    if any(r["split"] not in ("TRAIN", "VAL", "TEST") for r in splits):
        errors.append("split.csv must use only TRAIN, VAL, TEST")
    if not all(v in splits_by_id.values() for v in ("TRAIN", "VAL", "TEST")):
        errors.append("All TRAIN, VAL, TEST partitions must be nonempty")
    train_set = {eid for eid, sp in splits_by_id.items() if sp == "TRAIN"}
    label_ids = {r["entity_id"] for r in train}
    if label_ids != train_set:
        errors.append("train_labels.csv must have ALL and ONLY TRAIN entity IDs")
    if any(r["is_fraud"] not in ("0", "1") for r in train):
        errors.append("train_labels.csv is_fraud must be 0 or 1; unknown -1 is excluded")
    if {r["is_fraud"] for r in train} != {"0", "1"}:
        errors.append("Current EM requires both confirmed normal and fraud TRAIN entities")
    train_det_fraud = {
        r["entity_id"] for r in train if r["is_fraud"] == "1"
        and r["entity_id"] in idset and bool(detected[ids.index(r["entity_id"])])
    }
    if not train_det_fraud:
        errors.append("Need at least one detected fraud TRAIN entity for global calibration")

    mode = prior.get("mode")
    if mode not in ("synthetic_known", "empirical_train", "external_fixed"):
        errors.append("delay_prior.mode must be synthetic_known/empirical_train/external_fixed")
    for name in ("definition", "source", "parameters", "version"):
        if name not in prior:
            errors.append(f"delay_prior.json missing {name}")
    if mode == "synthetic_known" and meta.get("dataset_kind") != "synthetic":
        errors.append("synthetic_known prior is prohibited for real datasets")
    if mode == "empirical_train" and not (bundle / "train_onset_for_prior.csv").is_file():
        errors.append("empirical_train requires explicit train_onset_for_prior.csv; "
                      "never infer from test/val gold")
    if mode == "empirical_train" and (bundle / "train_onset_for_prior.csv").is_file():
        prior_rows = read_csv(bundle / "train_onset_for_prior.csv", {"entity_id", "t_onset"})
        check_unique(prior_rows, "train_onset_for_prior.csv", errors)
        if any(row["entity_id"] not in train_set for row in prior_rows):
            errors.append("train_onset_for_prior.csv contains non-TRAIN users")
        for row in prior_rows:
            if row["entity_id"] not in train_set:
                continue
            idx = ids.index(row["entity_id"])
            try:
                onset = int(row["t_onset"])
                if detected[idx] != 1 or not 0 <= onset < td[idx]:
                    errors.append("train_onset_for_prior.csv has invalid detected onset/delay")
            except (ValueError, TypeError):
                errors.append("train_onset_for_prior.csv has noninteger onset")

    gold_path = bundle / "gold_onset.csv"
    if check_gold and gold_path.exists():
        try:
            gold = read_csv(gold_path, {"entity_id", "is_fraud", "onset_label_type", "t_onset"})
            check_unique(gold, "gold_onset.csv", errors)
            for row in gold:
                eid = row["entity_id"]
                if eid not in idset:
                    errors.append("gold_onset.csv contains unknown entity_id")
                    continue
                label_type = row["onset_label_type"]
                if label_type not in ("EXACT", "INTERVAL", "UNRESOLVED", "NOT_FRAUD"):
                    errors.append(f"{eid}: invalid onset_label_type")
                if label_type == "EXACT":
                    if row["is_fraud"] != "1":
                        errors.append(f"{eid}: EXACT requires is_fraud=1")
                    try:
                        onset = int(row["t_onset"])
                        idx = ids.index(eid)
                        if not (0 <= onset < lengths[idx]):
                            errors.append(f"{eid}: onset outside observed length")
                        if detected[idx] == 1 and not (onset < td[idx]):
                            errors.append(f"{eid}: onset must be strictly before t_detect")
                    except (ValueError, TypeError):
                        errors.append(f"{eid}: invalid integer EXACT t_onset")
                elif label_type == "INTERVAL" and meta.get("schema_version") == "1.1":
                    if row.get("t_onset", "") or row.get("onset_utc", ""):
                        errors.append(f"{eid}: INTERVAL must not contain a fabricated point onset")
                    if row.get("interval_closure") != "(lower,upper]":
                        errors.append(f"{eid}: INTERVAL closure must remain (lower,upper]")
                elif label_type == "NOT_FRAUD" and meta.get("dataset_kind") == "real":
                    if any(row.get(key, "") for key in ("onset_utc", "onset_lower_utc", "onset_upper_utc", "td_utc")):
                        errors.append(f"{eid}: censored normal control must not have a fabricated tf/td")
            stats["gold_rows"] = len(gold)
        except (ValueError, OSError) as ex:
            errors.append(f"Cannot read gold_onset.csv: {ex}")
    elif check_gold:
        warnings.append("gold_onset.csv missing: I/O is checkable, but onset MAE cannot be evaluated")

    stats.update({
        "detected": int((detected == 1).sum()),
        "censored": int((detected == 0).sum()),
        "train": sum(sp == "TRAIN" for sp in splits_by_id.values()),
        "val": sum(sp == "VAL" for sp in splits_by_id.values()),
        "test": sum(sp == "TEST" for sp in splits_by_id.values()),
        "prior_mode": mode,
    })
    return errors, warnings, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate frozen Exp04 I/O Contract v1.0")
    parser.add_argument("--bundle", type=Path, required=True,
                        help="Directory containing inputs.npz and companion manifests")
    args = parser.parse_args()
    errors, warnings, stats = validate_bundle(args.bundle)
    print("EXP04 I/O CONTRACT v1.0")
    print("SUMMARY:", json.dumps(stats, ensure_ascii=False, sort_keys=True))
    for warn in warnings:
        print("WARNING:", warn)
    for error in errors:
        print("ERROR:", error)
    print("RESULT:", "PASS (SCHEMA ONLY; prediction regression still pending)" if not errors else "FAIL")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
