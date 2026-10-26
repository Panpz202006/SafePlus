from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED_ARRAYS = ("x", "lengths", "detected", "t_detect", "t_onset", "entity_id")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    batch_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("batch_"))
    if len(batch_dirs) != 10:
        raise ValueError(f"Expected 10 batches, found {len(batch_dirs)}")

    output_dir = root / "combined_1000"
    output_dir.mkdir(exist_ok=False)
    arrays_by_key: dict[str, list[np.ndarray]] = {key: [] for key in REQUIRED_ARRAYS}
    index_rows: list[dict] = []
    timeline_rows: list[dict] = []
    summary_rows: list[dict] = []

    for batch_dir in batch_dirs:
        manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
        checks = json.loads((batch_dir / "validation_report.json").read_text(encoding="utf-8"))
        generation = manifest["generation"]
        with np.load(batch_dir / "model_input.npz", allow_pickle=False) as content:
            missing = set(REQUIRED_ARRAYS) - set(content.files)
            if missing:
                raise ValueError(f"{batch_dir.name} missing arrays: {sorted(missing)}")
            for key in REQUIRED_ARRAYS:
                arrays_by_key[key].append(content[key])
        batch_delay_mean = float(
            (arrays_by_key["t_detect"][-1] - arrays_by_key["t_onset"][-1]).mean()
        )

        batch_index = read_csv(batch_dir / "sequence_index.csv")
        signal_spec = generation.get("signal_strength", 1.0)
        behavior_mode = generation.get("behavior_mode")
        if behavior_mode is None:
            behavior_mode = (
                "gradual_ramp"
                if batch_index[0]["alignment_condition"] == "gradual_signal_begins_at_tf"
                else "aligned_step"
            )
        for row in batch_index:
            row["source_batch"] = batch_dir.name
            if not row.get("signal_strength"):
                if isinstance(signal_spec, (int, float)):
                    row["signal_strength"] = str(signal_spec)
                    row["signal_strength_source"] = "batch_protocol"
                else:
                    raise ValueError(f"{batch_dir.name} lacks per-entity signal strength")
            else:
                row["signal_strength_source"] = "per_entity_protocol"
            index_rows.append(row)

        for row in read_csv(batch_dir / "timeline_evaluation_only.csv"):
            row["source_batch"] = batch_dir.name
            timeline_rows.append(row)

        summary_rows.append(
            {
                "batch_id": manifest["batch_id"],
                "entity_count": checks["entity_count"],
                "model_entity_id_min": int(arrays_by_key["entity_id"][-1].min()),
                "model_entity_id_max": int(arrays_by_key["entity_id"][-1].max()),
                "behavior_mode": behavior_mode,
                "signal_strength_spec": json.dumps(signal_spec, ensure_ascii=False, sort_keys=True),
                "delay_policy_id": generation["delay_policy"]["id"],
                "delay_mean": checks.get("delay_mean", batch_delay_mean),
                "tf_source": generation["tf_source"],
                "td_source": generation["td_source"],
                "all_required_checks_pass": checks["all_required_checks_pass"],
            }
        )

    combined = {key: np.concatenate(parts, axis=0) for key, parts in arrays_by_key.items()}
    np.savez_compressed(output_dir / "model_input_all.npz", **combined)

    index_fields = list(index_rows[0].keys())
    for row in index_rows[1:]:
        for key in row:
            if key not in index_fields:
                index_fields.append(key)
    write_csv(output_dir / "sequence_index_all.csv", index_rows, index_fields)

    timeline_fields = list(timeline_rows[0].keys())
    for row in timeline_rows[1:]:
        for key in row:
            if key not in timeline_fields:
                timeline_fields.append(key)
    write_csv(output_dir / "timeline_evaluation_only_all.csv", timeline_rows, timeline_fields)
    write_csv(output_dir / "batch_summary.csv", summary_rows, list(summary_rows[0].keys()))

    split_counts: dict[str, int] = {}
    for row in index_rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    ids = combined["entity_id"]
    validation = {
        "batch_count": len(batch_dirs),
        "entity_count": int(len(ids)),
        "unique_entity_count": int(len(np.unique(ids))),
        "entity_id_min": int(ids.min()),
        "entity_id_max": int(ids.max()),
        "timeline_row_count": len(timeline_rows),
        "feature_shape": list(combined["x"].shape),
        "split_counts": split_counts,
        "all_finite_features": bool(np.isfinite(combined["x"]).all()),
        "all_have_normal_prefix": bool((combined["t_onset"] >= 4).all()),
        "all_tf_before_td": bool((combined["t_onset"] < combined["t_detect"]).all()),
        "all_td_within_horizon": bool((combined["t_detect"] < combined["lengths"]).all()),
        "all_detected": bool((combined["detected"] == 1).all()),
        "all_batch_checks_pass": bool(all(row["all_required_checks_pass"] for row in summary_rows)),
    }
    validation["all_required_checks_pass"] = bool(
        validation["batch_count"] == 10
        and validation["entity_count"] == 1000
        and validation["unique_entity_count"] == 1000
        and validation["entity_id_min"] == 1
        and validation["entity_id_max"] == 1000
        and validation["timeline_row_count"] == 24000
        and validation["all_finite_features"]
        and validation["all_have_normal_prefix"]
        and validation["all_tf_before_td"]
        and validation["all_td_within_horizon"]
        and validation["all_detected"]
        and validation["all_batch_checks_pass"]
    )
    (output_dir / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    manifest = {
        "dataset_name": "SafePlus model-ready controlled sequences v1",
        "release_entity_count": 1000,
        "batch_count": 10,
        "canonical_npz_keys": list(REQUIRED_ARRAYS),
        "tf_provenance": "controlled injection; not a real platform timestamp",
        "td_provenance": "prespecified controlled delay; not a platform detection or ban timestamp",
        "selection_policy": "generated from prespecified protocols; not selected by model score",
        "scope_limit": "positive-only controlled onset-recovery release; not a standalone AUPRC/AUROC benchmark",
        "batch_names": [path.name for path in batch_dirs],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "# SafePlus combined controlled release (1000 sequences)\n\n"
        "This folder combines ten prespecified 100-sequence batches into one model-ready release.\n"
        "Every sequence has a normal prefix, a controlled onset, and an in-window controlled detection time.\n\n"
        "## Provenance\n\n"
        "- t_onset/tf is a controlled injected action index, not a real platform timestamp.\n"
        "- t_detect/td is produced by a declared delay rule, not a real discovery or ban time.\n"
        "- Samples were not selected according to model performance.\n\n"
        "## Files\n\n"
        "- model_input_all.npz: 1000 canonical model sequences.\n"
        "- sequence_index_all.csv: entity-level protocol and provenance.\n"
        "- timeline_evaluation_only_all.csv: 24000 evaluation-only time-step rows.\n"
        "- batch_summary.csv: ten batch conditions.\n"
        "- validation_report.json: release-level checks.\n"
        "- manifest.json: release contract and limitations.\n\n"
        "## Limitation\n\n"
        "All 1000 sequences are detected positives. This release supports controlled onset-recovery studies but cannot alone provide valid discrimination metrics such as AUPRC, AUROC, precision, recall, or F1.\n",
        encoding="utf-8",
    )

    hash_targets = [
        output_dir / "model_input_all.npz",
        output_dir / "sequence_index_all.csv",
        output_dir / "timeline_evaluation_only_all.csv",
        output_dir / "batch_summary.csv",
        output_dir / "validation_report.json",
        output_dir / "manifest.json",
    ]
    hashes = {path.name: sha256(path) for path in hash_targets}
    (output_dir / "checksums_sha256.json").write_text(
        json.dumps(hashes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
