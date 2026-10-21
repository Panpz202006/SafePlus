from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

FIELDS = [
    "entity_id",
    "entity_role",
    "matched_pair_id",
    "raw_event_origin",
    "endpoint_origin",
    "t_f_proxy_origin",
    "t_f_proxy_finality",
    "constructed_operations",
    "difference_from_raw_real_data",
    "allowed_claim",
    "prohibited_claim",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def annotate(dataset: Path) -> dict:
    entities = read_csv(dataset / "entities.csv")
    pairs = read_csv(dataset / "matched_pairs.csv")
    pair_by_entity: dict[str, str] = {}
    for pair in pairs:
        pair_by_entity[pair["positive_entity_id"]] = pair["matched_pair_id"]
        pair_by_entity[pair["control_entity_id"]] = pair["matched_pair_id"]

    rows: list[dict] = []
    for entity in entities:
        detected = int(entity["detected"])
        if detected:
            role = "detected_malicious_behavior_candidate"
            endpoint_origin = "real_public_block_log_timestamp"
            tf_origin = "evidence_based_proxy_selected_from_real_revision_timeline"
            tf_finality = entity["review_status"]
            operations = (
                "30_day_window;entity_hashing;t_f_proxy_candidate_selection;"
                "left_censor_flag;activity_matching"
            )
            difference = (
                "raw data contains revisions and a real block endpoint but no native t_f; "
                "the benchmark adds a research-defined onset proxy and matching design"
            )
            allowed = (
                "publicly observed behavior followed by a public block, with a provisional "
                "earliest-observable malicious-behavior proxy"
            )
            prohibited = (
                "true latent malicious intent time; independently confirmed final t_f; "
                "financial fraud ground truth"
            )
        else:
            role = "right_censored_control_candidate"
            endpoint_origin = "constructed_30_day_followup_endpoint"
            tf_origin = "not_applicable_no_observed_block_in_window"
            tf_finality = entity["review_status"]
            operations = (
                "account_creation_cohort;30_day_followup;block_exclusion;"
                "entity_hashing;activity_matching"
            )
            difference = (
                "raw data contains real revisions, but the control label and censor endpoint "
                "come from the study window; absence of a block is not proof of normality"
            )
            allowed = "no public block observed during the defined 30-day follow-up"
            prohibited = "permanently normal user; verified non-malicious account"
        rows.append(
            {
                "entity_id": entity["entity_id"],
                "entity_role": role,
                "matched_pair_id": pair_by_entity.get(entity["entity_id"], ""),
                "raw_event_origin": "real_public_Wikipedia_revision_metadata",
                "endpoint_origin": endpoint_origin,
                "t_f_proxy_origin": tf_origin,
                "t_f_proxy_finality": tf_finality,
                "constructed_operations": operations,
                "difference_from_raw_real_data": difference,
                "allowed_claim": allowed,
                "prohibited_claim": prohibited,
            }
        )

    output = dataset / "data_layer_annotations.csv"
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "entities": len(rows),
        "role_counts": dict(Counter(row["entity_role"] for row in rows)),
        "raw_event_origin": "real public Wikipedia revision metadata",
        "note": (
            "real observations are preserved, while study windows, proxy labels, censoring, "
            "matching, and derived features are explicitly marked as constructed"
        ),
        "output": str(output),
    }
    (dataset / "data_layer_annotation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate real versus constructed layers in a Wikipedia pilot."
    )
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(json.dumps(annotate(args.dataset), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
