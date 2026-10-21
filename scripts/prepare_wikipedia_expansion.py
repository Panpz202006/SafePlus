from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ANNOTATION_FIELDS = [
    "entity_hash",
    "block_log_id",
    "t_detect",
    "event_count",
    "auto_reason_category",
    "annotation_status",
    "raw_data_origin",
    "t_detect_origin",
    "t_f_proxy_status",
    "constructed_fields",
    "eligible_for_tf_review",
    "review_priority",
    "difference_from_final_benchmark",
    "block_evidence_url",
    "block_reason_public",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def prepare(baseline_dir: Path, expanded_dir: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    baseline = read_csv(baseline_dir / "annotation_queue.csv")
    expanded = read_csv(expanded_dir / "annotation_queue.csv")
    baseline_ids = {row["entity_hash"] for row in baseline}
    new_rows = [row for row in expanded if row["entity_hash"] not in baseline_ids]
    new_ids = {row["entity_hash"] for row in new_rows}

    annotated: list[dict] = []
    for row in new_rows:
        event_count = int(row["event_count"])
        status = row["annotation_status"]
        eligible = status == "pending_double_review" and event_count > 0
        if not eligible:
            priority = "Exclude_or_policy_review"
        elif row["auto_reason_category"] in {
            "vandalism",
            "harassment_or_abuse",
            "spam_or_promotion",
        }:
            priority = "P1_behavior_evidence_review"
        else:
            priority = "P2_semantic_relevance_review"
        annotated.append(
            {
                "entity_hash": row["entity_hash"],
                "block_log_id": row["block_log_id"],
                "t_detect": row["t_detect"],
                "event_count": event_count,
                "auto_reason_category": row["auto_reason_category"],
                "annotation_status": status,
                "raw_data_origin": "real_public_Wikipedia_block_log_and_revision_metadata",
                "t_detect_origin": "observed_public_block_log_timestamp",
                "t_f_proxy_status": "not_labeled_pending_revision_review",
                "constructed_fields": (
                    "30_day_observation_window;entity_hash;automatic_reason_triage"
                ),
                "eligible_for_tf_review": int(eligible),
                "review_priority": priority,
                "difference_from_final_benchmark": (
                    "contains real observations and a real block endpoint, but no "
                    "independently reviewed t_f_proxy, no adjudication, and no matched control"
                ),
                "block_evidence_url": row["block_evidence_url"],
                "block_reason_public": row["block_reason_public"],
            }
        )

    with (output / "new_entities_annotated.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(annotated)

    event_count = 0
    with (
        (expanded_dir / "events_public.jsonl").open(encoding="utf-8") as source,
        (output / "new_events_public.jsonl").open("w", encoding="utf-8") as target,
    ):
        for line in source:
            if not line.strip():
                continue
            event = json.loads(line)
            if event["entity_hash"] not in new_ids:
                continue
            event["raw_data_origin"] = "real_public_Wikipedia_revision_metadata"
            event["label_status"] = "unlabeled_event_for_manual_review"
            target.write(json.dumps(event, ensure_ascii=False) + "\n")
            event_count += 1

    category_counts = Counter(row["auto_reason_category"] for row in new_rows)
    status_counts = Counter(row["annotation_status"] for row in new_rows)
    manifest = {
        "dataset": "Wikipedia real-data expansion after D12 deduplication",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "English Wikipedia public MediaWiki Action API",
        "baseline_entities": len(baseline),
        "expanded_crawl_entities": len(expanded),
        "overlap_entities": len(expanded) - len(new_rows),
        "new_unique_entities": len(new_rows),
        "new_unique_events": event_count,
        "new_entities_with_public_events": sum(
            int(row["event_count"]) > 0 for row in new_rows
        ),
        "new_entities_without_public_events": sum(
            int(row["event_count"]) == 0 for row in new_rows
        ),
        "eligible_for_tf_review": sum(
            row["eligible_for_tf_review"] for row in annotated
        ),
        "category_counts": dict(category_counts),
        "annotation_status_counts": dict(status_counts),
        "truth_boundary": {
            "real": [
                "public revision metadata",
                "public revision IDs and timestamps",
                "public block-log IDs, timestamps, reasons, and evidence URLs",
            ],
            "constructed": [
                "30-day observation window",
                "hashed entity identifiers",
                "automatic reason triage",
                "review priority",
            ],
            "not_yet_available": [
                "independently reviewed t_f_proxy",
                "adjudicated behavior category",
                "matched right-censored control",
            ],
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deduplicate and annotate a Wikipedia real-data expansion."
    )
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--expanded-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.baseline_dir, args.expanded_dir, args.output),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
