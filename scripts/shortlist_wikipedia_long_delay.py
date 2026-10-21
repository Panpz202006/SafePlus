from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

TRIGGER_PATTERN = re.compile(
    r"possible vandalism|blanking|references removed|grief|get blanked|"
    r"reverting anti-vandal bot|rapid reverts|spam|advertis",
    re.IGNORECASE,
)
FIELDS = [
    "entity_id",
    "block_log_id",
    "auto_reason_category",
    "event_count",
    "candidate_onset_event_id",
    "candidate_onset_time",
    "t_detect",
    "candidate_delay_hours",
    "candidate_event_reference",
    "detection_reference",
    "candidate_event_tags",
    "candidate_event_comment",
    "block_reason_public",
    "review_status",
]


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def shortlist(raw_dir: Path, output: Path, minimum_delay_hours: float) -> list[dict]:
    with (raw_dir / "annotation_queue.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        candidates = {
            row["entity_hash"]: row for row in csv.DictReader(handle)
        }
    with (raw_dir / "events_public.jsonl").open(encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]

    earliest_trigger_by_entity: dict[str, dict] = {}
    for event in sorted(events, key=lambda item: (item["timestamp"], item["revid"])):
        entity_id = event["entity_hash"]
        text = " ".join(event.get("tags", [])) + " " + event.get("comment", "")
        if entity_id not in earliest_trigger_by_entity and TRIGGER_PATTERN.search(text):
            earliest_trigger_by_entity[entity_id] = event

    rows: list[dict] = []
    for entity_id, event in earliest_trigger_by_entity.items():
        source = candidates[entity_id]
        delay_hours = (
            parse_time(source["t_detect"]) - parse_time(event["timestamp"])
        ).total_seconds() / 3600
        if delay_hours < minimum_delay_hours:
            continue
        rows.append(
            {
                "entity_id": entity_id,
                "block_log_id": source["block_log_id"],
                "auto_reason_category": source["auto_reason_category"],
                "event_count": source["event_count"],
                "candidate_onset_event_id": event["revid"],
                "candidate_onset_time": event["timestamp"],
                "t_detect": source["t_detect"],
                "candidate_delay_hours": round(delay_hours, 6),
                "candidate_event_reference": event["diff_url"],
                "detection_reference": source["block_evidence_url"],
                "candidate_event_tags": ";".join(event.get("tags", [])),
                "candidate_event_comment": event.get("comment", ""),
                "block_reason_public": source["block_reason_public"],
                "review_status": "screened_only_pending_manual_semantic_review",
            }
        )
    rows.sort(key=lambda row: (-row["candidate_delay_hours"], row["entity_id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a manual-review queue for long-delay Wikipedia candidates."
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-delay-hours", type=float, default=24.0)
    args = parser.parse_args()
    rows = shortlist(args.raw_dir, args.output, args.minimum_delay_hours)
    print(
        json.dumps(
            {
                "minimum_delay_hours": args.minimum_delay_hours,
                "shortlisted_candidates": len(rows),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
