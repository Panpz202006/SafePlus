from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

TAG_PATTERN = re.compile(
    r"possible vandalism|blanking|references removed|possible libel|"
    r"reverting anti-vandal bot|rapid reverts",
    re.IGNORECASE,
)
COMMENT_PATTERN = re.compile(
    r"\bgrief(?:ed|ing)?\b|\bget blanked\b|\bvandalis[em]\b",
    re.IGNORECASE,
)
FIELDS = [
    "entity_id",
    "block_log_id",
    "auto_reason_category",
    "event_count",
    "preannotation_status",
    "review_priority",
    "candidate_tf_proxy",
    "candidate_tf_event_id",
    "candidate_delay_hours",
    "candidate_confidence",
    "left_censored",
    "why_tf",
    "why_not_before",
    "why_not_after",
    "candidate_event_reference",
    "detection_reference",
    "annotation_method",
    "label_finality",
    "raw_data_origin",
    "constructed_annotation",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def event_signal(event: dict, category: str) -> tuple[str, str] | None:
    tags = ";".join(event.get("tags", []))
    comment = event.get("comment", "")
    tag_match = TAG_PATTERN.search(tags)
    comment_match = COMMENT_PATTERN.search(comment)
    if category == "vandalism" and (tag_match or comment_match):
        signal = tag_match.group(0) if tag_match else comment_match.group(0)
        return "Provisional-High", f"public behavior signal '{signal}'"
    if category == "harassment_or_abuse" and re.search(
        r"possible libel", tags, re.IGNORECASE
    ):
        return "Provisional-Medium", "public 'possible libel' moderation signal"
    if category == "spam_or_promotion":
        title = event.get("title", "")
        promotional_creation = (
            title.startswith(("Draft:", "User:"))
            and int(event.get("sizediff") or 0) >= 300
        )
        if promotional_creation:
            return (
                "Provisional-Medium",
                "substantial user/draft-page creation preceding a public spam block",
            )
    return None


def preannotate(entities_path: Path, events_path: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    entities = read_csv(entities_path)
    with events_path.open(encoding="utf-8") as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    events_by_entity: dict[str, list[dict]] = {}
    for event in events:
        events_by_entity.setdefault(event["entity_hash"], []).append(event)
    for timeline in events_by_entity.values():
        timeline.sort(key=lambda item: (item["timestamp"], item["revid"]))

    rows: list[dict] = []
    for entity in entities:
        entity_id = entity["entity_hash"]
        category = entity["auto_reason_category"]
        timeline = events_by_entity.get(entity_id, [])
        selected_index = None
        selected_signal = None
        if entity["eligible_for_tf_review"] == "1" and category in {
            "vandalism",
            "spam_or_promotion",
            "harassment_or_abuse",
        }:
            for index, event in enumerate(timeline):
                signal = event_signal(event, category)
                if signal:
                    selected_index = index
                    selected_signal = signal
                    break

        base = {
            "entity_id": entity_id,
            "block_log_id": entity["block_log_id"],
            "auto_reason_category": category,
            "event_count": entity["event_count"],
            "candidate_tf_proxy": "",
            "candidate_tf_event_id": "",
            "candidate_delay_hours": "",
            "candidate_confidence": "",
            "left_censored": "",
            "why_tf": "",
            "why_not_before": "",
            "why_not_after": "",
            "candidate_event_reference": "",
            "detection_reference": entity["block_evidence_url"],
            "annotation_method": "deterministic_evidence_screening_v1",
            "label_finality": "not_final_pending_independent_human_review",
            "raw_data_origin": "real_public_Wikipedia_records",
            "constructed_annotation": (
                "candidate t_f_proxy and reasons generated from an explicit screening rule"
            ),
        }
        if selected_index is None or selected_signal is None:
            if not timeline:
                status = "exclude_no_public_revision_evidence"
            elif entity["eligible_for_tf_review"] != "1":
                status = "policy_or_eligibility_review_required"
            else:
                status = "manual_event_review_required_no_rule_candidate"
            rows.append(
                {
                    **base,
                    "preannotation_status": status,
                    "review_priority": "P3_manual_or_exclusion",
                }
            )
            continue

        event = timeline[selected_index]
        confidence, signal_description = selected_signal
        delay_hours = (
            parse_time(entity["t_detect"]) - parse_time(event["timestamp"])
        ).total_seconds() / 3600
        if delay_hours < 0:
            raise ValueError(f"candidate event occurs after t_detect for {entity_id}")
        if selected_index == 0:
            why_before = (
                "No earlier public revision is present in the collected 30-day timeline; "
                "the candidate is left-censored and represents earliest observable evidence."
            )
        else:
            why_before = (
                f"{selected_index} earlier public revision(s) exist, but none matched the "
                "predefined category-specific evidence rule; human review must confirm this."
            )
        tags = ";".join(event.get("tags", []))
        size_change = int(event.get("sizediff") or 0)
        why_tf = (
            f"This is the earliest revision matching {signal_description}; category triage "
            f"is '{category}', size change is {size_change}, public tags are [{tags}], and "
            "a later public block-log entry supplies the detection endpoint."
        )
        why_after = (
            "The selected revision is the earliest event satisfying the screening rule; "
            "later revisions or the public block are confirmations rather than an earlier onset."
        )
        rows.append(
            {
                **base,
                "preannotation_status": "candidate_pending_independent_human_review",
                "review_priority": (
                    "P1_long_delay_candidate"
                    if delay_hours >= 24
                    else "P1_non_left_censored_candidate"
                    if selected_index > 0
                    else "P2_left_censored_short_delay"
                ),
                "candidate_tf_proxy": event["timestamp"],
                "candidate_tf_event_id": event["revid"],
                "candidate_delay_hours": round(delay_hours, 6),
                "candidate_confidence": confidence,
                "left_censored": int(selected_index == 0),
                "why_tf": why_tf,
                "why_not_before": why_before,
                "why_not_after": why_after,
                "candidate_event_reference": event["diff_url"],
            }
        )

    with (output / "preannotation_queue.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    candidates = [
        row
        for row in rows
        if row["preannotation_status"]
        == "candidate_pending_independent_human_review"
    ]
    remaining = [row for row in rows if row not in candidates]
    event_keys = {
        (event["entity_hash"], str(event["revid"])) for event in events
    }
    validation_errors: list[str] = []
    for row in candidates:
        key = (row["entity_id"], str(row["candidate_tf_event_id"]))
        if key not in event_keys:
            validation_errors.append(f"candidate event not found: {key}")
        if float(row["candidate_delay_hours"]) < 0:
            validation_errors.append(f"negative candidate delay: {key}")
        for field in ("why_tf", "why_not_before", "why_not_after"):
            if not row[field].strip():
                validation_errors.append(f"blank {field}: {key}")
        if row["label_finality"] != "not_final_pending_independent_human_review":
            validation_errors.append(f"incorrect label finality: {key}")
    if validation_errors:
        raise ValueError("; ".join(validation_errors))
    for name, subset in (
        ("preannotated_tf_candidates.csv", candidates),
        ("manual_review_or_exclusion.csv", remaining),
    ):
        with (output / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(subset)

    summary = {
        "entities": len(rows),
        "preannotated_tf_candidates": len(candidates),
        "status_counts": dict(Counter(row["preannotation_status"] for row in rows)),
        "candidate_category_counts": dict(
            Counter(row["auto_reason_category"] for row in candidates)
        ),
        "candidate_confidence_counts": dict(
            Counter(row["candidate_confidence"] for row in candidates)
        ),
        "candidate_priority_counts": dict(
            Counter(row["review_priority"] for row in candidates)
        ),
        "left_censored_candidates": sum(int(row["left_censored"]) for row in candidates),
        "label_finality": "all candidates pending independent human review",
    }
    (output / "preannotation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "preannotation_validation.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "candidate_rows_checked": len(candidates),
                "checks": [
                    "candidate event belongs to the same entity",
                    "candidate event occurs no later than t_detect",
                    "why_tf/why_not_before/why_not_after are non-empty",
                    "all labels remain pending independent human review",
                ],
                "errors": [],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create evidence-traceable first-pass Wikipedia t_f preannotations."
    )
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            preannotate(args.entities, args.events, args.output),
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
