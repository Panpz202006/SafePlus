from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

ENTITY_FIELDS = [
    "entity_id",
    "t_i",
    "t_f_proxy",
    "t_d",
    "detected",
    "endpoint_type",
    "event_count",
    "tf_confidence",
    "agreement",
    "review_status",
    "left_censored",
    "block_log_id",
    "observation_start",
    "observation_end",
    "source_dataset",
]
EVENT_FIELDS = [
    "entity_id",
    "time",
    "event_id",
    "event_type",
    "raw_reference",
    "title",
    "comment",
    "size",
    "sizediff",
    "minor",
    "new",
    "top",
    "tags",
]
EVIDENCE_FIELDS = [
    "entity_id",
    "candidate_tf",
    "tf_event_id",
    "why_tf",
    "why_not_before",
    "why_not_after",
    "tf_confidence",
    "agreement",
    "review_status",
    "left_censored",
    "tf_event_reference",
    "detection_reference",
]
MODEL_INPUT_FIELDS = ["entity_id", "t_i", "t_d", "detected", "endpoint_type"]
MODEL_EVENT_FIELDS = [
    "entity_id",
    "time",
    "event_id",
    "step_index",
    "time_since_t_i_hours",
    "delta_previous_event_hours",
    "size",
    "sizediff",
    "abs_sizediff",
    "minor",
    "new",
    "top",
]
MATCH_FIELDS = [
    "matched_pair_id",
    "positive_entity_id",
    "control_entity_id",
    "positive_event_count",
    "control_event_count",
    "event_count_absolute_difference",
]
REVIEW_FIELDS = [
    "entity_id",
    "block_log_id",
    "candidate_tf",
    "tf_event_id",
    "tf_event_reference",
    "detection_reference",
    "proposed_why_tf",
    "proposed_why_not_before",
    "proposed_why_not_after",
    "reviewer_b_include",
    "reviewer_b_onset_event_id",
    "reviewer_b_onset_time",
    "reviewer_b_confidence",
    "reviewer_b_why_tf",
    "reviewer_b_why_not_before",
    "reviewer_b_why_not_after",
    "reviewer_b_disagreement_notes",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def event_type_for_positive(event: dict, onset_revid: int, onset_time: str) -> str:
    if int(event["revid"]) == onset_revid:
        return "fraud_evidence_candidate"
    if event["timestamp"] < onset_time:
        return "pre_onset_observation"
    return "post_onset_pre_detection"


def build(
    blocked_dir: Path,
    control_dir: Path,
    decisions_path: Path,
    output: Path,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    blocked_rows = read_csv(blocked_dir / "annotation_queue.csv")
    blocked_by_log = {int(row["block_log_id"]): row for row in blocked_rows}
    blocked_events = read_jsonl(blocked_dir / "events_public.jsonl")
    blocked_events_by_entity: dict[str, list[dict]] = {}
    for event in blocked_events:
        blocked_events_by_entity.setdefault(event["entity_hash"], []).append(event)

    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    entities: list[dict] = []
    events: list[dict] = []
    evidence: list[dict] = []

    for decision in decisions:
        log_id = int(decision["block_log_id"])
        if log_id not in blocked_by_log:
            raise ValueError(f"block log {log_id} is absent from the raw candidate file")
        source = blocked_by_log[log_id]
        entity_id = source["entity_hash"]
        timeline = sorted(
            blocked_events_by_entity.get(entity_id, []),
            key=lambda item: (item["timestamp"], int(item.get("revid") or -1)),
        )
        onset_revid = int(decision["onset_revid"])
        matches = [item for item in timeline if int(item["revid"]) == onset_revid]
        if len(matches) != 1:
            raise ValueError(
                f"onset revision {onset_revid} for block {log_id} has {len(matches)} matches"
            )
        onset_event = matches[0]
        onset_time = onset_event["timestamp"]
        if onset_time > source["t_detect"]:
            raise ValueError(f"onset occurs after detection for block {log_id}")

        entities.append(
            {
                "entity_id": entity_id,
                "t_i": source["observation_start"],
                "t_f_proxy": onset_time,
                "t_d": source["t_detect"],
                "detected": 1,
                "endpoint_type": "public_block_log",
                "event_count": len(timeline),
                "tf_confidence": decision["confidence"],
                "agreement": "Pending",
                "review_status": "candidate_pending_independent_review",
                "left_censored": int(bool(decision["left_censored"])),
                "block_log_id": log_id,
                "observation_start": source["observation_start"],
                "observation_end": source["observation_end"],
                "source_dataset": "Wikipedia_public_logs_and_revisions",
            }
        )
        for item in timeline:
            events.append(
                {
                    "entity_id": entity_id,
                    "time": item["timestamp"],
                    "event_id": item["revid"],
                    "event_type": event_type_for_positive(
                        item, onset_revid, onset_time
                    ),
                    "raw_reference": item["diff_url"],
                    "title": item.get("title", ""),
                    "comment": item.get("comment", ""),
                    "size": item.get("size", ""),
                    "sizediff": item.get("sizediff", ""),
                    "minor": int(bool(item.get("minor", False))),
                    "new": int(bool(item.get("new", False))),
                    "top": int(bool(item.get("top", False))),
                    "tags": ";".join(item.get("tags", [])),
                }
            )
        evidence.append(
            {
                "entity_id": entity_id,
                "candidate_tf": onset_time,
                "tf_event_id": onset_revid,
                "why_tf": decision["why_tf"],
                "why_not_before": decision["why_not_before"],
                "why_not_after": decision["why_not_after"],
                "tf_confidence": decision["confidence"],
                "agreement": "Pending",
                "review_status": "candidate_pending_independent_review",
                "left_censored": int(bool(decision["left_censored"])),
                "tf_event_reference": onset_event["diff_url"],
                "detection_reference": source["block_evidence_url"],
            }
        )

    control_rows = read_csv(control_dir / "control_entities.csv")
    control_events = read_jsonl(control_dir / "control_events.jsonl")
    for row in control_rows:
        entities.append(
            {
                "entity_id": row["entity_hash"],
                "t_i": row["t_i"],
                "t_f_proxy": "",
                "t_d": row["censor_time"],
                "detected": 0,
                "endpoint_type": row["endpoint_type"],
                "event_count": row["event_count"],
                "tf_confidence": "Not_applicable",
                "agreement": "Not_applicable",
                "review_status": row["eligibility_status"],
                "left_censored": 0,
                "block_log_id": "",
                "observation_start": row["observation_start"],
                "observation_end": row["observation_end"],
                "source_dataset": "Wikipedia_public_logs_and_revisions",
            }
        )
    for item in control_events:
        events.append(
            {
                "entity_id": item["entity_hash"],
                "time": item["timestamp"],
                "event_id": item["revid"],
                "event_type": "right_censored_observation",
                "raw_reference": item["diff_url"],
                "title": item.get("title", ""),
                "comment": item.get("comment", ""),
                "size": item.get("size", ""),
                "sizediff": item.get("sizediff", ""),
                "minor": int(bool(item.get("minor", False))),
                "new": int(bool(item.get("new", False))),
                "top": int(bool(item.get("top", False))),
                "tags": ";".join(item.get("tags", [])),
            }
        )

    entities.sort(key=lambda row: (int(row["detected"]), row["entity_id"]), reverse=True)
    events.sort(key=lambda row: (row["entity_id"], row["time"], int(row["event_id"])))
    evidence.sort(key=lambda row: row["entity_id"])
    model_input = [{field: row[field] for field in MODEL_INPUT_FIELDS} for row in entities]
    entity_lookup = {row["entity_id"]: row for row in entities}
    model_events = []
    previous_time_by_entity: dict[str, datetime] = {}
    step_by_entity: Counter[str] = Counter()
    for row in events:
        sizediff = int(row["sizediff"] or 0)
        entity_id = row["entity_id"]
        event_time = parse_time(row["time"])
        start_time = parse_time(entity_lookup[entity_id]["t_i"])
        previous_time = previous_time_by_entity.get(entity_id)
        step_index = step_by_entity[entity_id]
        model_events.append(
            {
                "entity_id": entity_id,
                "time": row["time"],
                "event_id": row["event_id"],
                "step_index": step_index,
                "time_since_t_i_hours": round(
                    (event_time - start_time).total_seconds() / 3600, 6
                ),
                "delta_previous_event_hours": (
                    0.0
                    if previous_time is None
                    else round(
                        (event_time - previous_time).total_seconds() / 3600, 6
                    )
                ),
                "size": row["size"],
                "sizediff": sizediff,
                "abs_sizediff": abs(sizediff),
                "minor": row["minor"],
                "new": row["new"],
                "top": row["top"],
            }
        )
        previous_time_by_entity[entity_id] = event_time
        step_by_entity[entity_id] += 1
    review_form = []
    for row in evidence:
        review_form.append(
            {
                "entity_id": row["entity_id"],
                "block_log_id": entity_lookup[row["entity_id"]]["block_log_id"],
                "candidate_tf": row["candidate_tf"],
                "tf_event_id": row["tf_event_id"],
                "tf_event_reference": row["tf_event_reference"],
                "detection_reference": row["detection_reference"],
                "proposed_why_tf": row["why_tf"],
                "proposed_why_not_before": row["why_not_before"],
                "proposed_why_not_after": row["why_not_after"],
                "reviewer_b_include": "",
                "reviewer_b_onset_event_id": "",
                "reviewer_b_onset_time": "",
                "reviewer_b_confidence": "",
                "reviewer_b_why_tf": "",
                "reviewer_b_why_not_before": "",
                "reviewer_b_why_not_after": "",
                "reviewer_b_disagreement_notes": "",
            }
        )

    write_csv(output / "entities.csv", ENTITY_FIELDS, entities)
    write_csv(output / "events.csv", EVENT_FIELDS, events)
    write_csv(output / "tf_evidence.csv", EVIDENCE_FIELDS, evidence)
    write_csv(output / "model_input.csv", MODEL_INPUT_FIELDS, model_input)
    write_csv(output / "model_events.csv", MODEL_EVENT_FIELDS, model_events)
    write_csv(
        output / "independent_review_form.csv", REVIEW_FIELDS, review_form
    )

    positive_entities = sorted(
        (row for row in entities if int(row["detected"]) == 1),
        key=lambda row: (int(row["event_count"]), row["entity_id"]),
    )
    control_entities = sorted(
        (row for row in entities if int(row["detected"]) == 0),
        key=lambda row: (int(row["event_count"]), row["entity_id"]),
    )
    matched_pairs = []
    for index, (positive, control) in enumerate(
        zip(positive_entities, control_entities, strict=True), start=1
    ):
        positive_count = int(positive["event_count"])
        control_count = int(control["event_count"])
        matched_pairs.append(
            {
                "matched_pair_id": f"P{index:03d}",
                "positive_entity_id": positive["entity_id"],
                "control_entity_id": control["entity_id"],
                "positive_event_count": positive_count,
                "control_event_count": control_count,
                "event_count_absolute_difference": abs(
                    positive_count - control_count
                ),
            }
        )
    write_csv(output / "matched_pairs.csv", MATCH_FIELDS, matched_pairs)

    artifact_names = [
        "entities.csv",
        "events.csv",
        "tf_evidence.csv",
        "model_input.csv",
        "model_events.csv",
        "matched_pairs.csv",
        "independent_review_form.csv",
    ]

    status_counts = Counter(row["review_status"] for row in entities)
    positive_event_counts = sorted(
        int(row["event_count"]) for row in entities if int(row["detected"]) == 1
    )
    control_event_counts = sorted(
        int(row["event_count"]) for row in entities if int(row["detected"]) == 0
    )
    detection_delay_hours = sorted(
        round(
            (
                parse_time(row["t_d"]) - parse_time(row["t_f_proxy"])
            ).total_seconds()
            / 3600,
            6,
        )
        for row in entities
        if int(row["detected"]) == 1
    )
    manifest = {
        "dataset": "SafePlus Wikipedia t_f proxy pilot",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "English Wikipedia public MediaWiki Action API",
        "source_endpoint": "https://en.wikipedia.org/w/api.php",
        "counts": {
            "entities": len(entities),
            "detected_positive_candidates": sum(
                int(row["detected"]) for row in entities
            ),
            "right_censored_controls": sum(
                1 for row in entities if int(row["detected"]) == 0
            ),
            "events": len(events),
            "tf_evidence_rows": len(evidence),
            "left_censored_positive_candidates": sum(
                int(row["left_censored"])
                for row in entities
                if int(row["detected"]) == 1
            ),
        },
        "review_status_counts": dict(status_counts),
        "activity_balance": {
            "positive_event_counts": positive_event_counts,
            "control_event_counts": control_event_counts,
            "positive_event_total": sum(positive_event_counts),
            "control_event_total": sum(control_event_counts),
            "positive_event_median": median(positive_event_counts),
            "control_event_median": median(control_event_counts),
            "mean_sorted_pair_absolute_difference": (
                sum(
                    abs(positive - control)
                    for positive, control in zip(
                        positive_event_counts, control_event_counts, strict=True
                    )
                )
                / len(positive_event_counts)
            ),
        },
        "candidate_detection_delay_hours": {
            "values": detection_delay_hours,
            "minimum": min(detection_delay_hours),
            "median": median(detection_delay_hours),
            "maximum": max(detection_delay_hours),
        },
        "semantics": {
            "t_f_proxy": "earliest public revision judged to provide sufficient observable malicious-behavior evidence",
            "t_d_for_detected": "public block-log timestamp",
            "t_d_for_undetected": "right-censoring timestamp, not a detection event",
            "controls": "no observed block in the fixed window; not ground-truth normal labels",
        },
        "leakage_control": {
            "model_input_files": ["model_input.csv", "model_events.csv"],
            "contains_t_f_proxy": False,
            "excluded_from_model_events": [
                "event_type",
                "tags",
                "comment",
                "title",
                "raw_reference",
            ],
            "reason": "exclude onset labels, post-hoc moderation tags, free text, and evidence URLs from the baseline feature view",
        },
        "sha256": {
            name: file_sha256(output / name) for name in artifact_names
        },
        "release_status": (
            "pilot for supervisor review; positive onset candidates require an independent "
            "second annotation before benchmark use"
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the strict Wikipedia t_f pilot tables.")
    parser.add_argument("--blocked-dir", type=Path, required=True)
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = build(
        args.blocked_dir, args.control_dir, args.decisions, args.output
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
