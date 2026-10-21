from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def validate(dataset: Path) -> dict:
    entity_fields, entities = read_csv(dataset / "entities.csv")
    event_fields, events = read_csv(dataset / "events.csv")
    evidence_fields, evidence = read_csv(dataset / "tf_evidence.csv")
    model_fields, model_rows = read_csv(dataset / "model_input.csv")
    model_event_fields, model_events = read_csv(dataset / "model_events.csv")
    match_fields, matched_pairs = read_csv(dataset / "matched_pairs.csv")

    errors: list[str] = []
    entity_by_id = {row["entity_id"]: row for row in entities}
    if len(entity_by_id) != len(entities):
        errors.append("entity_id values are not unique")
    if any(not HASH_PATTERN.fullmatch(row["entity_id"]) for row in entities):
        errors.append("one or more entity IDs are not 64-character lowercase hashes")

    evidence_by_id = {row["entity_id"]: row for row in evidence}
    event_keys = {(row["entity_id"], row["event_id"]): row for row in events}
    if len(event_keys) != len(events):
        errors.append("duplicate entity_id/event_id pairs found")

    for row in entities:
        entity_id = row["entity_id"]
        detected = int(row["detected"])
        t_i = parse_time(row["t_i"])
        t_d = parse_time(row["t_d"])
        if t_i > t_d:
            errors.append(f"{entity_id}: t_i occurs after t_d")
        if detected == 1:
            if not row["t_f_proxy"]:
                errors.append(f"{entity_id}: detected entity has no t_f_proxy")
                continue
            t_f = parse_time(row["t_f_proxy"])
            if not t_i <= t_f <= t_d:
                errors.append(f"{entity_id}: expected t_i <= t_f_proxy <= t_d")
            if entity_id not in evidence_by_id:
                errors.append(f"{entity_id}: detected entity has no evidence row")
        elif detected == 0:
            if row["t_f_proxy"]:
                errors.append(f"{entity_id}: censored control contains t_f_proxy")
            if entity_id in evidence_by_id:
                errors.append(f"{entity_id}: censored control contains t_f evidence")
        else:
            errors.append(f"{entity_id}: detected must be 0 or 1")

    for row in events:
        entity_id = row["entity_id"]
        if entity_id not in entity_by_id:
            errors.append(f"event references unknown entity {entity_id}")
            continue
        if parse_time(row["time"]) > parse_time(entity_by_id[entity_id]["t_d"]):
            errors.append(f"{entity_id}/{row['event_id']}: event occurs after endpoint")
        if not row["raw_reference"].startswith("https://en.wikipedia.org/"):
            errors.append(f"{entity_id}/{row['event_id']}: unexpected source URL")

    for row in evidence:
        entity_id = row["entity_id"]
        key = (entity_id, row["tf_event_id"])
        if key not in event_keys:
            errors.append(f"{entity_id}: t_f event is absent from events.csv")
        elif event_keys[key]["event_type"] != "fraud_evidence_candidate":
            errors.append(f"{entity_id}: t_f event has the wrong event_type")
        for field in ("why_tf", "why_not_before", "why_not_after"):
            if not row[field].strip():
                errors.append(f"{entity_id}: {field} is blank")

    forbidden = {"t_f", "t_f_proxy", "candidate_tf", "tf_event_id"}
    leaked_fields = sorted(forbidden.intersection(model_fields))
    if leaked_fields:
        errors.append(f"model_input.csv leaks fields: {leaked_fields}")
    if len(model_rows) != len(entities):
        errors.append("model_input.csv entity count differs from entities.csv")

    forbidden_event_fields = {
        "t_f",
        "t_f_proxy",
        "candidate_tf",
        "tf_event_id",
        "event_type",
        "tags",
        "comment",
        "title",
        "raw_reference",
    }
    leaked_event_fields = sorted(forbidden_event_fields.intersection(model_event_fields))
    if leaked_event_fields:
        errors.append(f"model_events.csv leaks fields: {leaked_event_fields}")
    if len(model_events) != len(events):
        errors.append("model_events.csv row count differs from events.csv")
    model_event_keys = {
        (row["entity_id"], row["event_id"]) for row in model_events
    }
    if model_event_keys != set(event_keys):
        errors.append("model_events.csv event keys differ from events.csv")
    model_events_by_entity: dict[str, list[dict]] = {}
    for row in model_events:
        model_events_by_entity.setdefault(row["entity_id"], []).append(row)
        if float(row["time_since_t_i_hours"]) < 0:
            errors.append(
                f"{row['entity_id']}/{row['event_id']}: negative time_since_t_i_hours"
            )
        if float(row["delta_previous_event_hours"]) < 0:
            errors.append(
                f"{row['entity_id']}/{row['event_id']}: negative event-time delta"
            )
    for entity_id, timeline in model_events_by_entity.items():
        expected_steps = list(range(len(timeline)))
        actual_steps = [int(row["step_index"]) for row in timeline]
        if actual_steps != expected_steps:
            errors.append(f"{entity_id}: model event step indexes are not contiguous")

    seen_positive: set[str] = set()
    seen_control: set[str] = set()
    for pair in matched_pairs:
        positive_id = pair["positive_entity_id"]
        control_id = pair["control_entity_id"]
        if positive_id in seen_positive or control_id in seen_control:
            errors.append("matched_pairs.csv reuses an entity")
        seen_positive.add(positive_id)
        seen_control.add(control_id)
        if positive_id not in entity_by_id or int(entity_by_id[positive_id]["detected"]) != 1:
            errors.append(f"matched pair has invalid positive entity {positive_id}")
        if control_id not in entity_by_id or int(entity_by_id[control_id]["detected"]) != 0:
            errors.append(f"matched pair has invalid control entity {control_id}")
        if positive_id in entity_by_id and int(pair["positive_event_count"]) != int(
            entity_by_id[positive_id]["event_count"]
        ):
            errors.append(f"matched pair count mismatch for positive {positive_id}")
        if control_id in entity_by_id and int(pair["control_event_count"]) != int(
            entity_by_id[control_id]["event_count"]
        ):
            errors.append(f"matched pair count mismatch for control {control_id}")
    expected_positive = {
        row["entity_id"] for row in entities if int(row["detected"]) == 1
    }
    expected_control = {
        row["entity_id"] for row in entities if int(row["detected"]) == 0
    }
    if seen_positive != expected_positive or seen_control != expected_control:
        errors.append("matched_pairs.csv does not cover every entity exactly once")

    summary = {
        "status": "pass" if not errors else "fail",
        "entities": len(entities),
        "detected_candidates": sum(int(row["detected"]) for row in entities),
        "right_censored_controls": sum(
            1 for row in entities if int(row["detected"]) == 0
        ),
        "events": len(events),
        "evidence_rows": len(evidence),
        "checked_entity_fields": entity_fields,
        "checked_event_fields": event_fields,
        "checked_evidence_fields": evidence_fields,
        "model_input_fields": model_fields,
        "model_event_fields": model_event_fields,
        "matched_pair_fields": match_fields,
        "matched_pairs": len(matched_pairs),
        "errors": errors,
    }
    (dataset / "validation_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if errors:
        raise ValueError("; ".join(errors))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a Wikipedia t_f pilot package.")
    parser.add_argument("dataset", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.dataset), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
