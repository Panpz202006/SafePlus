from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from crawl_wikipedia_blocked_users import (
    classify_reason,
    entity_hash,
    redact_target,
)
from crawl_wikipedia_lifecycle import (
    CachedMediaWikiClient,
    collect_block_candidates,
    collect_full_contributions,
    get_profile,
    parse_utc,
    utc_now,
    write_csv,
)


AUTO_RELEVANT_CATEGORIES = {
    "vandalism",
    "spam_or_promotion",
    "sockpuppetry",
    "harassment_or_abuse",
}

ENTITY_FIELDS = [
    "platform",
    "entity_id",
    "registration_utc",
    "first_observed_utc",
    "last_observed_utc",
    "observation_end_utc",
    "detected",
    "td_candidate_utc",
    "td_candidate_log_id",
    "td_candidate_reason_category",
    "td_selection_method",
    "td_verification_status",
    "published_revision_count",
    "public_actor_log_count",
    "total_public_event_count",
    "history_span_days",
    "reached_public_revision_start",
    "revision_history_truncated",
    "registration_available",
    "history_left_censored",
    "collection_status",
]


def collect_target_block_history(client: CachedMediaWikiClient, username: str) -> list[dict]:
    rows: list[dict] = []
    continuation: dict = {}
    while True:
        data = client.get(
            {
                "action": "query",
                "list": "logevents",
                "leaction": "block/block",
                "letitle": f"User:{username}",
                "ledir": "newer",
                "lelimit": 500,
                "leprop": "ids|title|timestamp|comment|details|user",
                **continuation,
            }
        )
        rows.extend(data.get("query", {}).get("logevents", []))
        if "continue" not in data:
            break
        continuation = {"lecontinue": data["continue"]["lecontinue"]}
    rows.sort(key=lambda item: (item.get("timestamp", ""), int(item.get("logid") or -1)))
    return rows


def select_first_related_block(blocks: list[dict]) -> tuple[dict | None, str]:
    for block in blocks:
        category = classify_reason(block.get("comment", ""))
        if category in AUTO_RELEVANT_CATEGORIES:
            return block, "earliest_auto_relevant_public_block_reason"
    if blocks:
        return blocks[0], "earliest_public_block_pending_relevance_review"
    return None, "no_public_block_found"


def collect_actor_logs_before(
    client: CachedMediaWikiClient,
    username: str,
    end_utc: str,
    safety_cap: int = 5000,
) -> tuple[list[dict], bool]:
    rows: list[dict] = []
    continuation: dict = {}
    while len(rows) < safety_cap:
        data = client.get(
            {
                "action": "query",
                "list": "logevents",
                "leuser": username,
                "lestart": end_utc,
                "ledir": "older",
                "lelimit": min(500, safety_cap - len(rows)),
                "leprop": "ids|title|type|user|timestamp|comment|details",
                **continuation,
            }
        )
        rows.extend(data.get("query", {}).get("logevents", []))
        if "continue" not in data:
            break
        continuation = {"lecontinue": data["continue"]["lecontinue"]}
    truncated = "continue" in data and len(rows) >= safety_cap
    rows.sort(key=lambda item: (item.get("timestamp", ""), int(item.get("logid") or -1)))
    return rows, truncated


def crawl(
    output: Path,
    language: str,
    target_accounts: int,
    candidate_pool: int,
    candidate_offset: int,
    revision_safety_cap: int,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    endpoint = f"https://{language}.wikipedia.org/w/api.php"
    client = CachedMediaWikiClient(endpoint, output)
    seed_pool = collect_block_candidates(
        client,
        max(candidate_offset + candidate_pool, candidate_offset + target_accounts),
    )
    seeds = seed_pool[candidate_offset : candidate_offset + candidate_pool]
    entities: list[dict] = []
    private_rows: list[dict] = []
    event_rows: list[dict] = []
    action_rows: list[dict] = []
    skipped: Counter[str] = Counter()

    try:
        for seed in seeds:
            if len(entities) >= target_accounts:
                break
            username = seed["username"]
            blocks = collect_target_block_history(client, username)
            td, selection_method = select_first_related_block(blocks)
            if td is None:
                skipped["no_public_block_history"] += 1
                continue
            category = classify_reason(td.get("comment", ""))
            profile = get_profile(client, username)
            revisions, reached_start, revisions_truncated = collect_full_contributions(
                client, username, td["timestamp"], revision_safety_cap
            )
            actor_logs, actor_logs_truncated = collect_actor_logs_before(
                client, username, td["timestamp"]
            )
            if not revisions and not actor_logs:
                skipped["no_public_activity_before_td"] += 1
                continue

            entity_id = entity_hash(username)
            registration = profile.get("registration") or ""
            combined: list[dict] = []
            for revision in revisions:
                combined.append(
                    {
                        "platform": "wikipedia",
                        "entity_id": entity_id,
                        "source_type": "published_revision",
                        "source_event_id": str(revision.get("revid") or ""),
                        "parent_event_id": str(revision.get("parentid") or ""),
                        "timestamp_utc": revision.get("timestamp", ""),
                        "page_or_target_id": str(revision.get("pageid") or ""),
                        "title": redact_target(revision.get("title", ""), username),
                        "comment": redact_target(revision.get("comment", ""), username),
                        "size": revision.get("size", ""),
                        "size_delta": revision.get("sizediff", ""),
                        "minor": int("minor" in revision),
                        "new_page": int("new" in revision),
                        "tags_json": json.dumps(revision.get("tags", []), ensure_ascii=False),
                        "source_url": f"https://{language}.wikipedia.org/w/index.php?diff={revision.get('revid')}",
                        "content_fetch_status": "not_requested_stage1",
                    }
                )
            for log in actor_logs:
                combined.append(
                    {
                        "platform": "wikipedia",
                        "entity_id": entity_id,
                        "source_type": f"public_log:{log.get('type', 'unknown')}",
                        "source_event_id": str(log.get("logid") or ""),
                        "parent_event_id": "",
                        "timestamp_utc": log.get("timestamp", ""),
                        "page_or_target_id": str(log.get("pageid") or ""),
                        "title": redact_target(log.get("title", ""), username),
                        "comment": redact_target(log.get("comment", ""), username),
                        "size": "",
                        "size_delta": "",
                        "minor": "",
                        "new_page": "",
                        "tags_json": "[]",
                        "source_url": f"https://{language}.wikipedia.org/w/index.php?title=Special:Log&logid={log.get('logid')}",
                        "content_fetch_status": "not_applicable_public_log",
                    }
                )
            combined.sort(
                key=lambda item: (
                    item["timestamp_utc"],
                    item["source_type"],
                    item["source_event_id"],
                )
            )
            if not registration:
                skipped["registration_unavailable"] += 1
                continue
            if not reached_start or revisions_truncated or actor_logs_truncated:
                skipped["incomplete_public_history"] += 1
                continue
            if parse_utc(combined[0]["timestamp_utc"]) < parse_utc(registration):
                skipped["activity_before_reported_registration"] += 1
                continue
            for sequence_index, event in enumerate(combined):
                event["sequence_index"] = sequence_index
                event_rows.append(event)

            first_observed = combined[0]["timestamp_utc"]
            last_observed = combined[-1]["timestamp_utc"]
            history_span_days = round(
                (parse_utc(td["timestamp"]) - parse_utc(first_observed)).total_seconds()
                / 86400,
                6,
            )
            registration_available = 1
            left_censored = 0
            verification = (
                "pending_human_first_related_block_confirmation"
                if category in AUTO_RELEVANT_CATEGORIES
                else "pending_human_relevance_review"
            )
            entities.append(
                {
                    "platform": "wikipedia",
                    "entity_id": entity_id,
                    "registration_utc": registration,
                    "first_observed_utc": first_observed,
                    "last_observed_utc": last_observed,
                    "observation_end_utc": td["timestamp"],
                    "detected": 1,
                    "td_candidate_utc": td["timestamp"],
                    "td_candidate_log_id": td.get("logid", ""),
                    "td_candidate_reason_category": category,
                    "td_selection_method": selection_method,
                    "td_verification_status": verification,
                    "published_revision_count": len(revisions),
                    "public_actor_log_count": len(actor_logs),
                    "total_public_event_count": len(combined),
                    "history_span_days": history_span_days,
                    "reached_public_revision_start": int(reached_start),
                    "revision_history_truncated": int(revisions_truncated),
                    "registration_available": registration_available,
                    "history_left_censored": left_censored,
                    "collection_status": (
                        "collected_truncated"
                        if revisions_truncated or actor_logs_truncated
                        else "collected_stage1_complete_public_sources"
                    ),
                }
            )
            private_rows.append(
                {
                    "entity_id": entity_id,
                    "source_username": username,
                    "public_user_id": profile.get("userid", ""),
                }
            )
            for block in blocks:
                action_rows.append(
                    {
                        "entity_id": entity_id,
                        "action_type": "public_block_log",
                        "action_time_utc": block.get("timestamp", ""),
                        "action_log_id": block.get("logid", ""),
                        "action_reason_public": redact_target(
                            block.get("comment", ""), username
                        ),
                        "reason_category_triage_only": classify_reason(block.get("comment", "")),
                        "is_selected_td_candidate": int(block.get("logid") == td.get("logid")),
                        "verification_status": "pending_human_review",
                        "source_url": f"https://{language}.wikipedia.org/w/index.php?title=Special:Log&logid={block.get('logid')}",
                    }
                )
    finally:
        client.write_log()

    write_csv(output / "entities.csv", ENTITY_FIELDS, entities)
    event_fields = [
        "platform", "entity_id", "sequence_index", "source_type", "source_event_id",
        "parent_event_id", "timestamp_utc", "page_or_target_id", "title", "comment",
        "size", "size_delta", "minor", "new_page", "tags_json", "source_url",
        "content_fetch_status",
    ]
    write_csv(output / "events.csv", event_fields, event_rows)
    action_fields = [
        "entity_id", "action_type", "action_time_utc", "action_log_id",
        "action_reason_public", "reason_category_triage_only", "is_selected_td_candidate",
        "verification_status", "source_url",
    ]
    write_csv(output / "platform_actions.csv", action_fields, action_rows)
    selected_actions = {
        row["entity_id"]: row
        for row in action_rows
        if int(row["is_selected_td_candidate"]) == 1
    }
    td_review_rows = []
    for entity in entities:
        action = selected_actions[entity["entity_id"]]
        td_review_rows.append(
            {
                "entity_id": entity["entity_id"],
                "registration_utc": entity["registration_utc"],
                "first_observed_utc": entity["first_observed_utc"],
                "td_candidate_utc": entity["td_candidate_utc"],
                "td_candidate_log_id": entity["td_candidate_log_id"],
                "reason_category_triage_only": entity["td_candidate_reason_category"],
                "action_reason_public": action["action_reason_public"],
                "block_evidence_url": action["source_url"],
                "reviewer_id": "",
                "is_first_related_block": "",
                "final_td_utc": "",
                "decision": "",
                "decision_notes": "",
            }
        )
    write_csv(
        output / "td_review_queue.csv",
        [
            "entity_id", "registration_utc", "first_observed_utc",
            "td_candidate_utc", "td_candidate_log_id",
            "reason_category_triage_only", "action_reason_public",
            "block_evidence_url", "reviewer_id", "is_first_related_block",
            "final_td_utc", "decision", "decision_notes",
        ],
        td_review_rows,
    )
    private_dir = output / "private_do_not_publish"
    private_dir.mkdir(exist_ok=True)
    write_csv(
        private_dir / "entity_lookup.csv",
        ["entity_id", "source_username", "public_user_id"],
        private_rows,
    )
    manifest = {
        "dataset": "SafePlus Wikipedia full account lifecycle v1",
        "basis": "SafePlus-main canonical account-sequence contract",
        "created_at_utc": utc_now(),
        "stage": "raw_account_lifecycle_stage1",
        "counts": {
            "candidate_offset": candidate_offset,
            "candidate_accounts_scanned": len(seeds),
            "collected_accounts": len(entities),
            "public_events": len(event_rows),
            "public_block_actions": len(action_rows),
            "skipped": dict(skipped),
        },
        "observation_contract": {
            "start": "public registration when available plus earliest retrievable public activity",
            "end": "earliest automatically relevant public block candidate, pending human verification",
            "post_td_events_included": False,
            "tf_generated": False,
            "model_ready": False,
        },
        "next_required_stages": [
            "human verification of first relevant block td",
            "matched right-censored lifecycle controls",
            "revision diff/content acquisition",
            "causal feature extraction and temporal binning",
            "conversion to x/lengths/detected/t_detect/t_onset/entity_id NPZ",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect complete pre-block Wikipedia account lifecycles.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--language", default="en")
    parser.add_argument("--target-accounts", type=int, default=10)
    parser.add_argument("--candidate-pool", type=int, default=100)
    parser.add_argument("--candidate-offset", type=int, default=0)
    parser.add_argument("--revision-safety-cap", type=int, default=10000)
    args = parser.parse_args()
    print(
        json.dumps(
            crawl(
                args.output,
                args.language,
                args.target_accounts,
                args.candidate_pool,
                args.candidate_offset,
                args.revision_safety_cap,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
