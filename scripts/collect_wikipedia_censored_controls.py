from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from crawl_wikipedia_blocked_users import (
    api_get,
    collect_contributions,
    entity_hash,
    is_registered_account,
    redact_target,
)

CONTROL_FIELDS = [
    "entity_hash",
    "t_i",
    "censor_time",
    "detected",
    "endpoint_type",
    "event_count",
    "observation_start",
    "observation_end",
    "eligibility_status",
    "eligibility_evidence",
]


def recent_registered_editors(endpoint: str, limit: int = 500) -> list[str]:
    data = api_get(
        endpoint,
        {
            "action": "query",
            "list": "recentchanges",
            "rctype": "edit|new",
            "rcnamespace": 0,
            "rcshow": "!anon|!bot",
            "rclimit": min(limit, 500),
            "rcprop": "user|timestamp|ids|title|comment|tags",
        },
    )
    users: list[str] = []
    seen: set[str] = set()
    for change in data["query"]["recentchanges"]:
        user = change.get("user", "")
        normalized = user.casefold()
        if not is_registered_account(user) or normalized in seen:
            continue
        seen.add(normalized)
        users.append(user)
    return users


def newly_registered_accounts(
    endpoint: str,
    registration_latest: datetime,
    search_days: int = 7,
    limit: int = 500,
) -> list[dict]:
    registration_earliest = registration_latest - timedelta(days=search_days)
    data = api_get(
        endpoint,
        {
            "action": "query",
            "list": "logevents",
            "letype": "newusers",
            "lestart": registration_latest.isoformat().replace("+00:00", "Z"),
            "leend": registration_earliest.isoformat().replace("+00:00", "Z"),
            "ledir": "older",
            "lelimit": min(limit, 500),
            "leprop": "ids|title|timestamp|details",
        },
    )
    accounts: list[dict] = []
    seen: set[str] = set()
    for event in data["query"]["logevents"]:
        username = event.get("title", "").removeprefix("User:")
        normalized = username.casefold()
        if not is_registered_account(username) or normalized in seen:
            continue
        seen.add(normalized)
        accounts.append(
            {"username": username, "registration_time": event["timestamp"]}
        )
    return accounts


def user_profiles(endpoint: str, usernames: list[str]) -> dict[str, dict]:
    profiles: dict[str, dict] = {}
    for offset in range(0, len(usernames), 50):
        batch = usernames[offset : offset + 50]
        data = api_get(
            endpoint,
            {
                "action": "query",
                "list": "users",
                "ususers": "|".join(batch),
                "usprop": "blockinfo|editcount|registration",
            },
        )
        for profile in data["query"]["users"]:
            name = profile.get("name", "")
            profiles[name.casefold()] = profile
        time.sleep(0.1)
    return profiles


def has_block_evidence_in_window(
    endpoint: str, username: str, start: datetime, end: datetime
) -> bool:
    data = api_get(
        endpoint,
        {
            "action": "query",
            "list": "logevents",
            "leaction": "block/block",
            "letitle": f"User:{username}",
            "lestart": end.isoformat().replace("+00:00", "Z"),
            "leend": start.isoformat().replace("+00:00", "Z"),
            "ledir": "older",
            "lelimit": 1,
            "leprop": "ids|timestamp|title|comment|details",
        },
    )
    return bool(data["query"]["logevents"])


def is_currently_blocked(endpoint: str, username: str) -> bool:
    data = api_get(
        endpoint,
        {
            "action": "query",
            "list": "blocks",
            "bkusers": username,
            "bklimit": 1,
            "bkprop": "id|user|timestamp|expiry|reason",
        },
    )
    return bool(data["query"]["blocks"])


def collect_controls(
    output: Path,
    language: str = "en",
    max_controls: int = 10,
    observation_days: int = 30,
    max_contributions: int = 200,
    target_event_counts: list[int] | None = None,
    candidate_limit: int = 100,
) -> dict:
    endpoint = f"https://{language}.wikipedia.org/w/api.php"
    output.mkdir(parents=True, exist_ok=True)
    collection_time = datetime.now(timezone.utc).replace(microsecond=0)
    registration_latest = collection_time - timedelta(days=observation_days)
    candidates = newly_registered_accounts(
        endpoint, registration_latest, limit=max(candidate_limit, 100)
    )
    profiles = user_profiles(
        endpoint, [candidate["username"] for candidate in candidates[:candidate_limit]]
    )

    eligible: list[dict] = []
    for candidate in candidates[:candidate_limit]:
        username = candidate["username"]
        profile = profiles.get(username.casefold(), {})
        edit_count = int(profile.get("editcount") or 0)
        if not 1 <= edit_count <= max_contributions:
            continue
        if "blockid" in profile or "blockedby" in profile:
            continue
        observation_start = datetime.fromisoformat(
            candidate["registration_time"].replace("Z", "+00:00")
        )
        observation_end = observation_start + timedelta(days=observation_days)
        contributions = collect_contributions(
            endpoint,
            username,
            observation_end.isoformat().replace("+00:00", "Z"),
            observation_days,
            max_contributions,
        )
        contributions.sort(key=lambda item: (item["timestamp"], item.get("revid", -1)))
        if not contributions:
            continue
        if has_block_evidence_in_window(endpoint, username, observation_start, observation_end):
            continue
        eligible.append(
            {
                "username": username,
                "observation_start": observation_start,
                "observation_end": observation_end,
                "contributions": contributions,
            }
        )
        time.sleep(0.25)

    targets = target_event_counts or [2] * max_controls
    if len(targets) != max_controls:
        raise ValueError("target_event_counts must contain max_controls values")
    selected: list[dict] = []
    remaining = eligible[:]
    for target in sorted(targets):
        if not remaining:
            break
        best = min(
            remaining,
            key=lambda item: (
                abs(
                    math.log1p(len(item["contributions"]))
                    - math.log1p(target)
                ),
                abs(len(item["contributions"]) - target),
                item["username"].casefold(),
            ),
        )
        selected.append(best)
        remaining.remove(best)

    rows: list[dict] = []
    events: list[dict] = []
    for selected_item in selected:
        username = selected_item["username"]
        observation_start = selected_item["observation_start"]
        observation_end = selected_item["observation_end"]
        contributions = selected_item["contributions"]
        hashed = entity_hash(username)
        for contribution in contributions:
            events.append(
                {
                    "entity_hash": hashed,
                    "revid": contribution.get("revid"),
                    "parentid": contribution.get("parentid"),
                    "pageid": contribution.get("pageid"),
                    "timestamp": contribution.get("timestamp"),
                    "title": redact_target(contribution.get("title", ""), username),
                    "comment": redact_target(
                        contribution.get("comment", ""), username
                    ),
                    "size": contribution.get("size"),
                    "sizediff": contribution.get("sizediff"),
                    "minor": "minor" in contribution,
                    "new": "new" in contribution,
                    "top": "top" in contribution,
                    "tags": contribution.get("tags", []),
                    "diff_url": (
                        f"https://{language}.wikipedia.org/w/index.php?"
                        f"diff={contribution.get('revid')}"
                    ),
                }
            )
        rows.append(
            {
                "entity_hash": hashed,
                "t_i": observation_start.isoformat().replace("+00:00", "Z"),
                "censor_time": observation_end.isoformat().replace("+00:00", "Z"),
                "detected": 0,
                "endpoint_type": "right_censored_no_block_observed",
                "event_count": len(contributions),
                "observation_start": observation_start.isoformat().replace(
                    "+00:00", "Z"
                ),
                "observation_end": observation_end.isoformat().replace(
                    "+00:00", "Z"
                ),
                "eligibility_status": "control_candidate_not_final_normal_label",
                "eligibility_evidence": (
                    "registered non-bot editor; no current block and no public block "
                    f"event observed in the {observation_days}-day window"
                ),
            }
        )

    with (output / "control_entities.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTROL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "control_events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    manifest = {
        "dataset": "Wikipedia right-censored control pilot",
        "status": "control candidates; not verified normal labels",
        "endpoint": endpoint,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "observation_days": observation_days,
        "requested_controls": max_controls,
        "collected_controls": len(rows),
        "collected_events": len(events),
        "eligibility": {
            "registered_only": True,
            "temporary_accounts_excluded": True,
            "currently_blocked_excluded": True,
            "public_block_event_in_window_excluded": True,
            "minimum_public_events": 1,
            "registration_cohort": "accounts created at least one full observation window before collection",
            "activity_matching_targets": targets,
            "activity_matching_method": "greedy nearest neighbor on log1p(event_count)",
        },
        "semantics": {
            "detected": 0,
            "endpoint": "right censoring after a complete 30-day follow-up from public account creation",
            "warning": "absence of observed block is not a ground-truth normal label",
        },
        "privacy": {
            "username_stored": False,
            "entity_key": "SHA-256 of normalized public username",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect right-censored Wikipedia control candidates."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-controls", type=int, default=10)
    parser.add_argument("--observation-days", type=int, default=30)
    parser.add_argument("--max-contributions", type=int, default=200)
    parser.add_argument(
        "--target-event-counts",
        default="",
        help="Comma-separated positive event counts to match, one per control.",
    )
    parser.add_argument("--candidate-limit", type=int, default=100)
    args = parser.parse_args()
    targets = (
        [int(item) for item in args.target_event_counts.split(",") if item]
        if args.target_event_counts
        else None
    )
    manifest = collect_controls(
        args.output,
        language=args.language,
        max_controls=args.max_controls,
        observation_days=args.observation_days,
        max_contributions=args.max_contributions,
        target_event_counts=targets,
        candidate_limit=args.candidate_limit,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
