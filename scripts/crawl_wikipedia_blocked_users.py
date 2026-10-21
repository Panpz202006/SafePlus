from __future__ import annotations

import argparse
import csv
import hashlib
import ipaddress
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

USER_AGENT = (
    "SafePlusDatasetAudit/0.1 "
    "(https://github.com/Panpz202006/SafePlus; academic dataset collection)"
)
REASON_PATTERNS = [
    ("vandalism", re.compile(r"vandal", re.IGNORECASE)),
    ("spam_or_promotion", re.compile(r"spam|promot|advertis|linkspam", re.IGNORECASE)),
    ("sockpuppetry", re.compile(r"sock|block evasion", re.IGNORECASE)),
    ("username_policy", re.compile(r"username|user name", re.IGNORECASE)),
    ("harassment_or_abuse", re.compile(r"harass|personal attack|abuse|threat", re.IGNORECASE)),
]
ANNOTATION_FIELDS = [
    "entity_hash",
    "block_log_id",
    "observed_block_event",
    "t_detect",
    "t_detect_source",
    "block_reason_public",
    "auto_reason_category",
    "event_count",
    "observation_start",
    "observation_end",
    "block_evidence_url",
    "annotation_status",
    "annotator_a_include",
    "annotator_a_onset_revid",
    "annotator_a_onset_time",
    "annotator_a_category",
    "annotator_a_confidence",
    "annotator_a_notes",
    "annotator_b_include",
    "annotator_b_onset_revid",
    "annotator_b_onset_time",
    "annotator_b_category",
    "annotator_b_confidence",
    "annotator_b_notes",
    "adjudicated_label",
    "adjudicated_onset_revid",
    "adjudicated_onset_time",
    "adjudicated_category",
    "adjudicator_notes",
]


def entity_hash(username: str) -> str:
    normalized = " ".join(username.replace("_", " ").split()).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def redact_target(text: str, username: str) -> str:
    result = text or ""
    variants = {username, username.replace("_", " "), username.replace(" ", "_")}
    for variant in sorted(variants, key=len, reverse=True):
        if variant:
            result = re.sub(
                re.escape(variant), "[REDACTED_ENTITY]", result, flags=re.IGNORECASE
            )
    return result


def is_registered_account(target: str) -> bool:
    if not target or target.startswith(("~", "#")):
        return False
    try:
        ipaddress.ip_address(target)
        return False
    except ValueError:
        return "/" not in target


def classify_reason(reason: str) -> str:
    for label, pattern in REASON_PATTERNS:
        if pattern.search(reason or ""):
            return label
    return "other_manual_review"


def api_get(endpoint: str, params: dict, retries: int = 4) -> dict:
    query = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
    request = urllib.request.Request(
        f"{endpoint}?{urllib.parse.urlencode(query)}", headers={"User-Agent": USER_AGENT}
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code not in {429, 503} or attempt == retries - 1:
                raise
            time.sleep(float(error.headers.get("Retry-After", 2 ** (attempt + 1))))
        except (TimeoutError, urllib.error.URLError):
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


def collect_block_candidates(endpoint: str, max_users: int) -> list[dict]:
    candidates = []
    seen = set()
    continuation = {}
    while len(candidates) < max_users:
        data = api_get(
            endpoint,
            {
                "action": "query",
                "list": "logevents",
                "leaction": "block/block",
                "leprop": "ids|title|timestamp|comment|details",
                "lelimit": 100,
                "ledir": "older",
                **continuation,
            },
        )
        for event in data["query"]["logevents"]:
            title = event.get("title", "")
            target = title.removeprefix("User:")
            if (
                not is_registered_account(target)
                or target.casefold() in seen
                or "actionhidden" in event
            ):
                continue
            seen.add(target.casefold())
            candidates.append({"target": target, **event})
            if len(candidates) >= max_users:
                break
        if "continue" not in data:
            break
        continuation = {"lecontinue": data["continue"]["lecontinue"]}
        time.sleep(0.25)
    return candidates


def collect_contributions(
    endpoint: str,
    username: str,
    block_timestamp: str,
    observation_days: int,
    max_contributions: int,
) -> list[dict]:
    block_time = datetime.fromisoformat(block_timestamp.replace("Z", "+00:00"))
    start_time = block_time - timedelta(days=observation_days)
    contributions = []
    continuation = {}
    while len(contributions) < max_contributions:
        limit = min(500, max_contributions - len(contributions))
        data = api_get(
            endpoint,
            {
                "action": "query",
                "list": "usercontribs",
                "ucuser": username,
                "ucstart": block_time.isoformat().replace("+00:00", "Z"),
                "ucend": start_time.isoformat().replace("+00:00", "Z"),
                "ucdir": "older",
                "uclimit": limit,
                "ucprop": "ids|title|timestamp|comment|size|sizediff|flags|tags",
                **continuation,
            },
        )
        contributions.extend(data["query"]["usercontribs"])
        if "continue" not in data:
            break
        continuation = {"uccontinue": data["continue"]["uccontinue"]}
        time.sleep(0.25)
    return contributions


def crawl(
    output: Path,
    language: str = "en",
    max_users: int = 20,
    observation_days: int = 30,
    max_contributions: int = 500,
):
    endpoint = f"https://{language}.wikipedia.org/w/api.php"
    output.mkdir(parents=True, exist_ok=True)
    candidates = collect_block_candidates(endpoint, max_users)
    event_path = output / "events_public.jsonl"
    queue_path = output / "annotation_queue.csv"
    event_temporary = output / "events_public.jsonl.tmp"
    queue_temporary = output / "annotation_queue.csv.tmp"
    rows = []
    event_count = 0
    with event_temporary.open("w", encoding="utf-8") as event_file:
        for index, candidate in enumerate(candidates):
            target = candidate.pop("target")
            hashed = entity_hash(target)
            contributions = collect_contributions(
                endpoint,
                target,
                candidate["timestamp"],
                observation_days,
                max_contributions,
            )
            contributions.sort(key=lambda item: (item["timestamp"], item.get("revid", -1)))
            for contribution in contributions:
                record = {
                    "entity_hash": hashed,
                    "revid": contribution.get("revid"),
                    "parentid": contribution.get("parentid"),
                    "pageid": contribution.get("pageid"),
                    "timestamp": contribution.get("timestamp"),
                    "title": redact_target(contribution.get("title", ""), target),
                    "comment": redact_target(contribution.get("comment", ""), target),
                    "size": contribution.get("size"),
                    "sizediff": contribution.get("sizediff"),
                    "minor": "minor" in contribution,
                    "new": "new" in contribution,
                    "top": "top" in contribution,
                    "tags": contribution.get("tags", []),
                    "diff_url": f"https://{language}.wikipedia.org/w/index.php?diff={contribution.get('revid')}",
                }
                event_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            block_time = datetime.fromisoformat(candidate["timestamp"].replace("Z", "+00:00"))
            rows.append(
                {
                    "entity_hash": hashed,
                    "block_log_id": candidate["logid"],
                    "observed_block_event": 1,
                    "t_detect": candidate["timestamp"],
                    "t_detect_source": "public_block_log",
                    "block_reason_public": redact_target(
                        candidate.get("comment", ""), target
                    ),
                    "auto_reason_category": classify_reason(candidate.get("comment", "")),
                    "event_count": len(contributions),
                    "observation_start": (block_time - timedelta(days=observation_days)).isoformat(),
                    "observation_end": block_time.isoformat(),
                    "block_evidence_url": (
                        f"https://{language}.wikipedia.org/w/index.php?"
                        f"title=Special:Log&logid={candidate['logid']}"
                    ),
                    "annotation_status": (
                        "exclude_no_public_revision_evidence"
                        if not contributions
                        else "pending_policy_relevance_review"
                        if classify_reason(candidate.get("comment", ""))
                        == "username_policy"
                        else "pending_double_review"
                    ),
                }
            )
            event_count += len(contributions)
            if index + 1 < len(candidates):
                time.sleep(0.25)

    with queue_temporary.open("w", encoding="utf-8-sig", newline="") as queue_file:
        writer = csv.DictWriter(queue_file, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    event_temporary.replace(event_path)
    queue_temporary.replace(queue_path)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "dataset": "Wikipedia blocked-account annotation pilot",
        "status": "candidate collection only; no final labels",
        "endpoint": endpoint,
        "retrieved_at_utc": retrieved_at,
        "collection": {
            "block_action": "block/block",
            "registered_accounts_only": True,
            "observation_days_before_block": observation_days,
            "max_contributions_per_account": max_contributions,
            "requested_accounts": max_users,
            "collected_accounts": len(rows),
            "collected_events": event_count,
            "request_concurrency": 1,
            "minimum_delay_seconds": 0.25,
        },
        "semantics": {
            "t_detect": "public block-log timestamp",
            "t_onset": "blank until two independent annotators review revision evidence",
            "auto_reason_category": "triage aid only, not a final label",
        },
        "privacy": {
            "username_stored": False,
            "entity_key": "SHA-256 of normalized public username",
            "ip_and_temporary_accounts_excluded": True,
        },
    }
    manifest_temporary = output / "manifest.json.tmp"
    manifest_temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_temporary.replace(output / "manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Collect Wikipedia block candidates for annotation.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-users", type=int, default=20)
    parser.add_argument("--observation-days", type=int, default=30)
    parser.add_argument("--max-contributions", type=int, default=500)
    args = parser.parse_args()
    result = crawl(
        args.output,
        args.language,
        args.max_users,
        args.observation_days,
        args.max_contributions,
    )
    print(json.dumps(result["collection"], indent=2))


if __name__ == "__main__":
    main()
