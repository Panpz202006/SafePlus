from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from crawl_wikipedia_blocked_users import (
    USER_AGENT,
    classify_reason,
    entity_hash,
    is_registered_account,
    redact_target,
)


ENTITY_FIELDS = [
    "platform",
    "entity_id",
    "account_type",
    "registration_utc",
    "first_observed_utc",
    "last_observed_utc",
    "observation_end_utc",
    "detected",
    "td_utc",
    "td_type",
    "td_log_id",
    "event_count",
    "history_span_days",
    "history_left_censored",
    "contributions_complete_to_public_start",
    "truncated_by_safety_cap",
    "hidden_or_missing_event_count",
    "collection_status",
    "reason_category_triage_only",
    "block_evidence_url",
]

PRIVATE_FIELDS = ["entity_id", "source_username", "public_user_id"]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class CachedMediaWikiClient:
    def __init__(self, endpoint: str, output: Path, min_delay: float = 0.25):
        self.endpoint = endpoint
        self.cache_dir = output / "raw_api_cache_private"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = output / "crawl_requests.csv"
        self.min_delay = min_delay
        self.request_rows: list[dict] = []

    def _request_id(self, params: dict) -> str:
        canonical = json.dumps(params, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return sha256_bytes(self.endpoint.encode("utf-8") + b"\n" + canonical)

    def get(self, params: dict, retries: int = 5) -> dict:
        query = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
        request_id = self._request_id(query)
        cache_path = self.cache_dir / f"{request_id}.json"
        safe_params = dict(query)
        for key in ("ucuser", "ususers", "letitle"):
            if key in safe_params:
                safe_params[key] = "[PRIVATE_QUERY_VALUE]"
        if cache_path.exists():
            payload = cache_path.read_bytes()
            data = json.loads(payload)
            self.request_rows.append(
                {
                    "request_id": request_id,
                    "endpoint": self.endpoint,
                    "parameters_json": json.dumps(safe_params, ensure_ascii=False, sort_keys=True),
                    "requested_at_utc": utc_now(),
                    "http_status": 200,
                    "retry_count": 0,
                    "from_cache": 1,
                    "response_sha256": sha256_bytes(payload),
                    "cache_path": str(cache_path.relative_to(cache_path.parents[1])),
                    "error_type": "",
                }
            )
            return data

        url = f"{self.endpoint}?{urllib.parse.urlencode(query)}"
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        last_error = ""
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read()
                    status = int(response.status)
                data = json.loads(payload)
                temporary = cache_path.with_suffix(".json.tmp")
                temporary.write_bytes(payload)
                temporary.replace(cache_path)
                self.request_rows.append(
                    {
                        "request_id": request_id,
                        "endpoint": self.endpoint,
                        "parameters_json": json.dumps(safe_params, ensure_ascii=False, sort_keys=True),
                        "requested_at_utc": utc_now(),
                        "http_status": status,
                        "retry_count": attempt,
                        "from_cache": 0,
                        "response_sha256": sha256_bytes(payload),
                        "cache_path": str(cache_path.relative_to(cache_path.parents[1])),
                        "error_type": "",
                    }
                )
                time.sleep(self.min_delay)
                return data
            except urllib.error.HTTPError as error:
                last_error = f"HTTPError:{error.code}"
                if error.code not in {429, 500, 502, 503, 504} or attempt == retries - 1:
                    break
                time.sleep(float(error.headers.get("Retry-After", 2 ** (attempt + 1))))
            except (
                TimeoutError,
                ConnectionError,
                http.client.RemoteDisconnected,
                urllib.error.URLError,
                json.JSONDecodeError,
            ) as error:
                last_error = type(error).__name__
                if attempt == retries - 1:
                    break
                time.sleep(2 ** (attempt + 1))

        self.request_rows.append(
            {
                "request_id": request_id,
                "endpoint": self.endpoint,
                "parameters_json": json.dumps(safe_params, ensure_ascii=False, sort_keys=True),
                "requested_at_utc": utc_now(),
                "http_status": "",
                "retry_count": retries,
                "from_cache": 0,
                "response_sha256": "",
                "cache_path": "",
                "error_type": last_error,
            }
        )
        raise RuntimeError(f"MediaWiki request failed: {last_error}")

    def write_log(self) -> None:
        fields = [
            "request_id",
            "endpoint",
            "parameters_json",
            "requested_at_utc",
            "http_status",
            "retry_count",
            "from_cache",
            "response_sha256",
            "cache_path",
            "error_type",
        ]
        with self.log_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(self.request_rows)


def collect_block_candidates(client: CachedMediaWikiClient, max_users: int) -> list[dict]:
    candidates: list[dict] = []
    seen: set[str] = set()
    continuation: dict = {}
    while len(candidates) < max_users:
        data = client.get(
            {
                "action": "query",
                "list": "logevents",
                "leaction": "block/block",
                "leprop": "ids|title|timestamp|comment|details",
                "lelimit": 100,
                "ledir": "older",
                **continuation,
            }
        )
        for event in data.get("query", {}).get("logevents", []):
            username = event.get("title", "").removeprefix("User:")
            normalized = username.casefold()
            if (
                not is_registered_account(username)
                or normalized in seen
                or "actionhidden" in event
            ):
                continue
            seen.add(normalized)
            candidates.append({"username": username, **event})
            if len(candidates) >= max_users:
                break
        if "continue" not in data:
            break
        continuation = {"lecontinue": data["continue"]["lecontinue"]}
    return candidates


def get_profile(client: CachedMediaWikiClient, username: str) -> dict:
    data = client.get(
        {
            "action": "query",
            "list": "users",
            "ususers": username,
            "usprop": "editcount|registration|blockinfo",
        }
    )
    users = data.get("query", {}).get("users", [])
    return users[0] if users else {}


def collect_full_contributions(
    client: CachedMediaWikiClient,
    username: str,
    observation_end_utc: str,
    safety_cap: int,
) -> tuple[list[dict], bool, bool]:
    events: list[dict] = []
    continuation: dict = {}
    reached_start = False
    while len(events) < safety_cap:
        data = client.get(
            {
                "action": "query",
                "list": "usercontribs",
                "ucuser": username,
                "ucstart": observation_end_utc,
                "ucdir": "older",
                "uclimit": min(500, safety_cap - len(events)),
                "ucprop": "ids|title|timestamp|comment|size|sizediff|flags|tags",
                **continuation,
            }
        )
        events.extend(data.get("query", {}).get("usercontribs", []))
        if "continue" not in data:
            reached_start = True
            break
        continuation = {"uccontinue": data["continue"]["uccontinue"]}
    truncated = not reached_start and len(events) >= safety_cap
    events.sort(key=lambda item: (item.get("timestamp", ""), int(item.get("revid") or -1)))
    return events, reached_start, truncated


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def crawl(output: Path, language: str, max_users: int, safety_cap: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    endpoint = f"https://{language}.wikipedia.org/w/api.php"
    client = CachedMediaWikiClient(endpoint, output)
    candidates = collect_block_candidates(client, max_users)
    entities: list[dict] = []
    private_rows: list[dict] = []
    event_path = output / "events.jsonl"
    event_count = 0
    status_counts: Counter[str] = Counter()

    try:
        with event_path.open("w", encoding="utf-8") as event_file:
            for candidate in candidates:
                username = candidate["username"]
                entity_id = entity_hash(username)
                profile = get_profile(client, username)
                contributions, complete, truncated = collect_full_contributions(
                    client, username, candidate["timestamp"], safety_cap
                )
                registration = profile.get("registration") or ""
                first_observed = contributions[0]["timestamp"] if contributions else ""
                last_observed = contributions[-1]["timestamp"] if contributions else ""
                span_days = (
                    round((parse_utc(last_observed) - parse_utc(first_observed)).total_seconds() / 86400, 6)
                    if first_observed and last_observed
                    else 0.0
                )
                left_censored = int(not registration or not complete)
                status = "collected_complete_public_history" if complete else "collected_truncated"
                if not contributions:
                    status = "no_public_contributions_before_td"
                status_counts[status] += 1
                for index, contribution in enumerate(contributions):
                    record = {
                        "platform": "wikipedia",
                        "entity_id": entity_id,
                        "event_id": contribution.get("revid"),
                        "parent_event_id": contribution.get("parentid"),
                        "page_id": contribution.get("pageid"),
                        "sequence_index": index,
                        "timestamp_utc": contribution.get("timestamp"),
                        "event_type": "published_revision",
                        "title": redact_target(contribution.get("title", ""), username),
                        "comment": redact_target(contribution.get("comment", ""), username),
                        "size": contribution.get("size"),
                        "size_delta": contribution.get("sizediff"),
                        "minor": "minor" in contribution,
                        "new_page": "new" in contribution,
                        "top_revision_at_collection": "top" in contribution,
                        "tags": contribution.get("tags", []),
                        "source_url": f"https://{language}.wikipedia.org/w/index.php?diff={contribution.get('revid')}",
                        "content_fetch_status": "not_requested_stage1_metadata",
                        "collected_at_utc": utc_now(),
                    }
                    event_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                block_log_id = candidate.get("logid", "")
                entities.append(
                    {
                        "platform": "wikipedia",
                        "entity_id": entity_id,
                        "account_type": "registered",
                        "registration_utc": registration,
                        "first_observed_utc": first_observed,
                        "last_observed_utc": last_observed,
                        "observation_end_utc": candidate["timestamp"],
                        "detected": 1,
                        "td_utc": candidate["timestamp"],
                        "td_type": "public_block_log_proxy",
                        "td_log_id": block_log_id,
                        "event_count": len(contributions),
                        "history_span_days": span_days,
                        "history_left_censored": left_censored,
                        "contributions_complete_to_public_start": int(complete),
                        "truncated_by_safety_cap": int(truncated),
                        "hidden_or_missing_event_count": "unknown",
                        "collection_status": status,
                        "reason_category_triage_only": classify_reason(candidate.get("comment", "")),
                        "block_evidence_url": f"https://{language}.wikipedia.org/w/index.php?title=Special:Log&logid={block_log_id}",
                    }
                )
                private_rows.append(
                    {
                        "entity_id": entity_id,
                        "source_username": username,
                        "public_user_id": profile.get("userid", ""),
                    }
                )
                event_count += len(contributions)
    finally:
        client.write_log()

    write_csv(output / "entities.csv", ENTITY_FIELDS, entities)
    private_dir = output / "private_do_not_publish"
    private_dir.mkdir(exist_ok=True)
    write_csv(private_dir / "entity_lookup.csv", PRIVATE_FIELDS, private_rows)
    manifest = {
        "dataset": "SafePlus Wikipedia lifecycle collection pilot",
        "dataset_status": "stage1 metadata collection; no tf labels generated",
        "created_at_utc": utc_now(),
        "source_endpoint": endpoint,
        "counts": {
            "requested_entities": max_users,
            "collected_entities": len(entities),
            "collected_events": event_count,
            "complete_public_histories": sum(int(row["contributions_complete_to_public_start"]) for row in entities),
            "truncated_histories": sum(int(row["truncated_by_safety_cap"]) for row in entities),
        },
        "collection_status_counts": dict(status_counts),
        "boundaries": {
            "start": "earliest retrievable published contribution before td",
            "end": "first selected public block-log event",
            "registration": "stored when MediaWiki publicly returns it",
            "left_censoring": "true when registration is unavailable or pagination did not reach public-history start",
        },
        "safety_cap_events_per_entity": safety_cap,
        "content_policy": "stage1 records metadata for every published revision; revision text/diff is fetched in a later audited stage",
        "label_policy": "block is a td proxy only; neither maliciousness nor tf is inferred by the crawler",
        "private_files": [
            "raw_api_cache_private/",
            "private_do_not_publish/entity_lookup.csv",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect full pre-block Wikipedia account histories.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--language", default="en")
    parser.add_argument("--max-users", type=int, default=20)
    parser.add_argument("--safety-cap", type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps(crawl(args.output, args.language, args.max_users, args.safety_cap), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
