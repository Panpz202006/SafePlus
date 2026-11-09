from __future__ import annotations

import argparse
import csv
import datetime as dt
import email.utils
import hashlib
import json
import os
import pathlib
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable


ROOT = pathlib.Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "supplychain300.sqlite3"
CACHE_DIR = ROOT / "cache"
LOG_DIR = ROOT / "logs"
SOURCE_DIR = ROOT / "sources"
OPENSSF_REPO = SOURCE_DIR / "openssf-malicious-packages"
OPENSSF_PATH_CACHE = CACHE_DIR / "openssf_paths.txt"
SEED_PATH = SOURCE_DIR / "shai_hulud_compromised_packages.txt"
CURATED_SEEDS = SOURCE_DIR / "curated_takeover_seeds.json"
SEED_URL = "https://raw.githubusercontent.com/Cobenian/shai-hulud-detect/main/compromised-packages.txt"
USER_AGENT = "SafePlus-SupplyChain-Dataset/1.0 (research metadata collector)"


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    with connect() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id INTEGER PRIMARY KEY,
                ecosystem TEXT NOT NULL,
                package_name TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                alleged_malicious_version TEXT,
                discovery_source TEXT NOT NULL,
                discovery_url TEXT,
                campaign_id TEXT,
                status TEXT NOT NULL DEFAULT 'discovered',
                exclusion_reason TEXT,
                openssf_path TEXT,
                advisory_id TEXT,
                advisory_published TEXT,
                advisory_source TEXT,
                registry_exists INTEGER,
                registry_version_exists INTEGER,
                normal_prefix_count INTEGER,
                malicious_version_first TEXT,
                t_f TEXT,
                t_d TEXT,
                td_source_type TEXT,
                evidence_grade TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(ecosystem, normalized_name, alleged_malicious_version, discovery_source)
            );

            CREATE TABLE IF NOT EXISTS timeline (
                candidate_id INTEGER NOT NULL REFERENCES candidates(candidate_id),
                ecosystem TEXT NOT NULL,
                package_name TEXT NOT NULL,
                version TEXT NOT NULL,
                publish_time TEXT,
                publish_time_precision TEXT,
                source_url TEXT,
                source_revision_or_hash TEXT,
                publication_status TEXT,
                retrospective_label TEXT NOT NULL DEFAULT 'unknown',
                affected_by_campaign INTEGER,
                evidence_url TEXT,
                PRIMARY KEY(candidate_id, version)
            );

            CREATE TABLE IF NOT EXISTS evidence (
                evidence_id INTEGER PRIMARY KEY,
                candidate_id INTEGER NOT NULL REFERENCES candidates(candidate_id),
                evidence_type TEXT NOT NULL,
                advisory_id TEXT,
                source_organization TEXT,
                primary_url TEXT,
                report_published_time TEXT,
                supported_package_version TEXT,
                malicious_behavior_summary_zh TEXT,
                supports_tf INTEGER NOT NULL DEFAULT 0,
                supports_td INTEGER NOT NULL DEFAULT 0,
                verified_status TEXT NOT NULL DEFAULT 'pending',
                verification_time TEXT,
                raw_sha256 TEXT,
                UNIQUE(candidate_id, evidence_type, primary_url, supported_package_version)
            );

            CREATE TABLE IF NOT EXISTS processing_log (
                log_id INTEGER PRIMARY KEY,
                candidate_id INTEGER,
                stage TEXT NOT NULL,
                outcome TEXT NOT NULL,
                detail TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_candidates_status ON candidates(status);
            CREATE INDEX IF NOT EXISTS idx_candidates_pkg ON candidates(ecosystem, normalized_name);
            """
        )
        existing = {row[1] for row in con.execute("PRAGMA table_info(candidates)")}
        for name, declaration in {
            "label": "INTEGER",
            "sample_id": "TEXT",
            "dataset": "TEXT",
            "project_url": "TEXT",
            "observation_start": "TEXT",
            "observation_end": "TEXT",
            "tf_lower": "TEXT",
            "tf_upper": "TEXT",
            "tf_precision": "TEXT",
            "td_lower": "TEXT",
            "td_upper": "TEXT",
            "first_authoritative_public_alert_time": "TEXT",
            "official_platform_action_time": "TEXT",
            "final_removal_time": "TEXT",
            "early_window_hours": "REAL",
            "post_tf_observation_count": "INTEGER",
            "malicious_behavior_type": "TEXT",
            "anomaly_reason_zh": "TEXT",
            "evidence_url": "TEXT",
            "review_status": "TEXT",
            "audit_notes": "TEXT",
        }.items():
            if name not in existing:
                con.execute(f"ALTER TABLE candidates ADD COLUMN {name} {declaration}")
        con.execute("UPDATE candidates SET label=1 WHERE label IS NULL AND alleged_malicious_version IS NOT NULL")


def normalize_name(ecosystem: str, name: str) -> str:
    value = name.strip()
    if ecosystem.lower() == "pypi":
        return re.sub(r"[-_.]+", "-", value).lower()
    return value.lower()


def http_get(url: str, cache_key: str | None = None, timeout: int = 15) -> bytes:
    if cache_key:
        target = CACHE_DIR / cache_key
        if target.exists():
            return target.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = response.read()
            if cache_key:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            return data
        except urllib.error.HTTPError as exc:
            # Permanent absence is not improved by repeated requests.
            if exc.code in {400, 401, 403, 404, 410}:
                raise RuntimeError(f"HTTP {exc.code}: {url}") from exc
            last_error = exc
            time.sleep(2**attempt)
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"GET failed after retries: {url}: {last_error}")


def fetch_seed() -> None:
    data = http_get(SEED_URL)
    SEED_PATH.write_bytes(data)
    print(f"saved seed list: {SEED_PATH} ({len(data)} bytes)")


def parse_package_version(line: str) -> tuple[str, str] | None:
    text = line.strip()
    if not text or text.startswith("#"):
        return None
    text = re.sub(r"\s+#.*$", "", text).strip()
    if text.startswith("@"):
        pos = text.rfind(":")
        if pos <= text.find("/"):
            return None
        return text[:pos], text[pos + 1 :]
    if ":" not in text:
        return None
    name, version = text.rsplit(":", 1)
    if not name or not version:
        return None
    return name, version


def ingest_seed(campaign_id: str = "shai-hulud") -> None:
    if not SEED_PATH.exists():
        fetch_seed()
    now = utc_now()
    inserted = 0
    with connect() as con, SEED_PATH.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parsed = parse_package_version(line)
            if not parsed:
                continue
            name, version = parsed
            before = con.total_changes
            con.execute(
                """
                INSERT OR IGNORE INTO candidates(
                    ecosystem, package_name, normalized_name, alleged_malicious_version,
                    discovery_source, discovery_url, campaign_id, created_at, updated_at
                ) VALUES('npm', ?, ?, ?, 'community-campaign-list', ?, ?, ?, ?)
                """,
                (name, normalize_name("npm", name), version, SEED_URL, campaign_id, now, now),
            )
            inserted += con.total_changes - before
    print(f"seed candidates inserted: {inserted}")


def ingest_curated_seeds() -> None:
    records = json.loads(CURATED_SEEDS.read_text(encoding="utf-8"))
    now = utc_now()
    inserted = 0
    reused = 0
    with connect() as con:
        for item in records:
            ecosystem = str(item["ecosystem"]).lower()
            name = str(item["package_name"])
            version = str(item["malicious_version"])
            normalized = normalize_name(ecosystem, name)
            existing = con.execute(
                """
                SELECT candidate_id FROM candidates
                WHERE ecosystem=? AND normalized_name=? AND alleged_malicious_version=?
                ORDER BY candidate_id LIMIT 1
                """,
                (ecosystem, normalized, version),
            ).fetchone()
            if existing:
                candidate_id = existing[0]
                reused += 1
                con.execute(
                    """
                    UPDATE candidates SET discovery_source='curated-primary-advisory',
                        discovery_url=?, campaign_id=?, status='advisory_matched', label=1,
                        advisory_id=?, advisory_published=?, advisory_source='curated-primary-advisory',
                        updated_at=? WHERE candidate_id=?
                    """,
                    (
                        item["advisory_url"], item["campaign_id"], item["advisory_id"],
                        item["advisory_published"], now, candidate_id,
                    ),
                )
            else:
                candidate_id = con.execute(
                    """
                    INSERT INTO candidates(
                        ecosystem, package_name, normalized_name, alleged_malicious_version,
                        discovery_source, discovery_url, campaign_id, status, label,
                        advisory_id, advisory_published, advisory_source, created_at, updated_at
                    ) VALUES(?, ?, ?, ?, 'curated-primary-advisory', ?, ?, 'advisory_matched', 1,
                             ?, ?, 'curated-primary-advisory', ?, ?)
                    RETURNING candidate_id
                    """,
                    (
                        ecosystem, name, normalized, version, item["advisory_url"],
                        item["campaign_id"], item["advisory_id"], item["advisory_published"],
                        now, now,
                    ),
                ).fetchone()[0]
                inserted += 1
            con.execute(
                """
                INSERT OR REPLACE INTO evidence(
                    candidate_id, evidence_type, advisory_id, source_organization,
                    primary_url, report_published_time, supported_package_version,
                    malicious_behavior_summary_zh, supports_tf, supports_td,
                    verified_status, verification_time
                ) VALUES(?, 'curated_primary_advisory', ?, 'GitHub/PyPA Advisory Database',
                         ?, ?, ?, ?, 0, 1, 'source_record_matched_pending_time_check', ?)
                """,
                (
                    candidate_id, item["advisory_id"], item["advisory_url"],
                    item["advisory_published"], version, item["behavior_zh"], now,
                ),
            )
    print(f"curated seeds inserted: {inserted}; existing rows reused: {reused}")


def sync_curated_metadata() -> None:
    records = json.loads(CURATED_SEEDS.read_text(encoding="utf-8"))
    updated = 0
    with connect() as con:
        for item in records:
            ecosystem = str(item["ecosystem"]).lower()
            normalized = normalize_name(ecosystem, str(item["package_name"]))
            row = con.execute(
                """
                SELECT * FROM candidates
                WHERE ecosystem=? AND normalized_name=? AND alleged_malicious_version=?
                ORDER BY candidate_id LIMIT 1
                """,
                (ecosystem, normalized, str(item["malicious_version"])),
            ).fetchone()
            if not row:
                continue
            td = item["advisory_published"]
            con.execute(
                """
                UPDATE candidates SET campaign_id=?, advisory_id=?, advisory_published=?,
                    t_d=CASE WHEN status='candidate_B_pending_manual' AND t_f<? THEN ? ELSE t_d END,
                    observation_end=CASE WHEN status='candidate_B_pending_manual' AND t_f<? THEN ? ELSE observation_end END,
                    updated_at=? WHERE candidate_id=?
                """,
                (
                    item["campaign_id"], item["advisory_id"], td,
                    td, td, td, td, utc_now(), row["candidate_id"],
                ),
            )
            con.execute(
                """
                UPDATE evidence SET report_published_time=?
                WHERE candidate_id=? AND evidence_type='curated_primary_advisory'
                """,
                (td, row["candidate_id"]),
            )
            updated += 1
    print(f"curated metadata synchronized: {updated}")


def git_executable() -> str:
    configured = os.environ.get("SAFEPLUS_GIT")
    if configured:
        return configured
    return shutil.which("git") or "git"


def git_text(*args: str) -> str:
    proc = subprocess.run(
        [git_executable(), "-C", str(OPENSSF_REPO), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.stdout


def build_openssf_index() -> dict[tuple[str, str], list[str]]:
    if not OPENSSF_PATH_CACHE.exists():
        output = git_text(
            "ls-tree", "-r", "--name-only", "HEAD", "--", "osv/malicious/npm", "osv/malicious/pypi"
        )
        OPENSSF_PATH_CACHE.write_text(output, encoding="utf-8")
    index: dict[tuple[str, str], list[str]] = {}
    for raw in OPENSSF_PATH_CACHE.read_text(encoding="utf-8").splitlines():
        parts = raw.split("/")
        if len(parts) < 5 or not raw.endswith(".json"):
            continue
        ecosystem = parts[2].lower()
        package = "/".join(parts[3:-1])
        key = (ecosystem, normalize_name(ecosystem, package))
        index.setdefault(key, []).append(raw)
    return index


def advisory_versions(record: dict[str, Any], ecosystem: str, normalized_name: str) -> set[str]:
    versions: set[str] = set()
    for affected in record.get("affected", []):
        pkg = affected.get("package", {})
        pkg_ecosystem = str(pkg.get("ecosystem", "")).lower()
        pkg_name = normalize_name(pkg_ecosystem, str(pkg.get("name", "")))
        if pkg_ecosystem == ecosystem and pkg_name == normalized_name:
            versions.update(str(v) for v in affected.get("versions", []) if v)
    return versions


def attach_openssf(limit: int = 0) -> None:
    index = build_openssf_index()
    done = 0
    with connect() as con:
        rows = con.execute(
            "SELECT * FROM candidates WHERE status IN ('discovered','no_openssf_match') ORDER BY candidate_id"
        ).fetchall()
        for row in rows:
            if limit and done >= limit:
                break
            key = (row["ecosystem"].lower(), row["normalized_name"])
            paths = index.get(key, [])
            matched: tuple[str, dict[str, Any]] | None = None
            for path in paths:
                try:
                    record = json.loads(git_text("show", f"HEAD:{path}"))
                except (subprocess.CalledProcessError, json.JSONDecodeError):
                    continue
                versions = advisory_versions(record, key[0], key[1])
                alleged = row["alleged_malicious_version"]
                if not versions or alleged in versions:
                    matched = (path, record)
                    if alleged in versions:
                        break
            now = utc_now()
            if not matched:
                con.execute(
                    "UPDATE candidates SET status='no_openssf_match', updated_at=? WHERE candidate_id=?",
                    (now, row["candidate_id"]),
                )
                continue
            path, record = matched
            advisory_id = record.get("id")
            refs = record.get("references", [])
            primary_url = next((x.get("url") for x in refs if x.get("url")), None)
            origins = record.get("database_specific", {}).get("malicious-packages-origins", [])
            source = origins[0].get("source") if origins else None
            raw = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
            con.execute(
                """
                UPDATE candidates SET openssf_path=?, advisory_id=?, advisory_published=?,
                    advisory_source=?, status='advisory_matched', updated_at=?
                WHERE candidate_id=?
                """,
                (path, advisory_id, record.get("published"), source, now, row["candidate_id"]),
            )
            con.execute(
                """
                INSERT OR IGNORE INTO evidence(
                    candidate_id, evidence_type, advisory_id, source_organization,
                    primary_url, report_published_time, supported_package_version,
                    malicious_behavior_summary_zh, supports_tf, supports_td,
                    verified_status, verification_time, raw_sha256
                ) VALUES(?, 'malicious_advisory', ?, 'OpenSSF/GitHub Advisory Database',
                         ?, ?, ?, ?, 0, 0, 'source_record_matched', ?, ?)
                """,
                (
                    row["candidate_id"], advisory_id, primary_url, record.get("published"),
                    row["alleged_malicious_version"], record.get("summary"), now,
                    hashlib.sha256(raw).hexdigest(),
                ),
            )
            done += 1
            if done % 5 == 0:
                con.commit()
                print(f"OpenSSF matched: {done}")
    print(f"OpenSSF matching finished: {done}")


def registry_url(ecosystem: str, package_name: str) -> str:
    if ecosystem == "npm":
        return "https://registry.npmjs.org/" + urllib.parse.quote(package_name, safe="")
    return "https://pypi.org/pypi/" + urllib.parse.quote(package_name, safe="") + "/json"


def registry_cache_key(ecosystem: str, normalized_name: str) -> str:
    digest = hashlib.sha256(f"{ecosystem}:{normalized_name}".encode()).hexdigest()
    return f"registry/{ecosystem}/{digest}.json"


def npm_timeline(record: dict[str, Any]) -> list[tuple[str, str | None]]:
    times = record.get("time", {})
    versions = record.get("versions", {})
    # 撤下版本可能不再出现在 versions 中，但官方 time 字典仍保留发布时间。
    names = set(str(v) for v in versions.keys())
    names.update(str(v) for v in times.keys() if v not in {"created", "modified"})
    result = [(version, times.get(version)) for version in names]
    result.sort(key=lambda x: (x[1] is None, x[1] or "", x[0]))
    return result


def pypi_timeline(record: dict[str, Any]) -> list[tuple[str, str | None]]:
    result: list[tuple[str, str | None]] = []
    for version, files in record.get("releases", {}).items():
        dates = [f.get("upload_time_iso_8601") for f in files if f.get("upload_time_iso_8601")]
        result.append((str(version), min(dates) if dates else None))
    result.sort(key=lambda x: (x[1] is None, x[1] or "", x[0]))
    return result


def screen_registry(limit: int = 100) -> None:
    processed = 0
    with connect() as con:
        rows = con.execute(
            """
            SELECT * FROM candidates
            WHERE status IN ('advisory_matched','registry_retry','needs_archived_registry_evidence')
            ORDER BY candidate_id
            """
        ).fetchall()
        for row in rows:
            if limit and processed >= limit:
                break
            candidate_id = row["candidate_id"]
            url = registry_url(row["ecosystem"], row["package_name"])
            cache_key = registry_cache_key(row["ecosystem"], row["normalized_name"])
            now = utc_now()
            try:
                data = http_get(url, cache_key=cache_key)
                record = json.loads(data)
            except Exception as exc:
                con.execute(
                    "UPDATE candidates SET status='registry_retry', exclusion_reason=?, updated_at=? WHERE candidate_id=?",
                    (str(exc)[:500], now, candidate_id),
                )
                continue
            timeline = npm_timeline(record) if row["ecosystem"] == "npm" else pypi_timeline(record)
            alleged = row["alleged_malicious_version"]
            version_map = {version: publish_time for version, publish_time in timeline}
            if alleged not in version_map:
                con.execute(
                    """
                    UPDATE candidates SET registry_exists=1, registry_version_exists=0,
                        status='needs_archived_registry_evidence', exclusion_reason='恶意版本已从当前注册表移除，需可信存档证明发布时间',
                        updated_at=? WHERE candidate_id=?
                    """,
                    (now, candidate_id),
                )
                processed += 1
                continue
            tf = version_map[alleged]
            before = [v for v, published in timeline if published and tf and published < tf]
            prefix_count = len(before)
            active_versions = set(record.get("versions", {}).keys()) if row["ecosystem"] == "npm" else {
                version for version, files in record.get("releases", {}).items() if files
            }
            for version, published in timeline:
                label = "malicious" if version == alleged else ("normal" if published and tf and published < tf else "unknown")
                publication_status = "registry_current" if version in active_versions else "registry_time_only"
                con.execute(
                    """
                    INSERT OR REPLACE INTO timeline(
                        candidate_id, ecosystem, package_name, version, publish_time,
                        publish_time_precision, source_url, publication_status,
                        retrospective_label, affected_by_campaign, evidence_url
                    ) VALUES(?, ?, ?, ?, ?, 'timestamp', ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate_id, row["ecosystem"], row["package_name"], version, published,
                        url, publication_status, label, 1 if version == alleged else 0,
                        "https://github.com/ossf/malicious-packages/blob/main/" + str(row["openssf_path"]),
                    ),
                )
            if not tf:
                status = "exclude_missing_tf"
                reason = "官方注册表记录中恶意版本缺少精确发布时间"
            elif prefix_count < 2:
                status = "exclude_insufficient_normal_prefix"
                reason = f"恶意版本前仅有{prefix_count}个已发布版本，少于要求的2个"
            else:
                status = "needs_td_verification"
                reason = None
            con.execute(
                """
                UPDATE candidates SET registry_exists=1, registry_version_exists=1,
                    normal_prefix_count=?, malicious_version_first=?, t_f=?, status=?,
                    exclusion_reason=?, updated_at=? WHERE candidate_id=?
                """,
                (prefix_count, alleged, tf, status, reason, now, candidate_id),
            )
            con.execute(
                """
                INSERT OR IGNORE INTO evidence(
                    candidate_id, evidence_type, source_organization, primary_url,
                    supported_package_version, supports_tf, supports_td,
                    verified_status, verification_time, raw_sha256
                ) VALUES(?, 'official_registry_metadata', ?, ?, ?, 1, 0,
                         'verified', ?, ?)
                """,
                (
                    candidate_id, "npm registry" if row["ecosystem"] == "npm" else "PyPI",
                    url, alleged, now, hashlib.sha256(data).hexdigest(),
                ),
            )
            processed += 1
            if processed % 5 == 0:
                con.commit()
                print(f"registry screened: {processed}")
    print(f"registry screening finished: {processed}")


def verify_td_proxy(limit: int = 100) -> None:
    """Use an original GitHub malware advisory publication as an explicit B-grade proxy.

    This is deliberately not called an npm removal time. Records still require final
    human review and, where available, replacement by an official platform action.
    """
    processed = 0
    with connect() as con:
        rows = con.execute(
            """
            SELECT * FROM candidates
            WHERE status='needs_td_verification'
            ORDER BY candidate_id
            """
        ).fetchall()
        for row in rows:
            if limit and processed >= limit:
                break
            try:
                record = json.loads(git_text("show", f"HEAD:{row['openssf_path']}"))
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                continue
            origins = record.get("database_specific", {}).get("malicious-packages-origins", [])
            ghsa_origins = [x for x in origins if x.get("source") == "ghsa-malware"]
            ghsa_refs = [
                x.get("url") for x in record.get("references", [])
                if x.get("type") == "ADVISORY" and "github.com/advisories/GHSA-" in str(x.get("url"))
            ]
            td = record.get("published")
            tf = row["t_f"]
            now = utc_now()
            if not ghsa_origins or not ghsa_refs or not td:
                con.execute(
                    "UPDATE candidates SET status='needs_external_td_source', updated_at=? WHERE candidate_id=?",
                    (now, row["candidate_id"]),
                )
                continue
            if not tf or tf >= td:
                con.execute(
                    """
                    UPDATE candidates SET status='exclude_invalid_time_order',
                        exclusion_reason=?, updated_at=? WHERE candidate_id=?
                    """,
                    (f"无法证明 t_f < t_d：t_f={tf}, advisory_published={td}", now, row["candidate_id"]),
                )
                continue
            ghsa_id = ghsa_origins[0].get("id") or next(
                (alias for alias in record.get("aliases", []) if str(alias).startswith("GHSA-")), None
            )
            con.execute(
                """
                UPDATE candidates SET t_d=?, td_source_type='github_advisory_public_alert_proxy',
                    evidence_grade='B', status='candidate_B_pending_manual', updated_at=?
                WHERE candidate_id=?
                """,
                (td, now, row["candidate_id"]),
            )
            con.execute(
                """
                INSERT OR REPLACE INTO evidence(
                    candidate_id, evidence_type, advisory_id, source_organization,
                    primary_url, report_published_time, supported_package_version,
                    malicious_behavior_summary_zh, supports_tf, supports_td,
                    verified_status, verification_time
                ) VALUES(?, 'public_malware_alert_proxy_td', ?, 'GitHub Advisory Database',
                         ?, ?, ?, ?, 0, 1, 'proxy_time_extracted_pending_manual', ?)
                """,
                (
                    row["candidate_id"], ghsa_id, ghsa_refs[0], td,
                    row["alleged_malicious_version"],
                    'GitHub恶意软件公告明确将该版本标为恶意；此时间仅作为B级公开警报代理t_d，不代表npm实际下架时间。',
                    now,
                ),
            )
            processed += 1
            if processed % 5 == 0:
                con.commit()
                print(f"B-grade t_d proxies prepared: {processed}")
    print(f"t_d proxy verification finished: {processed}")


def verify_curated_td() -> None:
    processed = 0
    rejected = 0
    with connect() as con:
        rows = con.execute(
            """
            SELECT * FROM candidates
            WHERE status='needs_td_verification'
              AND discovery_source='curated-primary-advisory'
            ORDER BY candidate_id
            """
        ).fetchall()
        for row in rows:
            tf = row["t_f"]
            td = row["advisory_published"]
            now = utc_now()
            if not tf or not td or tf >= td:
                con.execute(
                    """
                    UPDATE candidates SET status='exclude_invalid_time_order',
                        exclusion_reason=?, updated_at=? WHERE candidate_id=?
                    """,
                    (f"策展公告无法证明 t_f < t_d：t_f={tf}, t_d={td}", now, row["candidate_id"]),
                )
                rejected += 1
                continue
            con.execute(
                """
                UPDATE candidates SET t_d=?, td_source_type='curated_public_advisory_proxy',
                    evidence_grade='B', status='candidate_B_pending_manual',
                    observation_start=(SELECT MIN(publish_time) FROM timeline WHERE candidate_id=?),
                    observation_end=?, updated_at=? WHERE candidate_id=?
                """,
                (td, row["candidate_id"], td, now, row["candidate_id"]),
            )
            con.execute(
                """
                UPDATE evidence SET verified_status='proxy_time_extracted_pending_manual',
                    verification_time=?
                WHERE candidate_id=? AND evidence_type='curated_primary_advisory'
                """,
                (now, row["candidate_id"]),
            )
            processed += 1
    print(f"curated B-grade candidates prepared: {processed}; time-order rejected: {rejected}")


def recover_pypi_artifact_tf() -> None:
    seeds = json.loads(CURATED_SEEDS.read_text(encoding="utf-8"))
    recovered = 0
    with connect() as con:
        for item in seeds:
            artifact_url = item.get("artifact_url")
            if not artifact_url:
                continue
            ecosystem = str(item["ecosystem"]).lower()
            normalized = normalize_name(ecosystem, str(item["package_name"]))
            row = con.execute(
                """
                SELECT * FROM candidates
                WHERE ecosystem=? AND normalized_name=? AND alleged_malicious_version=?
                ORDER BY candidate_id LIMIT 1
                """,
                (ecosystem, normalized, str(item["malicious_version"])),
            ).fetchone()
            if not row:
                continue
            request = urllib.request.Request(artifact_url, method="HEAD", headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=30) as response:
                headers = {key.lower(): value for key, value in response.headers.items()}
            last_modified = headers.get("last-modified")
            file_version = headers.get("x-pypi-file-version")
            file_project = headers.get("x-pypi-file-project")
            if not last_modified or file_version != item["malicious_version"] or normalize_name("pypi", file_project or "") != normalized:
                continue
            tf_dt = email.utils.parsedate_to_datetime(last_modified).astimezone(dt.timezone.utc)
            tf = tf_dt.isoformat(timespec="seconds").replace("+00:00", "Z")
            registry_raw = http_get(
                registry_url(ecosystem, item["package_name"]),
                cache_key=registry_cache_key(ecosystem, normalized),
            )
            registry = json.loads(registry_raw)
            timeline = [(v, t) for v, t in pypi_timeline(registry) if t]
            before = [(v, t) for v, t in timeline if t < tf]
            if len(before) < 2:
                continue
            for version, published in timeline:
                con.execute(
                    """
                    INSERT OR REPLACE INTO timeline(
                        candidate_id, ecosystem, package_name, version, publish_time,
                        publish_time_precision, source_url, publication_status,
                        retrospective_label, affected_by_campaign, evidence_url
                    ) VALUES(?, 'pypi', ?, ?, ?, 'timestamp', ?, 'registry_current',
                             ?, 0, ?)
                    """,
                    (
                        row["candidate_id"], item["package_name"], version, published,
                        registry_url(ecosystem, item["package_name"]),
                        "normal" if published < tf else "unknown", artifact_url,
                    ),
                )
            con.execute(
                """
                INSERT OR REPLACE INTO timeline(
                    candidate_id, ecosystem, package_name, version, publish_time,
                    publish_time_precision, source_url, source_revision_or_hash,
                    publication_status, retrospective_label, affected_by_campaign, evidence_url
                ) VALUES(?, 'pypi', ?, ?, ?, 'timestamp', ?, ?,
                         'official_file_archive_current', 'malicious', 1, ?)
                """,
                (
                    row["candidate_id"], item["package_name"], item["malicious_version"], tf,
                    artifact_url, headers.get("etag"), item["advisory_url"],
                ),
            )
            header_blob = json.dumps(headers, ensure_ascii=False, sort_keys=True).encode("utf-8")
            con.execute(
                """
                INSERT OR REPLACE INTO evidence(
                    candidate_id, evidence_type, advisory_id, source_organization,
                    primary_url, report_published_time, supported_package_version,
                    malicious_behavior_summary_zh, supports_tf, supports_td,
                    verified_status, verification_time, raw_sha256
                ) VALUES(?, 'official_pypi_artifact_headers', ?, 'PyPI files.pythonhosted.org',
                         ?, ?, ?, ?, 1, 0, 'verified', ?, ?)
                """,
                (
                    row["candidate_id"], item["advisory_id"], artifact_url, tf,
                    item["malicious_version"],
                    'PyPI官方文件服务仍可访问该已撤下版本的构件；Last-Modified作为文件级实际发布时间证据。',
                    utc_now(), hashlib.sha256(header_blob).hexdigest(),
                ),
            )
            con.execute(
                """
                UPDATE candidates SET t_f=?, malicious_version_first=?, normal_prefix_count=?,
                    registry_exists=1, registry_version_exists=1,
                    status='needs_td_verification', exclusion_reason=NULL,
                    observation_start=?, updated_at=? WHERE candidate_id=?
                """,
                (tf, item["malicious_version"], len(before), before[0][1], utc_now(), row["candidate_id"]),
            )
            cache_target = CACHE_DIR / "official_file_headers" / f"{normalized}-{item['malicious_version']}.json"
            cache_target.parent.mkdir(parents=True, exist_ok=True)
            cache_target.write_text(json.dumps(headers, ensure_ascii=False, indent=2), encoding="utf-8")
            recovered += 1
    print(f"PyPI artifact t_f recovered: {recovered}")


def deduplicate_packages() -> None:
    """Keep the earliest screened malicious release per package/campaign."""
    with connect() as con:
        groups = con.execute(
            """
            SELECT ecosystem, normalized_name, campaign_id, COUNT(*) AS n
            FROM candidates
            WHERE t_f IS NOT NULL
            GROUP BY ecosystem, normalized_name, campaign_id
            HAVING COUNT(*) > 1
            """
        ).fetchall()
        removed = 0
        for group in groups:
            rows = con.execute(
                """
                SELECT candidate_id, t_f FROM candidates
                WHERE ecosystem=? AND normalized_name=? AND campaign_id=? AND t_f IS NOT NULL
                ORDER BY t_f, candidate_id
                """,
                (group["ecosystem"], group["normalized_name"], group["campaign_id"]),
            ).fetchall()
            for row in rows[1:]:
                con.execute(
                    """
                    UPDATE candidates SET status='exclude_later_malicious_version',
                        exclusion_reason='同一包同一攻击周期仅保留最早恶意版本', updated_at=?
                    WHERE candidate_id=?
                    """,
                    (utc_now(), row["candidate_id"]),
                )
                removed += 1
    print(f"later malicious-version duplicates excluded: {removed}")


def classify_campaigns() -> None:
    """Classify broad seed entries using the wording of their matched advisory."""
    updated = 0
    unresolved = 0
    with connect() as con:
        rows = con.execute(
            """
            SELECT * FROM candidates
            WHERE status='candidate_B_pending_manual' AND openssf_path IS NOT NULL
            ORDER BY candidate_id
            """
        ).fetchall()
        for row in rows:
            try:
                record = json.loads(git_text("show", f"HEAD:{row['openssf_path']}"))
            except (subprocess.CalledProcessError, json.JSONDecodeError):
                unresolved += 1
                continue
            details = str(record.get("details", "")).lower()
            if "second coming" in details or "shai-hulud 2" in details or "sha1-hulud 2" in details:
                campaign = "shai-hulud-wave-2-2025-11"
            elif "shai-hulud npm worm" in details or "shai-hulud" in details:
                campaign = "shai-hulud-wave-1-2025-09"
            else:
                campaign = "campaign-needs-manual-classification"
                unresolved += 1
            con.execute(
                "UPDATE candidates SET campaign_id=?, updated_at=? WHERE candidate_id=?",
                (campaign, utc_now(), row["candidate_id"]),
            )
            updated += 1
    print(f"campaigns classified: {updated}; unresolved: {unresolved}")


def collect_npm_controls(target: int = 100) -> None:
    """Create screened normal-control candidates from npm's official search/metadata APIs."""
    index = build_openssf_index()
    topics = [
        "keywords:javascript", "keywords:typescript", "keywords:testing",
        "keywords:react", "keywords:node", "keywords:cli", "keywords:css",
        "keywords:parser", "keywords:logging", "keywords:utility",
    ]
    names: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    for topic in topics:
        query = urllib.parse.urlencode(
            {"text": topic, "size": 250, "popularity": 1.0, "quality": 0.5, "maintenance": 0.5}
        )
        data = json.loads(http_get("https://registry.npmjs.org/-/v1/search?" + query))
        for obj in data.get("objects", []):
            package = obj.get("package", {})
            name = str(package.get("name", "")).strip()
            normalized = normalize_name("npm", name)
            if not name or normalized in seen:
                continue
            seen.add(normalized)
            names.append((name, package.get("links", {}).get("npm")))
    accepted = 0
    now = utc_now()
    with connect() as con:
        occupied = {
            row[0] for row in con.execute("SELECT normalized_name FROM candidates WHERE ecosystem='npm'")
        }
        for name, project_url in names:
            if accepted >= target:
                break
            normalized = normalize_name("npm", name)
            # A control is rejected if any malicious advisory exists for that package name.
            if normalized in occupied or ("npm", normalized) in index:
                continue
            url = registry_url("npm", name)
            try:
                raw = http_get(url, cache_key=registry_cache_key("npm", normalized))
                record = json.loads(raw)
            except Exception:
                continue
            timeline = [(v, t) for v, t in npm_timeline(record) if t]
            active = set(record.get("versions", {}).keys())
            continuous = [(v, t) for v, t in timeline if v in active]
            if len(continuous) < 3:
                continue
            candidate = con.execute(
                """
                INSERT INTO candidates(
                    ecosystem, package_name, normalized_name, discovery_source,
                    discovery_url, campaign_id, status, label, project_url,
                    observation_start, observation_end, evidence_grade,
                    created_at, updated_at
                ) VALUES('npm', ?, ?, 'npm_official_search', ?, 'normal-control',
                         'control_candidate_pending_manual', 0, ?, ?, ?, 'control_screened', ?, ?)
                RETURNING candidate_id
                """,
                (
                    name, normalized,
                    "https://registry.npmjs.org/-/v1/search",
                    project_url or url,
                    continuous[0][1], continuous[-1][1], now, now,
                ),
            ).fetchone()[0]
            for version, published in continuous:
                con.execute(
                    """
                    INSERT INTO timeline(
                        candidate_id, ecosystem, package_name, version, publish_time,
                        publish_time_precision, source_url, publication_status,
                        retrospective_label, affected_by_campaign, evidence_url
                    ) VALUES(?, 'npm', ?, ?, ?, 'timestamp', ?, 'registry_current',
                             'normal', 0, ?)
                    """,
                    (candidate, name, version, published, url, url),
                )
            con.execute(
                """
                INSERT INTO evidence(
                    candidate_id, evidence_type, source_organization, primary_url,
                    malicious_behavior_summary_zh, supports_tf, supports_td,
                    verified_status, verification_time, raw_sha256
                ) VALUES(?, 'normal_control_registry_history', 'npm registry', ?,
                         '官方注册表存在至少三个连续可访问版本；OpenSSF恶意记录按包名交叉检查未命中。该结论仍待最终人工复核。',
                         0, 0, 'screened_pending_manual', ?, ?)
                """,
                (candidate, url, now, hashlib.sha256(raw).hexdigest()),
            )
            occupied.add(normalized)
            accepted += 1
            if accepted % 5 == 0:
                con.commit()
                print(f"normal controls prepared: {accepted}")
    print(f"normal-control collection finished: {accepted}")


def export_status() -> None:
    out = ROOT / "exports" / "candidate_status.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with connect() as con, out.open("w", encoding="utf-8-sig", newline="") as handle:
        rows = con.execute(
            """
            SELECT candidate_id, label, ecosystem, package_name, alleged_malicious_version,
                   campaign_id, status, exclusion_reason, openssf_path, advisory_id,
                   advisory_published, advisory_source,
                   observation_start, observation_end, normal_prefix_count,
                   t_f, t_d, td_source_type, evidence_grade
            FROM candidates ORDER BY candidate_id
            """
        ).fetchall()
        writer = csv.writer(handle)
        writer.writerow(rows[0].keys() if rows else [])
        writer.writerows([tuple(row) for row in rows])
        counts = con.execute(
            "SELECT status, COUNT(*) AS n FROM candidates GROUP BY status ORDER BY n DESC"
        ).fetchall()
    print(f"exported: {out}")
    for row in counts:
        print(f"{row['status']}: {row['n']}")


def quality_checkpoint() -> None:
    with connect() as con:
        scalar = lambda sql: con.execute(sql).fetchone()[0]
        report = {
            "generated_at": utc_now(),
            "malicious_B_pending_manual": scalar(
                "SELECT COUNT(*) FROM candidates WHERE status='candidate_B_pending_manual'"
            ),
            "malicious_distinct_packages": scalar(
                "SELECT COUNT(DISTINCT ecosystem || ':' || normalized_name) FROM candidates WHERE status='candidate_B_pending_manual'"
            ),
            "malicious_time_order_violations": scalar(
                "SELECT COUNT(*) FROM candidates WHERE status='candidate_B_pending_manual' AND (t_f IS NULL OR t_d IS NULL OR t_f>=t_d)"
            ),
            "malicious_prefix_violations": scalar(
                "SELECT COUNT(*) FROM candidates WHERE status='candidate_B_pending_manual' AND COALESCE(normal_prefix_count,0)<2"
            ),
            "normal_controls_pending_manual": scalar(
                "SELECT COUNT(*) FROM candidates WHERE status='control_candidate_pending_manual'"
            ),
            "normal_controls_with_tf_or_td": scalar(
                "SELECT COUNT(*) FROM candidates WHERE status='control_candidate_pending_manual' AND (t_f IS NOT NULL OR t_d IS NOT NULL)"
            ),
            "normal_controls_under_three_versions": scalar(
                """
                SELECT COUNT(*) FROM candidates c
                WHERE c.status='control_candidate_pending_manual'
                  AND (SELECT COUNT(*) FROM timeline t WHERE t.candidate_id=c.candidate_id)<3
                """
            ),
            "timeline_rows": scalar("SELECT COUNT(*) FROM timeline"),
            "evidence_rows": scalar("SELECT COUNT(*) FROM evidence"),
            "campaign_count_current": scalar(
                "SELECT COUNT(DISTINCT campaign_id) FROM candidates WHERE status='candidate_B_pending_manual'"
            ),
            "pypi_malicious_current": scalar(
                "SELECT COUNT(*) FROM candidates WHERE status='candidate_B_pending_manual' AND ecosystem='pypi'"
            ),
        }
    target = ROOT / "exports" / "quality_checkpoint.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {target}")


def populate_derived_fields() -> None:
    behavior = {
        "shai-hulud-wave-1-2025-09": "凭据窃取与npm蠕虫式自传播",
        "shai-hulud-wave-2-2025-11": "凭据窃取、自传播、GitHub持久化及破坏行为",
        "eslint-account-takeover-2018": "npm令牌窃取",
        "event-stream-flatmap-2018": "定向加密货币钱包凭据窃取",
        "ua-parser-js-account-takeover-2021": "加密货币挖矿与凭据窃取",
        "coa-rc-account-takeover-2021": "安装阶段下载执行与凭据窃取",
        "solana-web3-publisher-compromise-2024": "私钥材料外传",
        "ultralytics-release-artifact-injection-2024": "发布构件注入与加密货币挖矿",
    }
    updated = 0
    with connect() as con:
        rows = con.execute(
            "SELECT * FROM candidates WHERE status='candidate_B_pending_manual' ORDER BY candidate_id"
        ).fetchall()
        for row in rows:
            tf = dt.datetime.fromisoformat(str(row["t_f"]).replace("Z", "+00:00"))
            td = dt.datetime.fromisoformat(str(row["t_d"]).replace("Z", "+00:00"))
            start = con.execute(
                "SELECT MIN(publish_time) FROM timeline WHERE candidate_id=? AND publish_time IS NOT NULL",
                (row["candidate_id"],),
            ).fetchone()[0]
            post_count = con.execute(
                """
                SELECT COUNT(*) FROM timeline
                WHERE candidate_id=? AND publish_time>? AND publish_time<=?
                """,
                (row["candidate_id"], row["t_f"], row["t_d"]),
            ).fetchone()[0]
            evidence_url = con.execute(
                """
                SELECT primary_url FROM evidence
                WHERE candidate_id=? AND supports_td=1 AND primary_url IS NOT NULL
                ORDER BY evidence_id LIMIT 1
                """,
                (row["candidate_id"],),
            ).fetchone()
            reason = behavior.get(row["campaign_id"], "公开恶意软件公告确认的供应链投毒")
            project_url = (
                f"https://www.npmjs.com/package/{row['package_name']}"
                if row["ecosystem"] == "npm"
                else f"https://pypi.org/project/{row['package_name']}/"
            )
            con.execute(
                """
                UPDATE candidates SET dataset='SafePlus_SupplyChain_300',
                    observation_start=?, observation_end=?,
                    tf_lower=t_f, tf_upper=t_f, tf_precision='timestamp',
                    td_lower=t_d, td_upper=t_d,
                    first_authoritative_public_alert_time=t_d,
                    early_window_hours=?, post_tf_observation_count=?,
                    malicious_behavior_type=?, anomaly_reason_zh=?, evidence_url=?,
                    project_url=COALESCE(project_url, ?),
                    review_status='pending_manual_final_review',
                    audit_notes='B级候选：t_d为公开安全公告代理，不等同于平台正式下架时间。',
                    updated_at=? WHERE candidate_id=?
                """,
                (
                    start, row["t_d"], (td - tf).total_seconds() / 3600.0, post_count,
                    reason, f"{reason}；首次恶意版本为 {row['malicious_version_first']}。",
                    evidence_url[0] if evidence_url else row["discovery_url"], project_url,
                    utc_now(), row["candidate_id"],
                ),
            )
            updated += 1
        con.execute(
            """
            UPDATE candidates SET dataset='SafePlus_SupplyChain_300',
                review_status='pending_manual_final_review',
                audit_notes='正常对照候选：至少三个连续版本且包名未命中OpenSSF恶意记录，仍待人工终审。',
                updated_at=? WHERE status='control_candidate_pending_manual'
            """,
            (utc_now(),),
        )
    print(f"derived malicious fields populated: {updated}")


def export_provisional_300() -> None:
    with connect() as con:
        malicious = [dict(row) for row in con.execute(
            "SELECT * FROM candidates WHERE status='candidate_B_pending_manual'"
        ).fetchall()]
        controls = [dict(row) for row in con.execute(
            "SELECT * FROM candidates WHERE status='control_candidate_pending_manual' ORDER BY normalized_name"
        ).fetchall()]
        non_shai = [row for row in malicious if not str(row["campaign_id"]).startswith("shai-hulud")]
        shai = [row for row in malicious if str(row["campaign_id"]).startswith("shai-hulud")]
        shai.sort(
            key=lambda row: hashlib.sha256(
                f"safeplus-selection-v1:{row['ecosystem']}:{row['normalized_name']}:{row['campaign_id']}".encode()
            ).hexdigest()
        )
        needed = 200 - len(non_shai)
        selected_malicious = sorted(non_shai + shai[:needed], key=lambda row: (row["campaign_id"], row["normalized_name"]))
        selected_controls = controls[:100]
        if len(selected_malicious) != 200 or len(selected_controls) != 100:
            raise RuntimeError(
                f"provisional quota unavailable: malicious={len(selected_malicious)}, controls={len(selected_controls)}"
            )
        for number, row in enumerate(selected_malicious, 1):
            row["sample_id"] = f"SC-M-{number:04d}"
        for number, row in enumerate(selected_controls, 1):
            row["sample_id"] = f"SC-N-{number:04d}"
        selected = selected_malicious + selected_controls
        sample_columns = [
            "sample_id", "dataset", "ecosystem", "package_name", "label", "campaign_id",
            "project_url", "observation_start", "observation_end", "malicious_version_first",
            "t_f", "tf_lower", "tf_upper", "tf_precision", "t_d", "td_lower", "td_upper",
            "td_source_type", "first_authoritative_public_alert_time", "official_platform_action_time",
            "final_removal_time", "early_window_hours", "normal_prefix_count",
            "post_tf_observation_count", "malicious_behavior_type", "anomaly_reason_zh",
            "evidence_grade", "evidence_url", "review_status", "audit_notes",
        ]
        sample_path = ROOT / "exports" / "provisional_samples_300.csv"
        with sample_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=sample_columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(selected)
        id_map = {row["candidate_id"]: row["sample_id"] for row in selected}
        timeline_path = ROOT / "exports" / "provisional_timeline.csv"
        with timeline_path.open("w", encoding="utf-8-sig", newline="") as handle:
            columns = [
                "sample_id", "ecosystem", "package_name", "version", "publish_time",
                "publish_time_precision", "source_url", "source_revision_or_hash",
                "publication_status", "retrospective_label", "affected_by_campaign", "evidence_url",
            ]
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for candidate_id, sample_id in id_map.items():
                for item in con.execute("SELECT * FROM timeline WHERE candidate_id=? ORDER BY publish_time, version", (candidate_id,)):
                    value = dict(item)
                    value["sample_id"] = sample_id
                    writer.writerow({column: value.get(column) for column in columns})
        evidence_path = ROOT / "exports" / "provisional_evidence.csv"
        with evidence_path.open("w", encoding="utf-8-sig", newline="") as handle:
            columns = [
                "sample_id", "evidence_type", "advisory_id", "source_organization", "primary_url",
                "report_published_time", "supported_package_version", "malicious_behavior_summary_zh",
                "supports_tf", "supports_td", "verified_status", "verification_time",
            ]
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for candidate_id, sample_id in id_map.items():
                for item in con.execute("SELECT * FROM evidence WHERE candidate_id=? ORDER BY evidence_id", (candidate_id,)):
                    value = dict(item)
                    value["sample_id"] = sample_id
                    writer.writerow({column: value.get(column) for column in columns})
        campaigns: dict[str, int] = {}
        ecosystems: dict[str, int] = {}
        for row in selected_malicious:
            campaigns[row["campaign_id"]] = campaigns.get(row["campaign_id"], 0) + 1
            ecosystems[row["ecosystem"]] = ecosystems.get(row["ecosystem"], 0) + 1
        report = {
            "selection_status": "provisional_pending_manual_final_review",
            "selection_rule": "include all non-Shai-Hulud campaigns, then deterministic SHA256 selection from Shai-Hulud candidates",
            "malicious": len(selected_malicious),
            "normal": len(selected_controls),
            "campaigns": campaigns,
            "malicious_ecosystems": ecosystems,
        }
        report_path = ROOT / "exports" / "provisional_selection_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {sample_path}")
    print(f"saved: {timeline_path}")
    print(f"saved: {evidence_path}")


def validate_provisional() -> None:
    sample_path = ROOT / "exports" / "provisional_samples_300.csv"
    timeline_path = ROOT / "exports" / "provisional_timeline.csv"
    evidence_path = ROOT / "exports" / "provisional_evidence.csv"
    with sample_path.open("r", encoding="utf-8-sig", newline="") as handle:
        samples = list(csv.DictReader(handle))
    with timeline_path.open("r", encoding="utf-8-sig", newline="") as handle:
        timeline = list(csv.DictReader(handle))
    with evidence_path.open("r", encoding="utf-8-sig", newline="") as handle:
        evidence = list(csv.DictReader(handle))
    timeline_counts: dict[str, int] = {}
    evidence_counts: dict[str, int] = {}
    for row in timeline:
        timeline_counts[row["sample_id"]] = timeline_counts.get(row["sample_id"], 0) + 1
    for row in evidence:
        evidence_counts[row["sample_id"]] = evidence_counts.get(row["sample_id"], 0) + 1
    malicious = [row for row in samples if row["label"] == "1"]
    normal = [row for row in samples if row["label"] == "0"]
    checks = {
        "sample_count_is_300": len(samples) == 300,
        "malicious_count_is_200": len(malicious) == 200,
        "normal_count_is_100": len(normal) == 100,
        "sample_id_unique": len({row["sample_id"] for row in samples}) == len(samples),
        "package_unique": len({(row["ecosystem"], normalize_name(row["ecosystem"], row["package_name"])) for row in samples}) == len(samples),
        "malicious_tf_before_td": all(row["t_f"] and row["t_d"] and row["t_f"] < row["t_d"] for row in malicious),
        "malicious_two_normal_prefixes": all(int(row["normal_prefix_count"] or 0) >= 2 for row in malicious),
        "normal_tf_td_null": all(not row["t_f"] and not row["t_d"] for row in normal),
        "normal_three_timeline_rows": all(timeline_counts.get(row["sample_id"], 0) >= 3 for row in normal),
        "every_sample_has_timeline": all(timeline_counts.get(row["sample_id"], 0) > 0 for row in samples),
        "every_sample_has_evidence": all(evidence_counts.get(row["sample_id"], 0) > 0 for row in samples),
    }
    report = {
        "generated_at": utc_now(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "samples": len(samples), "timeline": len(timeline), "evidence": len(evidence),
        },
        "important_limitations": [
            "All rows remain pending manual final review.",
            "Only one malicious PyPI sample currently passes the strict t_f evidence rule.",
            "Most malicious samples belong to two Shai-Hulud waves.",
            "Public-advisory proxy t_d is not the same as official registry removal time.",
        ],
    }
    target = ROOT / "exports" / "provisional_validation.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("provisional validation failed")


def review_provisional_r1() -> None:
    sample_path = ROOT / "exports" / "provisional_samples_300.csv"
    timeline_path = ROOT / "exports" / "provisional_timeline.csv"
    evidence_path = ROOT / "exports" / "provisional_evidence.csv"
    with sample_path.open("r", encoding="utf-8-sig", newline="") as handle:
        samples = list(csv.DictReader(handle))
    with timeline_path.open("r", encoding="utf-8-sig", newline="") as handle:
        timeline_rows = list(csv.DictReader(handle))
    with evidence_path.open("r", encoding="utf-8-sig", newline="") as handle:
        evidence_rows = list(csv.DictReader(handle))
    timeline_by_id: dict[str, list[dict[str, str]]] = {}
    evidence_by_id: dict[str, list[dict[str, str]]] = {}
    for row in timeline_rows:
        timeline_by_id.setdefault(row["sample_id"], []).append(row)
    for row in evidence_rows:
        evidence_by_id.setdefault(row["sample_id"], []).append(row)
    reviews: list[dict[str, Any]] = []
    reviewed_at = utc_now()
    for sample in samples:
        sample_id = sample["sample_id"]
        timeline = timeline_by_id.get(sample_id, [])
        evidence = evidence_by_id.get(sample_id, [])
        is_malicious = sample["label"] == "1"
        checks: dict[str, bool] = {
            "package_identity": bool(sample["ecosystem"] and sample["package_name"] and sample["project_url"]),
            "timeline_present": len(timeline) > 0,
            "evidence_present": len(evidence) > 0,
            "observation_window": bool(sample["observation_start"] and sample["observation_end"] and sample["observation_start"] <= sample["observation_end"]),
        }
        if is_malicious:
            malicious_version = sample["malicious_version_first"]
            exact_timeline = [
                row for row in timeline
                if row["version"] == malicious_version and row["retrospective_label"] == "malicious"
            ]
            prior_normal = [
                row for row in timeline
                if row["retrospective_label"] == "normal" and row["publish_time"] and row["publish_time"] < sample["t_f"]
            ]
            version_evidence = [
                row for row in evidence
                if row["supported_package_version"] == malicious_version
                and row["evidence_type"] in {"malicious_advisory", "curated_primary_advisory"}
            ]
            tf_evidence = [row for row in evidence if row["supports_tf"] == "1" and row["verified_status"] == "verified"]
            td_evidence = [row for row in evidence if row["supports_td"] == "1" and "pending_manual" in row["verified_status"]]
            checks.update(
                {
                    "exact_malicious_version_in_timeline": bool(exact_timeline),
                    "two_prior_normal_versions": len(prior_normal) >= 2,
                    "advisory_matches_version": bool(version_evidence),
                    "tf_source_verified": bool(tf_evidence),
                    "td_proxy_source_present": bool(td_evidence),
                    "tf_before_td": bool(sample["t_f"] and sample["t_d"] and sample["t_f"] < sample["t_d"]),
                    "evidence_grade_allowed": sample["evidence_grade"] in {"A", "B"},
                }
            )
            if all(checks.values()):
                decision = "R1_PASS_B"
                issue = ""
                note = "恶意版本、前序版本、t_f与公告代理t_d证据齐全。t_d仍为B级公开公告代理，不代表平台正式下架时间。"
            else:
                decision = "R1_REJECT"
                issue = ",".join(name for name, passed in checks.items() if not passed)
                note = "至少一项必要证据未通过。"
        else:
            normal_rows = [row for row in timeline if row["retrospective_label"] == "normal"]
            malicious_rows = [row for row in timeline if row["retrospective_label"] == "malicious"]
            control_evidence = [row for row in evidence if row["evidence_type"] == "normal_control_registry_history"]
            checks.update(
                {
                    "three_continuous_versions": len(normal_rows) >= 3,
                    "no_malicious_timeline_label": len(malicious_rows) == 0,
                    "tf_td_empty": not sample["t_f"] and not sample["t_d"],
                    "control_screen_evidence": bool(control_evidence),
                }
            )
            if all(checks.values()):
                decision = "R1_PASS_CONTROL_SCREENED"
                issue = ""
                note = "注册表连续版本与恶意记录交叉排除通过。该结论表示在核查范围内未发现投毒证据，不等于证明包在全部历史中绝对安全。"
            else:
                decision = "R1_REJECT"
                issue = ",".join(name for name, passed in checks.items() if not passed)
                note = "至少一项正常对照条件未通过。"
        reviews.append(
            {
                "reviewer_id": "R1",
                "reviewed_at": reviewed_at,
                "sample_id": sample_id,
                "ecosystem": sample["ecosystem"],
                "package_name": sample["package_name"],
                "label": sample["label"],
                "campaign_id": sample["campaign_id"],
                "malicious_version_first": sample["malicious_version_first"],
                "t_f": sample["t_f"],
                "t_d": sample["t_d"],
                "decision": decision,
                "failed_checks": issue,
                "review_note": note,
                **{f"check_{name}": "yes" if passed else "no" for name, passed in checks.items()},
            }
        )
    review_path = ROOT / "exports" / "review_r1.csv"
    base_columns = [
        "reviewer_id", "reviewed_at", "sample_id", "ecosystem", "package_name",
        "label", "campaign_id", "malicious_version_first", "t_f", "t_d",
        "decision", "failed_checks", "review_note",
    ]
    check_columns = sorted({key for row in reviews for key in row if key.startswith("check_")})
    columns = base_columns + check_columns
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(reviews)
    counts: dict[str, int] = {}
    for row in reviews:
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1
    report = {
        "reviewer_id": "R1",
        "reviewed_at": reviewed_at,
        "total": len(reviews),
        "decisions": counts,
        "rejected_sample_ids": [row["sample_id"] for row in reviews if row["decision"] == "R1_REJECT"],
        "scope": "Evidence-led first review of package identity, timeline, exact affected version, t_f, proxy t_d, and control eligibility.",
    }
    report_path = ROOT / "exports" / "review_r1_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {review_path}")


def review_provisional_r2() -> None:
    """Run a stricter second-pass evidence audit and compare it with R1.

    R2 keeps label validity separate from evidence completeness.  A sample can
    retain its proposed label while being flagged for a thin normal prefix,
    missing post-t_f registry observations, or a control screen that is still
    pending manual confirmation.
    """
    export_dir = ROOT / "exports"

    def read_csv(name: str) -> list[dict[str, str]]:
        with (export_dir / name).open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    samples = read_csv("provisional_samples_300.csv")
    timeline_rows = read_csv("provisional_timeline.csv")
    evidence_rows = read_csv("provisional_evidence.csv")
    r1_rows = read_csv("review_r1.csv")
    timeline_by_id: dict[str, list[dict[str, str]]] = {}
    evidence_by_id: dict[str, list[dict[str, str]]] = {}
    r1_by_id = {row["sample_id"]: row for row in r1_rows}
    for row in timeline_rows:
        timeline_by_id.setdefault(row["sample_id"], []).append(row)
    for row in evidence_rows:
        evidence_by_id.setdefault(row["sample_id"], []).append(row)

    reviewed_at = utc_now()
    reviews: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = sample["sample_id"]
        timeline = timeline_by_id.get(sample_id, [])
        evidence = evidence_by_id.get(sample_id, [])
        is_malicious = sample["label"] == "1"
        hard_checks: dict[str, bool] = {
            "sample_identity": bool(sample["ecosystem"] and sample["package_name"] and sample["project_url"]),
            "timeline_present": bool(timeline),
            "evidence_present": bool(evidence),
            "observation_order": bool(
                sample["observation_start"]
                and sample["observation_end"]
                and sample["observation_start"] <= sample["observation_end"]
            ),
        }
        quality_flags: list[str] = []

        if is_malicious:
            malicious_version = sample["malicious_version_first"]
            exact_rows = [row for row in timeline if row["version"] == malicious_version]
            exact_malicious_rows = [
                row for row in exact_rows
                if row["retrospective_label"] == "malicious" and row["affected_by_campaign"] == "1"
            ]
            prior_rows = [row for row in timeline if row["publish_time"] and row["publish_time"] < sample["t_f"]]
            prior_normal_rows = [row for row in prior_rows if row["retrospective_label"] == "normal"]
            prior_positive_rows = [
                row for row in prior_rows
                if row["retrospective_label"] == "malicious" or row["affected_by_campaign"] == "1"
            ]
            direct_tf_evidence = [
                row for row in evidence
                if row["supports_tf"] == "1"
                and row["supported_package_version"] == malicious_version
                and row["verified_status"] == "verified"
            ]
            td_evidence = [row for row in evidence if row["supports_td"] == "1"]
            hard_checks.update(
                {
                    "exact_first_malicious_version": bool(exact_malicious_rows),
                    "tf_matches_registry_publish_time": bool(
                        exact_malicious_rows
                        and any(row["publish_time"] == sample["t_f"] for row in exact_malicious_rows)
                    ),
                    "no_earlier_positive_version": len(prior_positive_rows) == 0,
                    "at_least_two_prior_normal_versions": len(prior_normal_rows) >= 2,
                    "direct_tf_evidence": bool(direct_tf_evidence),
                    "td_evidence": bool(td_evidence),
                    "tf_before_td": bool(sample["t_f"] and sample["t_d"] and sample["t_f"] < sample["t_d"]),
                }
            )
            if len(prior_normal_rows) < 3:
                quality_flags.append("normal_prefix_only_two_versions")
            if int(sample["post_tf_observation_count"] or 0) == 0:
                quality_flags.append("no_post_tf_registry_version")
            if "proxy" in sample["td_source_type"]:
                quality_flags.append("td_is_public_alert_proxy")
        else:
            normal_rows = [row for row in timeline if row["retrospective_label"] == "normal"]
            positive_rows = [
                row for row in timeline
                if row["retrospective_label"] == "malicious" or row["affected_by_campaign"] == "1"
            ]
            control_evidence = [row for row in evidence if row["evidence_type"] == "normal_control_registry_history"]
            hard_checks.update(
                {
                    "three_normal_versions": len(normal_rows) >= 3,
                    "no_positive_timeline_row": len(positive_rows) == 0,
                    "tf_td_empty": not sample["t_f"] and not sample["t_d"],
                    "control_screen_recorded": bool(control_evidence),
                }
            )
            if any(row["verified_status"] != "verified" for row in control_evidence):
                quality_flags.append("control_screen_pending_manual_confirmation")

        failed_hard = [name for name, passed in hard_checks.items() if not passed]
        if failed_hard:
            decision = "R2_REJECT"
            include_recommendation = "no"
            note = "必要证据条件未通过；暂不进入模型数据。"
        elif quality_flags:
            decision = "R2_PASS_WITH_LIMITATION"
            include_recommendation = "yes_with_limit"
            note = "核心标签链条成立，但存在需要在论文和后续复核中明确的证据限制。"
        else:
            decision = "R2_PASS"
            include_recommendation = "yes"
            note = "核心标签链条与补充质量条件均通过。"

        r1 = r1_by_id.get(sample_id, {})
        r1_included = r1.get("decision", "").startswith("R1_PASS")
        r2_included = include_recommendation != "no"
        reviews.append(
            {
                "reviewer_id": "R2",
                "reviewed_at": reviewed_at,
                "sample_id": sample_id,
                "ecosystem": sample["ecosystem"],
                "package_name": sample["package_name"],
                "label": sample["label"],
                "campaign_id": sample["campaign_id"],
                "malicious_version_first": sample["malicious_version_first"],
                "t_f": sample["t_f"],
                "t_d": sample["t_d"],
                "r1_decision": r1.get("decision", "missing"),
                "r2_decision": decision,
                "include_recommendation": include_recommendation,
                "binary_inclusion_agreement": "yes" if r1_included == r2_included else "no",
                "failed_hard_checks": ",".join(failed_hard),
                "quality_flags": ",".join(quality_flags),
                "review_note": note,
                **{f"check_{name}": "yes" if passed else "no" for name, passed in hard_checks.items()},
            }
        )

    base_columns = [
        "reviewer_id", "reviewed_at", "sample_id", "ecosystem", "package_name", "label",
        "campaign_id", "malicious_version_first", "t_f", "t_d", "r1_decision", "r2_decision",
        "include_recommendation", "binary_inclusion_agreement", "failed_hard_checks",
        "quality_flags", "review_note",
    ]
    check_columns = sorted({key for row in reviews for key in row if key.startswith("check_")})
    review_path = export_dir / "review_r2.csv"
    with review_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_columns + check_columns)
        writer.writeheader()
        writer.writerows(reviews)

    decision_counts: dict[str, int] = {}
    flag_counts: dict[str, int] = {}
    for row in reviews:
        decision_counts[row["r2_decision"]] = decision_counts.get(row["r2_decision"], 0) + 1
        for flag in filter(None, row["quality_flags"].split(",")):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    disagreements = [row for row in reviews if row["binary_inclusion_agreement"] == "no"]
    limited = [row for row in reviews if row["r2_decision"] == "R2_PASS_WITH_LIMITATION"]
    report = {
        "reviewer_id": "R2",
        "reviewed_at": reviewed_at,
        "total": len(reviews),
        "decisions": decision_counts,
        "quality_flags": flag_counts,
        "binary_inclusion_agreement_count": len(reviews) - len(disagreements),
        "binary_inclusion_agreement_rate": (len(reviews) - len(disagreements)) / len(reviews),
        "disagreement_sample_ids": [row["sample_id"] for row in disagreements],
        "limited_sample_ids": [row["sample_id"] for row in limited],
        "scope": "Second-pass cross-check of first malicious version, chronology, direct t_f evidence, proxy t_d evidence, normal-prefix depth, and control-screen status.",
    }
    (export_dir / "review_r2_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    disagreement_columns = [
        "sample_id", "ecosystem", "package_name", "label", "campaign_id", "r1_decision",
        "r2_decision", "include_recommendation", "failed_hard_checks", "quality_flags", "review_note",
    ]
    with (export_dir / "review_r1_r2_disagreements.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=disagreement_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(disagreements)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {review_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("fetch-seed")
    sub.add_parser("ingest-seed")
    sub.add_parser("ingest-curated-seeds")
    sub.add_parser("sync-curated-metadata")
    attach = sub.add_parser("attach-openssf")
    attach.add_argument("--limit", type=int, default=0)
    screen = sub.add_parser("screen-registry")
    screen.add_argument("--limit", type=int, default=100)
    verify = sub.add_parser("verify-td-proxy")
    verify.add_argument("--limit", type=int, default=100)
    sub.add_parser("verify-curated-td")
    sub.add_parser("recover-pypi-artifact-tf")
    sub.add_parser("deduplicate")
    sub.add_parser("classify-campaigns")
    controls = sub.add_parser("collect-npm-controls")
    controls.add_argument("--target", type=int, default=100)
    sub.add_parser("export-status")
    sub.add_parser("quality-checkpoint")
    sub.add_parser("populate-derived-fields")
    sub.add_parser("export-provisional-300")
    sub.add_parser("validate-provisional")
    sub.add_parser("review-provisional-r1")
    sub.add_parser("review-provisional-r2")
    args = parser.parse_args()
    init_db()
    if args.command == "init":
        print(DB_PATH)
    elif args.command == "fetch-seed":
        fetch_seed()
    elif args.command == "ingest-seed":
        ingest_seed()
    elif args.command == "ingest-curated-seeds":
        ingest_curated_seeds()
    elif args.command == "sync-curated-metadata":
        sync_curated_metadata()
    elif args.command == "attach-openssf":
        attach_openssf(args.limit)
    elif args.command == "screen-registry":
        screen_registry(args.limit)
    elif args.command == "verify-td-proxy":
        verify_td_proxy(args.limit)
    elif args.command == "verify-curated-td":
        verify_curated_td()
    elif args.command == "recover-pypi-artifact-tf":
        recover_pypi_artifact_tf()
    elif args.command == "deduplicate":
        deduplicate_packages()
    elif args.command == "classify-campaigns":
        classify_campaigns()
    elif args.command == "collect-npm-controls":
        collect_npm_controls(args.target)
    elif args.command == "export-status":
        export_status()
    elif args.command == "quality-checkpoint":
        quality_checkpoint()
    elif args.command == "populate-derived-fields":
        populate_derived_fields()
    elif args.command == "export-provisional-300":
        export_provisional_300()
    elif args.command == "validate-provisional":
        validate_provisional()
    elif args.command == "review-provisional-r1":
        review_provisional_r1()
    elif args.command == "review-provisional-r2":
        review_provisional_r2()


if __name__ == "__main__":
    main()
