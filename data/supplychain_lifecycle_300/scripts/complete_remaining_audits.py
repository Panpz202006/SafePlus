from __future__ import annotations

import csv
import hashlib
import json
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports"


def read_csv(name: str) -> list[dict[str, str]]:
    with (EXPORTS / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {name}")
    with (EXPORTS / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fetch_npm_registry(package_name: str) -> dict[str, object]:
    url = "https://registry.npmjs.org/" + urllib.parse.quote(package_name, safe="")
    last_error = ""
    for attempt in range(3):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "SafePlus-research-audit/1.0"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                data = json.load(response)
            return {"package_name": package_name, "registry_url": url, "data": data, "error": ""}
        except Exception as exc:  # network errors are recorded rather than hidden
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(0.5 * (attempt + 1))
    return {"package_name": package_name, "registry_url": url, "data": {}, "error": last_error}


def stable_bucket(sample_id: str) -> int:
    return int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16) % 100


def main() -> None:
    reviewed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    samples = read_csv("provisional_samples_300.csv")
    timeline_rows = read_csv("provisional_timeline.csv")
    evidence_rows = read_csv("provisional_evidence.csv")
    candidate_rows = read_csv("candidate_status.csv")
    timeline_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    evidence_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in timeline_rows:
        timeline_by_id[row["sample_id"]].append(row)
    for row in evidence_rows:
        evidence_by_id[row["sample_id"]].append(row)

    # P2: no package release between t_f and proxy t_d.  Detection is an evidence
    # event, so the absence of another release is acceptable when the evidence
    # timestamp and observation endpoint both agree with t_d.
    p2_rows: list[dict[str, object]] = []
    for sample in samples:
        if sample["label"] != "1" or int(sample["post_tf_observation_count"] or 0) != 0:
            continue
        sid = sample["sample_id"]
        timeline = timeline_by_id[sid]
        evidence = evidence_by_id[sid]
        exact_tf = [
            row for row in timeline
            if row["version"] == sample["malicious_version_first"]
            and row["publish_time"] == sample["t_f"]
            and row["retrospective_label"] == "malicious"
        ]
        td_evidence = [row for row in evidence if row["supports_td"] == "1"]
        td_time_match = any(row["report_published_time"] == sample["t_d"] for row in td_evidence)
        checks = {
            "exact_tf_event": bool(exact_tf),
            "tf_before_td": bool(sample["t_f"] < sample["t_d"]),
            "observation_end_equals_td": sample["observation_end"] == sample["t_d"],
            "td_evidence_present": bool(td_evidence),
            "td_evidence_time_matches": td_time_match,
            "post_tf_release_count_is_zero": int(sample["post_tf_observation_count"] or 0) == 0,
        }
        failed = [name for name, passed in checks.items() if not passed]
        decision = "P2_KEEP_DETECTION_EVIDENCE_ONLY" if not failed else "P2_HOLD"
        p2_rows.append(
            {
                "reviewer_id": "R2-P2",
                "reviewed_at": reviewed_at,
                "sample_id": sid,
                "package_name": sample["package_name"],
                "campaign_id": sample["campaign_id"],
                "t_f": sample["t_f"],
                "t_d": sample["t_d"],
                "td_source_type": sample["td_source_type"],
                "td_evidence_count": len(td_evidence),
                "failed_checks": ",".join(failed),
                "decision": decision,
                "include_recommendation": "yes_with_limit" if not failed else "hold",
                "review_note": (
                    "t_f后到代理t_d之间没有新版本；检测由公告证据事件表示，生命周期仍闭合。"
                    if not failed
                    else "t_f、t_d或公告证据之间存在不一致，需要人工处理。"
                ),
                **{f"check_{name}": "yes" if passed else "no" for name, passed in checks.items()},
            }
        )
    write_csv("review_priority_p2_no_post_tf_release.csv", p2_rows)

    # Control audit: compare the exported timeline with current official registry
    # metadata and the complete known-malicious candidate-name set.
    controls = [row for row in samples if row["label"] == "0"]
    malicious_names = {
        (row["ecosystem"].lower(), row["package_name"].lower())
        for row in candidate_rows if row["label"] == "1"
    }
    registry_results: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(fetch_npm_registry, row["package_name"]): row for row in controls}
        for future in as_completed(futures):
            result = future.result()
            registry_results[str(result["package_name"])] = result

    control_rows: list[dict[str, object]] = []
    for sample in controls:
        sid = sample["sample_id"]
        package_name = sample["package_name"]
        timeline = timeline_by_id[sid]
        evidence = evidence_by_id[sid]
        registry_result = registry_results[package_name]
        metadata = registry_result["data"]
        versions = metadata.get("versions", {}) if isinstance(metadata, dict) else {}
        times = metadata.get("time", {}) if isinstance(metadata, dict) else {}
        current_versions = {version for version in versions if version in times}
        timeline_versions = {row["version"] for row in timeline}
        matching_versions = timeline_versions & current_versions
        control_evidence = [
            row for row in evidence if row["evidence_type"] == "normal_control_registry_history"
        ]
        checks = {
            "registry_request_succeeded": not registry_result["error"],
            "timeline_has_three_versions": len(timeline_versions) >= 3,
            "current_registry_has_three_versions": len(current_versions) >= 3,
            "three_versions_cross_match": len(matching_versions) >= 3,
            "no_positive_timeline_label": not any(
                row["retrospective_label"] == "malicious" or row["affected_by_campaign"] == "1"
                for row in timeline
            ),
            "not_in_known_malicious_names": (sample["ecosystem"].lower(), package_name.lower()) not in malicious_names,
            "control_screen_evidence_present": bool(control_evidence),
            "tf_td_empty": not sample["t_f"] and not sample["t_d"],
        }
        failed = [name for name, passed in checks.items() if not passed]
        decision = "CONTROL_KEEP_SCREENED" if not failed else "CONTROL_HOLD"
        control_rows.append(
            {
                "reviewer_id": "R2-CONTROL",
                "reviewed_at": reviewed_at,
                "sample_id": sid,
                "package_name": package_name,
                "registry_url": registry_result["registry_url"],
                "registry_error": registry_result["error"],
                "timeline_version_count": len(timeline_versions),
                "current_registry_version_count": len(current_versions),
                "cross_matched_version_count": len(matching_versions),
                "failed_checks": ",".join(failed),
                "decision": decision,
                "include_recommendation": "yes_with_limit" if not failed else "hold",
                "review_note": (
                    "官方注册表历史、导出时间线和已知恶意包名筛查一致；保留为已筛查正常对照。"
                    if not failed
                    else "注册表或恶意记录交叉检查未完全通过，暂缓纳入。"
                ),
                **{f"check_{name}": "yes" if passed else "no" for name, passed in checks.items()},
            }
        )
    write_csv("review_priority_controls_final.csv", control_rows)

    p2_by_id = {row["sample_id"]: row for row in p2_rows}
    control_by_id = {row["sample_id"]: row for row in control_rows}
    # Outer test is campaign-held-out (Shai-Hulud wave 2).  Remaining rows use
    # a stable hash for train/validation.  This keeps the final test campaign
    # isolated while retaining a practical validation set.
    final_rows: list[dict[str, object]] = []
    for sample in samples:
        sid = sample["sample_id"]
        if sample["label"] == "1":
            p2 = p2_by_id.get(sid)
            include = not p2 or p2["decision"] == "P2_KEEP_DETECTION_EVIDENCE_ONLY"
            label_status = "provisional_b_public_alert_td"
        else:
            control = control_by_id[sid]
            include = control["decision"] == "CONTROL_KEEP_SCREENED"
            label_status = "screened_normal_control"
        if not include:
            split = "excluded"
        elif sample["campaign_id"] == "shai-hulud-wave-2-2025-11":
            split = "test"
        elif sample["label"] == "0" and stable_bucket(sid) < 15:
            split = "test"
        elif sample["label"] == "0" and stable_bucket(sid) < 30:
            split = "validation"
        elif sample["label"] == "1" and stable_bucket(sid) < 25:
            split = "validation"
        else:
            split = "train"
        final_rows.append(
            {
                "sample_id": sid,
                "ecosystem": sample["ecosystem"],
                "package_name": sample["package_name"],
                "label": sample["label"],
                "campaign_id": sample["campaign_id"],
                "t_f": sample["t_f"],
                "t_d": sample["t_d"],
                "label_status": label_status,
                "include": "yes" if include else "no",
                "split": split,
                "split_scheme": "campaign_holdout_outer_test_v1",
                "audit_note": (
                    "t_d为公开公告代理时间" if sample["label"] == "1"
                    else "正常对照经注册表和已知恶意包名交叉筛查"
                ),
            }
        )
    write_csv("model_ready_samples_v1.csv", final_rows)

    p2_counts = Counter(str(row["decision"]) for row in p2_rows)
    control_counts = Counter(str(row["decision"]) for row in control_rows)
    split_counts = Counter(
        (str(row["split"]), str(row["label"])) for row in final_rows if row["include"] == "yes"
    )
    report = {
        "generated_at": reviewed_at,
        "p2_total": len(p2_rows),
        "p2_decisions": dict(p2_counts),
        "control_total": len(control_rows),
        "control_decisions": dict(control_counts),
        "included_total": sum(row["include"] == "yes" for row in final_rows),
        "excluded_total": sum(row["include"] == "no" for row in final_rows),
        "split_counts": {
            f"{split_name}_label_{label}": count
            for (split_name, label), count in sorted(split_counts.items())
        },
        "test_campaigns": sorted(
            {row["campaign_id"] for row in final_rows if row["split"] == "test" and row["label"] == "1"}
        ),
        "limitations": [
            "Malicious t_d values remain public-advisory proxies rather than official platform removal times.",
            "Normal controls are screened controls, not proofs of absolute historical safety.",
            "The outer test isolates Shai-Hulud wave 2; train and validation may share non-test campaigns.",
        ],
    }
    (EXPORTS / "completion_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
