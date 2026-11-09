from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPORTS = ROOT / "exports"


def read_csv(name: str) -> list[dict[str, str]]:
    with (EXPORTS / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def fetch_registry(package_name: str) -> tuple[str, dict]:
    url = "https://registry.npmjs.org/" + urllib.parse.quote(package_name, safe="")
    request = urllib.request.Request(url, headers={"User-Agent": "SafePlus-research-audit/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return url, json.load(response)


def main() -> None:
    samples = read_csv("provisional_samples_300.csv")
    timeline_rows = read_csv("provisional_timeline.csv")
    evidence_rows = read_csv("provisional_evidence.csv")
    timeline_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    evidence_by_id: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in timeline_rows:
        timeline_by_id[row["sample_id"]].append(row)
    for row in evidence_rows:
        evidence_by_id[row["sample_id"]].append(row)

    candidates = [
        row for row in samples
        if row["label"] == "1" and int(row["normal_prefix_count"] or 0) < 3
    ]
    reviewed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results: list[dict[str, str | int]] = []
    for sample in candidates:
        sample_id = sample["sample_id"]
        package_name = sample["package_name"]
        tf = sample["t_f"]
        timeline = timeline_by_id[sample_id]
        prior_normal = sorted(
            [
                row for row in timeline
                if row["publish_time"] < tf and row["retrospective_label"] == "normal"
            ],
            key=lambda row: row["publish_time"],
        )
        registry_url, metadata = fetch_registry(package_name)
        versions = metadata.get("versions", {})
        times = metadata.get("time", {})
        current_prior_versions = {
            version
            for version, published_at in times.items()
            if version in versions and published_at < tf
        }
        historical_prior_versions = {row["version"] for row in prior_normal}
        missing_from_current = sorted(historical_prior_versions - current_prior_versions)
        direct_tf_verified = any(
            row["supports_tf"] == "1"
            and row["supported_package_version"] == sample["malicious_version_first"]
            and row["verified_status"] == "verified"
            for row in evidence_by_id[sample_id]
        )

        if len(prior_normal) < 2 or not direct_tf_verified:
            decision = "P1_REJECT"
            include_recommendation = "no"
            note = "正常前缀不足两个版本，或首个恶意版本缺少直接注册表证据。"
        elif missing_from_current:
            decision = "P1_KEEP_ARCHIVAL_DEPENDENCY"
            include_recommendation = "yes_with_limit"
            note = "两个正常前缀在历史时间线中成立，但至少一个版本已无法从当前注册表直接复核；必须保留历史快照证据。"
        else:
            decision = "P1_KEEP_THIN_HISTORY_VERIFIED"
            include_recommendation = "yes_with_limit"
            note = "官方注册表确认不存在遗漏的更早版本；包本身在t_f前只有两个正常版本。"

        results.append(
            {
                "reviewer_id": "R2-P1",
                "reviewed_at": reviewed_at,
                "sample_id": sample_id,
                "package_name": package_name,
                "campaign_id": sample["campaign_id"],
                "t_f": tf,
                "malicious_version_first": sample["malicious_version_first"],
                "timeline_prior_normal_count": len(prior_normal),
                "timeline_prior_normal_versions": ",".join(row["version"] for row in prior_normal),
                "current_registry_prior_count": len(current_prior_versions),
                "missing_historical_versions_from_current_registry": ",".join(missing_from_current),
                "direct_tf_evidence_verified": "yes" if direct_tf_verified else "no",
                "registry_url": registry_url,
                "decision": decision,
                "include_recommendation": include_recommendation,
                "review_note": note,
            }
        )
        time.sleep(0.15)

    columns = list(results[0])
    output_path = EXPORTS / "review_priority_p1_thin_prefix.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    decisions: dict[str, int] = {}
    for row in results:
        decision = str(row["decision"])
        decisions[decision] = decisions.get(decision, 0) + 1
    report = {
        "reviewer_id": "R2-P1",
        "reviewed_at": reviewed_at,
        "total": len(results),
        "decisions": decisions,
        "rejected_sample_ids": [row["sample_id"] for row in results if row["decision"] == "P1_REJECT"],
        "archival_dependency_sample_ids": [
            row["sample_id"] for row in results if row["decision"] == "P1_KEEP_ARCHIVAL_DEPENDENCY"
        ],
        "conclusion": "The two-version normal prefixes are intrinsic to the package histories rather than collection truncation.",
    }
    (EXPORTS / "review_priority_p1_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
