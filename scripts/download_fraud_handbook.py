from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import urllib.request
from pathlib import Path

DEFAULT_COMMIT = "6e67dbd0a3bfe0d7ec33abc4bce5f37cd4ff0d6a"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download_one(day: dt.date, output: Path, commit: str) -> dict:
    name = f"{day.isoformat()}.pkl"
    destination = output / name
    url = (
        "https://raw.githubusercontent.com/Fraud-Detection-Handbook/"
        f"simulated-data-raw/{commit}/data/{name}"
    )
    if not destination.exists() or destination.stat().st_size == 0:
        temporary = destination.with_suffix(".pkl.part")
        request = urllib.request.Request(url, headers={"User-Agent": "safeplus-data-audit"})
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as handle,
        ):
            while block := response.read(1024 * 1024):
                handle.write(block)
        temporary.replace(destination)
    return {"file": name, "bytes": destination.stat().st_size, "sha256": digest(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the official Fraud Detection Handbook data."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2018, 4, 1))
    parser.add_argument("--end", type=dt.date.fromisoformat, default=dt.date(2018, 9, 30))
    parser.add_argument("--commit", default=DEFAULT_COMMIT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.end < args.start:
        raise ValueError("end must not precede start")
    args.output.mkdir(parents=True, exist_ok=True)
    days = [
        args.start + dt.timedelta(days=offset) for offset in range((args.end - args.start).days + 1)
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(lambda day: download_one(day, args.output, args.commit), days))
    records.sort(key=lambda row: row["file"])
    manifest = {
        "repository": "https://github.com/Fraud-Detection-Handbook/simulated-data-raw",
        "commit": args.commit,
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "files": records,
        "total_bytes": sum(row["bytes"] for row in records),
    }
    (args.output / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"files": len(records), "total_bytes": manifest["total_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
