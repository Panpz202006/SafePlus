from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_split(source: Path, split: str, suffix: str):
    x = np.load(
        source / f"X_{split}{suffix}.npy",
        allow_pickle=True,
        encoding="latin1",
    )
    lengths = np.load(source / f"T_{split}{suffix}.npy").astype(np.int64)
    event = np.load(source / f"C_{split}{suffix}.npy").astype(np.int64)
    return x, lengths, event


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapt the legacy SAFE Twitter/Wiki arrays to SafePlus NPZ files."
    )
    parser.add_argument("--dataset", choices=["twitter", "wiki"], required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-label",
        default="https://github.com/PanpanZheng/SAFE",
        help="Logical source recorded in manifest.json; never stores a local path.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    suffix = "_var" if args.dataset == "twitter" else ""
    loaded = {
        split: load_split(args.source, split, suffix)
        for split in ("train", "valid", "test")
    }
    max_length = max(int(lengths.max()) for _, lengths, _ in loaded.values())
    feature_dim = int(np.asarray(loaded["train"][0][0]).shape[1])

    offset = 0
    stats = {}
    for split, (x_obj, lengths, event) in loaded.items():
        padded = np.zeros((len(x_obj), max_length, feature_dim), dtype=np.float32)
        for index, row in enumerate(x_obj):
            row = np.asarray(row, dtype=np.float32)
            if len(row) != lengths[index]:
                raise ValueError(f"{split}[{index}] sequence length does not match T")
            padded[index, : len(row)] = row

        detected = event.astype(bool)
        t_detect = np.where(detected, lengths - 1, -1).astype(np.int64)
        t_onset = np.full(len(x_obj), -1, dtype=np.int64)
        entity_id = np.arange(offset, offset + len(x_obj), dtype=np.int64)
        offset += len(x_obj)

        np.savez_compressed(
            args.output / f"{split}.npz",
            x=padded,
            lengths=lengths,
            detected=detected,
            t_detect=t_detect,
            t_onset=t_onset,
            entity_id=entity_id,
        )
        stats[split] = {
            "n": len(x_obj),
            "events": int(event.sum()),
            "censored": int((1 - event).sum()),
            "event_rate": float(event.mean()),
            "length_min": int(lengths.min()),
            "length_max": int(lengths.max()),
            "feature_dim": feature_dim,
        }

    manifest = {
        "dataset": args.dataset,
        "source": args.source_label,
        "redistribution": "raw and adapted arrays are not included",
        "max_length": max_length,
        "feature_dim": feature_dim,
        "splits": stats,
        "semantics": {
            "C=1": "platform suspension/event observed at T",
            "C=0": "right-censored at last observation T",
            "t_detect": "zero-based T-1 for C=1; -1 for censored",
            "t_onset": "unknown (-1) for every entity; must not be used as ground truth",
        },
        "source_sha256": {
            path.name: digest(path) for path in sorted(args.source.glob("*.npy"))
        },
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
