from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_FEATURES = [
    "log1p_tx_count",
    "log1p_amount_sum",
    "log1p_amount_mean",
    "log1p_amount_max",
    "log1p_unique_merchants",
]


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _clean_string_column(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.strip("'")


def stratified_indices(labels: np.ndarray, ratios: tuple[float, float, float], seed: int):
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("split ratios must sum to one")
    rng = np.random.default_rng(seed)
    parts: list[list[int]] = [[], [], []]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        cut1 = int(len(indices) * ratios[0])
        cut2 = cut1 + int(len(indices) * ratios[1])
        for destination, values in zip(parts, (indices[:cut1], indices[cut1:cut2], indices[cut2:])):
            destination.extend(values.tolist())
    result = []
    for values in parts:
        values_array = np.asarray(values, dtype=np.int64)
        rng.shuffle(values_array)
        result.append(values_array)
    return result


def build_arrays(
    csv_path: Path,
    delay: int = 4,
    censor_rate: float = 0.2,
    seed: int = 20261005,
):
    if delay < 0:
        raise ValueError("delay must be non-negative")
    if not 0.0 <= censor_rate <= 1.0:
        raise ValueError("censor_rate must be between zero and one")

    frame = pd.read_csv(csv_path)
    required = {"step", "customer", "merchant", "category", "amount", "fraud"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing BankSim columns: {sorted(missing)}")

    for column in ("customer", "merchant", "category"):
        frame[column] = _clean_string_column(frame[column])
    frame["step"] = pd.to_numeric(frame["step"], errors="raise").astype(np.int64)
    frame["amount"] = pd.to_numeric(frame["amount"], errors="raise").astype(np.float32)
    frame["fraud"] = pd.to_numeric(frame["fraud"], errors="raise").astype(np.int8)
    if not set(frame["fraud"].unique()).issubset({0, 1}):
        raise ValueError("fraud must be binary")

    customers = np.asarray(sorted(frame["customer"].unique()))
    categories = sorted(frame["category"].unique())
    horizon = int(frame["step"].max()) + 1
    if int(frame["step"].min()) < 0:
        raise ValueError("step cannot be negative")

    grouped = frame.groupby(["customer", "step"], sort=False).agg(
        tx_count=("amount", "size"),
        amount_sum=("amount", "sum"),
        amount_mean=("amount", "mean"),
        amount_max=("amount", "max"),
        unique_merchants=("merchant", "nunique"),
    )
    category_counts = pd.crosstab([frame["customer"], frame["step"]], frame["category"]).reindex(
        columns=categories, fill_value=0
    )
    category_shares = category_counts.div(category_counts.sum(axis=1), axis=0)
    active = grouped.join(category_shares, how="left").reset_index()

    customer_to_index = {customer: index for index, customer in enumerate(customers)}
    row_index = active["customer"].map(customer_to_index).to_numpy(dtype=np.int64)
    time_index = active["step"].to_numpy(dtype=np.int64)
    x = np.zeros((len(customers), horizon, len(BASE_FEATURES) + len(categories)), dtype=np.float32)
    numeric = active[["tx_count", "amount_sum", "amount_mean", "amount_max", "unique_merchants"]]
    x[row_index, time_index, : len(BASE_FEATURES)] = np.log1p(numeric.to_numpy(dtype=np.float32))
    x[row_index, time_index, len(BASE_FEATURES) :] = active[categories].to_numpy(dtype=np.float32)

    onset_by_customer = frame.loc[frame["fraud"] == 1].groupby("customer")["step"].min().to_dict()
    t_onset = np.asarray(
        [onset_by_customer.get(customer, -1) for customer in customers], dtype=np.int64
    )
    candidate_detect = t_onset + delay
    detectable = (t_onset >= 0) & (candidate_detect < horizon)
    rng = np.random.default_rng(seed)
    simulated_censored = np.zeros(len(customers), dtype=bool)
    positive_indices = np.flatnonzero(detectable)
    simulated_censored[positive_indices] = rng.random(len(positive_indices)) < censor_rate
    detected = detectable & ~simulated_censored
    t_detect = np.where(detected, candidate_detect, -1).astype(np.int64)

    arrays = {
        "x": x,
        "lengths": np.full(len(customers), horizon, dtype=np.int64),
        "detected": detected,
        "t_detect": t_detect,
        "t_onset": t_onset,
        "entity_id": np.arange(len(customers), dtype=np.int64),
    }
    metadata = {
        "transactions": len(frame),
        "customers": len(customers),
        "horizon": horizon,
        "categories": categories,
        "feature_names": BASE_FEATURES + [f"category_share:{name}" for name in categories],
        "fraud_transactions": int(frame["fraud"].sum()),
        "fraud_transaction_rate": float(frame["fraud"].mean()),
        "positive_customers": int((t_onset >= 0).sum()),
        "observed_detections": int(detected.sum()),
        "simulated_censored_positive_customers": int(simulated_censored.sum()),
        "naturally_censored_positive_customers": int(((t_onset >= 0) & ~detectable).sum()),
        "active_customer_steps": len(active),
    }
    return arrays, metadata


def _split_stats(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, int | float]:
    onset = arrays["t_onset"][indices]
    detected = arrays["detected"][indices]
    return {
        "n": len(indices),
        "positive_customers": int((onset >= 0).sum()),
        "positive_rate": float((onset >= 0).mean()),
        "observed_detections": int(detected.sum()),
        "censored": int((~detected).sum()),
    }


def adapt_banksim(
    source: Path,
    output: Path,
    delay: int = 4,
    censor_rate: float = 0.2,
    seed: int = 20261005,
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    source_label: str = "BankSim synthetic bank payments dataset",
) -> dict:
    arrays, metadata = build_arrays(source, delay=delay, censor_rate=censor_rate, seed=seed)
    labels = (arrays["t_onset"] >= 0).astype(np.int8)
    split_indices = stratified_indices(labels, ratios, seed)
    output.mkdir(parents=True, exist_ok=True)

    split_names = ("train", "valid", "test")
    split_stats = {}
    for name, indices in zip(split_names, split_indices):
        np.savez_compressed(
            output / f"{name}.npz", **{key: value[indices] for key, value in arrays.items()}
        )
        split_stats[name] = _split_stats(arrays, indices)

    manifest = {
        "dataset": "BankSim",
        "source": source_label,
        "source_file": source.name,
        "source_sha256": file_digest(source),
        "redistribution": "raw and adapted arrays are kept local pending license confirmation",
        "construction": {
            "entity": "customer",
            "time_axis": "BankSim step, zero-filled to the global horizon",
            "aggregation": "customer-step",
            "t_onset": "first step containing a fraud-labelled transaction; evaluation only",
            "t_detect": f"t_onset + {delay} when observed, otherwise -1",
            "simulated_positive_censor_rate": censor_rate,
            "split": list(ratios),
            "split_unit": "customer",
            "split_stratification": "customer ever has fraud",
            "seed": seed,
        },
        **metadata,
        "splits": split_stats,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report = [
        "# BankSim 数据质量报告",
        "",
        f"- 原始交易数：{metadata['transactions']:,}",
        f"- 客户数：{metadata['customers']:,}",
        f"- 时间步：0–{metadata['horizon'] - 1}（共 {metadata['horizon']} 步）",
        f"- 欺诈交易数：{metadata['fraud_transactions']:,}",
        f"- 欺诈交易比例：{metadata['fraud_transaction_rate']:.4%}",
        f"- 至少发生一次欺诈的客户：{metadata['positive_customers']:,}",
        f"- 输入特征维数：{len(metadata['feature_names'])}",
        "- 标签隔离：fraud 只用于构造 t_onset，未写入 x。",
        "- 划分隔离：按 customer 划分，客户不会跨 train/valid/test。",
        "",
        "## 划分统计",
        "",
        "| split | 客户数 | 阳性客户 | 阳性率 | 可观测检测 | 删失 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in split_names:
        stats = split_stats[name]
        report.append(
            f"| {name} | {stats['n']} | {stats['positive_customers']} | "
            f"{stats['positive_rate']:.4%} | {stats['observed_detections']} | {stats['censored']} |"
        )
    report.extend(
        [
            "",
            "## 当前限制",
            "",
            "- t_detect 并非 BankSim 原生字段，而是为双时间风险实验显式设置的可复现实验机制。",
            "- 类别与欺诈标签相关性较强；它是交易发生时可见的输入，但必须单独做去类别特征消融。",
            "- 统一 180 步时间轴包含无交易日；这些位置保持全零，代表 null event。",
            "- 原始数据与生成的 NPZ 暂不提交仓库，待数据许可进一步确认。",
            "",
        ]
    )
    (output / "data_quality_report.md").write_text("\n".join(report), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adapt BankSim transactions to SafePlus sequences."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay", type=int, default=4)
    parser.add_argument("--censor-rate", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument(
        "--source-label",
        default="https://www.kaggle.com/datasets/ealaxi/banksim1",
    )
    args = parser.parse_args()
    manifest = adapt_banksim(
        source=args.source,
        output=args.output,
        delay=args.delay,
        censor_rate=args.censor_rate,
        seed=args.seed,
        source_label=args.source_label,
    )
    print(json.dumps(manifest["splits"], indent=2))


if __name__ == "__main__":
    main()
