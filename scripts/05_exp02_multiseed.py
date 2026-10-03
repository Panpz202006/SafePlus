from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

from safeplus.data.dataset import _split_indices


ROOT = Path(__file__).resolve().parents[1]

SEEDS = [0, 1, 2, 3, 4]

MODELS = {
    "LSTM-SAFE": {
        "config": ROOT / "configs" / "synthetic" / "lstm-safe.yaml",
        "output_dir": "runs/synthetic/exp02/lstm-safe",
    },
    "GRU-SAFE": {
        "config": ROOT / "configs" / "synthetic" / "safe.yaml",
        "output_dir": "runs/synthetic/exp02/gru-safe",
    },
}

REPORT_DIR = ROOT / "reports" / "experiment02"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = REPORT_DIR / "exp02_multiseed_results.csv"
JSON_PATH = REPORT_DIR / "exp02_multiseed_summary.json"

DATA_PATH = ROOT / "data" / "processed" / "synthetic_demo.npz"


def run_command(args):
    print()
    print("=" * 80)
    print("RUN:")
    print(" ".join(str(x) for x in args))
    print("=" * 80)

    subprocess.run(
        [str(x) for x in args],
        cwd=ROOT,
        check=True,
    )


def first_alarm(risk_curve, threshold):
    crossed = risk_curve >= threshold

    if not crossed.any():
        return -1

    return int(np.argmax(crossed))


def group_final_risk(risk, lengths, indices):
    values = []

    for i in indices:
        T = int(lengths[i] - 1)
        values.append(float(risk[i, T]))

    return float(np.mean(values))


def alarm_rate(risk, lengths, indices, threshold):
    alarms = 0

    for i in indices:
        T = int(lengths[i] - 1)

        alarm = first_alarm(
            risk[i, : T + 1],
            threshold,
        )

        if alarm >= 0:
            alarms += 1

    return alarms / len(indices)


def onset_analysis(
    risk,
    lengths,
    t_onset,
    indices,
    threshold,
):
    before = 0
    exactly = 0
    after = 0
    no_alarm = 0

    deltas = []
    risks_at_tf = []

    for i in indices:
        tf = int(t_onset[i])
        T = int(lengths[i] - 1)

        risks_at_tf.append(
            float(risk[i, tf])
        )

        alarm = first_alarm(
            risk[i, : T + 1],
            threshold,
        )

        if alarm < 0:
            no_alarm += 1
            continue

        delta = alarm - tf
        deltas.append(delta)

        if delta < 0:
            before += 1
        elif delta == 0:
            exactly += 1
        else:
            after += 1

    total = len(indices)

    return {
        "before_count": before,
        "before_rate": before / total,
        "exact_count": exactly,
        "after_count": after,
        "no_alarm_count": no_alarm,
        "mean_alarm_minus_tf": (
            float(np.mean(deltas))
            if deltas
            else np.nan
        ),
        "mean_risk_at_tf": float(
            np.mean(risks_at_tf)
        ),
    }


# ============================================================
# 读取原始 synthetic 数据
# ============================================================

data = np.load(DATA_PATH)

t_onset = data["t_onset"]
detected = data["detected"]
lengths = data["lengths"]
entity_id = data["entity_id"]


all_results = []


# ============================================================
# 依次运行两个模型 × 5 seeds
# ============================================================

for model_label, model_info in MODELS.items():

    print()
    print("#" * 80)
    print(f"MODEL: {model_label}")
    print("#" * 80)

    with open(
        model_info["config"],
        "r",
        encoding="utf-8",
    ) as f:
        base_config = yaml.safe_load(f)

    for seed in SEEDS:

        print()
        print(
            f">>> {model_label} | seed={seed}"
        )

        config = dict(base_config)

        config["seed"] = seed
        config["output_dir"] = (
            model_info["output_dir"]
        )

        # 单独复制嵌套字典，避免修改原配置
        config["data"] = dict(
            base_config["data"]
        )
        config["model"] = dict(
            base_config["model"]
        )
        config["training"] = dict(
            base_config["training"]
        )

        # ----------------------------------------------------
        # 临时 YAML
        # ----------------------------------------------------

        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix=f"exp02_{seed}_",
            dir=ROOT / "configs" / "synthetic",
            delete=False,
            encoding="utf-8",
        )

        try:
            yaml.safe_dump(
                config,
                temp_file,
                sort_keys=False,
            )

            temp_file.close()

            temp_config_path = Path(
                temp_file.name
            )

            # ------------------------------------------------
            # 训练
            # ------------------------------------------------

            run_command(
                [
                    sys.executable,
                    "-m",
                    "safeplus",
                    "train",
                    "--config",
                    temp_config_path,
                ]
            )

            run_dir = (
                ROOT
                / config["output_dir"]
                / f"seed_{seed}"
            )

            pred_path = (
                run_dir
                / "predictions.npz"
            )

            # ------------------------------------------------
            # 预测
            # ------------------------------------------------

            run_command(
                [
                    sys.executable,
                    "-m",
                    "safeplus",
                    "predict",
                    "--run-dir",
                    run_dir,
                    "--data",
                    DATA_PATH,
                    "--output",
                    pred_path,
                    "--batch-size",
                    str(
                        config["training"][
                            "batch_size"
                        ]
                    ),
                    "--device",
                    str(
                        config["training"][
                            "device"
                        ]
                    ),
                ]
            )

        finally:

            if os.path.exists(
                temp_file.name
            ):
                os.remove(
                    temp_file.name
                )

        # ====================================================
        # 读取普通 metrics
        # ====================================================

        metrics_path = (
            run_dir / "metrics.json"
        )

        with open(
            metrics_path,
            "r",
            encoding="utf-8",
        ) as f:
            metrics = json.load(f)

        # ====================================================
        # 读取逐时间预测
        # ====================================================

        pred = np.load(pred_path)

        risk = pred["risk"]

        threshold = float(
            pred["threshold"]
        )

        # ====================================================
        # 恢复这个 seed 对应的测试集
        # ====================================================

        _, _, test_idx = _split_indices(
            n=len(entity_id),
            ratios=[0.7, 0.1, 0.2],
            seed=seed,
        )

        normal_idx = []
        detected_fraud_idx = []
        censored_fraud_idx = []

        for i in test_idx:

            tf = int(t_onset[i])
            is_detected = bool(
                detected[i]
            )

            if tf < 0:
                normal_idx.append(i)

            elif is_detected:
                detected_fraud_idx.append(i)

            else:
                censored_fraud_idx.append(i)

        normal_idx = np.asarray(
            normal_idx
        )

        detected_fraud_idx = np.asarray(
            detected_fraud_idx
        )

        censored_fraud_idx = np.asarray(
            censored_fraud_idx
        )

        # ====================================================
        # onset 分析
        # ====================================================

        detected_onset = onset_analysis(
            risk,
            lengths,
            t_onset,
            detected_fraud_idx,
            threshold,
        )

        censored_onset = onset_analysis(
            risk,
            lengths,
            t_onset,
            censored_fraud_idx,
            threshold,
        )

        row = {
            "model": model_label,
            "seed": seed,

            "accuracy":
                metrics.get("accuracy"),

            "auroc":
                metrics.get("auroc"),

            "auprc":
                metrics.get("auprc"),

            "precision":
                metrics.get("precision"),

            "recall":
                metrics.get("recall"),

            "f1":
                metrics.get("f1"),

            "early_detected_rate":
                metrics.get(
                    "early_detected_rate"
                ),

            "mean_lead_time":
                metrics.get(
                    "mean_lead_time"
                ),

            "threshold":
                threshold,

            "normal_count":
                len(normal_idx),

            "detected_fraud_count":
                len(detected_fraud_idx),

            "censored_fraud_count":
                len(censored_fraud_idx),

            "normal_final_risk":
                group_final_risk(
                    risk,
                    lengths,
                    normal_idx,
                ),

            "detected_final_risk":
                group_final_risk(
                    risk,
                    lengths,
                    detected_fraud_idx,
                ),

            "censored_final_risk":
                group_final_risk(
                    risk,
                    lengths,
                    censored_fraud_idx,
                ),

            "normal_alarm_rate":
                alarm_rate(
                    risk,
                    lengths,
                    normal_idx,
                    threshold,
                ),

            "detected_before_tf_rate":
                detected_onset[
                    "before_rate"
                ],

            "detected_mean_alarm_minus_tf":
                detected_onset[
                    "mean_alarm_minus_tf"
                ],

            "censored_before_tf_rate":
                censored_onset[
                    "before_rate"
                ],

            "censored_mean_alarm_minus_tf":
                censored_onset[
                    "mean_alarm_minus_tf"
                ],
        }

        all_results.append(row)

        # ====================================================
        # 当前 seed 输出
        # ====================================================

        print()
        print("-" * 80)

        print(
            f"{model_label} | seed={seed}"
        )

        print(
            "normal alarm rate:",
            f"{row['normal_alarm_rate']:.4f}",
        )

        print(
            "detected before tf rate:",
            f"{row['detected_before_tf_rate']:.4f}",
        )

        print(
            "detected mean(alarm-tf):",
            f"{row['detected_mean_alarm_minus_tf']:.4f}",
        )

        print(
            "censored before tf rate:",
            f"{row['censored_before_tf_rate']:.4f}",
        )

        print(
            "censored mean(alarm-tf):",
            f"{row['censored_mean_alarm_minus_tf']:.4f}",
        )


# ============================================================
# 保存逐 seed CSV
# ============================================================

fieldnames = list(
    all_results[0].keys()
)

with open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        all_results
    )


# ============================================================
# 计算 5-seed 均值和标准差
# ============================================================

summary = {}

numeric_fields = [
    "accuracy",
    "auroc",
    "auprc",
    "precision",
    "recall",
    "f1",
    "early_detected_rate",
    "mean_lead_time",
    "normal_final_risk",
    "detected_final_risk",
    "censored_final_risk",
    "normal_alarm_rate",
    "detected_before_tf_rate",
    "detected_mean_alarm_minus_tf",
    "censored_before_tf_rate",
    "censored_mean_alarm_minus_tf",
]


for model_label in MODELS:

    rows = [
        r
        for r in all_results
        if r["model"] == model_label
    ]

    summary[model_label] = {}

    for field in numeric_fields:

        values = np.asarray(
            [
                float(r[field])
                for r in rows
            ],
            dtype=float,
        )

        summary[model_label][field] = {
            "mean": float(
                np.mean(values)
            ),
            "std": float(
                np.std(
                    values,
                    ddof=1,
                )
            ),
        }


with open(
    JSON_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 最终在终端打印最关键结果
# ============================================================

print()
print()
print("=" * 80)
print("EXPERIMENT 02: 5-SEED SUMMARY")
print("=" * 80)

for model_label in MODELS:

    s = summary[model_label]

    print()
    print(model_label)

    print(
        "normal alarm rate:",
        f"{s['normal_alarm_rate']['mean']:.4f}"
        " ± "
        f"{s['normal_alarm_rate']['std']:.4f}",
    )

    print(
        "detected before-tf rate:",
        f"{s['detected_before_tf_rate']['mean']:.4f}"
        " ± "
        f"{s['detected_before_tf_rate']['std']:.4f}",
    )

    print(
        "detected mean(alarm-tf):",
        f"{s['detected_mean_alarm_minus_tf']['mean']:.4f}"
        " ± "
        f"{s['detected_mean_alarm_minus_tf']['std']:.4f}",
    )

    print(
        "censored before-tf rate:",
        f"{s['censored_before_tf_rate']['mean']:.4f}"
        " ± "
        f"{s['censored_before_tf_rate']['std']:.4f}",
    )

    print(
        "censored mean(alarm-tf):",
        f"{s['censored_mean_alarm_minus_tf']['mean']:.4f}"
        " ± "
        f"{s['censored_mean_alarm_minus_tf']['std']:.4f}",
    )

    print(
        "AUROC:",
        f"{s['auroc']['mean']:.4f}"
        " ± "
        f"{s['auroc']['std']:.4f}",
    )

    print(
        "AUPRC:",
        f"{s['auprc']['mean']:.4f}"
        " ± "
        f"{s['auprc']['std']:.4f}",
    )

    print(
        "Precision:",
        f"{s['precision']['mean']:.4f}"
        " ± "
        f"{s['precision']['std']:.4f}",
    )

    print(
        "Recall:",
        f"{s['recall']['mean']:.4f}"
        " ± "
        f"{s['recall']['std']:.4f}",
    )


print()
print(
    "Saved CSV:",
    CSV_PATH,
)

print(
    "Saved JSON:",
    JSON_PATH,
)