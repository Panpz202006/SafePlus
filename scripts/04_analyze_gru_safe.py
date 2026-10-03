from __future__ import annotations

import numpy as np

from safeplus.data.dataset import _split_indices


DATA_PATH = "data/processed/synthetic_demo.npz"

PRED_PATH = (
    "runs/synthetic/safe/seed_0/"
    "predictions.npz"
)


# ============================================================
# 1. 读取 synthetic 数据里的“真实答案”
# ============================================================

data = np.load(DATA_PATH)

t_onset = data["t_onset"]
t_detect = data["t_detect"]
detected = data["detected"]
lengths = data["lengths"]
entity_id = data["entity_id"]


# ============================================================
# 2. 读取 GRU-SAFE 的预测
# ============================================================

pred = np.load(PRED_PATH)

risk = pred["risk"]
pred_entity_id = pred["entity_id"]

threshold = float(pred["threshold"])


# 确认预测文件和原始数据用户顺序一致
if not np.array_equal(
    entity_id,
    pred_entity_id,
):
    raise ValueError(
        "Prediction entity_id order does not match dataset."
    )


# ============================================================
# 3. 恢复训练时完全相同的 test split
# ============================================================

train_idx, val_idx, test_idx = _split_indices(
    n=len(entity_id),
    ratios=[0.7, 0.1, 0.2],
    seed=0,
)

print("=" * 70)
print("Dataset split")
print("=" * 70)

print("total:", len(entity_id))
print("train:", len(train_idx))
print("val:", len(val_idx))
print("test:", len(test_idx))


# ============================================================
# 4. 找第一次超过报警阈值的时间
# ============================================================

def first_alarm(risk_curve, threshold):
    crossed = risk_curve >= threshold

    if not crossed.any():
        return -1

    return int(np.argmax(crossed))


# ============================================================
# 5. 把 400 个测试用户分成三种
# ============================================================

normal_idx = []
detected_fraud_idx = []
censored_fraud_idx = []


for i in test_idx:

    tf = int(t_onset[i])
    is_detected = bool(detected[i])

    # A. 从未发生 fraud
    if tf < 0:
        normal_idx.append(i)

    # B. 已发生 fraud，而且后来检测到了
    elif is_detected:
        detected_fraud_idx.append(i)

    # C. 已发生 fraud，但观察窗口结束仍未检测
    else:
        censored_fraud_idx.append(i)


normal_idx = np.asarray(normal_idx)

detected_fraud_idx = np.asarray(
    detected_fraud_idx
)

censored_fraud_idx = np.asarray(
    censored_fraud_idx
)


print()
print("=" * 70)
print("Test-set groups")
print("=" * 70)

print(
    "A. Normal:",
    len(normal_idx),
)

print(
    "B. Detected fraud:",
    len(detected_fraud_idx),
)

print(
    "C. Censored fraud:",
    len(censored_fraud_idx),
)


# ============================================================
# 6. 看三组人的最终 fraud risk
# ============================================================

def print_group_summary(name, indices):

    values = []
    alarm_count = 0

    for i in indices:

        T = int(lengths[i] - 1)

        values.append(
            float(risk[i, T])
        )

        if first_alarm(
            risk[i, : T + 1],
            threshold,
        ) >= 0:
            alarm_count += 1

    values = np.asarray(values)

    print()
    print("-" * 70)

    print(name)

    print(
        "count:",
        len(indices),
    )

    print(
        "mean final risk:",
        f"{values.mean():.4f}",
    )

    print(
        "median final risk:",
        f"{np.median(values):.4f}",
    )

    print(
        "min final risk:",
        f"{values.min():.4f}",
    )

    print(
        "max final risk:",
        f"{values.max():.4f}",
    )

    print(
        "alarm rate:",
        f"{alarm_count / len(indices):.4f}",
    )


print()
print(
    "GRU-SAFE threshold:",
    f"{threshold:.4f}",
)

print_group_summary(
    "A. NORMAL",
    normal_idx,
)

print_group_summary(
    "B. DETECTED FRAUD",
    detected_fraud_idx,
)

print_group_summary(
    "C. CENSORED FRAUD",
    censored_fraud_idx,
)


# ============================================================
# 7. 对真正 fraud 的用户，直接和真实 tf 对比
# ============================================================

def analyze_true_onset(
    name,
    indices,
):

    risk_at_tf = []
    alarm_minus_tf = []

    alarms_before_tf = 0
    alarms_after_tf = 0
    alarms_at_tf = 0
    no_alarm = 0

    for i in indices:

        tf = int(t_onset[i])
        T = int(lengths[i] - 1)

        risk_at_tf.append(
            float(risk[i, tf])
        )

        alarm = first_alarm(
            risk[i, : T + 1],
            threshold,
        )

        if alarm == -1:
            no_alarm += 1
            continue

        delta = alarm - tf

        alarm_minus_tf.append(
            delta
        )

        if delta < 0:
            alarms_before_tf += 1

        elif delta == 0:
            alarms_at_tf += 1

        else:
            alarms_after_tf += 1

    risk_at_tf = np.asarray(
        risk_at_tf
    )

    print()
    print("=" * 70)

    print(name)

    print("=" * 70)

    print(
        "mean risk at true tf:",
        f"{risk_at_tf.mean():.4f}",
    )

    print(
        "median risk at true tf:",
        f"{np.median(risk_at_tf):.4f}",
    )

    print(
        "alarm before true tf:",
        alarms_before_tf,
    )

    print(
        "alarm exactly at true tf:",
        alarms_at_tf,
    )

    print(
        "alarm after true tf:",
        alarms_after_tf,
    )

    print(
        "no alarm:",
        no_alarm,
    )

    if len(alarm_minus_tf) > 0:

        print(
            "mean(first_alarm - tf):",
            f"{np.mean(alarm_minus_tf):.4f}",
        )


analyze_true_onset(
    "B. Detected fraud vs true tf",
    detected_fraud_idx,
)

analyze_true_onset(
    "C. Censored fraud vs true tf",
    censored_fraud_idx,
)


# ============================================================
# 8. 打印一个 detected fraud 具体案例
# ============================================================

def print_case(title, i):

    tf = int(t_onset[i])
    T = int(lengths[i] - 1)

    td = (
        int(t_detect[i])
        if detected[i]
        else None
    )

    alarm = first_alarm(
        risk[i, : T + 1],
        threshold,
    )

    print()
    print("=" * 70)

    print(title)

    print("=" * 70)

    print(
        "dataset index:",
        i,
    )

    print(
        "entity_id:",
        int(entity_id[i]),
    )

    print(
        "true tf:",
        tf,
    )

    print(
        "td:",
        td,
    )

    print(
        "observation end T:",
        T,
    )

    print(
        "detected:",
        bool(detected[i]),
    )

    print(
        "first alarm:",
        alarm,
    )

    print(
        "risk at true tf:",
        f"{risk[i, tf]:.4f}",
    )

    print(
        "risk at T:",
        f"{risk[i, T]:.4f}",
    )

    print()
    print("Risk curve:")

    for t in range(T + 1):

        marker = ""

        if t == tf:
            marker += "  <-- TRUE tf"

        if td is not None and t == td:
            marker += "  <-- td"

        if t == alarm:
            marker += "  <-- FIRST ALARM"

        print(
            f"t={t:2d} | "
            f"risk={risk[i, t]:.4f}"
            f"{marker}"
        )


if len(detected_fraud_idx) > 0:

    print_case(
        "CASE B: Detected fraud",
        int(detected_fraud_idx[0]),
    )


if len(censored_fraud_idx) > 0:

    print_case(
        "CASE C: Fraud occurred but not detected",
        int(censored_fraud_idx[0]),
    )