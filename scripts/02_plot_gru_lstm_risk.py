from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ==================================================
# 1. 文件路径
# ==================================================

DATA_PATH = Path(
    "data/processed/synthetic_demo.npz"
)

GRU_PRED_PATH = Path(
    "runs/synthetic/safe-r/seed_0/predictions.npz"
)

LSTM_PRED_PATH = Path(
    "runs/synthetic/lstm-r/seed_0/predictions.npz"
)

OUTPUT_DIR = Path(
    "reports/figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "exp01_gru_vs_lstm_risk.png"
)


# ==================================================
# 2. 读取 synthetic 真实数据
# ==================================================

data = np.load(DATA_PATH)

t_onset = data["t_onset"]
t_detect = data["t_detect"]
detected = data["detected"]


# ==================================================
# 3. 读取 GRU / LSTM 预测结果
# ==================================================

gru = np.load(GRU_PRED_PATH)
lstm = np.load(LSTM_PRED_PATH)

gru_risk = gru["risk"]
lstm_risk = lstm["risk"]

gru_threshold = float(
    gru["threshold"]
)

lstm_threshold = float(
    lstm["threshold"]
)


# ==================================================
# 4. 找模型第一次超过报警阈值的时间
# ==================================================

def first_alarm(
    risk_curve,
    threshold,
):
    crossed = (
        risk_curve >= threshold
    )

    if not crossed.any():
        return -1

    return int(
        np.argmax(crossed)
    )


# ==================================================
# 5. 使用刚才已经确认的代表性样本
# ==================================================

selected = 1067

tf = int(
    t_onset[selected]
)

td = int(
    t_detect[selected]
)

gru_alarm = first_alarm(
    gru_risk[selected],
    gru_threshold,
)

lstm_alarm = first_alarm(
    lstm_risk[selected],
    lstm_threshold,
)


# ==================================================
# 6. 打印关键信息
# ==================================================

print("=" * 60)

print(
    f"Selected entity_id: {selected}"
)

print(
    f"tf (fraud onset): {tf}"
)

print(
    f"td (detection):   {td}"
)

print()

print(
    f"GRU threshold: {gru_threshold:.3f} | "
    f"first alarm: {gru_alarm} | "
    f"lead time: {td - gru_alarm}"
)

print(
    f"LSTM threshold: {lstm_threshold:.3f} | "
    f"first alarm: {lstm_alarm} | "
    f"lead time: {td - lstm_alarm}"
)

print("=" * 60)


# ==================================================
# 7. 绘图
# ==================================================

time = np.arange(
    gru_risk.shape[1]
)

plt.figure(
    figsize=(11, 6)
)

plt.plot(
    time,
    gru_risk[selected],
    marker="o",
    label="GRU / SAFE-R risk",
)

plt.plot(
    time,
    lstm_risk[selected],
    marker="s",
    label="LSTM-R risk",
)


# 真实欺诈开始时间 tf
plt.axvline(
    tf,
    linestyle="--",
    label=f"tf = {tf}",
)


# 真实检测时间 td
plt.axvline(
    td,
    linestyle="--",
    label=f"td = {td}",
)


# GRU 报警阈值
plt.axhline(
    gru_threshold,
    linestyle=":",
    label=(
        f"GRU threshold = "
        f"{gru_threshold:.2f}"
    ),
)


# LSTM 报警阈值
plt.axhline(
    lstm_threshold,
    linestyle=":",
    label=(
        f"LSTM threshold = "
        f"{lstm_threshold:.2f}"
    ),
)


# 标记 GRU 第一次报警位置
if gru_alarm >= 0:
    plt.scatter(
        gru_alarm,
        gru_risk[
            selected,
            gru_alarm,
        ],
        s=90,
        label=(
            f"GRU first alarm = "
            f"{gru_alarm}"
        ),
    )


# 标记 LSTM 第一次报警位置
if lstm_alarm >= 0:
    plt.scatter(
        lstm_alarm,
        lstm_risk[
            selected,
            lstm_alarm,
        ],
        s=90,
        label=(
            f"LSTM first alarm = "
            f"{lstm_alarm}"
        ),
    )


plt.xlabel(
    "Time step"
)

plt.ylabel(
    "Cumulative detection risk"
)

plt.title(
    "Experiment 01: "
    "GRU vs LSTM Detection Risk\n"
    f"Entity {selected}"
)

plt.xticks(time)

plt.ylim(
    0,
    1.05,
)

plt.grid(
    alpha=0.25
)

plt.legend()

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
)

print(
    f"\nFigure saved to: "
    f"{OUTPUT_PATH}"
)

plt.show()