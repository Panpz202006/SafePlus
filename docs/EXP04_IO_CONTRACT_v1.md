# SafePlus Experiment 04 — I/O Contract v1.0 (封口协议)

**适用版本：** `exp04-synthetic-frozen-v1` (commit `fd6b55d`)。**接口适配分支**应独立创建，禁止改动旧 Tag。

## 0. 任务和边界

当前 Exp04 是 **Retrospective Fraud Onset Localization（已检测欺诈用户的事后起点回溯）**，不是独立的在线分类器。推断一个用户时，已知其检测事件 `detected=1` 和观测检测时间 `t_detect`，目标是由此前可用行为推断首次可核验欺诈时间 `t_onset`。未检测用户可以参与明确定义的行为训练，但当前已验证的最终融合不对其输出假定准确的 `t_onset`。`is_fraud` 不属于预测输入。

`T` 在旧融合文献里是 temperature；在数据集里使用 `L` 或 `lengths` 表示序列长度，避免混淆。

## 1. 文件组织：五个独立对象

每份数据集分别提交：

```
datasets/<dataset_id>/
  inputs.npz           # 模型可见输入（绝不包含 t_onset / is_fraud）
  split.csv            # entity_id,split; TRAIN / VAL / TEST
  train_labels.csv     # 仅 TRAIN：entity_id,is_fraud（0 或 1）
  metadata.json        # 数据来源、特征、时间语义及泄漏边界
  delay_prior.json     # 延迟先验的来源、规则、参数
  gold_onset.csv       # 可选、只能由评测器读取，绝不传进模型训练/预测
```

这里将模型输入、训练监督、评测真值、配置 **物理分离**。真实标注另保留原始证据表（case_id/event_id/time/evidence/annotation/仲裁），不能用模型猜测当 Ground Truth。

### `inputs.npz`（只允许五个键）

| 键 | 形状 / 类型 | 约束 |
|---|---|---|
| `entity_id` | `[N]`，整数或 Unicode 字符串 | 唯一且稳定；不得编码标签 |
| `x` | `[N,L,D]`，浮点数 | 按时间排序；有效观测区间 finite；不得带答案、平台处分或事后事件特征 |
| `lengths` | `[N]`，整数 | `1 <= lengths <= L`；标识实际序列有效边界 |
| `detected` | `[N]`，0/1 | 观察结束前是否真的发生了指定的检测/处置事件 |
| `t_detect` | `[N]`，整数 | `detected=1`：`1 <= d < lengths`；`detected=0`：等于 `lengths`，表示未观测到事件，不是实际检测日期 |

`detected=1` 的候选位置固定为 `k=0,...,d-1`，且 **只允许用 `x[0:d+1]`** 的观察行为；`x[d]` 必须是决策或检测结果注入前即可获得的行为特征。`x[d+1:]` 必须屏蔽或由预测包装层完全忽略。对 `detected=0`，无新观察到的 `t_d`，不能把 `t_detect=lengths` 解释为真实检测时间。

**变长序列注意：** 当前冻结 Exp04 EM 在部分位置仍根据 `x.shape[1]` 推断观察结束，没有完整使用 `lengths`。因此 v1 的回归验收先要求数据集中 `lengths` 全等于 `L`（定长），真实变长数据 **暂不准直接通过最终预测流程**。后续支持变长时，必须改进所有观测边界并通过“定长结果完全不变”回归测试；可以通过独立时间分箱/规范化适配定长，但不得把填充值当作真实行为。

### `split.csv`

每个 `entity_id` 恰好一行，`split` ∈ `TRAIN,VAL,TEST`。严格按实体分割，严禁同一实体跨集合；有时间信息时优先时间划分，并记录为何采用或不采用。`split.csv` 由数据准备过程固定，不允许训练/预测时再任意重新随机划分。

### `train_labels.csv`

只包含 TRAIN 实体的 `entity_id,is_fraud`，值必须为 `0` 或 `1`。未知标签为 `-1` 时不得进入当前 EM 训练。正常用户需要可靠正常定义，未检测用户不可默认当正常；正式数据严禁从 `t_onset` 存在性推导 `is_fraud`。

### `gold_onset.csv`（仅评测器可读）

建议列：`entity_id,is_fraud,onset_label_type,t_onset,t_onset_min,t_onset_max,source_event_id,evidence_ids,annotation_version`。其中 `onset_label_type` ∈ `EXACT,INTERVAL,UNRESOLVED,NOT_FRAUD`。主 MAE 仅对 `TEST` 中 `detected=1,is_fraud=1,onset_label_type=EXACT` 的实体计算。区间、不确定或没有客观真值的样本不应强行生成单点真值。

**许可例外**：如要利用 TRAIN 人工标注的真实 onset 估计 Delay Prior，须另行以明确的 `train_onset_for_prior.csv` 作为训练资源，只包含 TRAIN 样本，并在论文中如实说明此模型整体方案属于部分 onset 监督，不能再宣称训练完全无真实 onset。测试标签永远不得用于建模或调参。

### `metadata.json` 必须字段

- `schema_version`: `"1.0"`
- `dataset_id`、`dataset_kind`: `synthetic` 或 `real`
- `time_unit`、`time_origin`、`time_index_base`: 0
- `feature_names`: `[D]` 且顺序对应 `x` 最后一维
- `t_detect_definition`: 精确指明首次标记/处罚/封禁的哪一种（如 `simulated_detection_event`）
- `detected_bin_features`: `pre_detection_only`（若含检测事件生成的特征则不合格）
- `post_detection_policy`: `masked_zero` 或保证所有实现严格裁剪
- `split_method`, `preprocessing`, `data_source`, `license`, `annotation_guideline_version`

### `delay_prior.json`

- `synthetic_known`：仅用生成器预先公布的 **真实规则**，不能根据测试 `t_onset` 建模。
- `empirical_train`：严格使用明示的 TRAIN-onset 监督，明确 Laplace 平滑和支持集；独立于测试真值。
- `external_fixed`：独立外部来源，注明版本/统计时间/支持集。

字段：`mode`, `definition`, `source`, `parameters`, `version`。如果真实数据还没有合法先验，则接口可以记录 `unresolved`，但 **不允许宣称完整预测验收通过**。

## 2. 固定算法和运行约束

- 冻结结构：Two-state Gaussian Behavior EM + TRAIN-only Global Behavior Scale + Delay Prior + Fixed Fusion。
- 固定融合：`w=2/3, temperature=1/3`，等价于旧 `alpha=2`。
- Behavior EM 不得使用数值 `t_onset`；只用受许可的 TRAIN `is_fraud`、`detected`、`t_detect` 和因果 `x`。
- TRAIN/VAL/TEST 索引从固定的 `split.csv` 读取，不得内部重新打乱。
- 每数据集可以在 **固定训练协议** 下拟合自身 Behavior 参数和 TRAIN-only calibration；这不是改动算法封板。
- 不许用 TEST `t_onset` 选择权重、先验、特征、阈值、模型 checkpoint 或数据清洗规则。

## 3. 固定预测输出

输出 `predictions.csv` 与 `posterior.jsonl` 两类文件，以 `entity_id` 关联，至少保存：

`predictions.csv`: `entity_id,status,observed_td,pred_tf_median,pred_tf_mean,pred_tf_mode,pred_delay,posterior_max,model_version,delay_prior_version`。

`posterior.jsonl`: 每个合格用户一行 `{entity_id, candidate_times:[...], posterior_probs:[...]}`，保证每个概率有限、非负、和为 1、且候选严格落在 `0..d-1`。

对于无合法已检测事件、不满足任务协议、证据不够等输入，必须报告明确 `status`，不能虚构 `pred_tf`。

另存 `run_manifest.json`：git commit/tag、数据哈希、特征定义、划分哈希、EM 参数、Global Scale、Delay Prior/版本、随机种子、融合权重、Python 与依赖版本、运行时间等。

## 4. 固定评测协议

**主指标** `Posterior Median MAE = mean(|pred_tf_median - gold_tf|)`。

**辅助指标**：`Signed Error = mean(pred_tf_median-gold_tf)`；`Within-1`、`Within-2`；按真值 `delta=t_detect-gold_tf` 划 `1–2 / 3–5 / >=6` 的分组 MAE（主报告可用合成时间步，真实数据需说明分箱单位与可比性）；相同样本上相对 Behavior-only / Delay-only 的 paired bootstrap 95% CI；`n_eligible, n_exact, n_unresolved`。

可选指标：posterior NLL、credible-interval coverage/width；**必须先定义零概率截断和区间构建规则，未实现不得写“已完成”**。

AUPRC/AUROC/F1 与 Online Early-Detection 是另一项任务，不属于当前 Exp04 回溯模型的已完成主评测；如论文声称在线预警，必须另开不知道未来 `t_d` 的实验协议。

## 5. 验收顺序与状态

1. `45` 校验器对新 Bundle 成功、对标签泄漏和非法检测时间等坏样本明确拒绝。
2. `46` 将原始 Synthetic 转为以上分离式 Bundle，做 I/O 回归验收。
3. 新的统一训练/预测/评测包装层从 Bundle 读取，但不改变 Exp04 两种证据的数学公式。
4. **用相同 seed0、训练配置和 `synthetic_known` 复现既有 MAE ≈ 1.234848**；参数和每个实体预测尽可能逐一比对。
5. 完成变长兼容或注明定长限制，不允许沉默忽略 `lengths`。
6. 经数据/输出完整性、真值泄漏和先验泄漏测试后，在新分支创建新标签（例如 `exp04-io-frozen-v1`）。

**目前状态：** 旧 `exp04-synthetic-frozen-v1` 仅为模型/合成实验封板；`v1.0` I/O 为新工程接口契约，需在上述验收通过后才能称为“正式封口”。
