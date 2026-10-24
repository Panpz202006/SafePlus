# SafePlus 阶段交付：Wiki300 评价补充包（2026-10-24）

> **重要：这不是新的人类 Gold 数据集，也不是新训练。** 这是对已完成的 Exp05 Wiki300 首轮模型预测的**只读、探索性重评分**，并从冻结数据中生成分层独立人工复核任务清单。不会改动旧模型、源数据、标签、原 I/O 或原有 TEST。未向 GitHub 推送。

## 你现在已经拥有的阶段成果

1. 已收录 300 账号：61 个具有准确 UTC 起点记录，139 个起点区间，100 个右删失正常观察窗口。这300条已做 AI/结构审查，**尚无独立人工终审 Gold 的证据**。
2. Exp05 冻结主线已完成首轮真实拟合与推断：固定 TRAIN/VAL/TEST = 226/24/50，测试中精确起点9、区间29、右删失对照12。
3. 新增本包的 `scripts/evaluate_addon.py`：在**不改模型**的前提下，按账号分开输出精确UTC点误差、精确已发布事件位置误差、位置标准化误差、与 Behavior-only/Delay-only 的配对胜率、区间按宽度分层的命中率/距离。
4. `reports/independent_review_queue_36.csv`：基于原始证据来源和分层规则、**未使用模型预测**挑选的36个复核任务。默认状态全是 PENDING，不得自动计为 Gold。含4条出自已评测TEST，仅作为数据质检，不能拿来重新调参并自称独立TEST。另新增 `reports/train_prior_evidence_review_36.csv`：优先核查用于**拟合延迟先验**的36条TRAIN精确标签，它们与36条广覆盖质检队列不一定相同。
5. `reports/dataset_quality_audit_v2.json` 和 `reports/exploratory_score_v2.json`：由实际冻结文件读出并运行得到，绝非估算。
6. 新增5项单元测试，且已在本轮实际源包上成功运行。已有Exp05包报告的20项测试属于此前项目；本包不把两组混为同一轮CI结果。

## Windows复现（现有Anaconda Python 3.10+即可；不需要GPU或API）

解压既有 `SafePlus_Exp05_Wiki300_Run_v1.zip`，并把本包放在任意不覆盖旧文件的目录。切换到本包目录后运行：

```bat
python scripts\evaluate_addon.py --bundle "F:\research\SafePlus_Exp05_Wiki300_Run_v1\data_bundle" --run "F:\research\SafePlus_Exp05_Wiki300_Run_v1\real_run" --source "F:\research\SafePlus_Exp05_Wiki300_Run_v1\frozen_source\SafePlus_combined_current.jsonl" --out "reports\local_rescore"
python -m unittest discover -s tests -v
```

应输出 `PASS_READ_ONLY`，并生成4个新文件。此代码仅依赖Python标准库；**不要让训练代码读取 `gold_onset.csv` 或 `independent_review_queue_36.csv`。**

## 必须区分的研究问题

- **任务A：已知账号在 `t_d` 被封禁，逆向定位其首个可核查违规起点 `t_f`。** 当前 Exp05 只完成这个任务的检测账号条件定位。对正常观察窗口现有运行没有风险分数，不可报告其分类正确率。
- **任务B：在未知未来封禁时间的情况下，判断账号是否会违规。** 需要前缀风险输出、右删失语义、验证集阈值等，当前冻结实验尚未完成。
- **任务C：在违规发生之后、平台正式封禁之前作在线预警。** 不能把已知 `t_d` 的事后推断冒充在线提前发现；需要严格因果截断前缀并在检测前执行预测。

本阶段**只对任务A做可证实的评价升级**。B/C列为下一阶段研究，不阻碍 A 的阶段交付。

## 目录

- `docs/SCHEMA_AND_EVAL_V2.md`：冻结目标、标注类型、公式、评分边界与升级口径。
- `docs/ACTION_PLAN_TO_WWW.md`：周日、11/9和11/16交付、建议数量、模型改动优先级、采证与人审分工。
- `scripts/evaluate_addon.py`：只读增量评分与复核抽样，防止错把区间/正常对照算成点误差。
- `tests/test_evaluate_addon.py`：左开右闭区间、边界、错误重复ID、UTC要求等单元测试。
- `reports/exploratory_score_v2.json`：实际评测补充结果，**TEST已看过，非新独立验证**。
- `reports/dataset_quality_audit_v2.json`：实际数据分布与需要补强的子组。
- `reports/independent_review_queue_36.csv`：分层人工A/B审阅清单，初始均未完成。
- `reports/train_prior_evidence_review_36.csv`：模型当前延迟先验最依赖的36条TRAIN起点证据核查清单。
- `reports/per_account_exploratory_v2.csv`：逐账号误差便于复盘；该文件不可作为模型训练特征。

## Git建议

在 `exp05-io-contract-v1` 派生的真实数据分支上，与已有 Exp05 patch 共存。只提交本包的 `docs/ scripts/ tests/ reports`（如公开对人相关来源信息需做审查与脱敏，可只推聚合结果）。

不要修改历史标签或已有 TEST；每次运行记录源码/数据/输出 SHA256。未来要优化模型，先在VAL上调试；建立额外完全未暴露的账号集合进行新的最终测试。

## 数据/模型和概念的限制

- `t_d` 是**首次相关封禁**，不保证为第一次怀疑或发现；`t_f` 只在取得的公开历史与可核查证据范围内谈“最早”。
- Wiki 包含垃圾外链、破坏、攻击性编辑等，不能通称金融欺诈。旧接口字段 `is_fraud` 是项目沿用名称。
- 旧实验使用了36条 TRAIN 精确起点估计延迟先验，故整体**不属于完全无起点监督**。
- 真实 I/O 特征只有8项已发布编辑的累计元数据，没有文本语义；`t_d`端点快照不是额外编辑。
- 精确UTC的部分起点并非发布编辑，而是公开过滤等日志，不可以硬转成“第几条发布编辑”；右删失不是永远正常。
- 数据质量分层针对研究适用性，并不自动否定原始审核的证据判断。大量只有1条正常前缀的样本应保留并作弱历史组报告。

来源：已保存项目档案 `SafePlus_Exp05_Wiki300_Run_v1.zip`、`SafePlus_delivery_audit_20261024.zip`、`SafePlus_control_progress.json`。官方截止参见 https://www2027.thewebconf.org/important-dates/ 。
