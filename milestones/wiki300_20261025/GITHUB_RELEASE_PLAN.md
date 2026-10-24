# Wiki300 v1.2 / 评价v2.1 当前待提交版本

发布编号：`wiki300-v1.2-eval-v2.1-20261025`；正式审核修订：`review12_user_rules_v1_1_20261024`。当前补丁、记录、报告和完整ZIP采用同一版本。没有Push、创建远程Release或移动历史tag。

## 当前内容

300条现行记录及固定226/24/50分区；审核处置300/300（12参与讨论、288授权接受既有审核）。61 EXACT、138 INTERVAL、100 RIGHT_CENSORED、1 UNCERTAIN。独立重复标注不是本阶段门槛，不虚构其已进行。

新增只读评价v2.1、五方法250行结果、配对与分层、34项评价测试、未来Gold/盲测规则，完整报告见EVALUATION_REPORT.md。没有新训练或根据TEST选模型。所有现行TEST标签已与旧冻结标签逐账号核对一致；现行TRAIN修订不追溯伪造新训练结果。

当前补丁github_ready/milestone_files.patch包含代码、协议、报告和当前审核处置登记。精确文件/SHA清单见github_ready/INCLUDED_FILES.json。原适配补丁仍为artifacts/exp05/exp05_wiki_real.patch。完整ZIP另含源数据、证据、模型/预测、离线Git历史及大文件；资产依赖摘要见data_review/ASSET_REQUIREMENTS.json。

旧发布保存在history/before_metrics_v2_1，原始证据及冻结文件不改。Wikipedia内容保留来源URL/修订ID，来源条款和原项目许可分别适用，不声称新数据许可证。

## 在ChatGPT Work云端从完整ZIP核查

进入解压后的`milestones/wiki300_20261025`：

```bash
python scripts/verify_manifest.py
python scripts/validate_review_revision.py
python -m unittest discover -s tests -p test_review_revision.py -v
python -m unittest discover -s tests -p test_milestone_contract.py -v
python -m unittest discover -s code/evaluation_v2/tests -p 'test_evaluate*.py' -v
```

依赖Python、Git、numpy、pandas，版本见requirements-reproduce.txt。已有依赖后不需要网络、GPU、LLM API、用户F盘或本地Coding Agent。上述7项现行修订、8项历史队列、34项评价测试共49项在发布副本执行；42项记录/保护条件核查一致性，不等同于全部内容独立重复标注。

实际新副本发布验收命令及退出码见`reports/RELEASE_VALIDATION.json`和`logs/publication/`。此前33项模型接口工程测试、合成回归与评分属于历史运行，不包装成这轮新实验。

## 在干净仓库准备提交

Tag2 commit：06fa34d2097b8ca4fa209dbf60ba551fef31a33d；原适配commit：57555b536f2395dd39d4c329917a7eef1acdf77c。离线bundle还保留Tag1 commit fd6b55de5d70e1366853f308870c45cf814eb050。使用tag解引用后的commit，不把annotated tag对象SHA当成commit。

设置MILESTONE为完整ZIP解压后的阶段目录绝对路径：

```bash
MILESTONE=/absolute/path/to/milestones/wiki300_20261025
git clone --branch exp05-io-contract-v1 "$MILESTONE/code/exp05_history.bundle" SafePlus_stage_review
cd SafePlus_stage_review
git switch -c milestone/wiki300-20261025-v1.2
git apply --check "$MILESTONE/artifacts/exp05/exp05_wiki_real.patch"
git apply --index "$MILESTONE/artifacts/exp05/exp05_wiki_real.patch"
git apply --check "$MILESTONE/github_ready/milestone_files.patch"
git apply --index "$MILESTONE/github_ready/milestone_files.patch"
git diff --cached --stat
git diff --cached
```

这条补丁应用顺序已在新副本实际验证。若已精确处于原适配commit57555b5，跳过原适配补丁。不要直接套用到main的另一套GRU实验或覆盖未合并改动。

代码补丁不包含全部数据。需要在新副本运行检查时，复制完整ZIP的阶段内容，Git忽略的大文件只作为本地复现资源：

```bash
cp -R "$MILESTONE/." milestones/wiki300_20261025/
python milestones/wiki300_20261025/scripts/validate_review_revision.py
python -m unittest discover -s milestones/wiki300_20261025/tests -p test_review_revision.py -v
```

审查后可自行进行本地提交：

```bash
git commit -m "Prepare Wiki300 v1.2 evaluation milestone with formal review records and provenance"
```

本轮不执行提交或Push，不创建远程分支、tag或Release。保留真实审核来源，独立Gold栏不冒签；尝试时间不当作成功发布起点，区间不改成点。

## 重评分复现（全程只读冻结输入）

进入阶段目录，在新的输出路径运行，避免覆盖已封存结果：

```bash
python code/evaluation_v2/scripts/evaluate_v2_1.py --bundle artifacts/exp05/data_bundle --run artifacts/exp05/real_run --source data_review/SafePlus_combined_current.jsonl --out reports/evaluation_v2_1_reproduced
```

现行结果在reports/evaluation_v2_1；CLI会拒绝非空输出目录。输入与预测SHA必须符合封存清单，TEST类型/时间与现行正式数据必须一致。程序没有拟合、权重选择或窗口调整；旧v2原代码和评分保留。

## 阶段发布说明

“Wiki300首阶段完成300账号及审核处置登记，保留固定分区、冻结模型、证据与历史；评价v2.1完成账户自身标准化、五方法对照、配对胜率和分层。当前小样本显示优于末次编辑猜测，尚未证实融合优于行为模型。独立重复标注如实注明未进行且不作为本阶段门槛；第8条保持UNCERTAIN。未重新训练、TEST调参或Push。”
