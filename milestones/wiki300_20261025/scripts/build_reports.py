#!/usr/bin/env python3
"""Build milestone reports exclusively from completed acceptance evidence."""
from __future__ import annotations
import argparse, datetime, json, shutil
from pathlib import Path
from audit_dataset import csvrows, dump

def main():
    p=argparse.ArgumentParser();p.add_argument('--acceptance',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];run=a.acceptance.resolve()
    if (root/'data_review/REVISION_STATUS.json').exists():
        raise ValueError('This generator describes frozen v1 only; do not overwrite current v1.1 review and release reports.')
    acc=json.loads((run/'acceptance_results.json').read_text());assert acc['status']=='PASS'
    for d in ['audit','evaluation_v2','original_scorer']:
        shutil.copytree(run/d,root/'reports'/d,dirs_exist_ok=True)
    shutil.copytree(run/'logs',root/'logs/acceptance',dirs_exist_ok=True)
    shutil.copy2(run/'acceptance_results.json',root/'reports/acceptance_results.json')
    qa=json.loads((run/'audit/dataset_audit.json').read_text());sc=json.loads((run/'evaluation_v2/exploratory_score_v2.json').read_text())
    original=json.loads((run/'original_scorer/wiki_metrics.json').read_text());sel=json.loads((root/'HUMAN_REVIEW_QUEUE/SELECTION_MANIFEST.json').read_text())
    stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec='seconds')
    dataset_gates=[
        ('账号级任务、观察边界及标签定义可查询',True),('300条记录和原始证据来源可恢复',True),
        ('账号去重、时间与跨表完整性验收',True),('EXACT/INTERVAL/右删失分流并保持原标签',True),
        ('固定TRAIN/VAL/TEST和已知匹配组隔离',True),('输入、评测答案与TRAIN先验资源分离',True),
        ('数据质量分层和抽样限制有逐账号清单',True),('可重复的12条TRAIN人工复核材料齐备',True),
        ('12条分层独立人工复核实际完成',False),('人工分歧处理并冻结首版Gold子集',False)]
    stage_gates=[
        ('恢复三个指定成果包并验证嵌套哈希',True),('复核README、协议和运行记录',True),
        ('完成300条结构验收和源文件保护检查',True),('原评分与评价v2重新运行并对齐',True),
        ('现有20项及v2的5项测试重新通过',True),('完成质量报告与逐账号分层',True),
        ('按源证据选择12个TRAIN账号',True),('生成完整复核任务且Gold保持PENDING',True),
        ('提供离线代码恢复、执行脚本和实际日志',True),('形成报告、GitHub待提交补丁和可下载包',True)]
    status={'as_of_asia_shanghai':stamp,'deadline':'2026-10-25 Asia/Shanghai','dataset_v1_readiness_percent':80,
        'dataset_progress_basis':'8/10 equal-weight project acceptance gates; not annotation accuracy, model accuracy, or statistical confidence',
        'dataset_gates':[{'gate':k,'complete':v} for k,v in dataset_gates],
        'requested_stage_engineering_percent':100,'stage_progress_basis':'10/10 scoped engineering deliverables, excluding independent human review explicitly left PENDING',
        'stage_gates':[{'gate':k,'complete':v} for k,v in stage_gates],
        'dataset_rows':qa['row_count'],'independent_human_gold_completed':0,'human_review_queue_n':12,
        'test_count_this_run':acc['test_count'],'structural_checks':qa['structural_check_n'],
        'real_model_retrained_this_round':False,'frozen_artifacts_unchanged':acc['protected_files_unchanged'],
        'github_pushed':False,'model_effectiveness_established':False,'status':'ENGINEERING_ACCEPTED__HUMAN_GOLD_PENDING__NO_MODEL_SUPERIORITY_CLAIM'}
    dump(root/'MILESTONE_STATUS.json',status)
    lines=['# Wiki300 v1 第一阶段验收状态',f'\n验收时间：{stamp}；目标日期：2026年10月25日（北京时间）。',
        '\n**工程验收完成；人工Gold未完成；真实模型优于对照的结论尚未成立。**',
        '\n## 进度口径','\n|对象|当前完成度|计算方法|','|---|---:|---|',
        '|第一版科研可审查数据交付|80%|下面10个等权工作门槛完成8个|',
        '|本轮指定阶段工程成果|100%|下面10项工程交付完成10项；人审按要求保持PENDING|',
        '|独立人工复核首批12条|0%|0/12；300条均不自动称为人工Gold|',
        '\n这些百分比是本交付的项目管理清单，不是科研通用成熟度分数，也不是模型准确率。未用主观比例掩盖人审和模型问题。',
        '\n## 数据交付10项门槛']
    lines += [f'- [{"x" if done else " "}] {name}' for name,done in dataset_gates]
    lines += ['\n## 本轮工程10项交付']+[f'- [x] {name}' for name,_ in stage_gates]
    lines += ['\n## 实际验收结果',
        f'\n- 三个指定ZIP全部恢复；合并包内两个ZIP与独立包SHA256一致；缺失文件0。',
        f'- {qa["row_count"]}条，61 EXACT、139 INTERVAL、100右删失对照；TRAIN/VAL/TEST=226/24/50。',
        f'- {qa["structural_check_n"]}/{qa["structural_check_n"]}项结构条件通过；账号ID与标准化用户名无重复，跨表ID完整；100组显式匹配关系同分区。',
        f'- 本轮实际通过{acc["test_count"]}项测试=20项Exp04/Exp05+5项评价v2+8项交付约束测试。测试夹具内部会运行小型合成拟合；没有重新训练Wiki300真实模型。',
        '- 在全新目录从离线Git历史恢复Tag2并应用原补丁，代码树与原适配提交完全一致；原评分输出与已存结果相等，v2的5个输出全部复现。',
        '- 12/12复核对象来自TRAIN并复用原36条队列；其中3条是延迟先验使用的TRAIN精确样本；4个窄区间均≤1天。',
        '- 复核包含修改前后正文、原始URL、时间、历史、风险与缺口；人工A/B及仲裁均PENDING。完整任务可从缓存逐字重建。',
        '- 保护检查确认冻结数据、标签、分区、参数、预测及原评价包未被改写。没有推送GitHub。',
        '\n## 复现范围与边界',
        '\n本轮复现的是代码恢复、I/O检查、已保存预测的独立评分和单元测试；没有重新运行真实模型拟合，也不宣称已在另一套依赖环境验证从头训练的数值一致性。此前训练与合成回归的原日志保留在artifacts/exp05。',
        '\n本包代码分支可追溯到Tag2的commit `06fa34d2097b8ca4fa209dbf60ba551fef31a33d` 和适配commit `57555b536f2395dd39d4c329917a7eef1acdf77c`。Tag1/Tag2均未移动。',
        '\n## 剩余任务与下一步3件事',
        '\n1. 由真人完成本包12条TRAIN复核，先核3条先验样本及公开过滤日志样本；记录确认、修改或证据不足，不覆盖旧标签。',
        '2. 处理复核分歧，冻结有实际审阅记录的Gold子集；先补12案明确缺失的关键证据，不扩大随机采集。',
        '3. 审查待提交补丁与阶段文档后，再决定推送GitHub；后续模型改进只在TRAIN/VAL开展，另设未暴露账号作最终检验。',
        '\n## 阅读入口',
        '\n- DATASET_QUALITY_REPORT.md：质量分层、风险及计数。',
        '- EVALUATION_REPORT.md：真实模型与两个消融对照的实际结果。',
        '- HUMAN_REVIEW_QUEUE/REVIEW_ALL_12.html：可离线展开阅读的12份材料。',
        '- HUMAN_REVIEW_QUEUE/HUMAN_REVIEW_QUEUE.csv：真人登记表。',
        '- GITHUB_RELEASE_PLAN.md：待提交内容及复现命令。',
        '- reports/acceptance_results.json及logs/acceptance：实际命令、退出码、测试数和结果。']
    (root/'MILESTONE_STATUS.md').write_text('\n'.join(lines)+'\n')
    def table_counts(title,counts,denom):
        return '\n'+title+'\n\n|分层|数量|占该组比例|\n|---|---:|---:|\n'+'\n'.join(f'|{k}|{v}|{100*v/denom:.1f}%|' for k,v in counts.items())+'\n'
    quality=['# Wiki300 v1 数据质量报告',f'\n生成时间：{stamp}。来源为三个恢复成果包，统计以冻结JSONL、I/O和实际验收输出为准。',
        '\n## 数据是什么',
        '\n一条数据对应一个账号的指定观察窗。200条是有正常前缀和后续公开违规证据的转变，100条是已审窗口未观察到定义内违规的右删失对照。任务是给定首次相关封禁后的事后起点恢复；`td`不是保证的首次发现时间。所有既有判定为AI审核，独立人工Gold=0。',
        '\n61条EXACT中51条能对齐真实发布修订，10条是公开过滤等日志的精确UTC事件；后者不强行转成精确发布位置。139条INTERVAL保留`(lower,upper]`，没有填造点标签。',
        '\n## 核心验收',
        f'\n{qa["structural_check_n"]}项结构条件全部通过，错误0。其含义是已缓存文件内部满足规则，不代表重新证明每一条语义判断。',
        '\n|项目|实际结果|','|---|---|',
        '|记录数、ID和标准化用户名|300条、300个唯一账号、0重复|',
        '|源记录 / gold / split / timeline / raw_inputs / NPZ|账号ID完整对齐，未出现孤立记录|',
        '|固定分区|TRAIN226、VAL24、TEST50|',
        '|显式匹配关系|100个对照的匹配正样本均可按原account_id解析，同组不跨分区|',
        '|已声明组数|200；不保证未知关联身份或共享页面已经隔离|',
        '|时间|UTC有时区，非递减；起点和区间在观察边界内；发布编辑严格早于封禁|',
        '|同秒事件|9个账号存在相邻同秒记录；未猜测秒内因果次序|',
        '|序列|300×90×8存储；真实长度2–90；补零不当事件；200个终点快照复制最后发布状态|',
        '|输入保护|输入不含tf/判案理由/封禁理由；gold物理分离；先验仅36个TRAIN精确发布事件|',
        '|原始文件保护|旧manifest哈希均匹配；本轮读前读后保护检查一致|',
        '\n## 分区与标签', '\n|分区|EXACT|INTERVAL|右删失|合计|','|---|---:|---:|---:|---:|']
    for sp,d in qa['split_by_label'].items():quality.append(f'|{sp}|{d["EXACT"]}|{d["INTERVAL"]}|{d["NOT_FRAUD"]}|{sum(d.values())}|')
    quality += [table_counts('## 200条转变的正常前缀',qa['positive_normal_prefix_groups'],200),
        '\n只有1条前缀的118条占59%。这不自动推翻原纳入判定，但不足以代表长期正常习惯，研究行为变化时应单列。61条EXACT中，41条仅1次前缀、9条2次、11条至少3次。',
        table_counts('## 139条区间的宽度',qa['interval_width_groups'],139),
        f'\n区间中位宽度{qa["interval_width_median_days"]:.3f}天，最长{qa["interval_width_max_days"]:.3f}天。宽区间是真实不确定性，命中不能解释为精确定位。',
        table_counts('## 100条对照的观察窗',qa['control_window_groups'],100),
        '\n52条不足1天，28条不少于7天；其中不少于7天且≥5次编辑的仅14条。窗口长但编辑少仍不等于行为信息充分。正常仅限定已审窗口，不能声称终身正常。',
        '\n## 延迟、正文和证据形式',
        f'\n从精确tf或区间upper到首次相关封禁，54/200条间隔>30天，5/200条>365天。对INTERVAL，这里统计`td-upper`，只是实际延迟的保守下界，真实延迟属于`[td-upper,td-lower)`；不把未知tf填成upper。',
        '\n200/200转变账号的已标正常前缀都保留了修改前后文本，100/100对照的窗口内全部事件保留差异正文。197/200转变有标准增删正文字段，另3条是结构化移动日志及关联修订，并非凭空缺失的空白编辑。',
        '\n|结构化移动证据账号|实际证据形式|','|---|---|',
        '|wiki_u54084390|IPad移动至无关标题的move_log+linked_revision；INTERVAL|',
        '|wiki_u55358238|Instagram移动至攻击性无关标题的move_log+linked_revision；INTERVAL|',
        '|wiki_u56175585|人物条目移动至无关标题的move_log+linked_revision；EXACT|',
        '\n上述正文覆盖率只描述已挑出的证据字段；不表示所有中间贡献都有本地全文。12案逐条列出仅有元数据的中间事件和原始封禁API响应等缺口。本轮没有全站重抓取，也没有实时验证所有链接当前可读性。',
        '\n## 主要风险与处置',
        '\n|风险|证据与影响|程度 / 信心|最小下一步|','|---|---|---|---|',
        '|没有独立人工Gold|0条真人终审；AI共识不能充当独立真值|高 / 高|完成12条TRAIN复核并留理由、分歧、审阅者记录|',
        '|正常历史薄|118/200仅1条前缀；难以充分验证行为变化|高 / 高|保留PREFIX分层；新研究结论单列长前缀组|',
        '|宽区间|51/139超过30天；命中容易而定位证据弱|高 / 高|按宽度报告，不合并为准确率|',
        '|短对照及有限采样|52/100不足1天；候选来自有限框架|高 / 高|报告窗口、编辑数；不宣称全站代表性|',
        '|相关封禁是处置代理|54条保守延迟>30天；不等于首次发现|中 / 高|标明td定义和延迟单位|',
        '|同秒与日志事件|9个同秒账号、10个序列外EXACT|中 / 高|保留UTC并分流，不推断秒内次序|',
        '|正文与模型信息不同|模型仅8维累计元数据，未读取审核使用的正文语义|高 / 高|下一阶段先在TRAIN/VAL检查可用输入，不以TEST调参|',
        '\n## 可审查证据',
        '\n完整逐账号分层见reports/audit/dataset_strata_300.csv；所有结构条件见reports/audit/structural_checks.json。scripts/audit_dataset.py和notebooks/quality_review.ipynb保留可执行检查。质量分层不会自动剔除账号，也不会改写冻结标签。']
    (root/'DATASET_QUALITY_REPORT.md').write_text('\n'.join(quality)+'\n')
    evaluation=['# Wiki300 v1 模型评价验收',f'\n生成时间：{stamp}。本轮没有真实模型重训、重新预测或参数选择，只读取已封存的预测重新评分。',
        '\n## 方法与监督声明',
        '\n双状态对角高斯行为EM + 仅TRAIN的全局行为尺度 + 经验延迟先验 + 固定融合；w=2/3，T=1/3。整体延迟先验使用36条TRAIN精确起点，因此属于部分起点监督，不能称整个流程完全无起点监督。行为EM本身不读取数值tf。',
        '\n## 评分复现结果',
        '\n原50号评分器在独立输出目录重新生成结果，与原wiki_metrics.json内容相同；v2的评分、逐账号表、数据分布和两个36条队列共5个输出，与已存版本一致。v2测试5项、原接口/长度/评分20项均通过。本轮另有8项交付约束测试，共33项。',
        '\nTEST固定50条：9个精确UTC点、29个区间、12个正常窗口。38条转变账号有起点预测；12个对照输出NOT_DETECTED_UNSUPPORTED，不虚构tf。',
        '\n## 精确UTC点：同一批9个账号',
        '\n|方法|平均绝对误差（小时）↓|中位绝对误差（小时）↓|','|---|---:|---:|']
    for name,label in [('fusion','完整SafePlus'),('behavior','只看行为'),('delay','只看延迟')]:
        evaluation.append(f'|{label}|{sc["exact_point_utc_mae_days"][name]*24:.4f}|{sc["exact_point_utc_median_abs_days"][name]*24:.4f}|')
    evaluation += ['\n融合平均21.22小时，中位约11.63分钟；6/9误差<1小时，另外3条约47.43、66.74、75.88小时。需要同时看均值与尾部失败，不能事后只选较漂亮的中位数。',
        '\n融合相对只看行为的平均UTC误差约高14.8%，是这9条上的点估计，不能当作稳定差异；原配对bootstrap区间跨0，尚无稳定优势证据。',
        '\n## 精确发布位置：同一批8个账号',
        '\n另1条精确UTC起点落在两次发布编辑之间，不能当作精确编辑编号。以下分母均为8，不与9条混用。',
        '\n|方法|平均绝对位置误差↓|中位位置误差↓|完全对齐|差≤1次编辑|','|---|---:|---:|---:|---:|']
    for name,label in [('fusion','完整SafePlus'),('behavior','只看行为'),('delay','只看延迟')]:
        evaluation.append(f'|{label}|{sc["exact_event_abs_error_mean"][name]:.3f}|{sc["exact_event_abs_error_median"][name]:.2f}|{sc["exact_event_hit_exact"][name]}/8|{sc["exact_event_hit_within_1"][name]}/8|')
    evaluation += ['\n融合对只看行为：2胜、3负、3平；对只看延迟：2胜、5负、1平。单位是发布编辑位置，不能按固定小时换算。分数归一化是辅助观察，不会使短历史、长空窗或宽区间自动公平。',
        '\n## 区间：保留左开右闭并按宽度分层',
        '\n|真值区间宽度|账号数|融合命中|行为命中|延迟命中|融合离区间距离中位数（天）↓|','|---|---:|---:|---:|---:|---:|']
    for key,label in [('le_1d','≤1天'),('1_to_7d','>1至7天'),('7_to_30d','>7至30天'),('gt_30d','>30天')]:
        z=sc['interval_stratified'][key];evaluation.append(f'|{label}|{z["fusion"]["n"]}|{z["fusion"]["hits"]}|{z["behavior"]["hits"]}|{z["delay"]["hits"]}|{z["fusion"]["median_distance_days"]:.6f}|')
    evaluation += ['\n整体融合15/29、行为15/29、延迟17/29。≤7天的融合仅3/16，>30天为9/10；整体51.7%命中数会掩盖窄区间表现差的问题，不能称分类准确率。离区间平均距离分别为融合5.579天、行为0.474天、延迟10.702天；它们不是精确起点误差，不能与EXACT的MAE相加。',
        '\n若预测恰好等于lower，距离可为0但仍不命中；upper包含在区间内。该边界已有测试覆盖。',
        '\n## 已验证与尚未验证',
        '\n已验证：冻结输入/分区保护、标签类型分流、原评分与v2可重现、预测支持位置合法、TRAIN先验隔离、padding处理与人审队列规则。',
        '\n未验证/不支持：独立人工真值、全站泛化、在线提前发现、正常账号AUC/误报率、融合稳定超过对照。没有正常风险输出，12个对照不能算作12次分类正确。',
        '\n本轮没有重新拟合真实模型。原真实拟合33轮收敛、合成132个可评测对象MAE=1.234848步等信息来自原保留日志与原回归产物；本轮不把历史合成训练日志称为新执行的完整合成训练回归。',
        '\n## 下一步解释边界',
        '\n现有TEST已经看过，称exploratory test。现有结果应作为首轮基准公开保留；任何输入、先验或权重调整仅在TRAIN/VAL进行，最终效果需新的未暴露账号验证。本阶段不新增复杂模型、不以换单位获得更好的叙事。']
    (root/'EVALUATION_REPORT.md').write_text('\n'.join(evaluation)+'\n')
    print(json.dumps({'status':'REPORTS_GENERATED','dataset_percent':80,'stage_percent':100,'tests':acc['test_count'],'human_gold':0},ensure_ascii=False))

if __name__=='__main__':main()
