#!/usr/bin/env python3
"""Create and actually execute a plain nbformat-4 notebook using stdlib only.

Cells are compiled/executed sequentially in one namespace and real stdout is
captured. No fabricated execution counts or model outputs; Jupyter is optional.
"""
import contextlib, io, json, os, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
cells=[]
def markdown(s):cells.append({'cell_type':'markdown','id':'cell-'+str(len(cells)),'metadata':{},'source':s.splitlines(keepends=True)})
def code(s):cells.append({'cell_type':'code','id':'cell-'+str(len(cells)),'metadata':{},'source':s.splitlines(keepends=True),'execution_count':None,'outputs':[]})
markdown('# Wiki300 冻结v1历史核查（不是现行审核标签）\n\n只读缓存、重评分及人工待审状态；不训练真实模型、不改标签、不联网。结构通过不等于人类Gold。完整审计代码在scripts/audit_dataset.py。\n')
code("from pathlib import Path\nimport sys, json, tempfile, contextlib, io\nROOT=Path.cwd().resolve()\nwhile not (ROOT/'artifacts/exp05').is_dir() and ROOT != ROOT.parent:\n    ROOT=ROOT.parent\nassert (ROOT/'artifacts/exp05').is_dir(), 'Run from the extracted milestone directory'\nsys.path.insert(0,str(ROOT/'scripts'))\nfrom audit_dataset import run, csvrows\nprint('Source: frozen Wiki300; execution does not fit any model')\n")
code("with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):\n    audit=run(ROOT/'artifacts/exp05', Path(d))\nassert audit['status']=='PASS'\nprint(json.dumps({k:audit[k] for k in ['row_count','label_counts','split_counts','exact_subtypes','positive_normal_prefix_groups','interval_width_groups','control_window_groups','structural_check_n','structural_checks_passed','errors']},ensure_ascii=False,indent=2))\n")
markdown('## 已封存预测的分流评分\n\n9个EXACT UTC与8个精确发布位置的分母不同。区间命中不是分类准确率；正常窗口没有风险预测，人工Gold仍为0。\n')
code("scores=json.loads((ROOT/'reports/evaluation_v2/exploratory_score_v2.json').read_text())\nprint('EXACT UTC n=',scores['exact_point_utc_n'])\nprint('Mean absolute clock error (hours):', {k:round(v*24,6) for k,v in scores['exact_point_utc_mae_days'].items()})\nprint('Exact published event n=',scores['exact_published_event_n'])\nprint('Event-position MAE:',scores['exact_event_abs_error_mean'])\nprint('Interval strata:',json.dumps(scores['interval_stratified'],ensure_ascii=False,indent=2))\n")
code("from collections import Counter\nQUEUE_ROOT=ROOT/'history/review_before_rules_v1_1/HUMAN_REVIEW_QUEUE'\nif not QUEUE_ROOT.exists(): QUEUE_ROOT=ROOT/'HUMAN_REVIEW_QUEUE'\nqueue=csvrows(QUEUE_ROOT/'HUMAN_REVIEW_QUEUE.csv')\nassert len(queue)==12 and all(x['split']=='TRAIN' for x in queue)\nassert all(x['human_gold_status']=='PENDING' for x in queue)\nprint('Review tasks:',len(queue),'TRAIN:',len(queue),'Human Gold complete:',0)\nprint('Original source types:',dict(Counter(x['source_label_type'] for x in queue)))\nprint('Dataset readiness: 8/10 project gates = 80%; engineering deliverables: 10/10 = 100%. These are not accuracy estimates.')\n")
namespace={'__name__':'__main__'};count=0
with contextlib.chdir(ROOT):
    for cell in cells:
        if cell['cell_type']!='code':continue
        count+=1;capture=io.StringIO()
        with contextlib.redirect_stdout(capture):exec(compile(''.join(cell['source']),f'<notebook-cell-{count}>','exec'),namespace)
        cell['execution_count']=count
        cell['outputs']=[{'output_type':'stream','name':'stdout','text':capture.getvalue().splitlines(keepends=True)}]
notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':sys.version.split()[0]},'execution_provenance':'Cells actually compiled/executed sequentially by scripts/build_notebook.py; captured stdout only; no nbformat package required'},'cells':cells}
target=ROOT/'notebooks/quality_review.ipynb';target.parent.mkdir(exist_ok=True)
target.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':'PASS','executed_code_cells':count,'path':'notebooks/quality_review.ipynb'},ensure_ascii=False))
