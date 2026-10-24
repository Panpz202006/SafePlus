#!/usr/bin/env python3
"""Twelve TRAIN review tasks, selected from old evidence queues, never predictions."""
from __future__ import annotations
import argparse, hashlib, html, json
from pathlib import Path
from audit_dataset import body_present, csvrows, dump, jsonl, utc, writecsv

RULES=[
 ('exact_public_prefix1','精确发布起点；1条正常前缀'),
 ('exact_public_prefix2','精确发布起点；2条正常前缀'),
 ('exact_public_prefix3plus','精确发布起点；至少3条正常前缀'),
 ('exact_other_log','精确公开日志；不伪造发布编辑位置'),
 ('interval_narrow_1','窄区间≤1天，第1条'),('interval_narrow_2','窄区间≤1天，第2条'),
 ('interval_narrow_3','窄区间≤1天，第3条'),('interval_narrow_4','窄区间≤1天，第4条'),
 ('control_short','正常观察窗<1天'),('control_middle','正常观察窗1至不足7天'),
 ('control_long_few','正常观察窗≥7天，至多4次编辑'),('control_long_more','正常观察窗≥7天，至少5次编辑')]

def select(source, splits, existing, prior):
    old={x['entity_id'] for x in existing}; selected=[]; used=set()
    for key,description in RULES:
        pool=[]
        for r in source:
            uid='wiki_u'+str(r['userid'])
            if splits[uid]!='TRAIN' or uid in used: continue
            typ=r.get('onset_type','');n=int(r.get('normal_prefix_count') or 0)
            days=float(r.get('observation_duration_days') or 0)
            exact=typ.startswith('EXACT');other=typ=='EXACT_PUBLIC_LOG_TIMESTAMP'
            ok=(key=='exact_public_prefix1' and exact and not other and n==1 or
                key=='exact_public_prefix2' and exact and not other and n==2 or
                key=='exact_public_prefix3plus' and exact and not other and n>=3 or
                key=='exact_other_log' and other or
                key.startswith('interval_narrow_') and typ=='INTERVAL' and (utc(r['tf_upper_utc'])-utc(r['tf_lower_utc']))/86400<=1 or
                key=='control_short' and not typ and days<1 or
                key=='control_middle' and not typ and 1<=days<7 or
                key=='control_long_few' and not typ and days>=7 and len(r['events'])<=4 or
                key=='control_long_more' and not typ and days>=7 and len(r['events'])>=5)
            if ok:
                evidence=r.get('events',[]) if not typ else r['normal_prefix']+[r.get('onset_evidence',{})]
                complete=all(body_present(x) for x in evidence)
                rank=(uid not in old,not complete,uid not in prior,hashlib.sha256(('wiki300-milestone-review-v1|'+key+'|'+uid).encode()).hexdigest())
                pool.append((rank,uid,r))
        if not pool: raise ValueError('No source candidate for '+key)
        _,uid,r=min(pool,key=lambda x:x[0]);used.add(uid)
        selected.append((key,description,r,uid in old,uid in prior))
    return selected

def event_md(e):
    time=e.get('timestamp_utc') or e.get('timestamp') or '时间见原始证据'
    rid=e.get('revision_id') or e.get('revid') or e.get('event_id') or '不适用/未记录'
    url=e.get('source_url') or e.get('url') or ''
    text=[f'- 时间：`{time}`；修订/事件：`{rid}`；页面：{e.get("title", "见原始来源")}',f'- 原始来源：{url or "未记录，需补证"}']
    if e.get('edit_summary'):text.append('- 原编辑摘要：'+e['edit_summary'])
    if e.get('review'):text.append('- 原AI逐事件说明（非人工结论）：'+json.dumps(e['review'],ensure_ascii=False))
    if body_present(e):
        changes=e.get('changes') or [{'deleted':e.get('deleted',''),'added':e.get('added','')}]
        if not any('deleted' in x or 'added' in x for x in changes):changes=[e.get('values',{})]
        for i,c in enumerate(changes,1):
            deleted=c.get('deleted',c.get('old_wikitext',''));added=c.get('added',c.get('new_wikitext',''))
            for name,value in [('删除/修改前',deleted),('新增/修改后',added)]:
                # Fence length exceeds any literal backtick run in the cached evidence.
                fence='`'*max(3,max([len(x) for x in __import__('re').findall(r'`+',str(value))]+[0])+1)
                text += [f'\n{name}（差异块{i}；空串表示该侧没有文本）：',f'{fence}text\n{value}\n{fence}']
    else:text.append('- 缓存中没有标准增删正文；保持待补证，不将空缺当作空白编辑。')
    return '\n'.join(text)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path);a=p.parse_args();root=a.root.resolve()
    art=root/'artifacts/exp05';b=art/'data_bundle';out=(a.out or root/'HUMAN_REVIEW_QUEUE').resolve();out.mkdir(parents=True,exist_ok=True)
    if (root/'data_review/REVISION_STATUS.json').exists() and out==(root/'HUMAN_REVIEW_QUEUE').resolve():
        raise ValueError('Current formal review revision exists. This generator recreates frozen v1 only; use --out OTHER_DIR and do not overwrite the current queue.')
    source=jsonl(art/'frozen_source/SafePlus_combined_current.jsonl');splits={x['entity_id']:x['split'] for x in csvrows(b/'split.csv')}
    existing=csvrows(root/'code/evaluation_v2/reports/independent_review_queue_36.csv');prior={x['entity_id'] for x in csvrows(b/'train_onset_for_prior.csv')}
    raws={x['entity_id']:x for x in jsonl(b/'raw_inputs.jsonl')}
    selected=select(source,splits,existing,prior);tasks=[];human=[];sections=[]
    for number,(key,desc,r,in_old,in_prior) in enumerate(selected,1):
        uid='wiki_u'+str(r['userid']);typ=r.get('onset_type','RIGHT_CENSORED');iscontrol=typ=='RIGHT_CENSORED'
        ev=raws[uid]['events'];proof=r.get('events',[]) if iscontrol else r['normal_prefix'];body_ids={str(x.get('revision_id')) for x in proof}
        onset=r.get('onset_evidence',{})
        if r.get('onset_revision_id'):body_ids.add(str(r['onset_revision_id']))
        missing=[]
        if not iscontrol:
            if not body_present(onset):missing.append('起点增删正文缺标准字段，需检查原始来源/嵌套日志')
            if any(not body_present(x) for x in proof):missing.append('部分正常前缀正文缺失')
            missing.append('本包未保存首次封禁日志的独立API响应；仅保留日志ID、原始链接和既有摘录，需人工点击核对')
        unreviewed=[x for x in ev if str(x['revision_id']) not in body_ids]
        if unreviewed:missing.append(f'{len(unreviewed)}次其他已发布编辑仅有历史元数据；本包未逐条附其正文，不足以单凭本包重新证明绝对更早性')
        missing.append('未重新联网检查链接当前可读性；公开过滤/删除内容可能不可见，无法核实时记录INSUFFICIENT')
        flags=list(r.get('risk_flags',[]));reason=r.get('reason') or r.get('rationale') or ''
        if not iscontrol and int(r['normal_prefix_count'])<3:flags.append('正常历史较短；不得据此描述长期正常习惯')
        if typ=='EXACT_PUBLIC_LOG_TIMESTAMP':flags.append('起点为公开过滤日志，时间精确但不属于已发布编辑序列；不能伪造事件编号真值')
        interval=None
        if typ=='INTERVAL':interval={'lower_utc':r['tf_lower_utc'],'upper_utc':r['tf_upper_utc'],'closure':'(lower,upper]','width_seconds':utc(r['tf_upper_utc'])-utc(r['tf_lower_utc'])}
        task={'task_number':number,'entity_id':uid,'username':r['username'],'split':'TRAIN','selection_stratum':key,'selection_reason':desc,
            'reused_existing_queue':in_old,'used_by_train_prior':in_prior,'source_label_type':typ,
            'onset_utc':r.get('tf_utc'),'interval':interval,'td_utc':r.get('td_utc'),
            'observation_start_utc':raws[uid]['events'][0]['timestamp_utc'],'observation_end_utc':raws[uid]['observation_end_utc'],
            'normal_history_count':len(proof),'published_revision_count':len(ev),'original_ai_rationale':reason,
            'onset_source_url':r.get('onset_url'),'block_log_id':r.get('block_log_id'),'block_source_url':r.get('block_url'),
            'block_reason_excerpt':r.get('block_reason'),'normal_evidence':proof,'onset_evidence':onset,
            'coverage_claimed_by_previous_ai_review':r.get('coverage',{}),'all_published_metadata':ev,
            'risk_flags':flags,'missing_or_unverified':missing,'human_gold_status':'PENDING',
            'reviewer_A_status':'PENDING','reviewer_B_status':'PENDING','adjudication_status':'PENDING'}
        tasks.append(task);d=out/'accounts'/uid;d.mkdir(parents=True,exist_ok=True)
        dump(d/'review_task.json',task);dump(d/'source_record.json',r)
        md=[f'# 任务{number:02d}｜{uid}｜{r["username"]}',
            '\n**人工Gold：PENDING。以下判断和理由来自原AI审核，不是独立人工结论。**',
            f'\n抽样：TRAIN；{desc}。复用原队列：{in_old}；用于TRAIN延迟先验：{in_prior}。未读取模型预测或TEST误差进行选择。',
            '\n## 时间与标签',f'\n- 原类型：`{typ}`',f'- 观察窗口：`{task["observation_start_utc"]}` 至 `{task["observation_end_utc"]}`（UTC）',
            f'- 精确起点：`{r.get("tf_utc") or "不适用"}`',
            f'- 起点区间：`{json.dumps(interval,ensure_ascii=False) if interval else "不适用"}`',
            f'- 首次相关封禁：`{r.get("td_utc") or "不适用；正常对照的终点是右删失，不是封禁"}`',
            '\n## 原AI理由与待确认事项','\n'+reason,
            '\n'+('\n'.join('- '+x for x in flags) or '- 无单列风险标志；仍须逐项复核。'),
            '\n## 正常历史 / 正常观察窗逐事件证据']
        for i,e in enumerate(proof,1):md += [f'\n### 正常事件{i}',event_md(e)]
        md+=['\n## 违规证据（正常对照不适用）']
        if iscontrol:md+=['\n此窗口未观察到违规；不能确认终身正常，不填写tf。']
        else:md += [f'\n原始起点来源：{r.get("onset_url") or "未记录"}',event_md(onset)]
        md += ['\n## 封禁及覆盖范围',f'\n- 日志：{r.get("block_url") or "不适用/原包未提供独立日志链接"}',f'- 日志ID：`{r.get("block_log_id") or "不适用"}`',f'- 既有封禁理由摘录：{r.get("block_reason") or "不适用"}',
            '\n原AI声称的覆盖字段（须独立复核，不能自动当作人审通过）：', '```json\n'+json.dumps(r.get('coverage',{}),ensure_ascii=False,indent=2)+'\n```',
            '\n## 全部已缓存发布历史（不只展示挑出的证据）','\n|位置（0起）|时间UTC|修订ID|页面|','|---:|---|---|---|']
        for i,e in enumerate(ev):md.append(f'|{i}|{e["timestamp_utc"]}|{e["revision_id"]}|{e.get("title", "").replace("|", "/")}|')
        md+=['\n## 缺失、疑点和继续位置','\n'+'\n'.join('- '+x for x in missing),
            '\n## 人工操作清单',
            '\n1. 核对账号、原始修订ID、父版本和UTC时间，打开原始差异/日志。',
            '2. 阅读正常历史正文，确认可接受的正常行为及观察范围；不能只看编辑摘要。',
            '3. 对转变样本核对具体违规内容，检查更早贡献/过滤日志是否推翻“首次可核验”；正常对照则逐条核窗口内容。',
            '4. 对区间分别核下界正常证据和上界违规证据，保留左开右闭；不得改成精确点。',
            '5. 记录CONFIRM、REVISE或INSUFFICIENT，并写原始链接及理由；修改建议另存，禁止覆盖冻结标签。',
            '6. A/B审阅人分别填写；有分歧则仲裁。未实际审阅的字段保持PENDING。',
            '\nA状态：PENDING；B状态：PENDING；仲裁：PENDING；人工Gold：PENDING。']
        markdown='\n'.join(md)+'\n';(d/'REVIEW.md').write_text(markdown,encoding='utf-8')
        # A single offline HTML with every cached evidence block, safely escaped.
        urls=[x.get('source_url') or x.get('url') for x in proof]+[r.get('onset_url'),r.get('block_url')]
        links=''.join(f'<li><a href="{html.escape(u,quote=True)}" target="_blank" rel="noopener noreferrer">{html.escape(u)}</a></li>' for u in dict.fromkeys(u for u in urls if u and u.startswith('https://en.wikipedia.org/')))
        sections.append(f'<article id="{uid}"><h2>{number:02d}. {html.escape(uid)} · {html.escape(r["username"])}</h2><details><summary>展开完整审核材料（含修改前后正文）</summary><p>原始证据链接：</p><ul>{links}</ul><pre>{html.escape(markdown)}</pre></details></article>')
        human.append({'task_number':number,'entity_id':uid,'username':r['username'],'split':'TRAIN','source_label_type':typ,'selection_reason':desc,
            'used_by_train_prior':in_prior,'packet_path':f'accounts/{uid}/REVIEW.md','human_gold_status':'PENDING',
            'reviewer_A_name':'','reviewer_A_status':'PENDING','reviewer_A_identity_check':'PENDING','reviewer_A_normal_history_check':'PENDING','reviewer_A_violation_check':'PENDING','reviewer_A_earliestness_check':'PENDING','reviewer_A_boundary_check':'PENDING','reviewer_A_reason':'','reviewer_A_evidence_urls':'','reviewer_A_completed_utc':'',
            'reviewer_B_name':'','reviewer_B_status':'PENDING','reviewer_B_reason':'','reviewer_B_evidence_urls':'','reviewer_B_completed_utc':'',
            'proposed_label_type':'','proposed_onset_utc':'','proposed_lower_utc':'','proposed_upper_utc':'','adjudication_status':'PENDING','adjudicator_name':'','adjudication_reason':''})
    writecsv(out/'HUMAN_REVIEW_QUEUE.csv',human)
    (out/'HUMAN_REVIEW_QUEUE.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in tasks),encoding='utf-8')
    manifest={'n':len(tasks),'split_counts':{'TRAIN':len(tasks)},'exact_n':4,'narrow_interval_n':4,'control_n':4,
        'existing_queue_reused_n':sum(x['reused_existing_queue'] for x in tasks),'train_prior_review_n':sum(x['used_by_train_prior'] for x in tasks),
        'prediction_files_read_for_selection':False,'selection_rule_version':'TRAIN_SOURCE_STRATIFIED_V1','human_gold_completed':0,
        'rules':['4 exact: public prefix1/prefix2/prefix3plus plus other-log; 4 interval <=1 day; controls by duration and edit-count strata',
                 'Within each stratum: existing36 queue first, body coverage then prior-use then fixed SHA256 tie-break; no error-based choice'],
        'selected':[{'entity_id':x['entity_id'],'stratum':x['selection_stratum']} for x in tasks]}
    dump(out/'SELECTION_MANIFEST.json',manifest)
    intro='''# 12条TRAIN人工复核任务\n\n人工Gold完成0/12；全部PENDING。这是证据整理包，不是人工审核结果。\n\n打开 REVIEW_ALL_12.html 可离线阅读所有材料；每个账号的 REVIEW.md 和 source_record.json 保留详细证据。HUMAN_REVIEW_QUEUE.csv 用于登记真实审阅结果。证据缺口已逐案列出，未重新抓取正文或实时验证链接。\n\n4个精确起点包含1个公开过滤日志，4个窄区间均≤1天，4个正常窗口覆盖短/中/长窗口。全部来自TRAIN并复用原队列，不根据模型误差挑选。A/B是人工工作角色，AI不代填。\n\n人工应核身份、正常行为、具体违规、更早性、边界、来源链接。可填CONFIRM/REVISE/INSUFFICIENT；Gold批准必须记录真人与时间、理由及分歧处理。原AI理由只供核查，不是默认正确答案。正常对照不填tf，区间不自动收紧为点。\n'''
    (out/'README.md').write_text(intro,encoding='utf-8')
    doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>SafePlus｜12条人工复核材料</title><style>body{max-width:1100px;margin:36px auto;padding:0 24px;font:16px/1.7 system-ui;color:#152b3c;background:#f6f8fa}article{background:white;border:1px solid #ccd8df;border-radius:10px;padding:18px;margin:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.65 system-ui}summary{cursor:pointer;color:#075c83}h1{font-size:28px}.badge{background:#fff1cd;padding:10px}</style><h1>SafePlus｜12条TRAIN人工复核材料</h1><p class="badge">人工Gold：0/12。全部PENDING。原AI理由和缓存内容不代表独立人工确认。</p><p>本页面可离线阅读；原始Wikipedia链接需联网打开。每条保留正文、时间、历史、证据缺口和操作清单。完整结构化任务及来源记录随包提供。</p>'+''.join(sections)+'</html>'
    (out/'REVIEW_ALL_12.html').write_text(doc,encoding='utf-8');print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
