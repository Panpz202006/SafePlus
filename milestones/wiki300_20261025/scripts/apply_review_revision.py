#!/usr/bin/env python3
"""Apply explicit user-rule decisions to existing records, preserving frozen v1."""
from __future__ import annotations
import copy,csv,datetime,hashlib,html,json,shutil
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];REV='review12_user_rules_v1_1_20261024'
def jl(p):return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def dump(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def writejl(p,rows):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in rows),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    work=ROOT/'review_revision_work';current=ROOT/'data_review';history=ROOT/'history/review_before_rules_v1_1'
    current.mkdir(exist_ok=True)
    if (current/'REVISION_STATUS.json').exists():raise RuntimeError('Revision already applied: inspect existing results instead of duplicating history')
    history.mkdir(parents=True,exist_ok=True)
    shutil.copytree(ROOT/'HUMAN_REVIEW_QUEUE',history/'HUMAN_REVIEW_QUEUE',dirs_exist_ok=True)
    for name in ['README.md','MILESTONE_STATUS.md','MILESTONE_STATUS.json','DATASET_QUALITY_REPORT.md','GITHUB_RELEASE_PLAN.md','manifest_sha256.json']:
        shutil.copy2(ROOT/name,history/name)
    original=work/'current_source/SafePlus_combined_current.jsonl';frozen=ROOT/'artifacts/exp05/frozen_source/SafePlus_combined_current.jsonl'
    assert original.read_bytes()==frozen.read_bytes(),'Authoritative current source differs from preserved frozen version'
    originals=jl(original);decisions=json.loads((work/'decisions.json').read_text());byid={'wiki_u'+str(x['userid']):x for x in originals}
    split={r['entity_id']:r['split'] for r in csv.DictReader((ROOT/'artifacts/exp05/data_bundle/split.csv').open())}
    raw={r['entity_id']:r for r in jl(ROOT/'artifacts/exp05/data_bundle/raw_inputs.jsonl')}
    source_tasks={x['entity_id']:x for x in jl(history/'HUMAN_REVIEW_QUEUE/HUMAN_REVIEW_QUEUE.jsonl')}
    for d in decisions:assert split[d['entity_id']]=='TRAIN'
    shutil.copytree(work/'evidence',current/'evidence',dirs_exist_ok=True)
    shutil.copy2(work/'decisions.json',current/'DECISIONS_v1_1.json')
    diffs=jl(current/'evidence/derived_revision_diffs.jsonl');diffmap={x['revision_id']:x for x in diffs}
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    provenance={'mode':'USER_REPORTED_AI_ASSISTED_DISCUSSION','user_reported_discussion_complete':True,'independent_human_annotation':False,'independent_human_reviewers':[],
        'scope':'User supplies final rules and candidate-specific directions; AI performs evidence checks and writes decisions. No independent per-event user annotation is asserted.',
        'recorded_utc':stamp,'human_gold_status':'PENDING'}
    reviews=[];updated={};tasks=[];csvout=[]
    for d in decisions:
        uid=d['entity_id'];old=byid[uid];r=copy.deepcopy(old);task=copy.deepcopy(source_tasks[uid]);label=d['label'];kind=d['event_kind']
        normal=copy.deepcopy(old.get('normal_prefix',old.get('events',[])))
        revmap={str(x['revision_id']):x for x in raw[uid]['events']}
        for e in normal:
            meta=revmap.get(str(e.get('revision_id')))
            if meta:e['namespace']=int(meta['namespace'])
        positive=label in ['EXACT','INTERVAL']
        scope='REVIEWED_PUBLISHED_VIOLATION_ONSET' if positive and kind=='PUBLISHED_EDIT' else 'REVIEWED_VIOLATION_ATTEMPT_ONSET' if kind=='VIOLATION_ATTEMPT' else 'NO_CONFIRMED_VIOLATION_IN_ACTUAL_WINDOW' if label=='RIGHT_CENSORED' else 'VIOLATION_NOT_ESTABLISHED'
        annotation={'revision_id':REV,'task_number':d['number'],'entity_id':uid,'username':old['username'],'split':split[uid],
            'label_type':label,'onset_event_kind':kind,'label_scope':scope,'formal_review_status':d['status'],
            'human_participation_status':'DISCUSSION_REPORTED','ai_evidence_review_status':'CHECKED','human_gold_status':'PENDING',
            'independent_human_review_completed':False,'human_review_provenance':provenance,
            'tf_utc':old.get('tf_utc') if label=='EXACT' else None,
            'tf_lower_utc':(old.get('tf_lower_utc') or old.get('tf_utc')) if positive else None,
            'tf_upper_utc':(old.get('tf_upper_utc') or old.get('tf_utc')) if positive else None,
            'interval_closure':'(lower,upper]' if label=='INTERVAL' else None,
            'observation_start_utc':task['observation_start_utc'],'observation_end_utc':task['observation_end_utc'],'td_utc':old.get('td_utc'),
            'normal_prefix_evidence':normal,'reason':d['reason'],'unresolved_or_limited_evidence':d['issues'],'fixes':d['fixes'],
            'review_standard':'DATA_REVIEW_STANDARD.md v1.1','frozen_source_sha256':sha(frozen),
            'source_record_path':f'../artifacts/exp05/frozen_source/SafePlus_combined_current.jsonl',
            'earliestness_scope':'Within the actual reviewed public window; inaccessible histories are not claimed absent',
            'minimum_normal_days_required':None,'minimum_normal_edits_required':None,
            'length_or_window_used_as_exclusion':False,'td_substituted_for_tf':False,
            'eligible_for_confirmed_onset_dataset':positive,'eligible_for_right_censored_dataset':label=='RIGHT_CENSORED',
            'published_violation_onset_utc':old.get('tf_utc') if label=='EXACT' and kind=='PUBLISHED_EDIT' else None,
            'original_label_type':('RIGHT_CENSORED' if not old.get('onset_type') else 'EXACT' if old['onset_type'].startswith('EXACT') else old['onset_type'])}
        if d['number']==4:
            log=json.loads((current/'evidence/visable_abuse_details.json').read_text())['query']['abuselog'][0]
            assert log['result']=='warn' and log['timestamp']==annotation['tf_utc'] and len(log['details']['new_wikitext'])>0
            annotation['first_confirmed_violation_attempt_utc']=log['timestamp']
            annotation['attempt_evidence']={'log_id':44727043,'result':'warn','full_submitted_content_available':True,'source_url':'https://en.wikipedia.org/wiki/Special:AbuseLog/44727043','publication_proven_by_warn':False}
            annotation['later_save_metadata']={'log_id':44727049,'revision_id':1365980014,'timestamp_utc':'2026-07-25T15:06:59Z','status':'METADATA_ONLY_REVISION_BODY_UNAVAILABLE','confirms_same_content':False}
        if d['number'] in [5,6,7,8]:
            ids=[int(x['revision_id']) for x in raw[uid]['events']]
            annotation['additional_revision_checks']=[x for x in diffs if x['revision_id'] in ids]
        if d['number']==7:
            annotation['rd3_report']={'origin':'USER_REPORTED','operation_claim':'RD3: Purely disruptive material; four versions hidden together','api_verified':False,'admin':None,'timestamp_utc':None,
                'supports_if_authenticated':'A batch-level administrative finding concerning disruptive material',
                'does_not_support':['Every revision in the batch separately violated policy','The first covered revision must be the first violation','Impersonation of another person by a self-signed reply'],
                'verification_limit':'Target page log and nearby one-hour revision-delete log do not identify this batch; this is not proof that no later or inaccessible action exists'}
            annotation['direct_context_evidence']={'warning_revision_id':1367135911,'warning_time_utc':'2026-08-01T05:41:15Z','repeated_removal_revision_id':1367135958,'removal_time_utc':'2026-08-01T05:41:58Z','first_decline_signature':'Clerk013 themselves, not another named administrator'}
        if d['number']==8:
            compare=json.loads((current/'evidence/chimp_verified_compare.json').read_text())['compare']
            assert compare['fromrevid']==1373782788 and compare['torevid']==1373783294 and compare['fromid']==compare['toid']
            url='https://en.wikipedia.org/wiki/Special:Diff/1373782788/1373783294'
            annotation['verified_diff']={'url':url,'old_revision_id':1373782788,'new_revision_id':1373783294,'same_page_confirmed_by_api':True,'old_ids_found_incorrect':False,'ordinary_web_reader_status':'DISABLED_ERROR; official compare API succeeded'}
            annotation['verified_moves']=json.loads((current/'evidence/chimp_move_logs.json').read_text())['query']['logevents']
            annotation['candidate_onset_bounds_not_gold']={'lower_utc':old['tf_lower_utc'],'upper_utc':old['tf_upper_utc'],'source':'old v1 interval; retained only as a candidate'}
            r['candidate_onset_revision_id']=old.get('onset_revision_id');r['candidate_onset_bounds_not_gold']=annotation['candidate_onset_bounds_not_gold']
            r['onset_type']='UNCERTAIN';r['status']='uncertain';r['sample_type']='uncertain_candidate';r['entity_label']=None;r['event_observed']=None
            r['tf_utc']=None;r['tf_lower_utc']=None;r['tf_upper_utc']=None;r['onset_revision_id']=None;r['onset_url']=url
            r['onset_evidence']['source_url']=url;r['onset_evidence']['evidence_role']='CANDIDATE_EDIT_NOT_CONFIRMED_VIOLATION'
            r['qualifies_normal_to_violation']=None
        elif label=='RIGHT_CENSORED':
            r['onset_type']='RIGHT_CENSORED';r['status']='accepted';r['entity_label']=0;r['event_observed']=False;r['right_censored']=True
        else:r['status']='accepted'
        r['label_type']=label;r['onset_event_kind']=kind;r['label_scope']=scope;r['reason']=d['reason'];r['review_method']='Human-participatory rules and case discussion + AI raw-evidence verification; not independent human Gold'
        r['review_standard_version']='1.1';r['review_revision_id']=REV;r['formal_review_status']=d['status'];r['human_gold_status']='PENDING';r['independent_human_review_completed']=False
        r['human_review_provenance']=provenance;r['current_review']=annotation
        r['risk_flags']=[x for x in r.get('risk_flags',[]) if not any(k in x for k in ['正常前缀仅1条','正常历史较短','适合纳入但应在稳健性','不是长期正常编辑史','失败提交'])]
        r['review_limitations']=d['issues'];r['normal_prefix_namespace_counts']=dict(Counter(str(x['namespace']) for x in normal if 'namespace' in x))
        if positive:r['normal_prefix']=normal
        r['review_history']=r.get('review_history',[])+[{'version':'frozen300_AI_v1','original_annotation':{k:old.get(k) for k in ['onset_type','status','sample_type','entity_label','event_observed','tf_utc','tf_lower_utc','tf_upper_utc','onset_revision_id','onset_url','reason','rationale','risk_flags']},'source_sha256':sha(frozen)},{'version':REV,'recorded_utc':stamp,'label_type':label,'reason':d['reason'],'human_gold_status':'PENDING'}]
        if d['number']==4:
            r['first_confirmed_violation_attempt_utc']=annotation['tf_utc'];r['published_violation_onset_utc']=None;r['later_save_metadata']=annotation['later_save_metadata']
        updated[uid]=r;reviews.append(annotation)
        task['original_source_label_type']=task['source_label_type'];task['original_interval']=task.get('interval');task['original_onset_utc']=task.get('onset_utc')
        task['current_label_type']=label;task['onset_event_kind']=kind;task['formal_review_status']=d['status'];task['human_participation_status']='DISCUSSION_REPORTED';task['ai_evidence_review_status']='CHECKED';task['current_review']=annotation
        task['onset_utc']=annotation['tf_utc'];task['interval']={'lower_utc':annotation['tf_lower_utc'],'upper_utc':annotation['tf_upper_utc'],'closure':'(lower,upper]'} if label=='INTERVAL' else None
        task['risk_flags']=r['risk_flags'];task['missing_or_unverified']=d['issues'];tasks.append(task)
        dest=ROOT/'HUMAN_REVIEW_QUEUE/accounts'/uid
        dump(dest/'review_task.json',task);dump(dest/'formal_review.json',annotation)
        old_md=(history/'HUMAN_REVIEW_QUEUE/accounts'/uid/'REVIEW.md').read_text()
        header=f'# 当前正式审核｜{d["number"]:02d} {old["username"]}\n\n当前标签：**{label}**；事件种类：**{kind}**；正式状态：**{d["status"]}**。\n\n用户已报告完成AI辅助讨论；本条证据由AI核对，独立人工Gold仍为PENDING。\n\n## 本轮结论\n\n{d["reason"]}\n\n## 尚未确认\n\n'+('\n'.join('- '+x for x in d['issues']) or '- 无阻止当前标签使用的疑点；结论仅限实际已审窗口。')+'\n\n## 修复内容\n\n'+'\n'.join('- '+x for x in d['fixes'])+'\n\n## 原v1材料（历史引用，标签以本页上方为准）\n\n'
        (dest/'REVIEW.md').write_text(header+old_md,encoding='utf-8')
        csvout.append({'task_number':d['number'],'entity_id':uid,'username':old['username'],'split':split[uid],'original_label_type':annotation['original_label_type'],'current_label_type':label,'onset_event_kind':kind,
            'formal_review_status':d['status'],'human_participation_status':'DISCUSSION_REPORTED','ai_evidence_review_status':'CHECKED','human_gold_status':'PENDING','independent_human_review_completed':False,
            'tf_utc':annotation['tf_utc'] or '', 'tf_lower_utc':annotation['tf_lower_utc'] or '', 'tf_upper_utc':annotation['tf_upper_utc'] or '', 'interval_closure':annotation['interval_closure'] or '',
            'reason':d['reason'],'remaining_issues':' | '.join(d['issues']),'fixes':' | '.join(d['fixes']),'record_path':f'accounts/{uid}/formal_review.json',
            'reviewer_A_name':'','reviewer_A_status':'PENDING','reviewer_B_name':'','reviewer_B_status':'PENDING','independent_adjudication_status':'PENDING'})
    combined=[updated.get('wiki_u'+str(x['userid']),x) for x in originals]
    writejl(current/'SafePlus_combined_current.jsonl',combined);writejl(current/'review12_current.jsonl',reviews);writejl(ROOT/'HUMAN_REVIEW_QUEUE/HUMAN_REVIEW_QUEUE.jsonl',tasks)
    with (ROOT/'HUMAN_REVIEW_QUEUE/HUMAN_REVIEW_QUEUE.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(csvout[0]));w.writeheader();w.writerows(csvout)
    def labelof(x):return x.get('label_type') or ('RIGHT_CENSORED' if not x.get('onset_type') else 'EXACT' if x['onset_type'].startswith('EXACT') else x['onset_type'])
    counts=dict(Counter(x['label_type'] for x in reviews));allcounts=dict(Counter(labelof(x) for x in combined))
    status={'revision_id':REV,'recorded_utc':stamp,'records_revised':12,'record_revision_percent':100,'resolved_usable':11,'resolved_percent':100*11/12,'uncertain':1,'uncertain_percent':100/12,
        'review12_label_counts':counts,'exact_published_edit_n':3,'exact_violation_attempt_n':1,'independent_human_gold_n':0,'human_participatory_discussions_reported':12,
        'wiki300_rows':300,'wiki300_current_label_counts':allcounts,'wiki300_this_revision_coverage_percent':4,'unchanged_other_records':288,
        'split_sha256':sha(ROOT/'artifacts/exp05/data_bundle/split.csv'),'frozen_source_sha256':sha(frozen),'frozen_source_unchanged':True,
        'current_dataset_sha256':sha(current/'SafePlus_combined_current.jsonl'),'formal_dataset_path':'data_review/SafePlus_combined_current.jsonl','model_training_performed':False,'io_conversion_performed':False,
        'old_model_results_annotation_version':'frozen300_AI_v1; not rescored against this revision','blocking_case_ids':['wiki_u55507062'],'definition_decisions_needed':[],
        'evidence_followups':[{'entity_id':'wiki_u55025968','item':'RD3 source attribution remains unverified; not needed to invent an exact onset'},{'entity_id':'wiki_u54928085','item':'Published revision body 1365980014 unavailable; attempt exact time remains separate'},{'entity_id':'wiki_u55507062','item':'Need evidence distinguishing harmful rename from ordinary error before assigning a violation label'}]}
    dump(current/'REVISION_STATUS.json',status);dump(current/'RECORD_SCHEMA_v1_1.json',{'label_type':['EXACT','INTERVAL','RIGHT_CENSORED','UNCERTAIN'],'formal_review_status':['RESOLVED','EVIDENCE_PENDING'],'human_participation_status':['DISCUSSION_REPORTED'],'ai_evidence_review_status':['CHECKED'],'human_gold_status':['PENDING','HUMAN_GOLD_CONFIRMED'],'onset_event_kind':['PUBLISHED_EDIT','VIOLATION_ATTEMPT','NO_CONFIRMED_VIOLATION_IN_WINDOW','UNRESOLVED'],'human_gold_promotion_guard':'Requires real independent reviewer identity, completion time and evidence decisions; no promotion in this revision','uncertain_entity_label':None,'normal_window_minimum':None})
    # Safe, readable current HTML. Historical evidence is available but explicitly demoted.
    cards=[]
    for row,annotation in zip(csvout,reviews):
        uid=row['entity_id'];links=[]
        for e in annotation['normal_prefix_evidence']:
            if e.get('source_url'):links.append(e['source_url'])
        links.append(annotation.get('verified_diff',{}).get('url') or byid[uid].get('onset_url'))
        if byid[uid].get('block_url'):links.append(byid[uid]['block_url'])
        linkhtml=''.join(f'<li><a href="{html.escape(u,quote=True)}">{html.escape(u)}</a></li>' for u in dict.fromkeys(u for u in links if u))
        details=annotation.get('additional_revision_checks',[])
        diffhtml=''.join(f'<details><summary>修订{x["revision_id"]}｜{html.escape(x["timestamp"])}</summary><pre>{html.escape(chr(10).join(x["diff_lines"]))}</pre></details>' for x in details)
        originaltext=(history/'HUMAN_REVIEW_QUEUE/accounts'/uid/'REVIEW.md').read_text()
        cards.append(f'<article id="{uid}"><h2>{row["task_number"]:02d} {html.escape(row["username"])}</h2><p class="status">{row["current_label_type"]} · {row["onset_event_kind"]} · {row["formal_review_status"]}</p><p>用户参与讨论：已报告完成；AI证据核验：已执行；独立人工Gold：PENDING。</p><p>{html.escape(row["reason"])}</p><h3>未确认事项</h3><p>{html.escape(row["remaining_issues"] or "当前标签无阻塞事项，仍仅限实际已审窗口。")}</p><h3>原始证据入口</h3><ul>{linkhtml}</ul><h3>本轮补查的父子版本差异</h3>{diffhtml}<details><summary>原v1证据材料（历史标签不作本轮结论）</summary><pre>{html.escape(originaltext)}</pre></details></article>')
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>SafePlus Review All 12｜正式修订v1.1</title><style>body{max-width:1050px;margin:30px auto;padding:0 20px;background:#f4f7fa;color:#193047;font:16px/1.75 system-ui}article{background:white;border:1px solid #cad7df;border-radius:12px;padding:24px;margin:22px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 monospace}.status{background:#e9f2fb;padding:10px}summary{cursor:pointer;color:#075d83}a{overflow-wrap:anywhere}</style><h1>Review All 12｜正式修订v1.1</h1><p>记录修订12/12（100%）；可用标签11/12（91.7%）；UNCERTAIN 1/12（8.3%）；独立人工Gold 0。精确4条中，3条是已发布编辑，1条仅为已确认违规尝试。区间3条，右删失4条。</p><p>不因正常历史短或窗口短排除样本。旧证据与历史版本完整保留；本轮不训练、不转换模型输入、不改变分区。</p>'+''.join(cards)+'</html>'
    (ROOT/'HUMAN_REVIEW_QUEUE/REVIEW_ALL_12.html').write_text(page,encoding='utf-8')
    (ROOT/'HUMAN_REVIEW_QUEUE/README.md').write_text('# Review All 12 正式修订v1.1\n\n现行标签以HUMAN_REVIEW_QUEUE.csv、每案formal_review.json和../data_review/review12_current.jsonl为准。12条均已记录用户参与讨论，AI完成证据核对；独立人工Gold仍全部PENDING。11条有可用结论，Chimpanzeedeerkidney为UNCERTAIN待补证。\n\n原v1材料保存在../history/review_before_rules_v1_1/HUMAN_REVIEW_QUEUE。source_record.json仍是原始证据记录，不覆盖；不要把其中旧标签当作现行结论。原抽样清单只说明当时如何抽样，不是本轮最终标签分布。\n',encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
