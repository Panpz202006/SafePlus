#!/usr/bin/env python3
"""Validate official review records without fitting or converting model inputs."""
import csv,hashlib,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def jl(p):return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def validate():
    base=jl(ROOT/'artifacts/exp05/frozen_source/SafePlus_combined_current.jsonl');cur=jl(ROOT/'data_review/SafePlus_combined_current.jsonl');reviews=jl(ROOT/'data_review/review12_current.jsonl');tasks=jl(ROOT/'HUMAN_REVIEW_QUEUE/HUMAN_REVIEW_QUEUE.jsonl')
    before={'wiki_u'+str(x['userid']):x for x in base};now={'wiki_u'+str(x['userid']):x for x in cur};rr={x['entity_id']:x for x in reviews}
    checks=[]
    def ck(name,value):checks.append({'check':name,'passed':bool(value)})
    ck('300 unique rows retained',len(cur)==len(now)==300 and set(before)==set(now))
    changed={uid for uid in now if now[uid]!=before[uid]};ck('exactly twelve known rows revised',changed==set(rr) and len(changed)==12)
    ck('all remaining 288 source objects unchanged',sum(now[uid]==before[uid] for uid in now)==288)
    oldmanifest=json.loads((ROOT/'history/review_before_rules_v1_1/manifest_sha256.json').read_text())['files']
    frozen={n:v for n,v in oldmanifest.items() if n.startswith('artifacts/')}
    ck('all frozen source, input, labels, splits, model parameters and predictions unchanged',all(sha(ROOT/n)==v for n,v in frozen.items()))
    ck('review12 label counts',Counter(x['label_type'] for x in reviews)=={'EXACT':4,'INTERVAL':3,'RIGHT_CENSORED':4,'UNCERTAIN':1})
    ck('eleven resolved and one evidence pending',Counter(x['formal_review_status'] for x in reviews)=={'RESOLVED':11,'EVIDENCE_PENDING':1})
    ck('no false independent human Gold',all(x['human_gold_status']=='PENDING' and not x['independent_human_review_completed'] and not x['human_review_provenance']['independent_human_reviewers'] for x in reviews))
    ck('user participation recorded without impersonating independent annotation',all(x['human_participation_status']=='DISCUSSION_REPORTED' and x['human_review_provenance']['independent_human_annotation'] is False for x in reviews))
    with (ROOT/'artifacts/exp05/data_bundle/split.csv').open() as f:split={x['entity_id']:x['split'] for x in csv.DictReader(f)}
    ck('twelve remain TRAIN',all(split[x['entity_id']]=='TRAIN' for x in reviews))
    ck('no duration or edit-count minimum',all(x['minimum_normal_days_required'] is None and x['minimum_normal_edits_required'] is None and not x['length_or_window_used_as_exclusion'] for x in reviews))
    ck('normal evidence namespace retained',all('namespace' in e for x in reviews for e in x['normal_prefix_evidence']))
    for x in reviews:
        label=x['label_type'];uid=x['entity_id'];r=now[uid]
        ck(uid+' current combined and review agree',r['label_type']==label and r['current_review']==x)
        if label=='EXACT':ck(uid+' point with declared event kind',bool(x['tf_utc']) and x['onset_event_kind'] in ['PUBLISHED_EDIT','VIOLATION_ATTEMPT'])
        elif label=='INTERVAL':ck(uid+' interval not collapsed',x['tf_utc'] is None and x['tf_lower_utc']<x['tf_upper_utc'] and x['interval_closure']=='(lower,upper]')
        elif label=='RIGHT_CENSORED':ck(uid+' censored no fabricated tf',not x['tf_utc'] and not x['tf_lower_utc'] and not x['tf_upper_utc'] and r['right_censored'] is True)
        elif label=='UNCERTAIN':ck(uid+' uncertainty not forced to negative or positive',r['entity_label'] is None and r['event_observed'] is None and not r['tf_utc'] and not r['tf_lower_utc'] and not r['tf_upper_utc'] and not x['eligible_for_confirmed_onset_dataset'] and not x['eligible_for_right_censored_dataset'])
    case4=rr['wiki_u54928085'];ck('Warn not equated to publication or failure',case4['attempt_evidence']['publication_proven_by_warn'] is False and case4['published_violation_onset_utc'] is None and case4['later_save_metadata']['confirms_same_content'] is False)
    case7=rr['wiki_u55025968'];ck('RD3 report not falsely API verified or first-edit exact',not case7['rd3_report']['api_verified'] and case7['label_type']=='INTERVAL' and case7['tf_utc'] is None)
    case8=rr['wiki_u55507062'];ck('actual compare parents confirmed',case8['verified_diff']['same_page_confirmed_by_api'] and case8['verified_diff']['old_revision_id']==1373782788 and case8['verified_diff']['new_revision_id']==1373783294)
    ck('historical 12 evidence packets preserved',all((ROOT/'history/review_before_rules_v1_1/HUMAN_REVIEW_QUEUE/accounts'/uid/'source_record.json').exists() for uid in rr))
    ck('original account evidence records unmodified',all((ROOT/'HUMAN_REVIEW_QUEUE/accounts'/uid/'source_record.json').read_bytes()==(ROOT/'history/review_before_rules_v1_1/HUMAN_REVIEW_QUEUE/accounts'/uid/'source_record.json').read_bytes() for uid in rr))
    ck('queue tasks reflect all current decisions',len(tasks)==12 and all(x['current_review']==rr[x['entity_id']] for x in tasks))
    with (ROOT/'HUMAN_REVIEW_QUEUE/HUMAN_REVIEW_QUEUE.csv').open(encoding='utf-8-sig') as f:csvq=list(csv.DictReader(f))
    ck('official CSV reflects all current decisions',len(csvq)==12 and all(x['current_label_type']==rr[x['entity_id']]['label_type'] and x['formal_review_status']==rr[x['entity_id']]['formal_review_status'] for x in csvq))
    return {'status':'PASS' if all(x['passed'] for x in checks) else 'FAIL','checks_passed':sum(x['passed'] for x in checks),'checks_total':len(checks),'checks':checks,'frozen_files_checked':len(frozen),'current_dataset_sha256':sha(ROOT/'data_review/SafePlus_combined_current.jsonl')}
if __name__=='__main__':
    result=validate();target=ROOT/'data_review/VALIDATION_REPORT.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2));raise SystemExit(0 if result['status']=='PASS' else 1)
