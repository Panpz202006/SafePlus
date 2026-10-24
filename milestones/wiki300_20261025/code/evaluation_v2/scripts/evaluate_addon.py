#!/usr/bin/env python3
"""Additive evaluation and quality audit for SafePlus Wiki300 Exp05 v1.

READ-ONLY with respect to the frozen source, I/O bundle, and original run.
This is an exploratory re-scoring of an *already inspected* TEST set, NOT a fresh
untouched held-out evaluation. No model fitting or hyperparameter selection.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


def utc(s: str) -> float:
    if not s:
        raise ValueError("Expected a non-empty UTC timestamp")
    d = datetime.fromisoformat(s.replace('Z', '+00:00'))
    if d.tzinfo is None:
        raise ValueError('UTC timestamp lacks offset')
    return d.timestamp()


def read_csv(path: Path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path):
    with path.open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict], columns: list[str]):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def median(a):
    return statistics.median(a) if a else None


def mean(a):
    return sum(a)/len(a) if a else None


def interval_hit(pred: float, lo: float, hi: float) -> bool:
    return lo < pred <= hi


def interval_distance(pred: float, lo: float, hi: float) -> float:
    if pred < lo: return lo - pred
    if pred > hi: return pred - hi
    return 0.0


def width_group(days: float) -> str:
    if days <= 1: return 'le_1d'
    if days <= 7: return '1_to_7d'
    if days <= 30: return '7_to_30d'
    return 'gt_30d'


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def indexed_unique(rows, key: str):
    out={}
    for r in rows:
        k=r[key]
        if k in out: raise ValueError(f'Duplicate {key}: {k}')
        out[k]=r
    return out


def candidate_score(item: dict, bucket: str):
    token=f'safeplus-audit-queue-v1|{bucket}|{item["userid"]}'
    return hashlib.sha256(token.encode('utf8')).hexdigest()


def select_audit_queue(source_rows, existing_test_ids):
    """A reproducible *review queue*, not an automatic gold annotation.
    Stratification uses source descriptions only; NEVER reads model predictions.
    """
    pools=defaultdict(list)
    for r in source_rows:
        onset=r.get('onset_type')
        uid='wiki_u'+str(r['userid'])
        if onset and onset.startswith('EXACT'):
            n=int(r.get('normal_prefix_count') or 0)
            bucket='exact_pre3plus' if n>=3 else 'exact_pre2' if n==2 else 'exact_pre1'
        elif onset=='INTERVAL':
            days=(utc(r['tf_upper_utc'])-utc(r['tf_lower_utc']))/86400
            bucket='interval_'+width_group(days)
        else:
            days=float(r.get('observation_duration_days') or 0)
            bucket='control_ge7d' if days>=7 else ('control_1to7d' if days>=1 else 'control_lt1d')
        pools[bucket].append(r)
    quotas={
        'exact_pre3plus':4,'exact_pre2':3,'exact_pre1':5,
        'interval_le_1d':4,'interval_1_to_7d':2,'interval_7_to_30d':2,'interval_gt_30d':4,
        'control_lt1d':4,'control_1to7d':4,'control_ge7d':4,
    }
    out=[]
    for bucket,limit in quotas.items():
        population=sorted(pools[bucket],key=lambda x:candidate_score(x,bucket))
        if len(population)<limit: raise ValueError(f'Insufficient {bucket} candidates: {len(population)} < {limit}')
        for r in population[:limit]:
            uid='wiki_u'+str(r['userid'])
            if r.get('onset_type'):
                link=r.get('onset_url') or r.get('block_url') or ''
                width=((utc(r['tf_upper_utc'])-utc(r['tf_lower_utc']))/86400
                       if r.get('onset_type')=='INTERVAL' else None)
                prefix=r.get('normal_prefix_count')
                duration=''
            else:
                events=r.get('events',[])
                link=(events[0].get('source_url','') if events else '')
                width=None;prefix='';duration=r.get('observation_duration_days')
            out.append({
                'entity_id':uid,'review_bucket':bucket,'existing_split_test':uid in existing_test_ids,
                'source_evidence_url':link,'normal_prefix_count':prefix,
                'interval_width_days':round(width,4) if width is not None else '',
                'control_window_days':duration,
                'reviewer_A_decision':'PENDING','reviewer_B_decision':'PENDING',
                'adjudicated_gold':'NO','review_notes':'',
            })
    return out


def qa_from_source(rows):
    exact=[r for r in rows if r.get('onset_type') and r['onset_type'].startswith('EXACT')]
    interval=[r for r in rows if r.get('onset_type')=='INTERVAL']
    ctrl=[r for r in rows if not r.get('onset_type')]
    intervals=[(utc(r['tf_upper_utc'])-utc(r['tf_lower_utc']))/86400 for r in interval]
    normal_long=[r for r in ctrl if float(r.get('observation_duration_days') or 0)>=7]
    return {
        'dataset_rows':len(rows),'source_status':'AI-adjudicated; NOT independent human Gold',
        'exact':len(exact),'interval':len(interval),'right_censored_controls':len(ctrl),
        'exact_normal_prefix_one':sum(int(r.get('normal_prefix_count') or 0)==1 for r in exact),
        'exact_normal_prefix_at_least_2':sum(int(r.get('normal_prefix_count') or 0)>=2 for r in exact),
        'exact_normal_prefix_at_least_3':sum(int(r.get('normal_prefix_count') or 0)>=3 for r in exact),
        'all_positives_normal_prefix_one':sum(int(r.get('normal_prefix_count') or 0)==1 for r in exact+interval),
        'interval_width_group_counts':dict(Counter(width_group(d) for d in intervals)),
        'interval_width_median_days':median(intervals),
        'interval_over_30days':sum(d>30 for d in intervals),
        'control_under_1day':sum(float(r.get('observation_duration_days') or 0)<1 for r in ctrl),
        'control_at_least_7days':len(normal_long),
        'control_at_least_7days_and_at_least_5_edits':sum(len(r.get('events',[]))>=5 for r in normal_long),
    }


def score(bundle: Path, run: Path, source: Path, out: Path):
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))
    name=manifest.get('split')
    if name!='TEST': raise ValueError('Addon currently describes sealed TEST predictions only')
    for fn,manifest_field in [('inputs.npz','input_sha256'),('split.csv','split_sha256')]:
        if manifest.get(manifest_field)!=sha(bundle/fn):
            raise ValueError(f'Cannot score changed input/split: {fn}')
    gold=indexed_unique(read_csv(bundle/'gold_onset.csv'),'entity_id')
    p=indexed_unique(read_csv(run/'predictions.csv'),'entity_id')
    sp=indexed_unique(read_csv(bundle/'split.csv'),'entity_id')
    ti=indexed_unique(read_jsonl(bundle/'timeline.jsonl'),'entity_id')
    test={k for k,x in sp.items() if x['split']=='TEST'}
    if set(p)!=test: raise ValueError('Predictions and predefined TEST users differ')
    if not test<=set(gold) or not test<=set(ti):raise ValueError('Missing TEST ground truth or timeline')

    per=[]; exact_errors={n:[] for n in ('fusion','behavior','delay')}
    exact_step={n:[] for n in exact_errors}
    exact_norm={n:[] for n in exact_errors}
    interval_by_group=defaultdict(lambda:{k:{'hits':0,'n':0,'distance_days':[]} for k in exact_errors})
    pairwise={base:{'win':0,'loss':0,'tie':0,'n':0} for base in ('behavior','delay')}
    statuses=Counter(); labels=Counter(); snap_predictions=0
    for uid in sorted(test):
        g=gold[uid];pr=p[uid];tr=ti[uid];label=g['onset_label_type'];labels[label]+=1
        status=pr['status'];statuses[status]+=1
        kinds=tr['row_kinds'];times=tr['times_utc'];n_pub=sum(k=='published_revision' for k in kinds)
        if len(kinds)!=len(times):raise ValueError('Mismatch timeline rows')
        item={'entity_id':uid,'label_type':label,'status':status,'observed_published_edits':n_pub,
              'total_observation_days':round((utc(tr['observation_end_utc'])-utc(tr['observation_start_utc']))/86400,6),
              'exact_fusion_event_error':'','exact_behavior_event_error':'','exact_delay_event_error':'',
              'exact_fusion_norm_event_error':'','exact_fusion_norm_time_error':'',
              'interval_width_days':'','interval_width_group':'','interval_hit_fusion':'','interval_distance_days_fusion':'',
              'model_beats_behavior_on_event':'','model_beats_delay_on_event':''}
        if label=='NOT_FRAUD':
            if status!='NOT_DETECTED_UNSUPPORTED' or pr.get('pred_tf_median'):
                raise ValueError('Right-censored controls must NOT be treated as onset hits')
            per.append(item);continue
        if status!='OK':per.append(item);continue
        d=int(pr['observed_td'])
        if d!=n_pub:raise ValueError('Expected td at exactly the terminal pre-decision snapshot')
        predictions={}
        for name,column in [('fusion','pred_tf_median'),('behavior','behavior_only_median'),('delay','delay_only_median')]:
            k=int(pr[column]);
            if k<0 or k>=d: raise ValueError('Onset index is not on published-event support')
            predictions[name]=(k,utc(times[k]))
        origin=utc(tr['observation_start_utc']);end=utc(tr['observation_end_utc'])
        horizon=end-origin
        if label=='EXACT':
            target=utc(g['onset_utc'])
            e={name:abs(timestamp-target) for name,(_,timestamp) in predictions.items()}
            for name,v in e.items():exact_errors[name].append(v/86400)
            item['exact_fusion_norm_time_error']=round(e['fusion']/horizon,8) if horizon>0 else ''
            if g['point_is_published_event']=='true':
                actual=int(g['t_onset'])
                if actual<0 or actual>=d or abs(utc(times[actual])-target)>1e-4:
                    raise ValueError('Exact event mapping is not truly exact')
                for name,(k,_) in predictions.items():
                    err=abs(k-actual);exact_step[name].append(err)
                    exact_norm[name].append(err/(n_pub-1) if n_pub>1 else None)
                    item[f'exact_{name}_event_error']=err
                item['exact_fusion_norm_event_error']=round(abs(predictions['fusion'][0]-actual)/(n_pub-1),8) if n_pub>1 else ''
                for base in ('behavior','delay'):
                    stats=pairwise[base];stats['n']+=1
                    fb=abs(predictions['fusion'][0]-actual);bb=abs(predictions[base][0]-actual)
                    cmp='win' if fb<bb else ('loss' if fb>bb else 'tie')
                    stats[cmp]+=1;item[f'model_beats_{base}_on_event']=cmp
        elif label=='INTERVAL':
            lo=utc(g['onset_lower_utc']);hi=utc(g['onset_upper_utc'])
            if not lo<hi or g['onset_utc'] or g['t_onset']:
                raise ValueError('Interval is not a valid open-closed range')
            width=(hi-lo)/86400;group=width_group(width)
            item['interval_width_days']=round(width,5)
            item['interval_width_group']=group
            for name,(_,timestamp) in predictions.items():
                q=interval_by_group[group][name];q['n']+=1
                hit=interval_hit(timestamp,lo,hi)
                q['hits']+=int(hit)
                dist=interval_distance(timestamp,lo,hi)/86400
                q['distance_days'].append(dist)
                if name=='fusion':
                    item['interval_hit_fusion']=hit;item['interval_distance_days_fusion']=round(dist,6)
        else:raise ValueError('Unsupported onset type')
        per.append(item)

    score_summary={
        'scope':'EXPLORATORY_RESCORING_OF_ALREADY_OPENED_TEST__NOT_NEW_UNTOUCHED_TEST',
        'test_n':len(test),'test_gold_types':dict(labels),'prediction_status':dict(statuses),
        'exact_point_utc_n':len(exact_errors['fusion']),
        'exact_point_utc_mae_days':{k:mean(v) for k,v in exact_errors.items()},
        'exact_point_utc_median_abs_days':{k:median(v) for k,v in exact_errors.items()},
        'exact_published_event_n':len(exact_step['fusion']),
        'exact_event_abs_error_median':{k:median(v) for k,v in exact_step.items()},
        'exact_event_abs_error_mean':{k:mean(v) for k,v in exact_step.items()},
        'exact_event_normalized_error_median':{k:median([v0 for v0 in v if v0 is not None]) for k,v in exact_norm.items()},
        'exact_event_hit_exact':{k:sum(v0==0 for v0 in v) for k,v in exact_step.items()},
        'exact_event_hit_within_1':{k:sum(v0<=1 for v0 in v) for k,v in exact_step.items()},
        'paired_exact_event_comparison':pairwise,
        'interval_stratified':{
            group:{k:{'n':v['n'],'hits':v['hits'],'hit_rate':v['hits']/v['n'] if v['n'] else None,
                      'median_distance_days':median(v['distance_days'])} for k,v in by.items()}
            for group,by in sorted(interval_by_group.items())},
        'notes':['No single scalar combines exact UTC, interval, and censored controls.',
                 'Normalized event error is secondary and may be unstable with very short histories.',
                 'Interval hits are easier for wide ranges; always stratify by range width.',
                 'Censored controls have no fraud risk prediction in this existing runner: no false-positive rate can be reported.',
                 'First-block time is a proxy for detection, not necessarily first discovery.',
                 'Original TEST is already viewed: never retune to this score and claim untouched performance.'],
        'input_sha256':sha(bundle/'inputs.npz'),'split_sha256':sha(bundle/'split.csv'),
        'run_predictions_sha256':sha(run/'predictions.csv')}
    write_json(out/'exploratory_score_v2.json',score_summary)
    write_csv(out/'per_account_exploratory_v2.csv',per,list(per[0].keys()))
    source_rows=read_jsonl(source)
    qa=qa_from_source(source_rows)
    write_json(out/'dataset_quality_audit_v2.json',qa)
    queue=select_audit_queue(source_rows,test)
    write_csv(out/'independent_review_queue_36.csv',queue,list(queue[0].keys()))
    source_lookup={"wiki_u"+str(r["userid"]):r for r in source_rows}
    prior_rows=read_csv(bundle/'train_onset_for_prior.csv')
    prior_review=[]
    for r in prior_rows:
        uid=r['entity_id']
        if uid not in source_lookup or sp[uid]['split']!='TRAIN':
            raise ValueError('TRAIN prior contains non-TRAIN user or missing source')
        original=source_lookup[uid]
        prior_review.append({
            'entity_id':uid,'train_onset_step':int(gold[uid]['t_onset']),
            'normal_prefix_count':original.get('normal_prefix_count'),
            'precise_onset_evidence_url':r['evidence_url'],
            'independent_human_review':'PENDING','confirmed_onset':'','notes':'',
        })
    write_csv(out/'train_prior_evidence_review_36.csv',prior_review,list(prior_review[0].keys()))
    return score_summary,qa,queue


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    scores,qa,queue=score(a.bundle,a.run,a.source,a.out)
    print(json.dumps({'status':'PASS_READ_ONLY',
                      'test_n':scores['test_n'],
                      'exact_published_event_n':scores['exact_published_event_n'],
                      'human_review_queue_n':len(queue),
                      'source_rows':qa['dataset_rows'],
                      'warning':scores['scope']},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
