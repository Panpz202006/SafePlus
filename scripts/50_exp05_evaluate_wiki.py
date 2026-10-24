"""Independent Wiki UTC/interval evaluation; run only after prediction is sealed.

Never use an interval lower/upper endpoint as a point target. An exact UTC time
between published edits is evaluated in UTC, not as a fabricated exact edit index.
"""
from __future__ import annotations
import argparse,csv,json,hashlib
from datetime import datetime
from pathlib import Path
import numpy as np

def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sec(s):return datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
def integer(s):
    x=float(s)
    if not np.isfinite(x) or int(x)!=x:raise ValueError('Noninteger predicted event index')
    return int(x)
def ci_gain(base,new,seed=20261024):
    gains=np.asarray(base,dtype=float)-np.asarray(new,dtype=float)
    if not len(gains):return None
    rng=np.random.default_rng(seed);means=np.mean(gains[rng.integers(0,len(gains),(5000,len(gains)))],axis=1)
    lo,hi=np.quantile(means,[.025,.975])
    return {'n':len(gains),'mean_gain_seconds':float(gains.mean()),'ci95_seconds':[float(lo),float(hi)]}

def evaluate(bundle:Path,predictions:Path):
    manifest=json.loads((predictions/'run_manifest.json').read_text(encoding='utf-8'))
    split=manifest['split']
    wanted={r['entity_id'] for r in read_csv(bundle/'split.csv') if r['split']==split}
    rows=read_csv(predictions/'predictions.csv');lookup={r['entity_id']:r for r in rows}
    if len(lookup)!=len(rows) or set(lookup)!=wanted:raise ValueError('Predictions must match every declared split ID exactly')
    goldrows=read_csv(bundle/'gold_onset.csv');gold={r['entity_id']:r for r in goldrows}
    if len(gold)!=len(goldrows):raise ValueError('Duplicate gold ID')
    timeline={r['entity_id']:r for r in map(json.loads,(bundle/'timeline.jsonl').read_text(encoding='utf-8').splitlines())}
    if manifest['input_sha256']!=hashlib.sha256((bundle/'inputs.npz').read_bytes()).hexdigest():raise ValueError('Input changed after prediction')
    if manifest['split_sha256']!=hashlib.sha256((bundle/'split.csv').read_bytes()).hexdigest():raise ValueError('Split changed after prediction')
    ntypes={};status={};details=[];clock={name:[] for name in ['final','behavior_only','delay_only']};step={name:[] for name in clock};interval={name:[] for name in clock};hits={name:[] for name in clock}
    for eid in sorted(wanted):
        r=lookup[eid];g=gold[eid];t=timeline[eid]['times_utc'];typ=g['onset_label_type'];s=r['status']
        ntypes[typ]=ntypes.get(typ,0)+1;status[s]=status.get(s,0)+1
        detail={'entity_id':eid,'source_onset_type':g['original_onset_type'],'label_type':typ,'prediction_status':s,'true_tf_utc':g['onset_utc'],'lower_utc':g['onset_lower_utc'],'upper_utc':g['onset_upper_utc'],'pred_tf_utc':'','exact_abs_error_seconds':'','interval_distance_seconds':'','interval_hit':''}
        if typ=='NOT_FRAUD':
            if s!='NOT_DETECTED_UNSUPPORTED' or r['pred_tf_median']:raise ValueError('Censored control received an onset')
            details.append(detail);continue
        if s!='OK':details.append(detail);continue
        d=integer(r['observed_td'])
        points={}
        for name,col in [('final','pred_tf_median'),('behavior_only','behavior_only_median'),('delay_only','delay_only_median')]:
            k=integer(r[col])
            if not 0<=k<d<len(t):raise ValueError('Predicted onset outside strict pre-td support')
            points[name]=(k,sec(t[k]))
        detail['pred_tf_utc']=t[points['final'][0]]
        if typ=='EXACT':
            if not g['onset_utc']:raise ValueError('Missing exact UTC gold')
            for name,(k,p) in points.items():
                clock[name].append(abs(p-sec(g['onset_utc'])))
                if g['point_is_published_event']=='true':step[name].append(abs(k-integer(g['t_onset'])))
            detail['exact_abs_error_seconds']=clock['final'][-1]
        elif typ=='INTERVAL':
            if g['onset_utc'] or g['t_onset'] or g['interval_closure']!='(lower,upper]':raise ValueError('Interval changed to point')
            lo,hi=sec(g['onset_lower_utc']),sec(g['onset_upper_utc'])
            if not lo<hi:raise ValueError('Invalid interval')
            for name,(k,p) in points.items():interval[name].append(max(lo-p,0,p-hi));hits[name].append(lo<p<=hi)
            detail['interval_distance_seconds']=interval['final'][-1];detail['interval_hit']=hits['final'][-1]
        else:raise ValueError('Unexpected Wiki gold type')
        details.append(detail)
    mean=lambda x:float(np.mean(x)) if len(x) else None
    report={'dataset_id':manifest['dataset_id'],'split':split,'n_split':len(rows),'gold_counts':ntypes,'prediction_status_counts':status,
            'exact_utc_n':len(clock['final']),'exact_event_step_n':len(step['final']),
            'exact_mae_seconds':{name:mean(vals) for name,vals in clock.items()},
            'exact_mae_days':{name:mean(np.asarray(vals)/86400) for name,vals in clock.items()},
            'exact_event_position_mae':{name:mean(vals) for name,vals in step.items()},
            'paired_clock_error_gain_vs_behavior_only':ci_gain(clock['behavior_only'],clock['final']),
            'paired_clock_error_gain_vs_delay_only':ci_gain(clock['delay_only'],clock['final']),
            'interval_n':len(interval['final']),
            'interval_point_hit_rate_open_closed':{name:mean(vals) for name,vals in hits.items()},
            'interval_distance_days':{name:mean(np.asarray(vals)/86400) for name,vals in interval.items()},
            'interval_distance_is_not_exact_error':True,'td_is_first_related_block_not_first_discovery':True,
            'prior_supervision':manifest.get('onset_supervision_for_prior'),
            'notes':['Fixed first-run metadata features; no hyperparameters selected from this TEST evaluation',
                     'Only nine source EXACT samples are in this fixed TEST; small-sample results are not stable evidence',
                     'Delay prior operates in observed event-position steps, not elapsed calendar time',
                     'Normal means no observed violation in a reviewed right-censored window',
                     'Exact UTC points and UTC intervals are unchanged; no independent human sign-off is claimed']}
    (predictions/'wiki_metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (predictions/'wiki_per_account_evaluation.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True,type=Path);p.add_argument('--predictions',required=True,type=Path)
    a=p.parse_args();print(json.dumps(evaluate(a.bundle,a.predictions),ensure_ascii=False,indent=2))
