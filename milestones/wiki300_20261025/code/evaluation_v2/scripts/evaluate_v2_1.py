#!/usr/bin/env python3
"""Extend existing v2 helpers: account-normalized frozen-prediction evaluation.

No fitting, inference, feature changes, window changes, or TEST-driven selection.
"""
from __future__ import annotations
import argparse,hashlib,importlib.util,json,math,random,statistics
from collections import Counter,defaultdict
from pathlib import Path

_spec=importlib.util.spec_from_file_location('safeplus_original_evaluation_v2',Path(__file__).with_name('evaluate_addon.py'))
v2=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(v2)
METHODS=('last_edit','time_midpoint','behavior','delay','fusion')
COLUMNS={'behavior':'behavior_only_median','delay':'delay_only_median','fusion':'pred_tf_median'}
PROTOCOL=Path(__file__).resolve().parents[1]/'docs/SCHEMA_AND_EVAL_V2_1.md'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def integer(value):
    x=float(value)
    if not math.isfinite(x) or x!=int(x):raise ValueError('Event index must be a finite integer')
    return int(x)
def quantile(values,q):
    if not 0<=q<=1:raise ValueError('q outside [0,1]')
    a=sorted(values)
    if not a:return None
    rank=(len(a)-1)*q;i=int(rank);j=min(i+1,len(a)-1)
    return a[i]+(a[j]-a[i])*(rank-i)
def describe(values):
    a=[float(x) for x in values if x is not None]
    return {'n':len(a),'median':quantile(a,.5),'p90':quantile(a,.9),'mean':statistics.mean(a) if a else None,'min':min(a) if a else None,'max':max(a) if a else None}
def normalize(loss,denominator,complete=True):
    return loss/denominator if complete and denominator>0 else None
def wilson(success,n,z=1.959963984540054):
    if not 0<=success<=n:raise ValueError('Invalid rate counts')
    if not n:return {'success':0,'n':0,'rate':None,'ci95':None}
    p=success/n;denom=1+z*z/n;center=(p+z*z/(2*n))/denom
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return {'success':success,'n':n,'rate':p,'ci95':[max(0,center-half),min(1,center+half)]}
def compare_loss(new,baseline,tolerance=1e-12):
    if math.isclose(new,baseline,rel_tol=tolerance,abs_tol=tolerance):return 'tie'
    return 'win' if new<baseline else 'loss'
def simple_baselines(times,start,end):
    if not times:raise ValueError('No published-event support')
    midpoint=(start+end)/2
    return {'last_edit':len(times)-1,'time_midpoint':min(range(len(times)),key=lambda k:(abs(times[k]-midpoint),k))}
def history_group(n):return '1' if n==1 else '2_to_5' if n<=5 else '6_to_10' if n<=10 else '11_to_20' if n<=20 else 'gt_20'
def prefix_group(n):return '1' if n==1 else '2' if n==2 else '3plus' if n>=3 else 'NOT_APPLICABLE'
def delay_group(days):return v2.width_group(days)
def interval_delay_group(lo_days,hi_days):
    if not 0<=lo_days<hi_days:raise ValueError('Bad interval delay range')
    # Delay is [td-U,td-L); exclude the upper endpoint when assigning bins.
    left=delay_group(lo_days);right=delay_group(math.nextafter(hi_days,-math.inf))
    return left if left==right else 'AMBIGUOUS_RANGE'
def label_of(r):
    if r.get('label_type'):return r['label_type']
    onset=r.get('onset_type')
    if not onset or onset=='RIGHT_CENSORED':return 'RIGHT_CENSORED'
    return 'EXACT' if onset.startswith('EXACT') else onset
def window_complete(r):
    c=r.get('coverage',{})
    signals=[c[k] for k in ['contributions_complete','public_contributions_complete','public_contribution_prefix_complete'] if isinstance(c.get(k),bool)]
    return bool(signals) and all(signals)
def verify_current_label(g,r):
    typ='RIGHT_CENSORED' if g['onset_label_type']=='NOT_FRAUD' else g['onset_label_type']
    if label_of(r)!=typ:raise ValueError('Current TEST label differs from sealed gold; declare changed cohort explicitly')
    if typ=='EXACT' and v2.utc(g['onset_utc'])!=v2.utc(r['tf_utc']):raise ValueError('Current TEST exact timestamp changed')
    if typ=='INTERVAL':
        if v2.utc(g['onset_lower_utc'])!=v2.utc(r['tf_lower_utc']) or v2.utc(g['onset_upper_utc'])!=v2.utc(r['tf_upper_utc']):raise ValueError('Current TEST interval changed')
    return typ

def cohort_of(row):
    if row['label_type']=='EXACT':
        return ['exact_utc_all','exact_utc_'+row['exact_kind']]+(['exact_published_event'] if row['event_error'] is not None else [])
    return ['interval'] if row['label_type']=='INTERVAL' else []
def summarize_models(rows):
    result={}
    for method in METHODS:
        selected=[r for r in rows if r['method']==method]
        if not selected:continue
        metrics={k:describe(r[k] for r in selected) for k in ['event_error','event_norm','time_error_hours','time_norm','interval_distance_days','interval_norm']}
        events=[r for r in selected if r['event_error'] is not None]
        intervals=[r for r in selected if r['interval_hit'] is not None]
        result[method]={'accounts':len(selected),'metrics':metrics,
            'exact_at_0':wilson(sum(r['event_error']==0 for r in events),len(events)),
            'within_1':wilson(sum(r['event_error']<=1 for r in events),len(events)),
            'interval_hit':wilson(sum(r['interval_hit'] for r in intervals),len(intervals))}
    return result
def paired_rows(rows):
    grouped=defaultdict(dict)
    for r in rows:grouped[r['entity_id']][r['method']]=r
    comparisons=[]
    for uid,methods in sorted(grouped.items()):
        fusion=methods.get('fusion')
        if not fusion:continue
        for metric in ['event_norm','time_norm','interval_norm']:
            if fusion[metric] is None:continue
            for base in METHODS[:-1]:
                r=methods[base]
                if r[metric] is None:continue
                row={'entity_id':uid,'label_type':fusion['label_type'],'metric':metric,'baseline':base,'fusion_loss':fusion[metric],'baseline_loss':r[metric],
                    'outcome':compare_loss(fusion[metric],r[metric]),'gain':r[metric]-fusion[metric],
                    'interval_hit_outcome':compare_loss(1-int(fusion['interval_hit']),1-int(r['interval_hit'])) if metric=='interval_norm' else ''}
                comparisons.append(row)
    return comparisons
def summarize_pairs(rows):
    groups=defaultdict(list)
    for r in rows:groups[(r['metric'],r['baseline'])].append(r)
    out={};rng=random.Random(20261024)
    for (metric,baseline),group in sorted(groups.items()):
        n=len(group);counts=Counter(r['outcome'] for r in group);gains=[r['gain'] for r in group]
        samples=[sum(gains[rng.randrange(n)] for _ in range(n))/n for _ in range(2000)]
        out.setdefault(metric,{})[baseline]={'n':n,'wins':counts['win'],'losses':counts['loss'],'ties':counts['tie'],
            'win_rate':counts['win']/n,'loss_rate':counts['loss']/n,'tie_rate':counts['tie']/n,
            'win_wilson95':wilson(counts['win'],n)['ci95'],
            'paired_mean_gain':statistics.mean(gains),'paired_mean_gain_bootstrap95':[quantile(samples,.025),quantile(samples,.975)],
            'interval_hit_outcomes':dict(Counter(r['interval_hit_outcome'] for r in group)) if metric=='interval_norm' else None}
    return out

def score(bundle,run,source,out):
    bundle=bundle.resolve();run=run.resolve();source=source.resolve();out=out.resolve()
    if out.is_relative_to(bundle) or out.is_relative_to(run):raise ValueError('Output must not overwrite frozen inputs/predictions')
    if out.exists() and any(out.iterdir()):raise ValueError('Output directory already contains files; preserve previous results')
    protected=sorted(set([p for root in [bundle,run] for p in root.rglob('*') if p.is_file()]+[source,PROTOCOL,Path(__file__),Path(v2.__file__)]))
    before={str(p):sha(p) for p in protected}
    manifest=json.loads((run/'run_manifest.json').read_text())
    if manifest['split']!='TEST':raise ValueError('This version only rescores the fixed already-opened TEST')
    for filename,key in [('inputs.npz','input_sha256'),('split.csv','split_sha256')]:
        if sha(bundle/filename)!=manifest[key]:raise ValueError('Input/split differs from sealed prediction manifest')
    gold=v2.indexed_unique(v2.read_csv(bundle/'gold_onset.csv'),'entity_id');splits=v2.indexed_unique(v2.read_csv(bundle/'split.csv'),'entity_id')
    predicted=v2.indexed_unique(v2.read_csv(run/'predictions.csv'),'entity_id');timelines=v2.indexed_unique(v2.read_jsonl(bundle/'timeline.jsonl'),'entity_id')
    raw=v2.indexed_unique(v2.read_jsonl(bundle/'raw_inputs.jsonl'),'entity_id');source_rows=v2.read_jsonl(source)
    src=v2.indexed_unique([dict(x,entity_id='wiki_u'+str(x['userid'])) for x in source_rows],'entity_id')
    wanted={uid for uid,r in splits.items() if r['split']=='TEST'}
    if set(predicted)!=wanted:raise ValueError('Predictions do not match fixed TEST IDs exactly')
    if not wanted<=set(gold)&set(timelines)&set(src)&set(raw):raise ValueError('Missing TEST source/gold/history')
    # If the original artifact checksum list is available, enforce its sealed outputs too.
    artifact_manifest=run.parent/'manifest_sha256.json'
    if artifact_manifest.exists():
        frozen=json.loads(artifact_manifest.read_text())
        for name in ['real_run/predictions.csv','real_run/behavior_model_params.npz','data_bundle/gold_onset.csv']:
            if sha(run.parent/name)!=frozen[name]:raise ValueError('Original sealed artifact checksum mismatch: '+name)
    per=[];accounts=[];excluded=Counter()
    for uid in sorted(wanted):
        g=gold[uid];r=src[uid];p=predicted[uid];tr=timelines[uid];typ=verify_current_label(g,r)
        times=[v2.utc(x) for x in tr['times_utc']];kinds=tr['row_kinds']
        if len(times)!=len(kinds) or any(a>b for a,b in zip(times,times[1:])):raise ValueError('Invalid timeline order/shape')
        pub=[t for t,kind in zip(times,kinds) if kind=='published_revision'];n=len(pub)
        if kinds[:n]!=['published_revision']*n:raise ValueError('Published events must be original prefix')
        start=v2.utc(tr['observation_start_utc']);end=v2.utc(tr['observation_end_utc']);span=end-start
        if start!=times[0] or end!=times[-1] or span<0:raise ValueError('Observation bounds changed or malformed')
        complete=window_complete(r);prefix=int(r.get('normal_prefix_count') or 0)
        acct={'entity_id':uid,'username':r['username'],'label_type':typ,'published_edits':n,'history_group':history_group(n),'normal_prefix_count':prefix if typ!='RIGHT_CENSORED' else None,
            'prefix_group':prefix_group(prefix) if typ!='RIGHT_CENSORED' else 'NOT_APPLICABLE','window_start_utc':tr['observation_start_utc'],'window_end_utc':tr['observation_end_utc'],'window_days':span/86400,
            'window_complete':complete,'time_norm_reason':'OK' if complete and span>0 else 'ZERO_SPAN' if span==0 else 'INCOMPLETE_PUBLIC_WINDOW',
            'max_published_gap_days':max([b-a for a,b in zip(pub,pub[1:])],default=0)/86400,'same_second_pairs':sum(a==b for a,b in zip(pub,pub[1:])),
            'delay_stratum':'NOT_APPLICABLE','delay_min_group':'NOT_APPLICABLE','delay_min_days':None,'delay_max_days':None,'interval_width_days':None,'interval_width_group':'NOT_APPLICABLE','exact_kind':'NOT_APPLICABLE','event_norm_reason':'NOT_APPLICABLE'}
        if typ in ['RIGHT_CENSORED','UNCERTAIN']:
            if typ=='RIGHT_CENSORED' and (p['status']!='NOT_DETECTED_UNSUPPORTED' or any(p.get(col) for col in COLUMNS.values())):raise ValueError('Control received an invented onset')
            accounts.append(acct);excluded[typ]+=1
            for method in METHODS:per.append(dict(acct,method=method,predicted_event_index=None,predicted_utc=None,event_error=None,event_norm=None,time_error_hours=None,time_norm=None,interval_hit=None,interval_distance_days=None,interval_norm=None,score_status='NO_ONSET_METRIC'))
            continue
        if p['status']!='OK':raise ValueError('Unexpected missing onset prediction; cannot silently drop cases')
        if integer(p['observed_td'])!=n or not all(t<end for t in pub):raise ValueError('Snapshot mistaken for a published event or non-pre-decision event')
        ks=simple_baselines(pub,start,end)
        for method,column in COLUMNS.items():
            k=integer(p[column])
            if not 0<=k<n:raise ValueError('Prediction outside shared published-event support')
            ks[method]=k
        target=None;actual=None;lo=None;hi=None
        if typ=='EXACT':
            target=v2.utc(g['onset_utc'])
            if not start<=target<=end:raise ValueError('Exact point outside observation')
            if any(g.get(key) and v2.utc(g[key])!=target for key in ['onset_lower_utc','onset_upper_utc']):raise ValueError('EXACT bounds disagree with point')
            aligned=g['point_is_published_event']=='true';acct['exact_kind']='public' if aligned else 'other_log'
            acct['event_norm_reason']='N_LE_1' if aligned and n<=1 else 'OK' if aligned else 'NOT_ALIGNED_TO_PUBLISHED_EVENT'
            if aligned:
                actual=integer(g['t_onset']);events=raw[uid]['events']
                if not 0<=actual<n or pub[actual]!=target or str(events[actual]['revision_id'])!=str(g['source_event_id']):raise ValueError('Exact event ID and UTC do not align')
            acct['delay_min_days']=(end-target)/86400;acct['delay_max_days']=acct['delay_min_days'];acct['delay_stratum']=delay_group(acct['delay_min_days']);acct['delay_min_group']=acct['delay_stratum']
        elif typ=='INTERVAL':
            if g['onset_utc'] or g['t_onset'] or g['interval_closure']!='(lower,upper]':raise ValueError('Interval collapsed into point')
            lo=v2.utc(g['onset_lower_utc']);hi=v2.utc(g['onset_upper_utc'])
            if not start<=lo<hi<=end:raise ValueError('Invalid interval bounds')
            acct['interval_width_days']=(hi-lo)/86400;acct['interval_width_group']=v2.width_group(acct['interval_width_days'])
            acct['delay_min_days']=(end-hi)/86400;acct['delay_max_days']=(end-lo)/86400;acct['delay_stratum']=interval_delay_group(acct['delay_min_days'],acct['delay_max_days']);acct['delay_min_group']=delay_group(acct['delay_min_days'])
        else:raise ValueError('Unsupported label')
        accounts.append(acct)
        for method in METHODS:
            k=ks[method];timestamp=pub[k];event_error=abs(k-actual) if actual is not None else None
            seconds=abs(timestamp-target) if target is not None else None
            dist=v2.interval_distance(timestamp,lo,hi) if lo is not None else None
            row=dict(acct,method=method,predicted_event_index=k,predicted_utc=tr['times_utc'][k],event_error=event_error,event_norm=normalize(event_error,n-1) if event_error is not None else None,
                time_error_hours=seconds/3600 if seconds is not None else None,time_norm=normalize(seconds,span,complete) if seconds is not None else None,
                interval_hit=v2.interval_hit(timestamp,lo,hi) if lo is not None else None,interval_distance_days=dist/86400 if dist is not None else None,interval_norm=normalize(dist,span,complete) if dist is not None else None,score_status='SCORED')
            per.append(row)
    cohorts=defaultdict(list)
    for row in per:
        for name in cohort_of(row):cohorts[name].append(row)
    paired=paired_rows(per);stratified=[]
    for dimension in ['history_group','prefix_group','interval_width_group','delay_stratum','delay_min_group']:
        bins=defaultdict(list)
        for row in per:
            if row[dimension]=='NOT_APPLICABLE':continue
            for cohort in cohort_of(row):bins[(cohort,row[dimension])].append(row)
        for (cohort,group),rr in sorted(bins.items()):stratified.append({'dimension':dimension,'group':group,'cohort':cohort,'n_accounts':len({r['entity_id'] for r in rr}),'models':summarize_models(rr)})
    old_report=run.parent.parent/'reports/evaluation_v2/exploratory_score_v2.json'
    # Compare with original v2 saved in the enclosing stage, when present.
    old_report=Path(__file__).resolve().parents[3]/'reports/evaluation_v2/exploratory_score_v2.json'
    legacy_matches={}
    if old_report.exists() and manifest.get('dataset_id')=='wiki300_exp05_frozen_method_real_v1':
        old=json.loads(old_report.read_text())
        for name in ['behavior','delay','fusion']:
            legacy_matches[name]=math.isclose(statistics.mean(r['event_error'] for r in cohorts['exact_published_event'] if r['method']==name),old['exact_event_abs_error_mean'][name],rel_tol=1e-12)
        if not all(legacy_matches.values()):raise ValueError('Original event-error results changed')
    after={str(p):sha(p) for p in protected}
    if before!=after:raise ValueError('A protected input/prediction/source changed')
    summary={'version':'2.1','scope':'EXPLORATORY_RESCORING_OF_ALREADY_OPENED_TEST_NOT_NEW_BLIND_TEST','protocol_sha256':sha(PROTOCOL),'models':list(METHODS),'test_n':len(wanted),
        'test_label_counts':dict(Counter(a['label_type'] for a in accounts)),'current_dataset_label_counts':dict(Counter(label_of(x) for x in source_rows)),
        'cohorts':{name:{'n_accounts':len({r['entity_id'] for r in rr}),'models':summarize_models(rr)} for name,rr in sorted(cohorts.items())},
        'paired':summarize_pairs(paired),'excluded_from_onset_metrics':dict(excluded),'normal_control_fpr':{'status':'NOT_EVALUABLE_NO_RISK_OUTPUT','n':sum(a['label_type']=='RIGHT_CENSORED' for a in accounts),'false_positive_rate':None,'specificity':None,'auc':None},
        'protected_files_unchanged':True,'protected_file_count':len(before),'frozen_predictions_sha256':sha(run/'predictions.csv'),'frozen_inputs_sha256':sha(bundle/'inputs.npz'),'split_sha256':sha(bundle/'split.csv'),'source_sha256':sha(source),'original_v2_event_means_reproduced':legacy_matches,
        'training_performed':False,'weight_or_threshold_selection':False,'combined_total_score_created':False,
        'cautions':['Account-normalization changes cross-account aggregation, not within-account error ordering.',
                    'Only eight exact published-event cases and nine exact UTC cases; P90 and uncertainty intervals are unstable.',
                    'No duration/edit-count minimum for normal history; grouping is descriptive only.',
                    'INTERVAL delay bins are ranges; an ambiguous range is never replaced by its upper endpoint.',
                    'All-UTC cohort includes a separate nonpublished-log event; event-level cohort excludes it.',
                    'Bootstrap/Wilson intervals are descriptive and assume account independence; unknown related identities may violate it.']}
    out.mkdir(parents=True,exist_ok=True);v2.write_json(out/'metrics_v2_1.json',summary);v2.write_json(out/'stratified_metrics_v2_1.json',stratified)
    v2.write_csv(out/'per_account_methods_v2_1.csv',per,list(per[0]));v2.write_csv(out/'per_account_paired_v2_1.csv',paired,list(paired[0]));v2.write_csv(out/'account_cohorts_v2_1.csv',accounts,list(accounts[0]))
    v2.write_json(out/'protected_files_v2_1.json',{'before':before,'after':after,'unchanged':True})
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    s=score(a.bundle,a.run,a.source,a.out);print(json.dumps({'status':'PASS_READ_ONLY','test_n':s['test_n'],'cohort_n':{k:v['n_accounts'] for k,v in s['cohorts'].items()},'original_v2_reproduced':s['original_v2_event_means_reproduced'],'protected_files_unchanged':s['protected_files_unchanged']},ensure_ascii=False,indent=2))
