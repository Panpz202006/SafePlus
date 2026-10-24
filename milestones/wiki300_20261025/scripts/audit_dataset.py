#!/usr/bin/env python3
"""Read-only structural audit. No fitting, label rewriting, or network reads."""
from __future__ import annotations
import argparse, csv, hashlib, json, statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

def utc(value):
    d = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if d.tzinfo is None: raise ValueError('Timestamp must carry timezone')
    return d.timestamp()

def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def csvrows(path):
    with path.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')

def writecsv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def body_present(event):
    if not isinstance(event, dict): return False
    if 'deleted' in event and 'added' in event: return True
    if any('deleted' in x and 'added' in x for x in event.get('changes', []) if isinstance(x,dict)): return True
    values=event.get('values', {})
    return isinstance(values,dict) and 'old_wikitext' in values and 'new_wikitext' in values

def width_group(days):
    return 'le_1d' if days<=1 else '1_to_7d' if days<=7 else '7_to_30d' if days<=30 else 'gt_30d'

def run(root, out):
    b=root/'data_bundle'; source=jsonl(root/'frozen_source/SafePlus_combined_current.jsonl')
    goldrows=csvrows(b/'gold_onset.csv'); splitrows=csvrows(b/'split.csv')
    timerows=jsonl(b/'timeline.jsonl'); rawrows=jsonl(b/'raw_inputs.jsonl')
    errors=[]; checks=[]
    def check(name, condition, detail=''):
        checks.append({'check':name,'passed':bool(condition),'detail':detail})
        if not condition: errors.append({'check':name,'detail':detail})
    sourceids=['wiki_u'+str(x['userid']) for x in source]
    check('source_count_300',len(source)==300)
    for name,rows,keys in [('source',source,sourceids),('gold',goldrows,[r['entity_id'] for r in goldrows]),('split',splitrows,[r['entity_id'] for r in splitrows]),('timeline',timerows,[r['entity_id'] for r in timerows]),('raw_inputs',rawrows,[r['entity_id'] for r in rawrows])]:
        check(name+'_unique_ids',len(keys)==len(set(keys)))
        check(name+'_id_join',set(keys)==set(sourceids))
    names=[str(r['username']).replace('_',' ').strip().casefold() for r in source]
    check('source_normalized_username_unique',len(set(names))==len(names))
    gold={r['entity_id']:r for r in goldrows}; split={r['entity_id']:r['split'] for r in splitrows}
    source_alias={r['account_id']:'wiki_u'+str(r['userid']) for r in source}
    times={r['entity_id']:r for r in timerows}; raw={r['entity_id']:r for r in rawrows}
    groups=json.loads((b/'split_groups.json').read_text())
    check('groups_id_join',set(groups)==set(sourceids))
    gs=defaultdict(set)
    for uid,g in groups.items(): gs[g].add(split[uid])
    check('declared_groups_do_not_cross_splits',all(len(v)==1 for v in gs.values()))
    check('split_counts',Counter(split.values())=={'TRAIN':226,'VAL':24,'TEST':50})
    npz=np.load(b/'inputs.npz', allow_pickle=False)
    check('input_keys_no_gold',set(npz.files)=={'x','lengths','detected','t_detect','entity_id'})
    ids=[str(x) for x in npz['entity_id']]; index={uid:i for i,uid in enumerate(ids)}
    check('npz_unique_ids',len(ids)==len(set(ids)))
    check('npz_id_join',set(ids)==set(sourceids))
    check('x_finite',bool(np.isfinite(npz['x']).all()))
    check('x_shape',npz['x'].shape==(300,90,8))
    rows=[]; revids=defaultdict(set)
    for r in source:
        uid='wiki_u'+str(r['userid']); g=gold[uid]; ti=times[uid]; rr=raw[uid]; i=index[uid]
        ev=rr['events']; tt=list(map(utc,ti['times_utc'])); start=utc(ti['observation_start_utc']); end=utc(ti['observation_end_utc'])
        n=len(ev); length=int(npz['lengths'][i]); normal=r.get('normal_prefix',[]); typ=g['onset_label_type']
        check(uid+':time_nondecreasing',all(a<=b for a,b in zip(tt,tt[1:])))
        check(uid+':time_bounds',start==tt[0] and end==tt[-1] and all(start<=x<=end for x in tt))
        check(uid+':length_timeline',length==len(tt)==len(ti['row_kinds']))
        check(uid+':raw_timeline_alignment',[x['timestamp_utc'] for x in ev]==ti['times_utc'][:n])
        check(uid+':raw_revision_ids_unique',len(set(str(x['revision_id']) for x in ev))==n)
        check(uid+':zero_padding',not np.any(npz['x'][i,length:]!=0))
        for e in ev: revids[str(e['revision_id'])].add(uid)
        delay=None; width=None; prefixn=None
        if typ in ('EXACT','INTERVAL'):
            check(uid+':positive_flags',int(r['entity_label'])==1 and bool(r['event_observed']) and int(npz['detected'][i])==1)
            check(uid+':td_matches',end==utc(r['td_utc'])==utc(g['td_utc']))
            check(uid+':snapshot_kind',ti['row_kinds']==['published_revision']*n+['pre_decision_snapshot'])
            check(uid+':td_index',int(npz['t_detect'][i])==n)
            check(uid+':strict_pre_td',all(utc(x['timestamp_utc'])<end for x in ev))
            check(uid+':endpoint_state_is_carry_forward',bool(np.array_equal(npz['x'][i,n],npz['x'][i,n-1])))
            prefixn=int(r.get('normal_prefix_count') or 0)
            check(uid+':normal_prefix_count',prefixn==len(normal) and prefixn>=1)
            prefix_times=[utc(x['timestamp_utc']) for x in normal]
            check(uid+':prefix_time_order',all(a<=b for a,b in zip(prefix_times,prefix_times[1:])))
            lower=utc(r.get('tf_lower_utc') or r['tf_utc']); upper=utc(r.get('tf_upper_utc') or r['tf_utc'])
            check(uid+':onset_within_observation',start<=lower<=upper<=end)
            if typ=='EXACT':
                point=utc(r['tf_utc']); check(uid+':exact_source_type',r['onset_type'].startswith('EXACT'))
                check(uid+':exact_utc_preserved',point==utc(g['onset_utc'])==utc(g['onset_lower_utc'])==utc(g['onset_upper_utc']))
                check(uid+':normal_strictly_before_exact',all(x<point for x in prefix_times))
                if g['point_is_published_event']=='true':
                    k=int(g['t_onset']); check(uid+':exact_published_event_mapping',0<=k<n and tt[k]==point and str(ev[k]['revision_id'])==str(g['source_event_id']))
                else:
                    check(uid+':exact_nonpublished_is_other_log',r['onset_type']=='EXACT_PUBLIC_LOG_TIMESTAMP')
                delay=(end-point)/86400
            else:
                check(uid+':interval_source_type',r['onset_type']=='INTERVAL')
                check(uid+':interval_no_point',not g['onset_utc'] and not g['t_onset'] and not r.get('tf_utc') and g['interval_closure']=='(lower,upper]')
                check(uid+':interval_bounds_preserved',lower==utc(g['onset_lower_utc']) and upper==utc(g['onset_upper_utc']) and lower<upper)
                check(uid+':prefix_ends_at_lower',max(prefix_times)==lower)
                width=(upper-lower)/86400
                delay=(end-upper)/86400
        elif typ=='NOT_FRAUD':
            check(uid+':control_no_point_or_interval',all(not g[k] for k in ['t_onset','onset_utc','onset_lower_utc','onset_upper_utc']) and not r.get('tf_utc') and not r.get('td_utc'))
            check(uid+':control_flags',r.get('right_censored') is True and r.get('event_observed') is False and int(npz['detected'][i])==0 and int(r['entity_label'])==0)
            check(uid+':control_timeline',ti['row_kinds']==['published_revision']*n and int(npz['t_detect'][i])==length)
            check(uid+':control_event_count',n==int(r['observed_revision_count'])==len(r['events']))
            check(uid+':control_duration',abs((end-start)/86400-float(r['observation_duration_days']))<1e-8)
            match=r.get('matching_plan',{}).get('matched_positive_id')
            if match:
                mate=source_alias.get(match)
                check(uid+':matched_pair_same_split',mate in split and split[mate]==split[uid] and groups[uid]==groups[mate])
        else: check(uid+':known_label_type',False,typ)
        onset=r.get('onset_evidence',{})
        rows.append({'entity_id':uid,'username':r['username'],'split':split[uid],'label_type':typ,
            'exact_subtype':('EXACT_PUBLIC' if g['point_is_published_event']=='true' else 'EXACT_OTHER_LOG') if typ=='EXACT' else '',
            'normal_prefix_count':prefixn if prefixn is not None else '',
            'normal_prefix_group':('PREFIX_1' if prefixn==1 else 'PREFIX_2' if prefixn==2 else 'PREFIX_3PLUS') if prefixn is not None else '',
            'interval_width_days':width if width is not None else '', 'interval_width_group':width_group(width) if width is not None else '',
            'block_delay_days_from_exact_or_interval_upper':delay if delay is not None else '',
            'delay_definition':'td-tf' if typ=='EXACT' else 'td-upper; conservative minimum, true delay in [td-upper,td-lower)' if typ=='INTERVAL' else '',
            'control_window_days':(end-start)/86400 if typ=='NOT_FRAUD' else '',
            'published_edits':n,'observation_days':(end-start)/86400,
            'same_second_adjacent_pairs':sum(a==b for a,b in zip(tt,tt[1:])),
            'median_published_gap_seconds':statistics.median([utc(y['timestamp_utc'])-utc(x['timestamp_utc']) for x,y in zip(ev,ev[1:])]) if n>1 else '',
            'article_namespace_edits':sum(int(x['namespace'])==0 for x in ev),
            'normal_prefix_bodies_available':sum(body_present(x) for x in normal),
            'onset_body_available':body_present(onset) if typ!='NOT_FRAUD' else '',
            'control_event_bodies_available':sum(body_present(x) for x in r.get('events',[])) if typ=='NOT_FRAUD' else '',
            'human_gold_status':'PENDING'})
    prior=csvrows(b/'train_onset_for_prior.csv')
    check('prior_unique_36',len(prior)==36 and len({x['entity_id'] for x in prior})==36)
    check('prior_train_only',all(split[x['entity_id']]=='TRAIN' for x in prior))
    check('prior_exact_published_only',all(gold[x['entity_id']]['onset_label_type']=='EXACT' and gold[x['entity_id']]['point_is_published_event']=='true' for x in prior))
    check('prior_matches_gold',all(x['t_onset']==gold[x['entity_id']]['t_onset'] and x['source_tf_utc']==gold[x['entity_id']]['onset_utc'] for x in prior))
    # Existing source-manifest entries are relative to its original package root.
    old=json.loads((root/'manifest_sha256.json').read_text())
    for name,digest in old.items():
        if isinstance(digest,str): check('frozen_manifest:'+name,(root/name).is_file() and hashlib.sha256((root/name).read_bytes()).hexdigest()==digest)
    positives=[x for x in rows if x['label_type']!='NOT_FRAUD']; intervals=[x for x in rows if x['label_type']=='INTERVAL']; controls=[x for x in rows if x['label_type']=='NOT_FRAUD']
    summary={'status':'PASS' if not errors else 'FAIL','row_count':len(rows),'label_counts':dict(Counter(x['label_type'] for x in rows)),
        'split_counts':dict(Counter(split.values())),
        'split_by_label':{s:dict(Counter(x['label_type'] for x in rows if x['split']==s)) for s in ['TRAIN','VAL','TEST']},
        'exact_subtypes':dict(Counter(x['exact_subtype'] for x in rows if x['label_type']=='EXACT')),
        'positive_normal_prefix_groups':dict(Counter(x['normal_prefix_group'] for x in positives)),
        'exact_normal_prefix_groups':dict(Counter(x['normal_prefix_group'] for x in rows if x['label_type']=='EXACT')),
        'interval_width_groups':dict(Counter(x['interval_width_group'] for x in intervals)),
        'interval_width_median_days':statistics.median(x['interval_width_days'] for x in intervals),
        'interval_width_max_days':max(x['interval_width_days'] for x in intervals),
        'control_window_groups':dict(Counter('lt_1d' if x['control_window_days']<1 else '1_to_lt_7d' if x['control_window_days']<7 else 'ge_7d' for x in controls)),
        'conservative_block_delay_over_30d':sum(x['block_delay_days_from_exact_or_interval_upper']>30 for x in positives),
        'conservative_block_delay_over_365d':sum(x['block_delay_days_from_exact_or_interval_upper']>365 for x in positives),
        'normal_prefix_bodies_complete_accounts':sum(x['normal_prefix_bodies_available']==x['normal_prefix_count'] for x in positives),
        'onset_body_available_accounts':sum(bool(x['onset_body_available']) for x in positives),
        'control_bodies_complete_accounts':sum(x['control_event_bodies_available']==x['published_edits'] for x in controls),
        'accounts_with_same_second_events':sum(x['same_second_adjacent_pairs']>0 for x in rows),
        'cross_account_shared_published_revision_ids':{k:sorted(v) for k,v in revids.items() if len(v)>1},
        'group_count':len(gs),'matched_control_pairs':sum(bool(r.get('matching_plan',{}).get('matched_positive_id')) for r in source),
        'independent_human_gold_completed':0,'prior_train_only_n':len(prior),
        'structural_check_n':len(checks),'structural_checks_passed':sum(x['passed'] for x in checks),'errors':errors,
        'provenance_note':'Local cached bytes only; live link availability and semantic correctness are not established by structural checks.',
        'ordering_note':'Nondecreasing seconds checked; equal-second records retained, not interpreted as proven fine-grained causal order.'}
    out.mkdir(parents=True,exist_ok=True); dump(out/'dataset_audit.json',summary); dump(out/'structural_checks.json',checks); writecsv(out/'dataset_strata_300.csv',rows)
    print(json.dumps(summary,ensure_ascii=False,indent=2)); return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    result=run(a.artifact,a.out);raise SystemExit(0 if result['status']=='PASS' else 1)
