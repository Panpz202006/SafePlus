"""Separate evaluation of Exp04 I/O Bundle. GOLD IS READ ONLY IN THIS SCRIPT.

Example:
 python scripts/48_exp04_evaluate_bundle.py --bundle runs/io_contract_v1/synthetic_same_delay --predictions runs/io_contract_v1/synthetic_predictions --check-legacy-seed0
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd


def bootstrap_ci(baseline_error, final_error, seed=20261020, count=5000):
    gain=np.asarray(baseline_error,dtype=float)-np.asarray(final_error,dtype=float)
    n=len(gain)
    rng=np.random.default_rng(seed)
    means=np.empty(count)
    for j in range(count):
        means[j]=gain[rng.integers(0,n,n)].mean()
    lo,hi=np.quantile(means,[.025,.975])
    return {'gain':float(gain.mean()),'ci95_low':float(lo),'ci95_high':float(hi)}


def evaluate(bundle:Path,predictions_dir:Path,check_legacy_seed0:bool=False):
    pred=pd.read_csv(predictions_dir/'predictions.csv',dtype={'entity_id':str})
    gold=pd.read_csv(bundle/'gold_onset.csv',dtype={'entity_id':str})
    if pred.entity_id.duplicated().any() or gold.entity_id.duplicated().any():
        raise ValueError('Duplicate entity_id in predictions/gold')
    inp=np.load(bundle/'inputs.npz',allow_pickle=False)
    input_ids=list(map(str,inp['entity_id'].tolist()))
    idmap={eid:i for i,eid in enumerate(input_ids)}
    td=np.asarray(inp['t_detect'],dtype=np.int64)
    detected=np.asarray(inp['detected'],dtype=np.int64)
    if set(pred.entity_id)-set(idmap):
        raise ValueError('Predictions contain ID outside input')
    manifest=json.loads((predictions_dir/'run_manifest.json').read_text(encoding='utf-8'))
    split=pd.read_csv(bundle/'split.csv',dtype={'entity_id':str})
    subset=set(split.loc[split['split']==manifest['split'],'entity_id'])
    if set(pred.entity_id)!=subset:
        raise ValueError('Prediction IDs do not exactly match declared split')
    if not (gold['onset_label_type'].isin(['EXACT','INTERVAL','UNRESOLVED','NOT_FRAUD']).all()):
        raise ValueError('Unknown label types in gold')
    m=pred.merge(gold,on='entity_id',how='inner',validate='one_to_one')
    eligible=m[(m['status']=='OK')&(m['is_fraud']==1)&(m['onset_label_type']=='EXACT')].copy()
    if eligible.empty:
        raise RuntimeError('No qualified detected positive EXACT test truth to score')
    for c in ('pred_tf_median','behavior_only_median','delay_only_median','t_onset'):
        eligible[c]=pd.to_numeric(eligible[c],errors='raise')
    for eid in eligible.entity_id:
        j=idmap[eid]
        if detected[j]!=1 or not (0<=int(eligible.loc[eligible.entity_id==eid,'t_onset'].iloc[0])<td[j]):
            raise ValueError(f'Illegal detected/exact onset for {eid}')
    err=eligible['pred_tf_median'].to_numpy()-eligible['t_onset'].to_numpy()
    ae=np.abs(err)
    summary={
        'model_version':manifest['model_version'],
        'dataset_id':manifest['dataset_id'],
        'split':manifest['split'],
        'n_split':len(pred),
        'n_eligible_exact_detected_fraud':len(eligible),
        'n_gold_unresolved_in_split':int(m.onset_label_type.eq('UNRESOLVED').sum()),
        'mae_median':float(ae.mean()),
        'signed_error':float(err.mean()),
        'within_1':float((ae<=1).mean()),
        'within_2':float((ae<=2).mean()),
    }
    for name,col in [('behavior_only','behavior_only_median'),('delay_only','delay_only_median')]:
        errors=np.abs(eligible[col].to_numpy()-eligible['t_onset'].to_numpy())
        summary[name+'_mae']=float(errors.mean())
        summary['vs_'+name]=bootstrap_ci(errors,ae)
    gold_delay=np.array([int(td[idmap[eid]]-tf) for eid,tf in zip(eligible.entity_id,eligible.t_onset)],dtype=np.int64)
    buckets={'short_1_2':gold_delay<=2,'mid_3_5':(gold_delay>=3)&(gold_delay<=5),'long_6_plus':gold_delay>=6}
    summary['by_true_delay']={}
    for name,mask in buckets.items():
        summary['by_true_delay'][name]={'n':int(mask.sum()),'mae':float(ae[mask].mean()) if mask.sum() else None}
    (predictions_dir/'metrics.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print('EXP04 POSTERIOR MEDIAN EVALUATION (independent gold)')
    print('Eligible n:',len(eligible))
    for label,k in [('MAE','mae_median'),('Signed Error','signed_error'),('Within-1','within_1'),('Within-2','within_2')]:
        print(f'{label}: {summary[k]:.6f}')
    print(f"Behavior-only MAE: {summary['behavior_only_mae']:.6f}")
    print(f"Delay-only MAE:    {summary['delay_only_mae']:.6f}")
    if check_legacy_seed0:
        expected={'n_eligible_exact_detected_fraud':132,'mae_median':1.2348484848484849,'signed_error':0.29545454545454547,'within_1':0.6439393939393939,'within_2':0.8712121212121212}
        failed=[]
        for k,v in expected.items():
            measured=summary[k]
            if not np.isclose(measured,v,atol=1e-6,rtol=0):
                failed.append(f'{k}: got {measured}, expected {v}')
        if failed:
            print('REGRESSION: FAIL')
            for issue in failed:print('  -',issue)
            raise RuntimeError('Synthetic frozen-model regression mismatch; DO NOT TAG')
        print('REGRESSION: PASS (frozen Exp04 seed0 metrics)')
    print('METRICS SAVED:',predictions_dir/'metrics.json')
    return summary


def main():
    p=argparse.ArgumentParser(description='Evaluate Exp04 model predictions with separate gold onset')
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--check-legacy-seed0',action='store_true')
    a=p.parse_args()
    evaluate(a.bundle,a.predictions,a.check_legacy_seed0)

if __name__=='__main__': main()
