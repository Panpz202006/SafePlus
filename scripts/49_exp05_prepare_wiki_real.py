"""Prepare the frozen Wiki 300 for the Exp05 (not main/GRU) interface.

No fetching and no fitting. Explicit TRAIN exact-onset supervision is isolated in
train_onset_for_prior.csv. Original UTC truth and open/closed intervals survive.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,shutil
from pathlib import Path
import numpy as np

def read_csv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(p,fields,rows):
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write_json(p,data):p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def prepare(source:Path,out:Path,frozen:Path):
    if out.exists() and any(out.iterdir()):raise FileExistsError('Output must be new')
    out.mkdir(parents=True,exist_ok=True)
    # Copy input/splits verbatim. No choices are fitted using VAL/TEST onsets.
    for name in ['inputs.npz','split.csv','train_labels.csv','gold_onset.csv','timeline.jsonl','raw_inputs.jsonl','split_groups.json']:
        shutil.copy2(source/name,out/name)
    # Frozen Exp04 uses td only as an observation boundary, never as a feature.
    # At the endpoint read the last known published-event state unchanged.
    # This is a state observation, not an invented extra edit or an empty edit.
    with np.load(out/'inputs.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
    snapshot_rows=[]
    for i in np.flatnonzero(arrays['detected']==1):
        d=int(arrays['t_detect'][i])
        if d<1:raise ValueError('Missing pre-detection published state')
        arrays['x'][i,d]=arrays['x'][i,d-1]
        snapshot_rows.append(str(arrays['entity_id'][i]))
    np.savez_compressed(out/'inputs.npz',**arrays)
    meta=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    split={r['entity_id']:r['split'] for r in read_csv(out/'split.csv')}
    gold=read_csv(out/'gold_onset.csv')
    records=[json.loads(x) for x in frozen.read_text(encoding='utf-8').splitlines()]
    original={'wiki_u'+str(r['userid']):r for r in records}
    observations={r['entity_id']:r for r in map(json.loads,(out/'raw_inputs.jsonl').read_text(encoding='utf-8').splitlines())}
    if len(original)!=300 or set(original)!=set(split):raise ValueError('Expected all 300 frozen entities')
    for r in gold:
        original_row=original[r['entity_id']]
        if r['onset_utc']!=(original_row.get('tf_utc') or ''):raise ValueError('Original UTC tf changed')
        if original_row.get('onset_type')=='INTERVAL':
            if r['t_onset'] or r['onset_utc'] or r['interval_closure']!='(lower,upper]':raise ValueError('Interval promoted to point')
            if r['onset_lower_utc']!=original_row['tf_lower_utc'] or r['onset_upper_utc']!=original_row['tf_upper_utc']:raise ValueError('Interval changed')
        if r['onset_label_type']=='EXACT' and r['point_is_published_event']=='true':
            event=observations[r['entity_id']]['events'][int(r['t_onset'])]
            if event['revision_id']!=r['source_event_id'] or event['timestamp_utc']!=r['onset_utc']:
                raise ValueError('Exact event-index mapping must match both original revision ID and UTC')
    # Only exact TRAIN evidence whose timestamp equals an actual published event.
    # Between-event exact UTC points are not fabricated into exact integer steps.
    prior_rows=[{'entity_id':r['entity_id'],'t_onset':r['t_onset'],'source_tf_utc':r['onset_utc'],
                 'evidence_url':r['evidence_ids'],'annotation_source':'AI_public_evidence_review'}
                for r in gold if split[r['entity_id']]=='TRAIN' and r['onset_label_type']=='EXACT' and r['point_is_published_event']=='true']
    if not prior_rows:raise ValueError('No independently reviewed exact TRAIN event for prior')
    write_csv(out/'train_onset_for_prior.csv',list(prior_rows[0]),prior_rows)
    prior={'mode':'empirical_train','definition':'positive edit-position delay d-k; not elapsed calendar time',
           'source':'explicit train_onset_for_prior.csv, TRAIN ONLY, exact published-event UTC alignment',
           'parameters':{'laplace_smoothing':1.0,'support':'1..padded_L, same frozen runner definition'},
           'version':'wiki300-train-exact-step-delay-v1',
           'onset_supervision':{'used':True,'scope':'TRAIN_ONLY','n':len(prior_rows),
                                'annotation_source':'AI adjudication of original public evidence; not independent human signoff',
                                'claim':'partially onset-supervised delay prior; behavior EM remains onset-latent'},
           'val_or_test_gold_used':False}
    write_json(out/'delay_prior.json',prior)
    meta.update(schema_version='1.1',dataset_id='wiki300_exp05_frozen_method_real_v1',
                feature_version='causal_published_event_state_exp05_v1',
                variable_length_policy='slice_before_fit_and_predict',
                terminal_row_definition='last known published-event state carried to the endpoint; no feature uses td clock time; not an additional edit',
                model_lineage='exp05-io-contract-v1 @ 06fa34d2097b8ca4fa209dbf60ba551fef31a33d',
                model_architecture='two_state_diagonal_Gaussian_EM + TRAIN_global_scale + delay_prior + fixed_fusion',
                fixed_fusion={'w':2/3,'temperature':1/3,'equivalent_alpha':2},
                source_frozen300_sha256=sha(frozen),prior_onset_supervision=prior['onset_supervision'],
                frozen_exp04_prediction_ready=True,model_training_performed=False,
                frozen_engine_blockers=[],
                interval_evaluation='UTC (lower,upper] remains authoritative; not point MAE',
                exact_step_evaluation='only exact UTC points that coincide with a published event; between-event EXACT retained for UTC evaluation')
    write_json(out/'metadata.json',meta)
    report={'all_accounts':300,'source_sha256':sha(frozen),'input_npz_unchanged':sha(source/'inputs.npz')==sha(out/'inputs.npz'),
            'input_change':'endpoint features now use last published state, removing numeric td-derived features',
            'endpoint_rows_recomputed':len(snapshot_rows),
            'gold_csv_unchanged':sha(source/'gold_onset.csv')==sha(out/'gold_onset.csv'),
            'prior_train_exact_events':len(prior_rows),'prior_val_test_onsets_used':False,
            'source_sample_counts':{'EXACT':sum(r['onset_label_type']=='EXACT' for r in gold),
                                    'INTERVAL':sum(r['onset_label_type']=='INTERVAL' for r in gold),
                                    'NOT_FRAUD':sum(r['onset_label_type']=='NOT_FRAUD' for r in gold)},
            'supersedes_main_gru_adapter_for_this_task':True}
    write_json(out/'preparation_report.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--frozen',required=True,type=Path)
    a=p.parse_args();print(json.dumps(prepare(a.source,a.out,a.frozen),ensure_ascii=False,indent=2))
