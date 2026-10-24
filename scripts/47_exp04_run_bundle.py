"""SafePlus Experiment 04 frozen model, unified I/O bundle runner.

Only supported task: retrospective onset inference for detected cases.

IMPORTANT: NEVER READ gold_onset.csv here, not even to filter/test.
All predictions are made before external, separately run evaluation.

Example from repo root:
 python scripts/47_exp04_run_bundle.py --bundle runs/io_contract_v1/synthetic_same_delay --output runs/io_contract_v1/synthetic_predictions
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
from pathlib import Path

import numpy as np
import pandas as pd

FROZEN_W = 2.0 / 3.0
FROZEN_TEMPERATURE = 1.0 / 3.0
BASE_04A = Path('scripts/34_exp04_learned_behavior_evidence.py')
BASE_04F = Path('scripts/40_exp04_convex_fusion_audit.py')
BASE_03Y = Path('scripts/32_exp03_behavior_likelihood_identification.py')
VALIDATOR_PATH = Path('scripts/45_exp04_validate_io.py')


def module_from_path(path: Path, name: str):
    if not path.is_file():
        raise FileNotFoundError(f'Run from SafePlus repository root; missing {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rows_by_id(path: Path, id_column: str = 'entity_id'):
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if len({r[id_column] for r in rows}) != len(rows):
        raise ValueError(f'Duplicate ID in {path}')
    return {r[id_column]: r for r in rows}


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def build_delay_prior(prior: dict, bundle: Path, train_ids: set[str], index: dict[str, int], detected: np.ndarray, td: np.ndarray, max_delay: int):
    mode = prior['mode']
    if mode == 'synthetic_known':
        # The same implementation already used by the frozen Exp03/Exp04 baseline.
        legacy = module_from_path(BASE_03Y, 'safeplus_exp03y_prior')
        def logprob(delay):
            return math.log(max(float(legacy.delay_probability(int(delay))), 1e-30))
        return logprob, {'mode':mode,'source':prior['source']}

    if mode == 'empirical_train':
        rows = rows_by_id(bundle / 'train_onset_for_prior.csv')
        if not set(rows).issubset(train_ids):
            raise ValueError('train_onset_for_prior.csv includes non-TRAIN entity')
        counts = np.ones(max_delay, dtype=np.float64)  # Laplace(1), positive delay 1..max_delay
        valid_n = 0
        for eid, r in rows.items():
            i = index[eid]
            if detected[i] != 1:
                continue
            tf = int(r['t_onset'])
            delta = int(td[i]) - tf
            if not (1 <= delta <= max_delay):
                raise ValueError(f'Invalid train onset/delay for entity {eid}')
            counts[delta-1] += 1.0
            valid_n += 1
        if valid_n == 0:
            raise ValueError('No valid TRAIN labeled delays for empirical prior')
        probs = counts / counts.sum()
        def logprob(delay):
            d = int(delay)
            return math.log(max(float(probs[d-1]), 1e-30)) if 1 <= d <= max_delay else math.log(1e-30)
        return logprob, {'mode':mode,'n_train_delays':valid_n,'laplace_smoothing':1.0,'pmf':probs.tolist()}

    if mode == 'external_fixed':
        raw = prior.get('parameters', {}).get('probabilities')
        if not isinstance(raw, list) or not raw:
            raise ValueError('external_fixed requires parameters.probabilities for delays 1,2,...')
        prob = np.asarray(raw, dtype=np.float64)
        if not np.all(np.isfinite(prob)) or np.any(prob < 0) or prob.sum() <= 0:
            raise ValueError('Invalid external delay prior probability vector')
        prob /= prob.sum()
        def logprob(delay):
            d=int(delay)
            return math.log(max(float(prob[d-1]), 1e-30)) if 1 <= d <= len(prob) else math.log(1e-30)
        return logprob, {'mode':mode,'pmf':prob.tolist()}
    raise ValueError(f'Unknown prior mode {mode}')


def infer_for_one(x_user, td_i, behavior_params, scale, exp04a, exp04f, delay_logprob):
    candidates, raw_b = exp04f.behavior_candidate_scores(
        x_user=x_user, observed_td=td_i, behavior_params=behavior_params, exp04a=exp04a
    )
    if len(candidates) != int(td_i) or np.any(candidates != np.arange(int(td_i))):
        raise ValueError('Candidate support mismatch with frozen method')
    scaled_b = (raw_b - raw_b.mean()) / scale
    d = np.array([delay_logprob(int(td_i-k)) for k in candidates], dtype=np.float64)
    fusion_logits = (FROZEN_W * scaled_b + (1 - FROZEN_W) * d) / FROZEN_TEMPERATURE
    logits = {
        'final': fusion_logits,
        'behavior_only': raw_b - raw_b.mean(),
        'delay_only': d,
        'old_alpha_equivalent': 2.0 * scaled_b + d,
    }
    out = {}
    posteriors = {}
    for name, score in logits.items():
        posterior = exp04a.stable_softmax(score)
        if np.any(~np.isfinite(posterior)) or abs(float(posterior.sum()) - 1) > 1e-10:
            raise ValueError(f'Bad posterior for {name}')
        out[name] = {
            'median': int(exp04f.posterior_median(candidates, posterior)),
            'mean': float(np.dot(candidates, posterior)),
            'mode': int(candidates[np.argmax(posterior)]),
            'max': float(np.max(posterior)),
        }
        posteriors[name] = posterior
    if not np.allclose(posteriors['final'], posteriors['old_alpha_equivalent'], atol=1e-12, rtol=1e-12):
        raise RuntimeError('Final convex formula differs from alpha=2 algebraic equivalent')
    return candidates, out, posteriors


def run(bundle: Path, output: Path, split: str = 'TEST', allow_overwrite: bool = False):
    validator = module_from_path(VALIDATOR_PATH, 'safeplus_contract_validator')
    errors, warnings, stats = validator.validate_bundle(bundle, check_gold=False)
    if errors:
        raise ValueError('Input contract FAIL:\n' + '\n'.join(errors))
    if output.exists() and any(output.iterdir()) and not allow_overwrite:
        raise FileExistsError(f'Output directory not empty: {output}; choose new --output')
    output.mkdir(parents=True, exist_ok=True)

    with np.load(bundle / 'inputs.npz', allow_pickle=False) as input_file:
        x = np.asarray(input_file['x'], dtype=np.float64)
        lengths = np.asarray(input_file['lengths'], dtype=np.int64)
        detected = np.asarray(input_file['detected'], dtype=np.int64)
        td = np.asarray(input_file['t_detect'], dtype=np.int64)
        ids = list(map(str, input_file['entity_id'].tolist()))
    length_aware = bool(np.any(lengths != x.shape[1]))
    lookup = {eid:i for i,eid in enumerate(ids)}
    splits = rows_by_id(bundle / 'split.csv')
    labels = rows_by_id(bundle / 'train_labels.csv')
    train_ids = {eid for eid in ids if splits[eid]['split'] == 'TRAIN'}
    train_idx = np.asarray([lookup[eid] for eid in ids if eid in train_ids], dtype=np.int64)
    fraud_label = np.full(len(ids), -1, dtype=np.int64)
    for eid,r in labels.items():
        if eid not in train_ids:
            raise ValueError(f'Training label outside TRAIN split: {eid}')
        fraud_label[lookup[eid]] = int(r['is_fraud'])
    if np.any(fraud_label[train_idx] == -1):
        raise ValueError('Missing train is_fraud labels')

    metadata = json.loads((bundle / 'metadata.json').read_text(encoding='utf-8'))
    prior = json.loads((bundle / 'delay_prior.json').read_text(encoding='utf-8'))
    if split not in {'TEST','VAL'}:
        raise ValueError('Predict only VAL or TEST; TRAIN is not a held-out prediction set')
    predict_idx = np.asarray([i for i,eid in enumerate(ids) if splits[eid]['split'] == split], dtype=np.int64)
    exp04a = module_from_path(BASE_04A, 'safeplus_exp04a')
    exp04f = module_from_path(BASE_04F, 'safeplus_exp04f')

    # IMPORTANT: no gold_onset.csv access anywhere in this executable.
    print(f'FROZEN EXP04 | bundle={bundle} | split={split} | n={len(predict_idx)}')
    print('TRAIN behavior EM: using only train_labels, inputs, fixed split')
    params, history = exp04a.train_emission_em(
        x=x, fraud_label=fraud_label, detected=detected, td=td, train_idx=train_idx,
        **({'lengths': lengths} if length_aware else {})
    )
    scale = exp04f.estimate_global_behavior_scale(
        x=x, fraud_label=fraud_label, detected=detected, td=td, train_idx=train_idx,
        behavior_params=params, exp04a=exp04a
    )
    log_delay, prior_resolved = build_delay_prior(
        prior, bundle, train_ids, lookup, detected, td, int(x.shape[1])
    )

    # Save a reproducible model state; this is a fitting artifact, not a test label.
    save_params = {k:np.asarray(v) for k,v in params.items() if isinstance(v,(np.ndarray,np.number,int,float))}
    save_params['global_behavior_scale'] = np.asarray(scale, dtype=np.float64)
    np.savez_compressed(output / 'behavior_model_params.npz', **save_params)
    if hasattr(history, 'to_csv'):
        history.to_csv(output / 'em_history.csv', index=False)

    predictions = []
    posterior_rows = []
    for i in predict_idx:
        eid = ids[int(i)]
        d = int(td[i])
        row = {
            'entity_id': eid, 'status': 'NOT_DETECTED_UNSUPPORTED',
            'observed_td': '' if detected[i] != 1 else d,
            'pred_tf_median':'', 'pred_tf_mean':'','pred_tf_mode':'',
            'pred_delay':'','posterior_max':'',
            'behavior_only_median':'','delay_only_median':'',
            'model_version':'exp04-synthetic-frozen-v1',
            'delay_prior_version':prior['version'],
        }
        if detected[i] == 1:
            candidate, by_mode, prob = infer_for_one(
                x[int(i), :int(lengths[i])], d, params, scale, exp04a, exp04f, log_delay
            )
            final = by_mode['final']
            row.update({
                'status':'OK',
                'pred_tf_median':final['median'],
                'pred_tf_mean':final['mean'],
                'pred_tf_mode':final['mode'],
                'pred_delay':d-final['median'],
                'posterior_max':final['max'],
                'behavior_only_median':by_mode['behavior_only']['median'],
                'delay_only_median':by_mode['delay_only']['median'],
            })
            posterior_rows.append({
                'entity_id':eid,'candidate_times':candidate.tolist(),
                'posterior_probs':prob['final'].tolist(),
                'behavior_only_probs':prob['behavior_only'].tolist(),
                'delay_only_probs':prob['delay_only'].tolist(),
            })
        predictions.append(row)

    pd.DataFrame(predictions).to_csv(output / 'predictions.csv', index=False)
    with (output / 'posterior.jsonl').open('w', encoding='utf-8') as f:
        for r in posterior_rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n')

    manifest = {
        'schema_version':metadata.get('schema_version','1.0'),'task':'retrospective_onset_after_detection',
        'model_version':'exp04-synthetic-frozen-v1',
        'split':split,'n_split':int(len(predict_idx)),
        'n_detected_predicted':len(posterior_rows),
        'w':FROZEN_W,'temperature':FROZEN_TEMPERATURE,
        'global_behavior_scale':float(scale),
        'delay_prior_version':prior['version'],
        'delay_prior_resolved':prior_resolved,
        'dataset_id':metadata['dataset_id'],
        'feature_names':metadata['feature_names'],
        'time_unit':metadata['time_unit'],
        'input_sha256':sha256(bundle/'inputs.npz'),
        'train_labels_sha256':sha256(bundle/'train_labels.csv'),
        'split_sha256':sha256(bundle/'split.csv'),
        'prior_sha256':sha256(bundle/'delay_prior.json'),
        'sequence_handling':'per_entity_real_lengths' if length_aware else 'unchanged_fixed_length_path',
        'onset_supervision_for_prior':prior.get('onset_supervision', 'synthetic_or_external'),
        'train_onset_for_prior_sha256':sha256(bundle/'train_onset_for_prior.csv') if prior['mode']=='empirical_train' else None,
        'source_code_sha256':{p.name:sha256(p) for p in (BASE_04A,BASE_04F,BASE_03Y) if p.is_file()},
        'python':platform.python_version(), 'numpy':np.__version__, 'pandas':pd.__version__,
        'warnings':warnings,
    }
    (output/'run_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(f'PREDICTED: {len(posterior_rows)} detected users in {split} (others marked unsupported)')
    print(f'RESULT: PASS (INFERENCE; test gold never loaded) | saved {output}')
    return manifest


def main():
    p = argparse.ArgumentParser(description='Run fixed SafePlus Exp04 model on unified I/O Bundle')
    p.add_argument('--bundle', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--split', choices=['VAL','TEST'], default='TEST')
    p.add_argument('--allow-overwrite', action='store_true')
    a = p.parse_args()
    run(a.bundle,a.output,a.split,a.allow_overwrite)


if __name__=='__main__':
    main()
