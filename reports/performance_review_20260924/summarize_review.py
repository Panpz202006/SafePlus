"""Summarize completed runs and benchmark CPU inference; run from repository root."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from safeplus.models import build_model

root = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
frame = pd.read_csv(root / 'runs.csv')
timing = pd.DataFrame(json.loads((root / 'timings.json').read_text()))
frame = frame.merge(timing, on=['model', 'seed'], validate='one_to_one')
with np.load('data/processed/synthetic_demo.npz') as data:
    x = torch.tensor(data['x'][:64], dtype=torch.float32)
    lengths = torch.tensor(data['lengths'][:64], dtype=torch.long)
latency = []
for name in frame.model.unique():
    for seed in range(5):
        run = root / 'runs' / name / f'seed_{seed}'
        cfg = yaml.safe_load((run / 'config.yaml').read_text())
        options = dict(cfg['model'])
        options.pop('name')
        model = build_model(name, x.shape[-1], **options).eval()
        model.load_state_dict(torch.load(run / 'model.pt', weights_only=True, map_location='cpu'))
        with torch.inference_mode():
            for _ in range(20):
                model(x, lengths)
            samples = []
            for _ in range(200):
                start = time.perf_counter_ns()
                model(x, lengths)
                samples.append((time.perf_counter_ns() - start) / 1e6)
        latency.append({'model': name, 'seed': seed,
                        'inference_batch64_median_ms': float(np.median(samples)),
                        'inference_batch64_p95_ms': float(np.percentile(samples, 95)),
                        'parameters': sum(p.numel() for p in model.parameters())})
frame = frame.merge(pd.DataFrame(latency), on=['model', 'seed'], validate='one_to_one')
frame.to_csv(root / 'review_runs.csv', index=False)
metrics = ['auprc', 'auroc', 'f1', 'early_detected_rate', 'mean_lead_time',
           'onset_mae', 'onset_nll', 'wall_seconds', 'epochs_completed',
           'inference_batch64_median_ms', 'inference_batch64_p95_ms', 'parameters']
summary = frame.groupby('model')[metrics].agg(['mean', 'std'])
summary.to_csv(root / 'review_summary.csv')
rng = np.random.default_rng(0)
paired = []
for metric in ['auprc', 'auroc', 'f1']:
    pivot = frame.pivot(index='seed', columns='model', values=metric)
    for baseline in ['gru', 'safe', 'safe-r']:
        differences = (pivot['safeplus'] - pivot[baseline]).to_numpy()
        bootstrap = rng.choice(differences, (10000, len(differences)), replace=True).mean(axis=1)
        lo, hi = np.quantile(bootstrap, [0.025, 0.975])
        paired.append({'metric': metric, 'comparison': f'safeplus - {baseline}',
                       'mean_difference': float(differences.mean()),
                       'bootstrap_95_low': float(lo), 'bootstrap_95_high': float(hi)})
pd.DataFrame(paired).to_csv(root / 'paired_comparisons.csv', index=False)
print(summary.to_string())
print(json.dumps(paired, indent=2))
