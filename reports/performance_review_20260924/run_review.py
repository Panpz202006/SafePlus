"""Run from the repository root with the project's dependencies installed."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import torch
import yaml

from safeplus.cli import main


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='reports/performance_review_20260924')
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = Path('data/processed/synthetic_demo.npz')
    metadata = {
        'python': sys.version, 'executable': sys.executable,
        'torch': torch.__version__, 'platform': platform.platform(),
        'device': 'cpu', 'torch_threads': 1,
        'data_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'seeds': list(range(5)), 'models': ['gru', 'safe', 'safe-r', 'safeplus'],
        'timing_scope': 'CLI train: data loading, training, threshold selection, test evaluation and saving',
    }
    (root / 'environment.json').write_text(json.dumps(metadata, indent=2))
    timings = []
    for model in metadata['models']:
        cfg = yaml.safe_load(Path('configs/synthetic/safeplus.yaml').read_text())
        cfg['model']['name'] = model
        cfg['data']['path'] = str(source.resolve())
        cfg['output_dir'] = str(root / 'runs' / model)
        config_path = root / f'{model}.yaml'
        config_path.write_text(yaml.safe_dump(cfg))
        for seed in metadata['seeds']:
            start = time.perf_counter()
            main(['train', '--config', str(config_path), '--seed', str(seed)])
            seconds = time.perf_counter() - start
            run_dir = root / 'runs' / model / f'seed_{seed}'
            history = json.loads((run_dir / 'history.json').read_text())
            metrics = json.loads((run_dir / 'metrics.json').read_text())
            row = {'model': model, 'seed': seed, 'wall_seconds': seconds,
                   'epochs_completed': len(history),
                   'best_epoch': max(history, key=lambda x: x['val_auprc'])['epoch'] + 1}
            timings.append(row)
            (root / 'timings.json').write_text(json.dumps(timings, indent=2))
            print(f"DONE {model} seed={seed} epochs={len(history)} seconds={seconds:.1f} AUPRC={metrics['auprc']:.4f}", flush=True)
    main(['aggregate', '--runs', str(root / 'runs'), '--output', str(root)])


if __name__ == '__main__':
    run()
