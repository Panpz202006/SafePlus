"""Reproduce the exploratory, overlap-filtered legacy SAFE evaluation.

Run in a fresh Python 3.11 process for each dataset. The archived inputs
have no account IDs or feature timestamps, so feature equality is the
only cross-split overlap check this runner can enforce.
"""
import json
from lib2to3.refactor import RefactoringTool
import os
from pathlib import Path
import random
import sys
from tempfile import TemporaryDirectory
from types import ModuleType

import numpy as np
import tensorflow.compat.v1 as tf

if len(sys.argv) != 2 or sys.argv[1] not in ("twitter", "wiki"):
    raise SystemExit("Usage: python -m experiments.comparison.safe_baseline.run {twitter|wiki}")

# Load the frozen legacy graph and data loader under TensorFlow 2.12.
# Keep the compatibility changes and checkpoints outside the SAFE checkout.
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from experiments.run_paths import new_run_directory
from safeplus.data import filter_legacy_overlap, fixed_prefixes, legacy_records
from safeplus.evaluation import prefix_metrics
from safeplus.training import fit_baseline, length_batches

safe_root = repo_root / "baselines" / "safe"
legacy = safe_root / "frameWork" / "safe.py"
os.chdir(legacy.parent)
tf.disable_v2_behavior()
tf.set_random_seed(7)
random.seed(7)
rnn = tf.nn.rnn_cell

helper_path = safe_root / "safdKit.py"
helper = ModuleType("safdKit")
helper_source = helper_path.read_text().replace(
    "from keras.preprocessing.sequence import pad_sequences",
    "from tensorflow.keras.preprocessing.sequence import pad_sequences",
)
exec(compile(helper_source, str(helper_path), "exec"), helper.__dict__)
sys.modules["safdKit"] = helper

def load_array(path):
    # The bundled legacy ragged arrays use Python 2 pickles.
    return np.load(path, allow_pickle=True, encoding="latin1")

checkpoint_dir = TemporaryDirectory(prefix="safe-audit-")
source = legacy.read_text().split("mini_batch_size =", 1)[0]
source = str(RefactoringTool(["lib2to3.fixes.fix_print"]).refactor_string(source, str(legacy)))
source = source.replace("import tensorflow as tf\nfrom tensorflow.contrib import rnn\n", "")
source = source.replace("np.load(source_path + ", "load_array(source_path + ")
source = source.replace("save_dir = '../checkpoints/'", "save_dir = checkpoint_dir.name")
exec(compile(source, str(legacy), "exec"))

horizon = 5
train = legacy_records(X_train, T_train, C_train, n_input)
valid = legacy_records(X_valid, T_valid, C_valid, n_input)
test = legacy_records(X_test, T_test, C_test, n_input)
train_kept, valid_kept, test_kept = filter_legacy_overlap(train, valid, test)
counts = [
    (len(kept), len(original))
    for kept, original in ((train_kept, train), (valid_kept, valid), (test_kept, test))
]
print("retained/total train validation test", counts, flush=True)

batches = length_batches(train_kept, 16, seed=7)
validation_x, validation_y = fixed_prefixes(valid_kept, horizon)
grid = np.arange(0.10, 1.0, 0.02)
prefix_mask = upp_tri_mat(np.zeros((horizon, horizon)))

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    def predict(features):
        return session.run(
            survivals, {X: features, batch_size: len(features), mask: prefix_mask}
        )

    def train_batch(features, labels):
        _, loss = session.run(
            [train_mle_op, mle_loss], {X: features, C: labels, batch_size: len(features)}
        )
        return loss

    best, epoch = fit_baseline(
        train_batch, batches, lambda: predict(validation_x), validation_y, grid,
        lambda: saver.save(session, save_path),
    )
    # Test inference starts only after restoring the validation-selected model.
    saver.restore(session, save_path)
    test_x, test_y = fixed_prefixes(test_kept, horizon)
    prediction = predict(test_x)

if not np.isfinite(prediction).all() or (np.diff(prediction, axis=1) > 1e-7).any():
    raise ValueError("expected finite, nonincreasing SAFE survival scores")
metrics = prefix_metrics(test_y, prediction <= best[2])
for index, values in enumerate(metrics, start=1):
    print("prefix", index, values)
print("means", np.mean(metrics, axis=0))
print("best checkpoint", best, "test positives", test_y.sum(), "last epoch", epoch)

# Persist the selected operating point and test metrics for later comparisons.
out = new_run_directory("safe_baseline", sys.argv[1])
summary = {
    "dataset": sys.argv[1],
    "seed": 7,
    "horizon": horizon,
    "exploratory": True,
    "counts": dict(zip(("train", "validation", "test"), counts)),
    "validation_accuracy": float(best[0]),
    "selected_epoch": best[1],
    "threshold": float(best[2]),
    "last_epoch": epoch,
    "test_positives": int(test_y.sum()),
    "metric_order": ["precision", "recall", "f1", "accuracy"],
    "test_metrics_by_prefix": metrics,
    "test_metric_means": np.mean(metrics, axis=0).tolist(),
}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"Results: {out}", file=sys.stderr)
