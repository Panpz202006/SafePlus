"""Fast tests for the Exp04 I/O contract. No actual training or original dataset required."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


validator = load("45_exp04_validate_io.py", "validator")
converter = load("46_exp04_prepare_synthetic_io.py", "converter")


class TestContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        n, l, d = 50, 12, 3
        rng = np.random.default_rng(99)
        x = rng.normal(size=(n, l, d))
        fraud = np.array([1] * 25 + [0] * 25, dtype=np.int64)
        detected = np.array([1] * 23 + [0] * 27, dtype=np.int64)
        tf = np.array([3] * 25 + [-1] * 25, dtype=np.int64)
        td = np.where(detected == 1, 7, l)
        self.source = self.base / "synthetic.npz"
        np.savez_compressed(self.source, x=x, detected=detected, t_detect=td,
                            t_onset=tf, is_fraud=fraud,
                            lengths=np.full(n, l, dtype=np.int64), entity_id=np.arange(n))
        self.bundle = self.base / "bundle"
        converter.convert(self.source, self.bundle)

    def inspect(self):
        return validator.validate_bundle(self.bundle)[0]

    def mutate_npz(self, updates):
        p = self.bundle / "inputs.npz"
        with np.load(p, allow_pickle=False) as f:
            values = {name: f[name] for name in f.files}
        values.update(updates)
        np.savez_compressed(p, **values)

    def test_good_contract_passes(self):
        self.assertEqual(self.inspect(), [])
        with np.load(self.bundle / "inputs.npz", allow_pickle=False) as f:
            self.assertNotIn("t_onset", f.files)
            self.assertNotIn("is_fraud", f.files)
            self.assertTrue(np.all(f["x"][0, 8:, :] == 0))

    def test_leaked_truth_is_rejected(self):
        self.mutate_npz({"t_onset": np.zeros(50, dtype=np.int64)})
        self.assertIn("forbidden extra keys", " ".join(self.inspect()))

    def test_wrong_detect_time_is_rejected(self):
        with np.load(self.bundle / "inputs.npz", allow_pickle=False) as f:
            td = f["t_detect"].copy()
        td[0] = 12
        self.mutate_npz({"t_detect": td})
        self.assertIn("Invalid t_detect", " ".join(self.inspect()))

    def test_variable_length_not_silently_accepted(self):
        with np.load(self.bundle / "inputs.npz", allow_pickle=False) as f:
            lengths = f["lengths"].copy()
        lengths[0] = 9
        self.mutate_npz({"lengths": lengths})
        self.assertIn("FROZEN_ENGINE_UNSUPPORTED", " ".join(self.inspect()))

    def test_post_detection_data_is_rejected(self):
        with np.load(self.bundle / "inputs.npz", allow_pickle=False) as f:
            x = f["x"].copy()
        x[0, 9, 0] = 12.0
        self.mutate_npz({"x": x})
        self.assertIn("masked to zero", " ".join(self.inspect()))

    def test_train_labels_cannot_contain_test_entity(self):
        p = self.bundle / "train_labels.csv"
        with p.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        rows.pop()
        rows.append({"entity_id": "999999", "is_fraud": "1"})
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["entity_id", "is_fraud"])
            w.writeheader(); w.writerows(rows)
        self.assertIn("ALL and ONLY TRAIN", " ".join(self.inspect()))

    def test_real_data_cannot_use_synthetic_prior(self):
        p = self.bundle / "metadata.json"
        m = json.loads(p.read_text(encoding="utf-8"))
        m["dataset_kind"] = "real"
        p.write_text(json.dumps(m), encoding="utf-8")
        self.assertIn("prohibited for real", " ".join(self.inspect()))

    def test_gold_onset_after_detection_fails(self):
        p = self.bundle / "gold_onset.csv"
        with p.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
            rows = list(reader)
        rows[0]["t_onset"] = "7"
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader(); w.writerows(rows)
        self.assertIn("strictly before t_detect", " ".join(self.inspect()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
