"""Small integration test for frozen Exp04 bundle runner. It does not need project synthetic.
Tests the training/prediction/ground-truth boundary and posterior validity.
"""
from __future__ import annotations
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def load(name,filename):
    spec=importlib.util.spec_from_file_location(name, ROOT/'scripts'/filename)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

converter=load('safeplus_converter2','46_exp04_prepare_synthetic_io.py')
runner=load('safeplus_runner2','47_exp04_run_bundle.py')
evaluator=load('safeplus_evaluator2','48_exp04_evaluate_bundle.py')

class TestStage2(unittest.TestCase):
    def test_prediction_does_not_need_gold_and_posterior_normalized(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td); src=td/'toy.npz'; bundle=td/'bundle'; pred=td/'pred'
            n,l,d=60,12,3
            rng=np.random.default_rng(7)
            x=rng.normal(size=(n,l,d))
            fraud=np.array([1]*30+[0]*30,dtype=np.int64)
            tf=np.array([3]*30+[-1]*30,dtype=np.int64)
            x[:30,3:,:]+=.65
            detected=np.array([1]*28+[0]*32,dtype=np.int64)
            observed=np.where(detected==1,7,l).astype(np.int64)
            np.savez_compressed(src,x=x,lengths=np.full(n,l),detected=detected,t_detect=observed,t_onset=tf,is_fraud=fraud,entity_id=np.arange(n))
            converter.convert(src,bundle)
            truth=bundle/'gold_onset.csv'
            hidden=bundle/'gold_hidden_in_test.csv'
            truth.rename(hidden)
            try:
                # Must not inspect/read t_f to fit or predict.
                manifest=runner.run(bundle,pred,split='TEST')
            finally:
                hidden.rename(truth)
            self.assertEqual(manifest['n_split'],12)
            import pandas as pd
            output=pd.read_csv(pred/'predictions.csv')
            self.assertEqual(len(output),12)
            rec=[json.loads(x) for x in (pred/'posterior.jsonl').read_text(encoding='utf-8').splitlines()]
            for r in rec:
                self.assertAlmostEqual(sum(r['posterior_probs']),1,places=10)
                self.assertEqual(len(r['candidate_times']),len(r['posterior_probs']))
            metrics=evaluator.evaluate(bundle,pred)
            self.assertGreater(metrics['n_eligible_exact_detected_fraud'],0)
            self.assertTrue(np.isfinite(metrics['mae_median']))

if __name__=='__main__': unittest.main()
