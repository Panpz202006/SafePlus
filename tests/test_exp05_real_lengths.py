"""Fixed-formula Exp05 real-data adapter: lengths, prior isolation and regression."""
import contextlib,csv,importlib.util,io,json,subprocess,tempfile,unittest
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
base=load('real_base',ROOT/'scripts/34_exp04_learned_behavior_evidence.py')
converter=load('real_converter',ROOT/'scripts/46_exp04_prepare_synthetic_io.py')
validator=load('real_validator',ROOT/'scripts/45_exp04_validate_io.py')
runner=load('real_runner',ROOT/'scripts/47_exp04_run_bundle.py')

class RealLengthTest(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);self.p=Path(self.t.name)
        self.n,self.l,self.d=40,9,3;rng=np.random.default_rng(9)
        self.x=rng.normal(size=(self.n,self.l,self.d));self.labels=np.r_[np.ones(20),np.zeros(20)].astype(np.int64)
        self.detected=np.r_[np.ones(15),np.zeros(25)].astype(np.int64);self.td=np.where(self.detected,5,self.l).astype(np.int64)
        self.lengths=np.array([6+i%4 for i in range(self.n)],dtype=np.int64)
        self.td[self.detected==0]=self.lengths[self.detected==0]
        self.idx=np.arange(self.n)
    def train(self,x,lengths=None,module=base):
        old=module.MAX_EM_ITERS;module.MAX_EM_ITERS=3
        try:
            with contextlib.redirect_stdout(io.StringIO()):return module.train_emission_em(x,self.labels,self.detected,self.td,self.idx,**({'lengths':lengths} if lengths is not None else {}))
        finally:module.MAX_EM_ITERS=old
    def test_all_equal_lengths_retain_frozen_results_exactly(self):
        source=subprocess.check_output(['git','show','06fa34d2097b8ca4fa209dbf60ba551fef31a33d:scripts/34_exp04_learned_behavior_evidence.py'],cwd=ROOT)
        p=self.p/'frozen.py';p.write_bytes(source);frozen=load('old_frozen',p)
        a,h=self.train(self.x,module=frozen);b,k=self.train(self.x,np.full(self.n,self.l))
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(h.to_numpy(),k.to_numpy())
    def test_padding_never_changes_fitted_parameters_even_for_censored_fraud(self):
        a,h=self.train(self.x,self.lengths);changed=self.x.copy()
        for i,n in enumerate(self.lengths):changed[i,n:]=999999
        b,k=self.train(changed,self.lengths)
        for key in a:np.testing.assert_array_equal(a[key],b[key])
        np.testing.assert_array_equal(h.to_numpy(),k.to_numpy())
    def test_exact_observed_point_count_ignores_storage_padding(self):
        p=base.initialize_emissions(self.x,self.labels,self.detected,self.td,self.idx,lengths=self.lengths)
        self.assertEqual(p['n_normal_points'],sum(self.lengths[self.labels==0]))
    def toy_bundle(self):
        src=self.p/'src.npz';out=self.p/'bundle'
        x=self.x.copy();tf=np.where(self.labels,2,-1)
        # Converter intentionally handles only fixed-length source data.
        td=np.where(self.detected,5,self.l).astype(np.int64)
        np.savez_compressed(src,x=x,lengths=np.full(self.n,self.l),is_fraud=self.labels,detected=self.detected,t_detect=td,t_onset=tf,entity_id=np.arange(self.n))
        with contextlib.redirect_stdout(io.StringIO()):converter.convert(src,out)
        with np.load(out/'inputs.npz') as z:arrays={k:z[k].copy() for k in z.files}
        arrays['lengths']=self.lengths;arrays['t_detect'][self.detected==0]=self.lengths[self.detected==0]
        for i,n in enumerate(self.lengths):arrays['x'][i,n:]=0
        np.savez_compressed(out/'inputs.npz',**arrays)
        meta=json.loads((out/'metadata.json').read_text());meta.update(schema_version='1.1',variable_length_policy='slice_before_fit_and_predict');(out/'metadata.json').write_text(json.dumps(meta))
        return out
    def test_variable_length_requires_explicit_policy(self):
        b=self.toy_bundle();self.assertEqual(validator.validate_bundle(b)[0],[])
        m=json.loads((b/'metadata.json').read_text());m.pop('variable_length_policy');(b/'metadata.json').write_text(json.dumps(m))
        self.assertIn('FROZEN_ENGINE_UNSUPPORTED',' '.join(validator.validate_bundle(b)[0]))
    def test_nonzero_padding_rejected(self):
        b=self.toy_bundle()
        with np.load(b/'inputs.npz') as z:arrays={k:z[k].copy() for k in z.files}
        arrays['x'][0,-1]=1;np.savez_compressed(b/'inputs.npz',**arrays)
        self.assertIn('storage padding',' '.join(validator.validate_bundle(b)[0]))
    def test_non_train_prior_onsets_rejected(self):
        b=self.toy_bundle()
        with (b/'split.csv').open() as f:split={r['entity_id']:r['split'] for r in csv.DictReader(f)}
        eid=next(k for k,v in split.items() if v=='TEST')
        (b/'train_onset_for_prior.csv').write_text('entity_id,t_onset\n'+eid+',2\n')
        p=json.loads((b/'delay_prior.json').read_text());p['mode']='empirical_train';(b/'delay_prior.json').write_text(json.dumps(p))
        self.assertIn('non-TRAIN',' '.join(validator.validate_bundle(b)[0]))
    def test_variable_runner_can_train_and_predict_without_gold(self):
        b=self.toy_bundle();(b/'gold_onset.csv').unlink()
        with contextlib.redirect_stdout(io.StringIO()):manifest=runner.run(b,self.p/'pred')
        self.assertEqual(manifest['sequence_handling'],'per_entity_real_lengths')
        for row in map(json.loads,(self.p/'pred/posterior.jsonl').read_text().splitlines()):
            self.assertAlmostEqual(sum(row['posterior_probs']),1,places=10)
        self.assertTrue((self.p/'pred/behavior_model_params.npz').exists())

if __name__=='__main__':unittest.main()
