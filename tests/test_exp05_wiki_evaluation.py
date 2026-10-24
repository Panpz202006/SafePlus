"""Independent scorer fixtures: no real model predictions are manufactured."""
import csv,hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('wiki_eval',ROOT/'scripts/50_exp05_evaluate_wiki.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class WikiEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);p=Path(self.t.name);self.b=p/'b';self.o=p/'o';self.b.mkdir();self.o.mkdir()
        (self.b/'inputs.npz').write_bytes(b'UNIT_TEST_HASH_FIXTURE')
        self.write(self.b/'split.csv',['entity_id','split'],[{'entity_id':e,'split':'TEST'} for e in ['e','i','n']])
        times=[f'2026-01-01T00:0{k}:00Z' for k in range(5)]
        (self.b/'timeline.jsonl').write_text(''.join(json.dumps({'entity_id':e,'times_utc':times})+'\n' for e in ['e','i','n']))
        self.gold=[]
        for eid,typ in [('e','EXACT'),('i','INTERVAL'),('n','NOT_FRAUD')]:
            self.gold.append({'entity_id':eid,'onset_label_type':typ,'original_onset_type':typ,'onset_utc':times[1] if eid=='e' else '',
                'onset_lower_utc':times[1] if eid=='i' else '','onset_upper_utc':times[3] if eid=='i' else '',
                't_onset':'0' if eid=='e' else '','interval_closure':'(lower,upper]' if eid=='i' else '',
                'point_is_published_event':'false'})
        self.write(self.b/'gold_onset.csv',list(self.gold[0]),self.gold)
        self.rows=[{'entity_id':eid,'status':'NOT_DETECTED_UNSUPPORTED' if eid=='n' else 'OK','observed_td':'' if eid=='n' else '4',
                    'pred_tf_median':'' if eid=='n' else ('1' if eid=='i' else '0'),
                    'behavior_only_median':'' if eid=='n' else '0','delay_only_median':'' if eid=='n' else '0'} for eid in ['e','i','n']]
        self.write(self.o/'predictions.csv',list(self.rows[0]),self.rows)
        manifest={'split':'TEST','dataset_id':'UNIT_TEST_ONLY','input_sha256':hashlib.sha256((self.b/'inputs.npz').read_bytes()).hexdigest(),'split_sha256':hashlib.sha256((self.b/'split.csv').read_bytes()).hexdigest()}
        (self.o/'run_manifest.json').write_text(json.dumps(manifest))
    def write(self,p,fields,rows):
        with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    def test_exact_utc_not_mistaken_for_exact_event_index(self):
        r=m.evaluate(self.b,self.o);self.assertEqual(r['exact_utc_n'],1);self.assertEqual(r['exact_event_step_n'],0)
        self.assertEqual(r['exact_mae_seconds']['final'],60)
    def test_open_lower_boundary_zero_distance_but_not_hit(self):
        r=m.evaluate(self.b,self.o);self.assertEqual(r['interval_distance_days']['final'],0)
        self.assertEqual(r['interval_point_hit_rate_open_closed']['final'],0)
    def test_interval_cannot_be_given_point_truth(self):
        self.gold[1]['t_onset']='1';self.write(self.b/'gold_onset.csv',list(self.gold[0]),self.gold)
        with self.assertRaises(ValueError):m.evaluate(self.b,self.o)
    def test_control_onset_is_rejected(self):
        self.rows[2]['pred_tf_median']='1';self.write(self.o/'predictions.csv',list(self.rows[0]),self.rows)
        with self.assertRaises(ValueError):m.evaluate(self.b,self.o)

if __name__=='__main__':unittest.main()
