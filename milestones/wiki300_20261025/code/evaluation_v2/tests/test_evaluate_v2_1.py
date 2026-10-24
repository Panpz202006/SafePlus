import csv,hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
SPEC=importlib.util.spec_from_file_location('evaluation_v21',Path(__file__).resolve().parents[1]/'scripts/evaluate_v2_1.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)

class MetricDefinitions(unittest.TestCase):
    def test_own_event_denominator(self):self.assertAlmostEqual(m.normalize(3,19),3/19)
    def test_zero_denominator_is_not_zero_error(self):self.assertIsNone(m.normalize(0,0))
    def test_incomplete_window_not_normalized(self):self.assertIsNone(m.normalize(1,100,False))
    def test_p90_linear_interpolation(self):self.assertAlmostEqual(m.quantile([0,1,2,3],.9),2.7)
    def test_empty_summary_is_unavailable(self):self.assertIsNone(m.describe([])['median']);self.assertEqual(m.describe([])['n'],0)
    def test_midpoint_uses_fixed_window_and_earlier_tie(self):self.assertEqual(m.simple_baselines([0,4,6],0,10),{'last_edit':2,'time_midpoint':1})
    def test_open_lower_is_not_hit_even_zero_distance(self):self.assertFalse(m.v2.interval_hit(0,0,4));self.assertEqual(m.v2.interval_distance(0,0,4),0)
    def test_interval_upper_is_hit(self):self.assertTrue(m.v2.interval_hit(4,0,4))
    def test_interval_delay_ambiguity_not_hidden(self):self.assertEqual(m.interval_delay_group(.5,10),'AMBIGUOUS_RANGE')
    def test_interval_delay_exclusive_endpoint(self):self.assertEqual(m.interval_delay_group(1.1,7),'1_to_7d')
    def test_positive_scaling_cannot_change_within_account_win(self):self.assertEqual(m.compare_loss(2,3),m.compare_loss(2/100,3/100))
    def test_rate_uncertainty_not_perfect_certainty(self):self.assertLess(m.wilson(3,3)['ci95'][0],1);self.assertIsNone(m.wilson(0,0)['rate'])
    def test_contradictory_completeness_not_accepted(self):self.assertFalse(m.window_complete({'coverage':{'contributions_complete':True,'public_contributions_complete':False}}))

class FrozenFixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.b=self.root/'bundle';self.r=self.root/'run';self.b.mkdir();self.r.mkdir();self.source=self.root/'source.jsonl'
        (self.b/'inputs.npz').write_bytes(b'UNIT_TEST_HASH_ONLY_NO_MODEL')
        self.times=['2026-01-01T00:00:00Z','2026-01-01T00:04:00Z','2026-01-01T00:08:00Z','2026-01-01T00:10:00Z']
        self.gold=[];self.pred=[];self.sources=[];self.timelines=[];self.raw=[]
        for num,typ in enumerate(['EXACT','INTERVAL','RIGHT_CENSORED','UNCERTAIN'],1):
            uid='wiki_u'+str(num);censored=typ in ['RIGHT_CENSORED','UNCERTAIN']
            self.gold.append({'entity_id':uid,'onset_label_type':'NOT_FRAUD' if typ=='RIGHT_CENSORED' else typ,'onset_utc':self.times[1] if typ=='EXACT' else '',
                'onset_lower_utc':self.times[0] if typ=='INTERVAL' else self.times[1] if typ=='EXACT' else '',
                'onset_upper_utc':self.times[1] if typ in ['EXACT','INTERVAL'] else '',
                't_onset':'1' if typ=='EXACT' else '','point_is_published_event':'true' if typ=='EXACT' else 'false','interval_closure':'(lower,upper]' if typ=='INTERVAL' else '',
                'source_event_id':'2' if typ=='EXACT' else ''})
            self.pred.append({'entity_id':uid,'status':'NOT_DETECTED_UNSUPPORTED' if typ=='RIGHT_CENSORED' else 'REVIEW_PENDING' if typ=='UNCERTAIN' else 'OK',
                'observed_td':'' if censored else '3','pred_tf_median':'' if censored else '1' if typ=='EXACT' else '0','behavior_only_median':'' if censored else '0' if typ=='EXACT' else '1','delay_only_median':'' if censored else '2'})
            self.sources.append({'userid':num,'username':'fixture'+str(num),'label_type':typ,'onset_type':typ,'tf_utc':self.times[1] if typ=='EXACT' else None,
                'tf_lower_utc':self.times[0] if typ=='INTERVAL' else None,'tf_upper_utc':self.times[1] if typ=='INTERVAL' else None,'normal_prefix_count':1,'coverage':{'contributions_complete':True}})
            self.timelines.append({'entity_id':uid,'times_utc':self.times[:3] if censored else self.times,'row_kinds':['published_revision']*3+([] if censored else ['pre_decision_snapshot']),
                'observation_start_utc':self.times[0],'observation_end_utc':self.times[2] if censored else self.times[3]})
            self.raw.append({'entity_id':uid,'events':[{'revision_id':str(i+1),'timestamp_utc':t} for i,t in enumerate(self.times[:3])]})
        self.write()
    def csv(self,p,rows):
        with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def write(self):
        self.csv(self.b/'gold_onset.csv',self.gold);self.csv(self.b/'split.csv',[{'entity_id':x['entity_id'],'split':'TEST'} for x in self.gold]);self.csv(self.r/'predictions.csv',self.pred)
        for name,rows in [('timeline.jsonl',self.timelines),('raw_inputs.jsonl',self.raw)]: (self.b/name).write_text(''.join(json.dumps(x)+'\n' for x in rows))
        self.source.write_text(''.join(json.dumps(x)+'\n' for x in self.sources))
        (self.r/'run_manifest.json').write_text(json.dumps({'dataset_id':'UNIT_TEST_ONLY','split':'TEST','input_sha256':hashlib.sha256((self.b/'inputs.npz').read_bytes()).hexdigest(),'split_sha256':hashlib.sha256((self.b/'split.csv').read_bytes()).hexdigest()}))
    def score(self):return m.score(self.b,self.r,self.source,self.root/'out')
    def test_same_account_five_methods_and_separate_types(self):
        result=self.score();self.assertEqual(result['cohorts']['exact_published_event']['n_accounts'],1);self.assertEqual(result['cohorts']['interval']['n_accounts'],1)
        self.assertEqual(result['models'],list(m.METHODS));self.assertEqual(result['excluded_from_onset_metrics'],{'RIGHT_CENSORED':1,'UNCERTAIN':1})
    def test_normal_control_not_counted_as_correct(self):
        result=self.score();self.assertEqual(result['normal_control_fpr']['status'],'NOT_EVALUABLE_NO_RISK_OUTPUT');self.assertIsNone(result['normal_control_fpr']['false_positive_rate'])
    def test_duplicate_prediction_rejected(self):
        self.pred.append(self.pred[0].copy());self.write()
        with self.assertRaises(ValueError):self.score()
    def test_current_label_change_cannot_be_silently_scored(self):
        self.sources[0]['tf_utc']=self.times[2];self.write()
        with self.assertRaisesRegex(ValueError,'timestamp changed'):self.score()
    def test_noninteger_prediction_rejected(self):
        self.pred[0]['pred_tf_median']='1.5';self.write()
        with self.assertRaises(ValueError):self.score()
    def test_wrong_event_revision_id_rejected(self):
        self.gold[0]['source_event_id']='999';self.write()
        with self.assertRaisesRegex(ValueError,'ID and UTC'):self.score()
    def test_interval_cannot_be_point_truth(self):
        self.gold[1]['onset_utc']=self.times[1];self.write()
        with self.assertRaisesRegex(ValueError,'collapsed'):self.score()
    def test_control_must_not_receive_onset(self):
        self.pred[2]['pred_tf_median']='0';self.write()
        with self.assertRaisesRegex(ValueError,'invented onset'):self.score()
    def test_unaligned_exact_utc_does_not_get_event_error(self):
        self.gold[0]['point_is_published_event']='false';self.gold[0]['onset_utc']='2026-01-01T00:02:00Z';self.sources[0]['tf_utc']=self.gold[0]['onset_utc'];self.write()
        self.gold[0]['onset_lower_utc']=self.gold[0]['onset_utc'];self.gold[0]['onset_upper_utc']=self.gold[0]['onset_utc'];self.write()
        result=self.score();self.assertNotIn('exact_published_event',result['cohorts']);self.assertEqual(result['cohorts']['exact_utc_other_log']['n_accounts'],1)
    def test_exact_bounds_cannot_hide_an_interval(self):
        self.gold[0]['onset_upper_utc']=self.times[2];self.write()
        with self.assertRaisesRegex(ValueError,'EXACT bounds'):self.score()
    def test_one_event_has_explicit_unavailable_normalization(self):
        self.timelines[0]['times_utc']=[self.times[0],self.times[3]];self.timelines[0]['row_kinds']=['published_revision','pre_decision_snapshot']
        self.raw[0]['events']=self.raw[0]['events'][:1]
        for key in ['onset_utc','onset_lower_utc','onset_upper_utc']:self.gold[0][key]=self.times[0]
        self.gold[0]['t_onset']='0';self.gold[0]['source_event_id']='1';self.sources[0]['tf_utc']=self.times[0]
        for col in ['pred_tf_median','behavior_only_median','delay_only_median']:self.pred[0][col]='0'
        self.pred[0]['observed_td']='1';self.write();result=self.score()
        metrics=result['cohorts']['exact_published_event']['models']['fusion']['metrics'];self.assertEqual(metrics['event_error']['n'],1);self.assertEqual(metrics['event_norm']['n'],0)
        rows=m.v2.read_csv(self.root/'out/per_account_methods_v2_1.csv');self.assertTrue(all(x['event_norm_reason']=='N_LE_1' for x in rows if x['entity_id']=='wiki_u1'))
    def test_incomplete_window_utc_error_retained_normalization_unavailable(self):
        self.sources[0]['coverage']['contributions_complete']=False;self.write();result=self.score()
        metrics=result['cohorts']['exact_utc_all']['models']['fusion']['metrics'];self.assertEqual(metrics['time_error_hours']['n'],1);self.assertEqual(metrics['time_norm']['n'],0)
    def test_open_boundary_distance_tie_does_not_become_hit(self):
        self.score();rows=m.v2.read_csv(self.root/'out/per_account_paired_v2_1.csv')
        x=next(x for x in rows if x['entity_id']=='wiki_u2' and x['baseline']=='behavior' and x['metric']=='interval_norm')
        self.assertEqual(x['outcome'],'tie');self.assertEqual(x['interval_hit_outcome'],'loss')
    def test_read_only_source_and_predictions(self):
        before=(self.source.read_bytes(),(self.r/'predictions.csv').read_bytes());self.score();self.assertEqual(before,(self.source.read_bytes(),(self.r/'predictions.csv').read_bytes()))
    def test_output_cannot_overwrite_original_bundle(self):
        with self.assertRaisesRegex(ValueError,'overwrite frozen'):m.score(self.b,self.r,self.source,self.b/'scoring')
    def test_existing_result_not_overwritten(self):
        self.score()
        with self.assertRaisesRegex(ValueError,'already contains'):self.score()

if __name__=='__main__':unittest.main()
