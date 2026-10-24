"""Current formal review checks; distinct from historical frozen-v1 metrics."""
import importlib.util,json,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('review_validator',ROOT/'scripts/validate_review_revision.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
class ReviewRevision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.result=v.validate()
    def test_current_annotations_and_evidence_integrity(self):self.assertEqual(self.result['status'],'PASS',self.result)
    def test_uncertain_retained_in_300(self):
        r=v.jl(ROOT/'data_review/SafePlus_combined_current.jsonl');x=next(x for x in r if x['username']=='Chimpanzeedeerkidney');self.assertEqual(len(r),300);self.assertEqual(x['onset_type'],'UNCERTAIN');self.assertIsNone(x['entity_label'])
    def test_attempt_not_published_onset(self):
        x=next(x for x in v.jl(ROOT/'data_review/review12_current.jsonl') if x['task_number']==4);self.assertEqual(x['onset_event_kind'],'VIOLATION_ATTEMPT');self.assertIsNone(x['published_violation_onset_utc'])
    def test_rd3_not_converted_into_point(self):
        x=next(x for x in v.jl(ROOT/'data_review/review12_current.jsonl') if x['task_number']==7);self.assertEqual(x['label_type'],'INTERVAL');self.assertFalse(x['rd3_report']['api_verified']);self.assertIsNone(x['tf_utc'])
    def test_namespace_user_and_user_talk_preserved(self):
        x=next(x for x in v.jl(ROOT/'data_review/review12_current.jsonl') if x['task_number']==3)
        from collections import Counter
        self.assertEqual(Counter(e['namespace'] for e in x['normal_prefix_evidence']),{2:10,3:3})
    def test_ordinary_self_correction_stays_censored(self):
        rows=v.jl(ROOT/'data_review/review12_current.jsonl');self.assertTrue(all(x['label_type']=='RIGHT_CENSORED' for x in rows if x['task_number']>=9))
    def test_old_generator_cannot_overwrite_formal_records(self):
        p=ROOT/'HUMAN_REVIEW_QUEUE/HUMAN_REVIEW_QUEUE.jsonl';before=p.read_bytes()
        q=subprocess.run([sys.executable,str(ROOT/'scripts/build_review_queue.py')],capture_output=True,text=True)
        self.assertNotEqual(q.returncode,0);self.assertIn('Current formal review revision exists',q.stderr);self.assertEqual(before,p.read_bytes())
if __name__=='__main__':unittest.main()
