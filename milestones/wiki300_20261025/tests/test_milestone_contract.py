"""Milestone acceptance requirements checked against immutable source records."""
import copy, importlib.util, json, sys, unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
QUEUE_ROOT=ROOT/'history/review_before_rules_v1_1/HUMAN_REVIEW_QUEUE'
if not QUEUE_ROOT.exists():QUEUE_ROOT=ROOT/'HUMAN_REVIEW_QUEUE'
sys.path.insert(0,str(ROOT/'scripts'))
import build_review_queue as review
from audit_dataset import csvrows, jsonl, utc

class MilestoneContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.art=ROOT/'artifacts/exp05';b=cls.art/'data_bundle'
        cls.source=jsonl(cls.art/'frozen_source/SafePlus_combined_current.jsonl')
        cls.byid={'wiki_u'+str(x['userid']):x for x in cls.source}
        cls.splits={x['entity_id']:x['split'] for x in csvrows(b/'split.csv')}
        cls.existing=csvrows(ROOT/'code/evaluation_v2/reports/independent_review_queue_36.csv')
        cls.prior={x['entity_id'] for x in csvrows(b/'train_onset_for_prior.csv')}
        cls.selected=review.select(cls.source,cls.splits,cls.existing,cls.prior)

    def test_selection_twelve_train_unique_and_existing_queue(self):
        ids=['wiki_u'+str(x[2]['userid']) for x in self.selected]
        self.assertEqual(len(ids),12);self.assertEqual(len(set(ids)),12)
        self.assertTrue(all(self.splits[x]=='TRAIN' for x in ids))
        self.assertTrue(all(x[3] for x in self.selected))

    def test_selection_has_exact_interval_censored_4_each(self):
        types=[x[2].get('onset_type','RIGHT_CENSORED') for x in self.selected]
        self.assertEqual(sum(x.startswith('EXACT') for x in types),4)
        self.assertEqual(types.count('INTERVAL'),4)
        self.assertEqual(types.count('RIGHT_CENSORED'),4)
        for _,_,r,_,_ in self.selected:
            if r.get('onset_type')=='INTERVAL':self.assertTrue(0<utc(r['tf_upper_utc'])-utc(r['tf_lower_utc'])<=86400)

    def test_selection_invariant_to_source_order(self):
        other=review.select(list(reversed(self.source)),self.splits,list(reversed(self.existing)),self.prior)
        self.assertEqual([x[2]['userid'] for x in self.selected],[x[2]['userid'] for x in other])

    def test_packet_status_never_automatically_human_gold(self):
        tasks=jsonl(QUEUE_ROOT/'HUMAN_REVIEW_QUEUE.jsonl')
        self.assertEqual(len(tasks),12)
        for t in tasks:
            for key in ['human_gold_status','reviewer_A_status','reviewer_B_status','adjudication_status']:self.assertEqual(t[key],'PENDING')
            self.assertEqual(t['split'],'TRAIN')

    def test_packet_cached_text_equals_original_source(self):
        for t in jsonl(QUEUE_ROOT/'HUMAN_REVIEW_QUEUE.jsonl'):
            s=self.byid[t['entity_id']]
            self.assertEqual(t['normal_evidence'],s.get('normal_prefix',s.get('events',[])))
            self.assertEqual(t['onset_evidence'],s.get('onset_evidence',{}))
            self.assertEqual(t['original_ai_rationale'],s.get('reason') or s.get('rationale') or '')

    def test_packet_interval_and_control_have_no_invented_point(self):
        for t in jsonl(QUEUE_ROOT/'HUMAN_REVIEW_QUEUE.jsonl'):
            if t['source_label_type']=='INTERVAL':self.assertFalse(t['onset_utc']);self.assertEqual(t['interval']['closure'],'(lower,upper]')
            if t['source_label_type']=='RIGHT_CENSORED':self.assertFalse(t['onset_utc']);self.assertFalse(t['td_utc']);self.assertIsNone(t['interval'])

    def test_legacy_account_aliases_resolve_all_matched_pairs(self):
        alias={r['account_id']:'wiki_u'+str(r['userid']) for r in self.source}
        groups=json.loads((self.art/'data_bundle/split_groups.json').read_text())
        pairs=0
        for r in self.source:
            match=r.get('matching_plan',{}).get('matched_positive_id')
            if match:
                uid='wiki_u'+str(r['userid']);self.assertIn(match,alias)
                self.assertEqual(groups[uid],groups[alias[match]]);self.assertEqual(self.splits[uid],self.splits[alias[match]]);pairs+=1
        self.assertEqual(pairs,100)

    def test_review_html_contains_no_external_code(self):
        text=(QUEUE_ROOT/'REVIEW_ALL_12.html').read_text()
        self.assertNotIn('<script',text.lower());self.assertNotIn('<iframe',text.lower())
        self.assertEqual(text.count('<article id='),12)
        self.assertIn('https://en.wikipedia.org/',text)

if __name__=='__main__':unittest.main()
