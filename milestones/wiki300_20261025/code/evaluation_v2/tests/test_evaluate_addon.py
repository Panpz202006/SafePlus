import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

SPEC=importlib.util.spec_from_file_location('evaluate_addon',Path(__file__).resolve().parents[1]/'scripts'/'evaluate_addon.py')
m=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)

class TestMetrics(unittest.TestCase):
    def test_open_closed_boundary(self):
        self.assertFalse(m.interval_hit(10,10,20))
        self.assertTrue(m.interval_hit(20,10,20))
        self.assertEqual(m.interval_distance(10,10,20),0)
    def test_interval_outside_distance(self):
        self.assertEqual(m.interval_distance(5,10,20),5)
        self.assertEqual(m.interval_distance(22,10,20),2)
    def test_width_group(self):
        self.assertEqual(m.width_group(1),'le_1d')
        self.assertEqual(m.width_group(7),'1_to_7d')
        self.assertEqual(m.width_group(30),'7_to_30d')
        self.assertEqual(m.width_group(30.0001),'gt_30d')
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):m.indexed_unique([{'id':1},{'id':1}],'id')
    def test_utc_requires_timezone(self):
        with self.assertRaises(ValueError):m.utc('2026-10-24T12:00:00')

if __name__=='__main__':unittest.main()
