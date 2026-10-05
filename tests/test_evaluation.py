"""Check benchmark semantics independently of the trained model."""
import unittest
import numpy as np
from reaction_lens.evaluation import ranking_metrics


class EvaluationTests(unittest.TestCase):
    def test_missing_catalog_labels_count_as_misses(self):
        ap, hit, found, gold = ranking_metrics(np.array([3.,2.,1.]), ['b','absent'], ['a','b','c'])
        self.assertEqual((ap,hit,found,gold),(.25,0,1,2))

    def test_ties_use_catalog_order(self):
        ap,hit,_,_ = ranking_metrics(np.array([1.,1.]),['b'],['a','b'])
        self.assertEqual((ap,hit),(.5,0))

    def test_empty_gold_and_nonfinite_scores_fail(self):
        for score,gold in [(np.array([1.]),[]),(np.array([np.nan]),['a'])]:
            with self.assertRaises(ValueError):
                ranking_metrics(score,gold,['a'])

if __name__ == '__main__':
    unittest.main()
