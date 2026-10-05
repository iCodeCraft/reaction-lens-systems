"""Scoring contracts, without encoders, training data or network access."""
import unittest
import numpy as np
import torch
from reaction_lens import LocalCatalogScorer, decision_payload
from reaction_lens.text import segments, segment_texts


class CoreContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = LocalCatalogScorer(input_dim=8, projection_dim=4, option_block=3).eval()
        self.articles = torch.randn(2, 8)
        self.options = torch.randn(7, 8)
        self.sentences = torch.randn(2, 3, 8)
        self.mask = torch.tensor([[True, True, False], [True, False, False]])

    def test_option_permutation_and_subset(self):
        with torch.inference_mode():
            full = self.model(self.articles, self.options, self.sentences, self.mask)
            order = torch.tensor([6, 1, 3, 0, 2, 5, 4])
            shuffled = self.model(self.articles, self.options[order], self.sentences, self.mask)
            subset = self.model(self.articles, self.options[[1, 4]], self.sentences, self.mask)
        torch.testing.assert_close(shuffled, full[:, order])
        torch.testing.assert_close(subset, full[:, [1, 4]])

    def test_padding_cannot_change_scores(self):
        changed = self.sentences.clone()
        changed[~self.mask] = 1e6
        with torch.inference_mode():
            a = self.model(self.articles, self.options, self.sentences, self.mask)
            b = self.model(self.articles, self.options, changed, self.mask)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_all_options_and_empty_or_multiple_selections(self):
        options = [{"reaction_id": str(i), "description": "option " + str(i)} for i in range(7)]
        logits = np.array([0, 3, 3, -1, 2, 1, -2], dtype=np.float32)
        result = decision_payload(options, logits, 2)
        self.assertEqual(result["options_scored"], 7)
        self.assertEqual(result["proposed_reaction_ids"], ["1", "2", "4"])
        self.assertEqual([r["rank"] for r in result["options"]], [5, 1, 2, 6, 3, 4, 7])
        self.assertEqual(decision_payload(options, logits, 4)["proposed_reaction_ids"], [])

    def test_offsets_and_abbreviation(self):
        row = {"title": "A title", "abstract": "Dr. Smith reports activation. B follows.\nC ends."}
        spans = segments(row)
        self.assertEqual(segment_texts(row, spans), ["A title", "Dr. Smith reports activation.", "B follows.", "C ends."])
        self.assertTrue(all(row[s["field"]][s["start"]:s["end"]].strip() for s in spans))


if __name__ == "__main__":
    unittest.main()
