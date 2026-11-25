import unittest

from scripts.spelling_robustness import build_robustness_dataframe, evaluate_robustness


class SpellingRobustnessTests(unittest.TestCase):
    def test_pairs_and_scoring(self):
        sample = build_robustness_dataframe()
        self.assertEqual(len(sample), 10)
        self.assertTrue((sample["original"] != sample["variant"]).all())
        predictions = dict(zip(sample["original"], sample["reference"]))
        predictions.update(zip(sample["variant"], sample["reference"]))
        predictions[sample.iloc[0]["variant"]] = ""
        results, scores = evaluate_robustness(sample, predictions.__getitem__)
        self.assertEqual(scores["original"]["exact_match"], 1)
        self.assertEqual(scores["variant"]["exact_match"], 0.9)
        self.assertGreater(scores["delta_cer"], 0)
        self.assertIn("variant_prediction", results.columns)


if __name__ == "__main__":
    unittest.main()
