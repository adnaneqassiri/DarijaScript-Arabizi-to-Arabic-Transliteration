import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.metrics import calculate_metrics, edit_distance, evaluate_csv, evaluate_dataframe


class MetricsTests(unittest.TestCase):
    def test_dataframe_columns_and_missing_values(self):
        frame = pd.DataFrame({"Darija_arabic": ["أ", "ب"], "output": ["أ", "ت"]})
        result = evaluate_dataframe(frame, reference_col="Darija_arabic", prediction_col="output")
        self.assertEqual(result["example_count"], 2)
        self.assertEqual(result["cer"], 0.5)
        self.assertEqual(result["exact_match"], 0.5)
        with self.assertRaisesRegex(ValueError, "exactly one column"):
            evaluate_dataframe(frame)
        frame.loc[0, "output"] = None
        with self.assertRaisesRegex(ValueError, "missing values"):
            evaluate_dataframe(frame, "Darija_arabic", "output")

    def test_corpus_scores_and_normalization(self):
        result = calculate_metrics(["  ا\u0654  بيت ", "abc"], ["أ بيت", "ax"])
        self.assertEqual(result["example_count"], 2)
        self.assertEqual(result["character_errors"], 2)
        self.assertEqual(result["reference_characters"], 8)
        self.assertEqual(result["word_errors"], 1)
        self.assertEqual(result["reference_words"], 3)
        self.assertEqual(result["cer"], 0.25)
        self.assertEqual(result["wer"], 1 / 3)
        self.assertEqual(result["exact_match"], 0.5)

    def test_edit_operations_and_empty_references(self):
        self.assertEqual(edit_distance("kitten", "sitting"), 3)
        result = calculate_metrics([""], ["a b"])
        self.assertIsNone(result["cer"])
        self.assertIsNone(result["wer"])
        self.assertEqual(result["character_errors"], 3)
        self.assertEqual(result["word_errors"], 2)
        with self.assertRaises(ValueError):
            calculate_metrics([], [])
        with self.assertRaises(ValueError):
            calculate_metrics(["a"], [])

    def test_csv_coverage_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory) / "predictions.csv"
            split = Path(directory) / "split.csv"
            with predictions.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["example_id", "reference", "prediction", "model_id", "run_id"])
                writer.writerows([["1", "أ", "أ", "baseline", "run1"],
                                  ["2", "ب", "ت", "baseline", "run1"]])
            split.write_text("example_id\n1\n2\n", encoding="utf-8")
            self.assertEqual(evaluate_csv(predictions, split)["exact_match"], 0.5)
            split.write_text("example_id\n2\n1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "order"):
                evaluate_csv(predictions, split)
            split.write_text("example_id\n1\n3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "coverage"):
                evaluate_csv(predictions, split)


if __name__ == "__main__":
    unittest.main()
