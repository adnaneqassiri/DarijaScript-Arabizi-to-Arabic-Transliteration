import tempfile
import unittest
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from scripts.data_cleaning import clean_data, prepare_data, split_data


class DataCleaningTests(unittest.TestCase):
    def test_clean_normalizes_and_removes_invalid_duplicate_pairs(self):
        data = pd.DataFrame({
            "Darija_latin": ["  abc ", "abc", "xyz", "", None],
            "Darija_arabic": ["ا\u0654  ب", "أ ب", "ت", "ت", "ت"],
        })
        result = clean_data(data).reset_index(drop=True)
        self.assertEqual(result.to_dict("records"), [
            {"Darija_latin": "abc", "Darija_arabic": "أ ب"},
            {"Darija_latin": "xyz", "Darija_arabic": "ت"},
        ])

    def test_splits_are_repeatable_and_keep_groups_together(self):
        data = pd.DataFrame({"Darija_latin": [f"word{i}" for i in range(20)] + ["word0"],
                             "Darija_arabic": ["أ"] * 20 + ["ب"]})
        first, second = split_data(data, 42), split_data(data, 42)
        for name in first:
            assert_frame_equal(first[name], second[name])
        groups = [set(part["Darija_latin"]) for part in first.values()]
        self.assertTrue(all(not a & b for i, a in enumerate(groups) for b in groups[i + 1:]))
        self.assertEqual(sum(len(part) for part in first.values()), len(data))
        self.assertEqual(sum((part["Darija_latin"] == "word0").sum() == 2
                             for part in first.values()), 1)

    def test_prepare_writes_expected_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            pd.DataFrame({"Darija_latin": ["abc", "def"],
                          "Darija_arabic": ["أ", "ب"]}).to_csv(source, index=False)
            prepare_data(source, root / "out")
            for name in ("train.csv", "val.csv", "test.csv", "split_manifest.json", "doda_train_clean.csv"):
                self.assertTrue((root / "out" / name).is_file())


if __name__ == "__main__":
    unittest.main()
