"""Build and score a small, paired Arabizi spelling-variation sample."""

from pathlib import Path
from collections.abc import Callable

import pandas as pd

from scripts.metrics import evaluate_dataframe


# Each replacement is limited to one known test example, not applied globally.
VARIANTS = {
    9: ("ch7al", "chhal"),
    77: ("bzzaf", "bzaaf"),
    161: ("wakha", "wa5a"),
    255: ("wach", "wash"),
    559: ("nl3eb", "nl3ab"),
    667: ("swwelti", "swelti"),
    698: ("khrej", "5rej"),
    760: ("chi", "shi"),
    827: ("nchoufouk", "nchufouk"),
    2137: ("3omrek", "3umrek"),
}
DEFAULT_TEST_CSV = Path(__file__).resolve().parents[1] / "data/processed/test.csv"


def build_robustness_dataframe(test_csv: str | Path = DEFAULT_TEST_CSV) -> pd.DataFrame:
    """Return original and variant inputs paired with their test references."""
    data = pd.read_csv(test_csv, dtype={"example_id": str}, keep_default_na=False)
    ids = [str(example_id) for example_id in VARIANTS]
    sample = data.loc[data["example_id"].isin(ids),
                      ["example_id", "Darija_latin", "Darija_arabic"]].copy()
    if len(sample) != len(ids) or set(sample["example_id"]) != set(ids):
        raise ValueError("Expected each robustness sample ID exactly once in the test split")
    variants = []
    for row in sample.itertuples(index=False):
        original, replacement = VARIANTS[int(row.example_id)]
        if row.Darija_latin.count(original) != 1:
            raise ValueError(f"Expected one {original!r} in example {row.example_id}")
        variants.append(row.Darija_latin.replace(original, replacement, 1))
    sample = sample.rename(columns={"Darija_latin": "original", "Darija_arabic": "reference"})
    sample["variant"] = variants
    return sample.reset_index(drop=True)


def evaluate_robustness(sample: pd.DataFrame, transliterate: Callable[[str], str]) -> tuple[pd.DataFrame, dict]:
    """Run a method on both inputs and return paired predictions and metrics."""
    results = sample.copy()
    for kind in ("original", "variant"):
        results[f"{kind}_prediction"] = results[kind].map(transliterate)
    scores = {kind: evaluate_dataframe(results, "reference", f"{kind}_prediction")
              for kind in ("original", "variant")}
    original_cer, variant_cer = scores["original"]["cer"], scores["variant"]["cer"]
    scores["delta_cer"] = (variant_cer - original_cer
                           if original_cer is not None and variant_cer is not None else None)
    return results, scores
