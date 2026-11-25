"""Reproducible transliteration metrics and CSV evaluation.

Input CSV columns: example_id, reference, prediction, model_id, run_id.
Use --expected-ids with a split CSV containing example_id to check coverage.
Alternative spellings are counted as errors unless they match the reference
under NFC and whitespace normalization.
"""

import argparse
import csv
import json
import unicodedata
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")
REQUIRED_COLUMNS = {"example_id", "reference", "prediction", "model_id", "run_id"}


def normalize_text(text: str) -> str:
    """Apply NFC and collapse all whitespace to single spaces."""
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    return " ".join(unicodedata.normalize("NFC", text).split())


def edit_distance(reference: Sequence[T], prediction: Sequence[T]) -> int:
    """Return Levenshtein distance using memory proportional to the shorter input."""
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for i, ref_item in enumerate(reference, 1):
        current = [i]
        for j, pred_item in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (ref_item != pred_item)))
        previous = current
    return previous[-1]


def calculate_metrics(references: Iterable[str], predictions: Iterable[str]) -> dict:
    """Calculate corpus CER, WER, exact match, and their underlying counts.

    Undefined error rates (zero reference characters or words) are returned as
    None. Empty datasets and mismatched input lengths are rejected.
    """
    references, predictions = list(references), list(predictions)
    if len(references) != len(predictions):
        raise ValueError("References and predictions must have the same length")
    if not references:
        raise ValueError("At least one example is required")
    char_errors = word_errors = reference_characters = reference_words = exact_matches = 0
    for reference, prediction in zip(references, predictions):
        reference, prediction = normalize_text(reference), normalize_text(prediction)
        ref_words, pred_words = reference.split(), prediction.split()
        char_errors += edit_distance(reference, prediction)
        word_errors += edit_distance(ref_words, pred_words)
        reference_characters += len(reference)
        reference_words += len(ref_words)
        exact_matches += reference == prediction
    return {
        "example_count": len(references),
        "cer": char_errors / reference_characters if reference_characters else None,
        "wer": word_errors / reference_words if reference_words else None,
        "exact_match": exact_matches / len(references),
        "character_errors": char_errors,
        "reference_characters": reference_characters,
        "word_errors": word_errors,
        "reference_words": reference_words,
        "exact_matches": exact_matches,
    }


def evaluate_dataframe(dataframe, reference_col: str = "reference",
                       prediction_col: str = "prediction") -> dict:
    """Score two DataFrame columns without modifying the original data.

    Every reference and prediction must be a string; missing values are not
    silently dropped. Column names can be changed to match the DataFrame.
    """
    for column in (reference_col, prediction_col):
        count = sum(name == column for name in dataframe.columns)
        if count != 1:
            raise ValueError(f"Expected exactly one column named {column!r}; found {count}")
        if not all(isinstance(value, str) for value in dataframe[column]):
            raise ValueError(f"Column {column!r} must contain only strings (no missing values)")
    return calculate_metrics(dataframe[reference_col], dataframe[prediction_col])


def read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no examples")
    if any(None in row or any(row.get(column) is None for column in required) for row in rows):
        raise ValueError(f"{path}: malformed CSV row")
    ids = [row["example_id"] for row in rows]
    if any(not example_id.strip() for example_id in ids):
        raise ValueError(f"{path}: example IDs must not be empty")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: duplicate example IDs")
    return rows


def evaluate_csv(path: Path, expected_ids: Path | None = None) -> dict:
    """Evaluate one model/run and optionally verify IDs and order against a split."""
    rows = read_csv(path, REQUIRED_COLUMNS)
    for field in ("model_id", "run_id"):
        if any(not row[field].strip() for row in rows) or len({row[field] for row in rows}) != 1:
            raise ValueError(f"{field} must identify one non-empty value across the file")
    if expected_ids is not None:
        expected = [row["example_id"] for row in read_csv(expected_ids, {"example_id"})]
        actual = [row["example_id"] for row in rows]
        missing, extra = set(expected) - set(actual), set(actual) - set(expected)
        if missing or extra:
            raise ValueError(f"ID coverage mismatch: {len(missing)} missing, {len(extra)} unexpected")
        if actual != expected:
            raise ValueError("Prediction IDs are not in the expected split order")
    return {"model_id": rows[0]["model_id"], "run_id": rows[0]["run_id"],
            **calculate_metrics((row["reference"] for row in rows),
                                (row["prediction"] for row in rows))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="CSV containing saved predictions")
    parser.add_argument("--expected-ids", type=Path, help="Split CSV used to verify IDs and order")
    args = parser.parse_args()
    try:
        result = evaluate_csv(args.predictions, args.expected_ids)
    except (OSError, ValueError, csv.Error) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
