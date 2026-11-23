"""Clean Darija pairs and create reproducible splits grouped by Latin input."""

import argparse
import json
import unicodedata
from pathlib import Path

import pandas as pd

COLUMNS = ["Darija_latin", "Darija_arabic"]
PROJECT_DIR = Path(__file__).resolve().parent


def normalize_text(text: str) -> str:
    """Normalize Unicode and collapse whitespace."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Remove missing, empty, and duplicate pairs after normalization."""
    missing = set(COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    clean = data.dropna(subset=COLUMNS).copy()
    for column in COLUMNS:
        clean[column] = clean[column].apply(normalize_text)
    clean = clean.replace({column: {"": pd.NA} for column in COLUMNS})
    return clean.dropna(subset=COLUMNS).drop_duplicates(subset=COLUMNS)


def split_data(data: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Assign identical Latin inputs to the same approximately 80/10/10 split."""
    groups = data["Darija_latin"].drop_duplicates().sample(frac=1, random_state=seed).tolist()
    n_train, n_val = int(len(groups) * 0.8), int(len(groups) * 0.1)
    split_groups = {
        "train": set(groups[:n_train]),
        "val": set(groups[n_train:n_train + n_val]),
        "test": set(groups[n_train + n_val:]),
    }
    examples = data.reset_index(drop=True).copy()
    examples.insert(0, "example_id", examples.index)
    return {name: examples[examples["Darija_latin"].isin(inputs)].copy()
            for name, inputs in split_groups.items()}


def prepare_data(input_path: str | Path, output_dir: str | Path, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Read a CSV, save cleaned data and split files, and return the splits."""
    data = clean_data(pd.read_csv(input_path))
    splits = split_data(data, seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "doda_train_clean.csv", index=False)
    manifest = {"seed": seed, "group_by": "Darija_latin", "splits": {}}
    for name, part in splits.items():
        part.to_csv(output_dir / f"{name}.csv", index=False)
        manifest["splits"][name] = part["example_id"].tolist()
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_DIR / "data/raw/doda_train.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    splits = prepare_data(args.input, args.output_dir, args.seed)
    print({name: len(part) for name, part in splits.items()})


if __name__ == "__main__":
    main()
