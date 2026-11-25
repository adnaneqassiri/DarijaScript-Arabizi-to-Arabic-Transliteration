"""Download the DODa-10K dataset into the project's raw data directory."""

from pathlib import Path

from datasets import load_dataset


def main() -> None:
    dataset = load_dataset("MBZUAI-Paris/DODa-10K")
    output_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["Darija_latin", "Darija_arabic"]

    for split_name, split_data in dataset.items():
        split_data = split_data.select_columns(columns)
        split_data.to_csv(output_dir / f"doda_{split_name}.csv", index=False)
        split_data.to_parquet(output_dir / f"doda_{split_name}.parquet")

    print(f"Dataset saved to {output_dir}")


if __name__ == "__main__":
    main()
