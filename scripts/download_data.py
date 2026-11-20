from datasets import load_dataset
from pathlib import Path

# Download dataset
dataset = load_dataset("MBZUAI-Paris/DODa-10K")

# Local directory
output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

# Keep only the two relevant columns
columns = ["Darija_latin", "Darija_arabic"]

for split_name, split_data in dataset.items():
    split_data = split_data.select_columns(columns)

    split_data.to_csv(output_dir / f"doda_{split_name}.csv", index=False)
    split_data.to_parquet(output_dir / f"doda_{split_name}.parquet")

print("Dataset saved locally!")