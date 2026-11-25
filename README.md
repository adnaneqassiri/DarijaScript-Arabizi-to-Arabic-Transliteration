# DarijaScript: Arabizi-to-Arabic transliteration

This project explores methods for transliterating Moroccan Darija written in
Arabizi into Arabic script. The character-level notebook contains a simple
mapping baseline. The shared evaluation protocol is described in
[Testing and Validation](Resources/TESTING_STRATEGY.md).

## Setup

From the project root, create an environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

## Data

`data/raw/` contains the DODa-10K data downloaded from
[MBZUAI-Paris/DODa-10K](https://huggingface.co/datasets/MBZUAI-Paris/DODa-10K).
The CSV is the input to the cleaning script; the Parquet file is an additional
copy of the downloaded data. Consult the source for dataset terms and details.
The tracked files in `data/processed/` are the fixed train, validation and test
splits and the split manifest. Use validation for development and reserve test
for final evaluation. Do not regenerate or change the tracked splits to improve
results.

To download the source again, run `python scripts/download_data.py`. This writes
to `data/raw/` and can overwrite the tracked raw files. To inspect how splits
would be generated without overwriting the fixed ones, run:

```bash
python scripts/data_cleaning.py --output-dir /tmp/darijascript-prepared
```

The script defaults to writing into `data/processed/`, so use the explicit
output directory above when checking the cleaning process. Its seed defaults to
42 and can be set with `--seed`. The cleaning notebook uses the same functions
and does not overwrite the splits.

## Evaluation and tests

Open `notebooks/characterlevel_transliterate.ipynb` to run the baseline on the
validation split. `scripts/metrics.py` provides DataFrame and CSV interfaces for
CER, WER, and exact match; see the testing strategy for examples and scoring
rules. Run the tests from the project root with:

```bash
python -m unittest discover -s tests -v
```
