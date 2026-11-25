# Testing and validation strategy

Use this protocol to compare Arabizi-to-Arabic transliteration methods on the same data.

## Data and evaluation

1. Use the fixed train, validation, and test splits for every method. Build rules, vocabularies, and other learned statistics from the training split only.
2. Use validation data to select settings and, for neural models, the checkpoint with the lowest decoded character error rate (CER). Record the decoding method and length limit.
3. Evaluate the selected method on the test split for final reporting. Do not use test results for tuning.
4. Save one prediction per test example in split order, with `example_id`, `reference`, `prediction`, `model_id`, and `run_id`. Keep the original text in the file.
5. Score all methods with the same script, normalization, and references. Report the number of examples alongside the scores.

The scoring script applies Unicode NFC and collapses whitespace to single spaces. It retains punctuation and diacritics. Alternative valid spellings count as errors unless they match the reference after normalization; identify these separately during error review.

## Metrics

Edit distance counts the minimum insertions, deletions, and substitutions needed to turn a prediction into its reference. The error rates below use totals across the dataset, rather than averaging per-example rates.

| Metric | Calculation | Interpretation |
|---|---|---|
| **CER** (primary) | Character edit distance ÷ reference characters | Measures character-level errors; lower is better. Spaces count as characters. |
| **WER** | Word edit distance ÷ reference words | Measures word-level errors; lower is better. Words are split on whitespace. |
| **Exact match** | Exactly matching examples ÷ total examples | Measures complete matches after normalization; higher is better. |

CER and WER may exceed 1 when predictions contain many insertions. They are undefined if the corresponding total reference length is zero.

From the project root, evaluate a saved prediction file and verify its IDs and order against the test split:

```bash
python scripts/metrics.py predictions.csv --expected-ids data/processed/test.csv
```

## Spelling robustness

Test whether a method handles alternate Arabizi spellings of the same expression:

- Select a fixed sample of about 100 test sentences and create one variant per sentence, such as `wach` → `wash` or `khoya` → `5oya`.
- Check that each variant preserves the meaning and Arabic reference. Use the same validated pairs for every method; do not use them for training or tuning.
- Report CER, WER, and exact match for the original and variant inputs. Also report **ΔCER = variant CER − original CER**. A smaller increase suggests greater robustness, but compare absolute CER too.

`scripts/spelling_robustness.py` provides a starter sample of ten paired test examples. Review the variants with a Darija speaker before interpreting results, and expand the sample for a more useful comparison. From the project root, pass a function that transliterates one string:

```python
from scripts.spelling_robustness import build_robustness_dataframe, evaluate_robustness

sample = build_robustness_dataframe()
predictions, scores = evaluate_robustness(sample, transliterate)
```

Here `transliterate` is the method being evaluated. The returned DataFrame contains both predictions; `scores` contains the metrics for each input and ΔCER.

## Error review and reproducibility

Review a documented sample of predictions and group errors by type, distinguishing actual errors from acceptable spelling variants. If reporting efficiency, include model size or latency along with hardware, batch size, and inference settings.

For each run, record the model and run IDs, configuration, random seed where applicable, and decoding settings. Keep the predictions and split IDs so scores can be reproduced and coverage checked.
