# Testing and Validation

This protocol keeps transliteration results comparable and reproducible across project methods.

## Data and splits

- Use one fixed train/validation/test split for every method. Record the split seed and example IDs; never change the split to improve a result.
- Use training data to fit models and derive any mapping rules or exceptions. Use validation data to choose settings and checkpoints. Use test data once for final evaluation; do not tune against test results.
- Apply the same documented text normalization to predictions and references. Keep raw text unchanged in saved predictions. At minimum, use Unicode NFC and trim or collapse whitespace. Do not remove letters, diacritics, or punctuation unless the team documents that choice.
- Build vocabularies and training-derived statistics from the training split only.

## Model validation

- For neural models, select the checkpoint with the lowest validation character error rate (CER), measured on decoded predictions. Record the decoding method and any length limit.
- Keep the inference and scoring procedure consistent across methods. A deterministic mapping baseline does not require training metrics.
- Save one prediction per test example, in the original order, with a stable example ID, reference, prediction, model ID, and run ID. Check that no test IDs are missing or duplicated.

## Metrics

Report the following for every method on the same test examples:

- **CER (primary):** total character edit distance divided by the total number of reference characters.
- **Word error rate (WER):** total word edit distance divided by the total number of reference words, using whitespace tokenization after shared normalization.
- **Exact match:** fraction of examples whose normalized prediction exactly matches the normalized reference.

Report example counts with all scores. Lower CER and WER are better; higher exact match is better. State how alternative valid spellings are handled. If using additional metrics, apply the same implementation and settings to all methods.

## Recommended analyses

- **Overall performance:** compare all methods using the metrics above.
- **Unseen words:** identify test examples containing source words absent from training, and report CER separately for seen and unseen groups with their sizes.
- **Spelling robustness:** evaluate validated Arabizi spelling variants as paired examples. Report scores for original and variant inputs, and document how variants were checked.
- **Error review:** inspect a documented sample of test predictions. Distinguish genuine transliteration errors from acceptable spelling variants; record the categories and examples reviewed.
- **Efficiency (if measured):** report latency and model size, and state the hardware, batch size, and inference settings. Mark training time as not applicable for a non-trained baseline.

Treat small subgroup results cautiously and report group sizes. Do not use test findings to change a model or its settings.

## Reproducibility checklist

- [ ] Same fixed split, references, normalization, and scoring code for every method.
- [ ] No test data used for fitting, model selection, or tuning.
- [ ] Saved predictions cover every test ID exactly once.
- [ ] Model/run ID, configuration, random seed, and decoding method are recorded.
- [ ] Scores include example counts and can be regenerated from saved predictions.
