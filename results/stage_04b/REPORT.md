# Stage 04B — completed supervised DeBERTa trial

## Decision

Reject this fixed three-epoch DeBERTa-v3-base recipe as a replacement for the Stage 03 development reference. All five prompt-grouped folds completed; pooled development balanced accuracy is **22.7639%**, versus **35.4131%** for character plus style. The difference is **-12.6493 percentage points** (conditional paired prompt-bootstrap 95% interval **-14.6927 to -10.7037 points**). Every fold worsened. This was not an official Kaggle submission and does not measure the public ensemble's performance.

## Execution and checks

Private Kaggle V1 `359319680` succeeded in 3022.4 seconds (about 50m 22s); runner time was 3002.06 seconds. The declared model/revision, original training-data hash and reference OOF hash match the protocol. All five folds used the fixed final epoch 3, 405 optimiser updates each, no validation-driven epoch or parameter selection, and zero AMP overflow steps. The runner-recorded warning list is empty; console model-initialisation notices are retained in the log. GPU was Tesla T4 with Transformers 4.56.2 and one visible device. Full environment, fold checkpoint hashes and metrics were captured from the rendered Download logs data link.

This is a negative result for one particular fine-tuning recipe, not proof that all DeBERTa models or supervised encoders are ineffective. No post-result tuning, retry or official submission was performed.

## Evidence and limits

- `metrics.json`: exact JSON between TRIAL_COMPLETE_JSON markers in the completed run log.
- `KAGGLE_COMPLETE.log`: complete console log exported from the page's visible Download logs link.
- `completed_fold_comparison.csv`: all five final-epoch scores and matching frozen reference scores.
- The immutable Kaggle Output tab retains 45 files, including OOF predictions/logits, class recall, length diagnostics, tokenization metadata and checkpoints. Those binary/full prediction outputs have not been downloaded and independently recomputed locally in this status check. The recorded score is the completed runner's score, not an independently rerun local fit.
- Bootstrap intervals are conditional on fitted development predictions and exclude refitting/selection uncertainty and multiplicity adjustment. All prompts remain reused development data.

Run: https://www.kaggle.com/code/watchman777/name-that-llm-stage-04b-deberta?scriptVersionId=359319680

The separate S02 public ensemble reproduction succeeded at **0.549 official public balanced accuracy**. Its score must not be confused with this DeBERTa development result.
