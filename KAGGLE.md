# Kaggle workspace

Private editor: https://www.kaggle.com/code/watchman777/name-that-llm-baseline-01/edit

| Version | Recipe | Validation result | Saved run |
|---|---|---:|---|
| [1](https://www.kaggle.com/code/watchman777/name-that-llm-baseline-01?scriptVersionId=359150821) | Baseline 01: word + character TF-IDF | 30.19% | Successful, 118.0 seconds |
| [2](https://www.kaggle.com/code/watchman777/name-that-llm-baseline-01?scriptVersionId=359158002) | Stage 03: character TF-IDF + 111 surface-style features | 35.41% | Successful, 105.7 seconds |

Both are private Save & Run All versions with the competition input attached, CPU execution and internet disabled. The scores in this table are prompt-grouped development validation.

## Official submission

**S01**, using the frozen version 2 and its `submission.parquet`, succeeded in Kaggle's hidden-test rerun. Its public balanced accuracy is **0.354 (35.4%)**, exactly as displayed. Public rank was **651 of 914 ranked teams** at the recorded check; the tenth-place public score was 0.549. These are time-specific public observations, not final private results. See [the submission report](results/submissions/S01_REPORT.md), [machine-readable record](results/submissions/S01.json) and [Kaggle submissions](https://www.kaggle.com/competitions/name-that-llm/submissions). Only this one candidate was submitted; final selection was left to Kaggle's existing automatic behaviour.

The current editor contains the same code as `scripts/stage03_candidate.py`; the code was compared through the editor before saving version 2. The original recipe remains in `scripts/baseline.py` and version 1.

GitHub is the source and experiment record; Kaggle is the execution/submission environment. They are saved separately; automatic two-way synchronization is not configured. Future changes should be committed here and copied to a new private Kaggle version.
