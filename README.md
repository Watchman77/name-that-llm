# Name That LLM

Private competition workspace for reproducible LLM authorship attribution.

## Current stage

Stage 02 is complete. On identical prompt-grouped folds: **character-only 30.86%**, **combined 30.19%**, **word-only 26.26%** balanced accuracy. The character-only lead is small and uncertain; both character-only and combined remain references. See [the full Stage 02 report](results/stage_02/REPORT.md) and [stage tracker](STAGES.md).

The private [Kaggle baseline](https://www.kaggle.com/code/watchman777/name-that-llm-baseline-01?scriptVersionId=359150821) is still version 1. Stage 02 results are local development comparisons, not a competition submission.

## First baseline

| Measure | Result |
|---|---:|
| Pooled balanced accuracy | 30.1877% |
| Uniform-random expected score | 5.5556% |
| Fold range | 28.3333–32.0370% |
| Convergence warnings | 0 |

These are local five-fold results, not a Kaggle leaderboard score. Every fold holds out 30 complete prompts. The fixed model combines word 1–2-gram and character 3–5-gram TF-IDF with a class-weighted linear SVM. Vocabularies and IDF are learned separately in each fold. No parameter search has been performed.

## Project layout

- `notebooks/Name_That_LLM_Baseline.ipynb`: portable notebook, with outputs cleared for version control.
- `scripts/baseline.py`: identical code cells as a runnable script.
- `results/baseline_01/`: measured metrics, source hashes, package versions, fold assignments and out-of-fold predictions. No passage text is included.
- `EXPERIMENTS.md`: experiment log and next comparisons.

## Run locally

Use Python 3.12 and install `requirements.txt`. Download the competition data after accepting its rules, extract it into `data/` in the project root, then run:

```bash
python scripts/baseline.py
```

Or set `NAME_THAT_LLM_DATA_DIR` to the extracted data directory. To run the notebook locally, start it from the project root or set that environment variable. Generated diagnostics go to `baseline_outputs/`; the output file is `submission.parquet` in the working directory.

## Run on Kaggle

Attach the **Name That LLM** competition as an input, import the notebook or script, select CPU, and turn internet off. Save a private version. The downloaded test file has three preview rows; the full hidden test is supplied only during competition scoring. All input paths are detected without network requests. Never upload the three-row preview output as if it were the hidden-test predictions.

## Data and collaboration

Competition source: https://www.kaggle.com/competitions/name-that-llm

The original training and test data are excluded from git. Keep this repository private during the competition and limit access to the owner and official teammates. Check the competition rules before any public release, private code sharing outside the team, paid synthetic-data generation or data redistribution. Winner licensing obligations will be addressed if applicable; no open-source licence is assigned to this private working repository yet.
