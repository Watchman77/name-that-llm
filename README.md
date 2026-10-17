# Name That LLM

Private competition workspace for reproducible LLM authorship attribution.

The competitive goal is **first place in the final private evaluation**. The public leading score was 58.4% at the 17 October 2026 UK-time check; this is a benchmark, not a fixed winning threshold.

## Current stage

The best development reference remains the Stage 03 **character + 111 style features** candidate at **35.41% balanced accuracy**, versus 30.86% for character-only and 30.19% for the original combined baseline. It improved all five saved prompt-grouped folds. See [the Stage 03 report](results/stage_03/REPORT.md) for uncertainty and regressions.

The fixed Stage 04 frozen MPNet experiment is complete: **14.59%** alone and **34.12%** added to character/style. Both additions are rejected. All ten new fold fits completed with zero fit warnings and zero truncated passages. See [the complete Stage 04 report](results/stage_04/REPORT.md) and [executed evaluation companion](notebooks/Stage_04_Evaluation.ipynb). The [next supervised DeBERTa protocol](plans/STAGE_04B_SUPERVISED_ENCODER.md) is implemented and launched as private offline T4 version 1; GPU preflight and results are pending. These remain development comparisons.

The frozen candidate passed [private offline Kaggle verification in version 2](https://www.kaggle.com/code/watchman777/name-that-llm-baseline-01?scriptVersionId=359158002), reproducing 35.41%. Its first official submission, **S01**, also succeeded and scored **0.354 (35.4%) public balanced accuracy**. Public rank was 651 of 914 ranked teams at the recorded check; the final private score remains unknown. See [the submission record](results/submissions/S01_REPORT.md) and [stage tracker](STAGES.md). Baseline version 1 is preserved.

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
