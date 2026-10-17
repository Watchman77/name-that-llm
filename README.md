# Name That LLM

Private competition workspace for reproducible LLM authorship attribution.

The competitive goal is **first place in the final private evaluation**. The public leading score was 60.0% at the latest 17 October 2026 check (previously 59.3%); this is a benchmark, not a fixed winning threshold.

## Current stage

**Our best confirmed official public score is now 0.549 (54.9%), from S02**, the unchanged Lucas Boesen V3 reproduction. This improves S01's 0.354 by **19.5 percentage points** and matches the source score at displayed precision. At the latest check we rank **181 of 1,212 teams**; tenth is 0.559 and the leader 0.600. The final private result remains unknown. See [S02 outcome](results/submissions/S02_REPORT.md), [source attribution](vendor/NOTICE.md) and [reproduction protocol](plans/STAGE_05_PUBLIC_REPRODUCTION.md). No paid services were used.

Stage 06A verified all 2,699 public LoRA held-out predictions. Stage 06B has now completed a fixed CPU DeBERTa centering comparison and repeated it with identical scenario scores. Mean component development BA improves from **46.57% to 47.25%** across three lexical context simulations (**+0.68 percentage points**, paired prompt 95% interval **+0.09 to +1.30**). Some classes regress: Qwen recall falls 8 points in the three-response scenario. This is a provisional component gain, not a new S02 official score or a full-ensemble result. No paid services or GPU were used. See [results and limits](results/stage_06b/REPORT.md), [component inventory](results/stage_06/COMPONENT_INVENTORY.json), and [next mixed-grouping confirmation](plans/STAGE_06C_MIXED_GROUP_VALIDATION.md). 70% remains a stretch goal.

Stage 06C private V1 [359405443](https://www.kaggle.com/code/watchman777/name-that-llm-stage-06c-mixed-grouping?scriptVersionId=359405443) has been queued with internet off and GPU T4 x2 requested. No actual CUDA preflight, extracted cache or result has yet been observed. It saves fixed GLM layers 8/3/24 and tests the same paired DeBERTa change using S02 mixed grouping. See [run receipt](results/stage_06c/KAGGLE_RUN.json). The next endpoint is [fixed available-component ensemble benefit](plans/STAGE_06D_FIXED_ENSEMBLE_PROXY.md), not a component-to-leaderboard extrapolation.

Stage 06D has begun with its independent CPU prerequisite: exact S02 stylegram OOF, **38.4484% development BA**, all 2,699 rows held out by whole prompt, zero convergence warnings. Full probabilities and independent checks are saved. This is one component; ensemble fusion has not yet run. Stage 06C remains queued with no CUDA preflight or result. See [Stage 06D report](results/stage_06d/REPORT.md) and [fixed ensemble protocol](plans/STAGE_06D_FIXED_ENSEMBLE_PROXY.md).

The original single-passage development reference is the Stage 03 **character + 111 style features** candidate at **35.41% balanced accuracy**, versus 30.86% for character-only and 30.19% for the original combined baseline. It improved all five saved prompt-grouped folds. See [the Stage 03 report](results/stage_03/REPORT.md) for uncertainty and regressions.

The fixed Stage 04 frozen MPNet experiment failed: **14.59%** alone and **34.12%** with character/style. The separate Stage 04B supervised DeBERTa trial has now completed all five folds at **22.76% development balanced accuracy**, below the **35.41%** character/style reference on every fold. Both encoder experiments are rejected as replacements for that development reference. See [MPNet report](results/stage_04/REPORT.md) and [completed DeBERTa report](results/stage_04b/REPORT.md). The public ensemble's 54.9% official score is a separate evaluation, not an independently measured development score.

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
