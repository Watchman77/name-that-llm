# Stage 12A — fixed SVC topic-correction component screen

## Decision and scope

This prospective CPU pilot asks whether the dormant topic-bias correction in Lightning V28 `362014737`, code cell 5, improves an independently cross-fitted four-view SVC. Our official best is S05 **0.642 public balanced accuracy**. The pilot does not reproduce the full S05 ensemble and cannot establish an improvement to it.

The source defines `adaptive_svc_topic_correction` but its production orchestrator imports it without calling it. This is a candidate hypothesis, not a demonstrated bug. Other S05 members already use centering; a component gain may be redundant in the ensemble.

The runner, this protocol and checks must be committed before real fitting. `--run` requires that actual pre-execution Git commit SHA. No real or toy classifier is fitted during preflight or self-test.

## Frozen data and folds

- Official training file: local private `stage07_private/train.csv`, 2,699 rows, 150 prompts, 18 classes; SHA256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`.
- Canonical fold assignment: `results/stage_07b/oof_predictions.csv`; SHA256 `72d7d43119367ef100511d0f8174b9483ec86bf28bfc7b7e61d8fb599ee1209d`. Only `id`, `prompt_id`, `model`, `fold` are read; earlier prediction columns are excluded.
- Preserve all five existing outer folds, numbered 1–5. Every prompt must belong to exactly one fold; all 18 classes must occur in each training and validation set. Validate exact IDs, prompt IDs and labels against the original file.
- Known prompt IDs are used only for splitting and evaluation diagnostics/bootstrap. The classifier and correction receive text; inferred validation groups receive only that fold's held-out texts.
- No external rows, full-fit supervised checkpoint, cached supervised features, official hidden data or new API calls.

These development folds have been reused in earlier work. Even after a prospective freeze this is exploratory evidence, not independent confirmation.

## Matched two-arm method

One vectorizer stack and one LinearSVC are fitted inside each outer training fold and shared by both arms. The four source-style views are:

| View | N-grams | Maximum features | Feature weight |
| --- | --- | --- | --- |
| Character | 3–5 | 250,000 | 1.0 |
| Word, including punctuation tokens | 1–3 | 180,000 | 0.7 |
| Shape character view | 2–6 | 60,000 | 0.5 |
| Function-word/punctuation view | 1–3 | 50,000 | 0.5 |

Fit TF-IDF on the outer training text only; transform validation without fitting its vocabulary/IDF. Fixed `lowercase=False`, `sublinear_tf=True`, float32 and `min_df=2`; shape and function-word transforms match the source mathematical AST.

The original S05 SVC fitting hyperparameters are not established by its cached inference source. This diagnostic fixes **C=1**, balanced class weights, `dual='auto'`, `tol=1e-4`, `max_iter=10000`, random seed 42. It is **source-style**, not an exact S05 SVC reproduction. Convergence warnings are counted and prevent admission; no automatic retry or parameter change.

- **Control:** `clipped_logp(raw_validation_SVC_margins)`; predict its argmax.
- **Candidate:** apply the exact source `adaptive_svc_topic_correction` once to those control scores and the same margins. Infer topics by the source word TF-IDF/cosine average-linkage function, threshold **0.85**, using that fold's validation texts only. Correct only inferred groups of size **8–18**. Subtract group mean margins; use source uncertainty gate and fixed **0.62** strength. Predict corrected argmax.

Do not apply Sinkhorn, an ensemble, temperature calibration, labels in grouping, hyperparameter search, true prompt-group centering or any second candidate. Empty-vocabulary/source clustering errors fail the run; no silent fallback is introduced.

## Evaluation and fixed screen

Primary quantity: pooled outer-fold **development balanced accuracy**, candidate minus control in percentage points. Report each fold, all 18 class recalls, both 18-by-18 confusion matrices, corrected/broken counts and convergence warnings. Report inferred-group size/eligible coverage, majority purity and pairwise precision/recall against true prompt IDs as diagnostics only; singleton-heavy purity must not be interpreted without coverage.

Use **10,000 paired prompt bootstrap draws**, seed **12012**. Resample the 150 known prompts jointly across both arms and all classes; recalculate class denominators and balanced accuracy in each draw. Use percentile 2.5%/97.5% interval. Interpret it conditionally on these reused development folds.

Admit only to further ensemble validation if all fixed gates pass:

1. Component balanced-accuracy gain is at least **0.5 percentage points**.
2. The paired prompt 95% interval lower bound is strictly above zero.
3. No class recall declines by more than **5 percentage points**.
4. No classifier convergence warning occurs.

Otherwise reject this exact correction candidate. No automatic strength, threshold, feature or split variants. Passing does not authorize an official submission or establish improvement to S05.

## Resources, privacy and output

Run on CPU with all BLAS/OpenMP limits set to one. A parent process enforces a **900-second whole-child process-group deadline**, covering preflight, five fits, grouping, bootstrap and output. It terminates then kills the group on timeout. A timeout or error is terminal incomplete execution; completed partial folds are not a result. No GPU, paid service or fitted model serialization.

Public repository output is aggregate only: `PREFLIGHT.json`, `RESULT.json`, `FOLDS.json`, `GROUP_DIAGNOSTICS.json`, per-class recalls, confusion counts, `EXECUTION.json` and status. Original IDs, prompt IDs, row predictions and margins are written only to `stage12_private/stage12a`, outside the repository; raw text is not written by the runner.

## Commands after review

From the workspace, preflight and synthetic mathematical checks can run before the freeze:

```bash
python stage07_work/scripts/stage12_topic_correction_pilot.py --preflight --receipt stage07_work/results/stage_12a/PREFLIGHT_PREPARATION.json
python stage07_work/scripts/stage12_topic_correction_pilot.py --self-test --receipt stage07_work/results/stage_12a/SYNTHETIC_VALIDATION.json
```

Only after the actual runner/protocol/checks commit is published:

```bash
python stage07_work/scripts/stage12_topic_correction_pilot.py --run --freeze-commit ACTUAL_40_CHARACTER_PRE_EXECUTION_COMMIT
```

No real fitting has occurred at protocol preparation. Real execution and results must be recorded separately.
