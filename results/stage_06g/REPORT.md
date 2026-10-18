# Stage 06G: completed comparison, fixed correction rejected

The one predeclared attention-pair correction **failed its development improvement screen**. Retain the unchanged control and retire this exact rule. Best confirmed official S02 remains **54.9% public balanced accuracy**; no new official submission or final-selection change occurred.

Private Kaggle [V2 359531626](https://www.kaggle.com/code/watchman777/name-that-llm-stage-06g-pair-complement?scriptVersionId=359531626) completed successfully in **22.5 seconds on CPU**, internet off, with 355.49 kB output. This is a valid negative development result, not an execution failure. The implementation performed no fitting, model inference, regrouping, balancing or parameter search.

## Fixed result

| Context | Control BA | Candidate BA | Change, pp | Corrected | Broken | Changed, still wrong |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 52.6381% | 52.4899% | -0.1481 | 26 | 30 | 7 |
| 3 | 55.3067% | 54.9734% | -0.3333 | 27 | 36 | 6 |
| 6 | 56.4919% | 56.5290% | +0.0370 | 29 | 28 | 5 |
| Mean / repeated totals | **54.8122%** | **54.6641%** | **-0.1481** | **82** | **94** | **18** |

The paired prompt bootstrap gives a conditional 95% interval of **-0.6420 to +0.3580 percentage points**. Both the positive mean and positive lower-bound requirements fail; the fixed per-context floor passes. The interval includes zero and does not establish population-wide harm. The small context-6 increase cannot select a candidate that failed the frozen primary screen.

The same 2,699 passages recur across contexts. Counts above are repeated-context observations, not unique passages. Target supports are 150 each, and all sixteen other class recalls are unchanged, so the net loss of twelve correct context predictions reconciles to mean balanced-accuracy change `-12/8100`, rather than row accuracy divided by 2,699.

Mean Gemini 3.7 recall rose **6.8889 points**, while Gemini 3.8 recall fell **9.5556 points**. The rule traded a larger loss in Gemini 3.8 for a smaller gain in Gemini 3.7. All outside-scope predictions and pair-entry masks remained unchanged; eighteen nonpair decisions switched between two wrong Gemini destinations. There were no exact-zero margins. See [detailed findings](FINDINGS.md).

## Source, execution and validation

The unchanged [prospective plan](../../plans/STAGE_06G_PAIR_COMPLEMENT.md) defines one equal-sum attention rule inside the source raw-control exact-top-two pair scope, with native arithmetic and ties retaining control. Fourteen independent synthetic fixtures and source/package review passed before execution. Saved private Quick V1 359531282 matched all five reviewed source strings, exact input versions 170484/359514801, CPU and offline metadata, null execution counts and empty outputs. Git source freeze `cbdd06c11bb62d72c04781bf7c75bceae08a59fb` preceded the single actual V2 execution, whose saved source and settings also matched.

Runtime strict input hashes, source prediction replay, numerical invariants and numeric/Unicode NPZ exact-readback checks passed. Independent terminal validation matched all **fourteen recovered artifact byte hashes** and recalculated scores from six full confusion matrices. It independently reproduced the conditional interval from 450 prompt-count rows without importing the production bootstrap. Fold, class, scope and false-entry tables reconcile. See [validation](INDEPENDENT_TERMINAL_VALIDATION.json), [run receipt](KAGGLE_RUN.json), [result](pair_complement_result.json) and [full log](KAGGLE_V2.log).

The saved notebook's summary table and two-panel chart were visually inspected; extra table columns use horizontal scrolling. Original PNG, numeric NPZ and status JSON remain on [immutable Kaggle output](https://www.kaggle.com/code/watchman777/name-that-llm-stage-06g-pair-complement/output?scriptVersionId=359531626). Independent local review did not reread row IDs, native pair margins or candidate arrays from that NPZ. The pair and recipe were selected using reused development evidence; this is an incomplete available-component proxy, with no untouched-validation or full-S02 improvement claim.

## Next decision

No alternative G weight or threshold was tested. Close this correction experiment and inspect representation coverage before selecting another substantive hypothesis. The [coverage audit](REPRESENTATION_COVERAGE.md) distinguishes representations already used by the full S02 baseline from evidence missing in this development proxy. Reproducing an existing S02 component improves the reference for future experiments; it is not itself a competitive gain. A genuinely new component still needs a separately frozen marginal full-18-class comparison. No follow-on candidate has executed.
