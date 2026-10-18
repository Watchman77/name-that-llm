# Stage 07C — fixed ensemble diagnostic rejected

The matched-mass external-data classifier did **not** improve the existing seven-member development ensemble at the single prospectively fixed 5% hard-vote weight. Retain the unchanged base. This is a completed negative comparison, not an execution failure. It is not a test of full S02, and it supplies no new official score.

## Result

All values below are development balanced accuracy, in percent. Contexts repeat the same 2,699 passages; their mean is not an evaluation on three independent datasets.

| Context size | Unchanged base | Base + original-only vote | Base + augmented vote |
| --- | ---: | ---: | ---: |
| 1 | 52.638081 | 52.377827 | 52.451653 |
| 3 | 55.306736 | 55.046980 | 55.046980 |
| 6 | 56.491921 | 56.009197 | 56.083271 |
| Mean | **54.812246** | **54.478001** | **54.527301** |

- Augmented addition versus original-only addition: **+0.049300 percentage points**, conditional 95% paired-prompt interval **[-0.185268, +0.320738]** points.
- Augmented addition versus untouched base: **-0.284945 percentage points**, interval **[-0.790123, +0.208383]** points.
- Both required positive-lower-bound conditions fail; the positive-mean condition also fails against base. The -0.5-point context floor passes, which does not rescue the failed primary screen.
- Versus base, corrected/broken counts are 50/55, 28/35 and 30/41 in contexts 1/3/6. These are repeated decisions, not distinct additional passages.

Stage 07B's standalone gain of +1.185434 points therefore did not demonstrate useful marginal ensemble value in this diagnostic. The observed small advantage over the equally weighted original-only vote is uncertain and neither vote improves on leaving the base alone. Do not add the standalone gain to official 54.9%, and do not infer that every possible external-data approach or calibrated blend must fail.

## Execution and verification

- Source and prospective protocol frozen at `92ba57ed4f4d617b178e71533db14d9073ddb94a` before this cache was tested. Consumer SHA-256 `5daac0ce119ddfcb777f9ebe717b3b98a120cda9a612d7e676fac6a0ffb459c7`.
- User restored cloud sign-in to `watchman777`. Its private [Stage 06E V2 output](https://www.kaggle.com/code/watchman777/name-that-llm-stage-06e-canonical-cpu-evidence/output?scriptVersionId=359514801) was accessible, marked private and Version 2 of 2. The previous account-access blocker is resolved.
- The browser delivered the binary with a `.zip` extension. Its 12,065,373 bytes exactly match the pinned NPZ SHA-256 `9348eeed149ad48bdd828d8055e945f57e384d8917c654125c2ebbe8d4b99519`; no transformation or refit was required. The browser download reporting call timed out, but the synchronized complete file was verified before execution; no duplicate download was made.
- Frozen original train and Stage 07B OOF hashes passed. All row, label, class-order, prompt, canonical-fold, context, cached-argmax, per-member prompt-fold and base-score checks passed. Runtime completed successfully under Python 3.12.14 / NumPy 2.3.5.
- The consumer performed no fitting, calibration, neural inference, regrouping or balancing. It used raw original-control probabilities only. It ran 10,000 joint prompt bootstrap draws with seed 20261018, preserving repeated contexts together.
- Separate code review found no substantive implementation blocker. Independent numeric validation passed all 21 checks, separately reconstructing vote mixtures, scores, paired/per-class counts and both joint prompt-bootstrap intervals without importing consumer helpers. Its evidence and limits are in `VALIDATION.json`.

`SOURCE_FREEZE.json` is the pre-run receipt: its `actual_cache_test_started: false` field describes the time of freezing, not current status. Current terminal state is in `STATUS.json` and `result.json`.

## Decision and limits

**Retain base; retire this exact 5% hard-vote candidate; no automatic weight variants or official submission.** The last verified official S02 result remains 0.549, which is not freshly rechecked here.

The bootstrap is conditional on fitted predictions and reused development prompts. It does not account for adaptive experiment selection, pre-existing source-weight selection or unresolved semantic overlap with external topics. Per-member fold consistency depends on existing cache provenance for historical training exclusion. The proxy contains seven branches, while full S02 contains sixteen active branches. Native SVM margins were not interpreted as calibrated probabilities.

The next prerequisite is a more faithful development reference and a clearly differentiated representation hypothesis. The existing coverage audit identifies our verified GLM cache as a prepared missing-proxy branch. GLM already contributes to full S02: reconstructing its prompt-held-out centered contribution would close a reference gap, not constitute a competitive improvement. Freeze that evaluation definition and verify common context/alignment before any future marginal test. This run did not launch that reconstruction or a new candidate.

Only code, receipts and aggregate results are stored in the public repository. Original text, row predictions, binary caches and fitted models are not published. No paid service, laptop computation, competition submission or final-selection change occurred.
