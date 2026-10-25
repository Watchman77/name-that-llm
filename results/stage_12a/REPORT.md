# Stage 12A — topic correction passes the component screen

The fixed CPU pilot completed normally in **58.99 seconds**. Pooled prompt-disjoint development balanced accuracy increased from **37.8931% to 40.0425%**, a gain of **2.1494 percentage points**. The independent 10,000-draw paired prompt bootstrap gives a conditional 95% interval of **[+1.0380, +3.2593] points**. All four prespecified gates passed; this admits the correction to further ensemble validation.

This is a **component result**, not a gain to our S05 ensemble or a Kaggle score. Our latest verified official public balanced accuracy remains **S05 64.2%**. The same development folds have been used before, and the bootstrap holds clustered predictions fixed, so the interval is exploratory and conditional rather than independent confirmation.

## What ran

One four-view SVC was fitted inside each of the five canonical outer training folds, then shared by both arms. The candidate applied the exact dormant source topic correction once: validation-text-only clustering, groups of 8–18 rows, source uncertainty gate and fixed 0.62 strength. No external rows, GPU, full-fit supervised checkpoints, balancing, parameter search or second candidate were used.

The real fit followed the published pre-execution freeze **4f08c924c05636b6b06eada719f1a1f517816de6**. Local runner, protocol and preflight/test hashes still match the reviewed freeze receipt. This validator checked local evidence without accessing GitHub. The runner hash is `3eec88f4d49e28f554ef72a8347401ff400a8ce4094c64b37570b5f587120be2`.

| Fold | Control development BA | Corrected development BA | Gain, percentage points |
| --- | --- | --- | --- |
| 1 | 35.7407% | 39.4444% | +3.7037 |
| 2 | 35.0000% | 36.8519% | +1.8519 |
| 3 | 38.5185% | 39.8148% | +1.2963 |
| 4 | 40.5556% | 42.9630% | +2.4074 |
| 5 | 39.6552% | 41.1494% | +1.4943 |

## Why the screen passed

| Fixed gate | Verified outcome |
| --- | --- |
| Gain at least 0.5 points | +2.1494 points |
| Paired prompt interval lower bound above zero | +1.0380 points |
| No class loses more than 5 recall points | Largest loss 4.0 points, GPT-5.6-terra |
| No convergence warning | Zero recorded across five fits |

There were **146 corrected and 88 broken responses**, for **58 additional correct responses** (1,023 to 1,081). Eleven classes improved, five declined and two tied. The largest gains were GPT-6-astra **+10.67 points** and GPT-5.5 **+8.67 points**; class-level regressions remain visible in [per-class recalls](per_class_recall.csv).

The recorded inferred groups made **2,121 of 2,699 rows eligible (78.58%)**. Recorded purity and pairwise precision are 1.0 in each fold, while pairwise recall ranges from 0.7100 to 0.8122. These grouping diagnostics describe the development folds; they do not establish comparable hidden-test grouping. The terminal review checked their aggregate consistency without regenerating clusters.

Independent terminal recalculation passed **37/37 checks**, reproducing pooled and fold scores, all class recalls, both confusion matrices, corrected/broken counts, bootstrap and screen decisions. Score-array IDs, label axes, folds, finite values and argmax predictions also match. No fitting was rerun. Raw IDs, row predictions and score arrays remain outside the repository; public outputs are aggregate.

## Next decision

Propose one prospectively frozen, matched complementarity test against a trustworthy cross-fitted reference: compare the same reference plus the raw SVC with the same reference plus the corrected SVC at identical fixed weights. Verify reference fold exclusion first; an incomplete reference must be labeled a proxy. No such test, official submission or variant has been implemented or launched by this review. A standalone gain can be redundant with centering already present in S05.

[Protocol](PROTOCOL.md) · [result](RESULT.json) · [independent terminal validation](TERMINAL_VALIDATION.json) · [execution](EXECUTION.json) · [source freeze](SOURCE_FREEZE.json).
