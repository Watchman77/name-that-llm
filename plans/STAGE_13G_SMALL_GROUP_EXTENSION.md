# Stage 13G: extend distinct-label assignment to groups of 10 or 11

Stage 13F established the value of S07's existing assignment on 12–18-row groups in the fixed, held-out development proxy. It left 810 rows unassigned. Its published size histogram already fixes this experiment's opportunity: six 10-row groups and two 11-row groups, 82 rows in eight groups (3.04% of 2,699). A pure group must correspond to a different true prompt, because two such groups cannot fit inside one 18-row prompt. This modest opportunity is not a path to an automatic leaderboard gain.

## Frozen test

Use the original canonical five folds, competition train snapshot 170484, GLM probabilities from 368292220 and the frozen pretrained DeBERTa features from 358745836. Reconstruct the identical text-only Stage 13F groups within each fold, with no grouping threshold or order changes. Reconcile the frozen per-fold cluster histogram, the 1,889 existing eligible rows and the 810 untouched rows. Report the geometry of all untouched sizes, but score only the prospectively selected sizes **10 and 11**.

Before touching the probability cache, require all eight selected groups to be pure by true prompt and to contain no repeated true model label. These labels are diagnostic only after unsupervised grouping, never inputs to the rule at inference. If any group fails, do not fit or score the extension. Do not score a post-hoc subset or try a different cutoff in this run.

If geometry passes, rebuild the exact Stage 13C per-fold balanced 75% GLM / 25% DeBERTa scores and require the original 47.565001% confusion replay. Then reproduce Stage 13F assignment-on exactly: **61.461844%**, 1,659 correct, confusion SHA-256 `6b3a2cd77f8b7d5c24f343dd685e3be63aaa63b3635d59b6ccefc2528d1ae696`. The control assigns distinct labels within 12–18-row groups. The candidate changes only the inclusion of 10–11-row groups; all original predictions outside those groups must remain identical. Keep score arrays, fold balancing, text grouping and the typography-guard exclusion identical.

Report the fixed pooled balanced-accuracy difference, paired corrections and regressions, per-fold and class differences, and the conditional interval from 5,000 shared true-prompt bootstrap draws within folds. A positive exploratory screen requires a gain of at least +0.50 percentage points, a strictly positive lower 95% bound, and no class losing more than five points of recall. This compares fixed predictions on previously used prompts; there are only eight groups that can change. Even a pass warrants replication, not a production decision or a score claim.

One private offline CPU run; no paid service, GPU, competition-test access, row-level export, official submission or final-selection change. S07's verified official public balanced accuracy remains **65.1%**.
