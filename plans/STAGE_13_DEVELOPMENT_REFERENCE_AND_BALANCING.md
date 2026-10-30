# Stage 13 — clean development reference, then one balancing hypothesis

Date: 30 October 2026. Status: **planned; no new fit, candidate evaluation, GPU run or official submission launched**.

## Starting point and decision

Keep exact S07 / private Stage12 V10 **363626171** frozen as our current official public reference: **0.651 balanced accuracy**. The verified public improvement over S05 is **+0.900 percentage points**. The 30 October snapshot places us at 713/3967, with tenth at 0.692 and the leader at 0.736. A 70% public target is a milestone, not the current winning threshold. Final private performance remains unknown.

The next useful work is a clean development reference and one falsifiable hypothesis. We do not yet have evidence for a new component that will improve the full S07 ensemble, and no gain is forecast here.

## What the source and completed experiments establish

S07 already contains style/n-gram evidence, adaptive centering, LoRA ensembles, DeBERTa, complementary models and ATLAS. Generic advice to add these mechanisms does not identify a new contribution.

There is **no honest full-S07 out-of-fold prediction set**. The source averages fold and full-data adapters, uses full-data-adapted Gaussian features, and performs seen-training sanity checks. Retraining only downstream heads in `--sim` does not undo upstream training exposure. Fold filenames alone do not establish held-out prompts. The hidden public score supplies no row-level ground truth or confusion matrix.

Completed negative recipes remain retired: the Stage06E centering modification, Stage06G attention-pair rule, Stage07C fixed external-data addition and exact S06 topic correction. Standalone gains were not reliable estimates of ensemble gains. This does not reject all future work in those modelling families.

Evidence: [Stage12 source review](../results/stage_12/SOURCE_REVIEW.md), [Stage06E](../results/stage_06e/REPORT.md), [Stage06G](../results/stage_06g/REPORT.md), [Stage07C](../results/stage_07c/REPORT.md), [Stage12A](../results/stage_12a/REPORT.md), [official S07 result](../results/submissions/S07_FINAL_RESULT.json).

## Stage 13A — first job: provenance and clean-reference preparation

Prepare one CPU admission job before any new model fitting. It must produce a branch ledger recording checkpoint/source versions, official/external training exposure, prompt-fold exclusions, feature/readout/calibration fits, available caches and unresolved provenance. Categorize each branch as verified held-out, full-fit or unknown. Do not upgrade full-fit/unknown components by assigning folds after training.

Use a **reduced, explicitly labelled development proxy** if full-S07 exclusion cannot be established. Start with verified frozen-base feature caches and fold-local text features/readouts. A frozen-base DeBERTa cache must be the pretrained encoder, not S07's fine-tuned DeBERTa checkpoint. Every model and learned transform contributing to a scoring batch must exclude **all prompts in that batch**. Exclude supervised adapted/full-fit components without sufficient common-fold evidence.

The best-prepared feature-cache lead is the existing GLM extraction from own Stage06C V1 **359405443**:
- `glm_features.npz`, SHA256 `2b41585a9482cef825962a8cc096697933c0b1237e8ede0712d1f8ab361d9414`.
- Shape `(2699,3,4096)`, ordered layers `[8,3,24]`; L8+L3 form the classifier representation and L24 supports grouping.
- Existing records establish aligned IDs, finite features and exact bytes, not a fitted held-out GLM classifier.

Reconstructing this reference closes a coverage gap; GLM already contributes to S07, so it is **not a new competitive gain**. Reverify actual cached bytes and source provenance before use. Keep all eighteen classes, canonical whole-prompt folds and matched context scenarios 1/3/6. Previously used folds remain exploratory development data.

The old Stage06E cache has useful raw predictions, but its balanced diagnostic can suffer mixed-source-fold peer supervision. Do **not** apply coupled balancing to pooled mixed-fold OOF and call it clean validation. Refit from eligible frozen-base features under common outer folds or leave the batch-coupled test blocked.

Before fitting, freeze exact implementations, input versions/hashes, folds, class axes, estimator settings, context construction and resource budget. Archive aligned held-out probability/log-evidence caches privately for cheap later comparisons. Public outputs contain code and aggregate evidence.

## Stage 13B — provisional first hypothesis: balancing robustness

The exact S07 source passes `--sinkhorn 1.0`. In code cell 3, for test sets of at least eighteen rows, it iterates column-mass adjustment toward `N/18` and row normalization one hundred times, then uses the balanced log evidence. The three-row preview does **not** exercise that branch.

Hypothesis: full uniform class-marginal balancing may be sensitive to class-mixture changes. This is a stress-test hypothesis, **not a claim that the hidden test has a particular class distribution or that disabling balancing improves S07**.

Only after Stage13A admits a common-fold reference:
1. Freeze one control using the source balancing arithmetic and one fixed no-balancing arm. No alpha, threshold or blend sweep. The reduced screen isolates balancing: both arms consume the same frozen pre-balancing probability/temperature values; neither adds typography guards or Hungarian assignment. Any later full-S07 confirmation must hold those downstream source operations identical. Uniform probability column mass is not a guarantee of equal final argmax label counts.
2. Freeze balanced and unequal-class stress designs, seeds, batch sizes, scoring weights and class supports before viewing candidate outcomes. Keep every class represented and score both arms on identical rows. Explain which changes affect grouping/context as well as class totals.
3. Apply both arms within batches whose complete model/head training excluded every scoring prompt. Do not combine incompatible supervised folds.
4. Before scoring, verify synthetic 17-row/18-row branch-boundary fixtures, finite row/column arithmetic and unchanged upstream probabilities. These fixtures are technical checks, not accuracy samples. Report full eighteen-class balanced accuracy, per-class recall, corrected/broken counts and paired prompt uncertainty. Preserve neutral/negative results. Do not select the best stress scenario after seeing results.
5. Separate exploratory proxy evidence from full-ensemble evidence. A proxy gain does not add to 65.1% or identify hidden-test errors.

This document selects a hypothesis family and order of work. **It is not an execution freeze for Stage13B**: exact stress masks, clean-reference membership, endpoints, numerical fixtures and admission thresholds must be reviewed and committed before evaluation.

Source: [frozen production notebook](https://github.com/Watchman77/name-that-llm/blob/55cf8622c592690c98f67b65c3e9b6381a726b32/notebooks/Stage_12_Public_651_Reproduction.ipynb). The source and existing repository evidence were read for this plan; no new live competition-rule refresh is asserted.

## Stage 13C — admission to another competitive run

A screen must supply positive marginal evidence, disclose class harms and pass independent numerical/provenance checks before proposing a full-pipeline candidate. A successful reduced proxy is still only a reason for further testing.

If admitted, freeze one S07 counterfactual with exactly the chosen intervention, unchanged other assets and a guard that proves the relevant branch was exercised. Use a test size that reaches any size-gated code path; the ordinary three-row preview cannot establish that. Recheck actual free quota, current competition rules and runtime before GPU launch. Hidden runtime remains unknown.

Submit at most one prospectively frozen candidate after its technical gates pass. Compare only a real displayed official score to S07 using exact decimal percentage points. Preserve final-selection settings unless separately requested.

## Deferred choices and stop rules

- Re-enabling omitted Qwen3.5 token attention is not the first GPU spend. Source comments estimate 1.7–2.7 hours fixed cost; marginal benefit here is unmeasured.
- S05 has a distinct full-text Qwen branch, potentially relevant to S07's 384-token main-LoRA cap, but its released final adapter fitted all 2699 official rows. Training-set blending is not honest validation. Assess genuine outer-checkpoint or refitting feasibility before using it as evidence.
- Learned stacking requires nested/cross-fitted meta-training. Shared components make arbitrary S05/S07 averaging a weakly justified experiment.
- If common-fold exclusion cannot be established within a practical free budget, report the blocker and retain S07. Do not relabel in-sample predictions as validation or invent an expected route to 70%.
