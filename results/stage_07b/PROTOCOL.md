# Stage 07B: matched-class-mass data test (prospective)

The user authorized the follow-up after Stage 07A. This is a new, single fixed experiment, not an undisclosed revision of 07A. It is designed **after** seeing 07A on these same official prompt folds; even a positive result is not independent confirmation.

## Question

Does adding the same admitted external GPT data help when total weighted loss mass for every class is identical in the original-only and augmented arms? Stage 07A increased the mass of augmented GPT classes by 50%, so it could not isolate new-example value from changed class preference.

## Frozen inputs and fixed change

Reuse the audited 750 first-party OpenAI passages from prvsiyan's Extended Name That LLM Dataset V4 (`new`, clean, direct `openai` route only), 150 complete five-GPT-label prompt blocks. Use the same original 2,699 passages, all 18 classes and five whole-original-prompt folds. Exact sources, admitted IDs, hashes, passage-overlap screening and topic-reconstruction limitations remain those of 07A. Re-run the deterministic admission checks before fitting. No new passages, providers, API calls, prompts, punctuation transformations or other preprocessing are introduced.

For fold-training count N and original class count n_c, control original weights are N/(18*n_c), totaling N/18 for every class.

For each of the five augmented classes, candidate original examples get two-thirds of their control weight. Its external examples jointly receive N/(18*3), divided evenly across the 150 external examples. Thus its original mass is 2N/(18*3) and combined mass N/18, exactly matching control. All other classes retain their original control weights and receive no external examples. The 2:1 official/external ratio is the normalized version of 07A's fixed 1:0.5 ratio, not a searched new ratio. Total objective weight across all classes also matches N.

Audit all 18 class masses in each of the five folds before fitting and inside the run. Fit the same feature map once per fold on official training passages only, reused in both arms: character TF-IDF 3–5 grams (min_df=3, sublinear TF, max_features=120000) plus the existing 111 standardized and normalized surface-style features. Both LinearSVC heads use C=1, dual=True, tol=0.0001, max_iter=10000 and seed=20261018. No parameter tuning or alternative fractions.

The estimator remains an undistributed private classifier under the same bounded first-party use assessment as 07A. Do not serialize trained models, run test inference, submit predictions, alter final selections or use paid services. User explicitly intends the repository to be public for now; save code, public external-data IDs and aggregate findings there. Do not add original row-level labels/predictions, native-margin arrays or fitted models to the public repository. No laptop access is needed.

## Evaluation and decision fixed before fitting

Primary: pooled original held-out-prompt development balanced accuracy, all 18 classes. Report difference, per-class recall, fold scores, corrected/broken original passages, GPT-vs-other-class recall and prediction counts, and fit warnings. Refit the control and verify its predictions exactly match 07A, given identical inputs/seed/environment. This reproduction check is diagnostic, not another validation split.

Use the same 10,000-draw paired original-prompt bootstrap, seed=20261018, recomputing class recalls on each resample. Its interval is conditional on fitted folds, with neither all retraining uncertainty nor validation-adaptation uncertainty included. Semantic prompt independence cannot be certified from guessed original instructions.

The fixed screen requires a positive overall difference, lower 95% interval endpoint above zero and no fit warnings. If it fails, retain the control and do not automatically try more fractions, class weights, thresholds or another data subset. If it passes, label it a provisional matched-mass signal requiring fresh validation and ensemble assessment. Neither outcome establishes an official score or replaces S02. Compare 07A only as an exploratory historical diagnostic; do not select the more favorable recipe.

Freeze this protocol, runner, dependency checksums, preflight and all-fold weight-mass audit in Git before fitting. A source/manifest change must prevent execution. After fitting, use the separate count-based metric/bootstrap validator and independently check class masses, prior control reproduction and saved artifact hashes. No independent agent review is claimed.
