# Stage 13C — fixed DeBERTa complement on the held-out GLM folds

Date: 11 November 2026. Prospective protocol. Freeze this plan, runner, notebook, fixtures and input bindings before scoring the real comparison. S07 remains the best verified official public balanced accuracy at 65.1%.

## Question and scope

Does adding one readout of a frozen pretrained DeBERTa-v3-large representation help the standalone held-out GLM reference when both arms use the same uniform balancing? This is a reduced development proxy on previously used prompts, not a full-S07 ablation, a claim of a novel component, a leaderboard forecast or a blind confirmation set. S07 already uses DeBERTa and other evidence; this experiment does not estimate their marginal effect in that ensemble. Do not revive the SVC topic correction, fit calibration, or change the prior balancing decision.

## Input and provenance admission

Use a new private Kaggle notebook, accelerator `None`, Internet off, and exactly two pinned inputs: `kernelVersion` **368292220** for Stage 13 GLM V3 probabilities and `kernelVersion` **358745836** for the pretrained encoder output. Run once within a 600-second execution limit. Verify saved input bindings and settings before execution. Preserve the source notebooks unchanged.

- GLM cache: unique `heldout_probabilities.npz` from 368292220 and its matching `readout_report.json`; cache SHA256 `565df6f8e652a9850ff11cf4e8b6a81fcccbc7db593186ff7f53358e7190bd16`. Reuse the Stage 13B admission contract: 2,699 unique rows, 150 prompts, exact ordered 18-class axis, canonical prompt-to-fold mapping digest `a55b035686cb7b31f6fbe852ac990668709aec4b913c5d0928d89d843bdbfe62`, finite positive float64 probabilities, and documented held-out fold exclusion.
- DeBERTa cache: unique `enc_debv3l2_train.npz` from [the pinned pretrained extractor](https://www.kaggle.com/code/lucasboesen/name-that-llm-extract-encoders2?scriptVersionId=358745836); SHA256 `11eed58762ad128f97457dde9cc4ab03c7932e1b4b04845ec6664c834b1b1dda`. Require aligned, unique row IDs in exactly the GLM order; a finite `(2699, 9, 1024)` float16 embedding tensor; and exactly one each of layers 24 and 6. Read numeric arrays with `allow_pickle=False`. If IDs have object dtype, inspect only the bounded `ids.npy` member's header and literal `pickletools` opcodes to verify identity and order. Never unpickle or execute opcodes, and fail on ambiguous or mismatched IDs. Do not use S07's supervised DeBERTa checkpoint or previously mixed-fold predictions.

## Frozen fitting and comparison

Require scikit-learn **1.6.1**. Concatenate layers 24 then 6 into a float64 feature matrix (a lossless conversion from the cached float16 values) of shape `(2699, 2048)`. In each of the five canonical folds, fit a fresh `StandardScaler` and `LogisticRegression(C=0.005, solver='lbfgs', tol=1e-4, max_iter=5000, random_state=0)` on its training rows only, with numerical thread pools limited to two threads. Score all rows in that fold with its training-only fitted scaler and classifier. Require the exact class axis, one held-out probability vector per row, no convergence warnings and finite positive normalized probabilities. No classifier, scaler or other learned transform may see any of the 30 prompts in the fold being scored. Folds 1–4 have 540 held-out rows and fold 5 has 539. Do not refit GLM.

- **Control:** original frozen GLM probabilities.
- **Candidate:** fixed arithmetic probability mixture `0.75 * GLM + 0.25 * DeBERTa` before balancing. There is no learned blend, geometric pooling, threshold, prompt centering, fitted calibration or parameter search.

Apply the unchanged Stage 13B uniform source operation (`alpha=1`, 100 column-mass/row-normalization iterations, `log(Q + 1e-12)`, canonical first maximizer) to both arms separately within each complete held-out fold. Never balance a batch pooled across folds. Leave both raw and balanced control decisions available for admission: reproduce the prior raw confusion counts, 1,239 correct and exact balanced accuracy `61547/134100`; reproduce the prior whole-fold balanced confusion counts, 1,244 correct and exact balanced accuracy `18539/40230`. Stop without a candidate decision if any binding, provenance, numerical or readback check fails.

## Metrics and fixed decision

Compare pooled 18-class balanced accuracy for the candidate minus the control, in percentage points. Report per-fold scores, class supports, recalls and differences, aggregate confusion counts, changed/corrected/broken decisions and fit/numerical checks. Report the conditional paired prompt bootstrap interval from 5,000 draws, seed **20261111**. In each draw resample 30 prompts with replacement within each fold, using the same multiplicities for both arms; pool class denominators across folds and recompute the difference. If any draw lacks a class, fail rather than discard it. Do not refit or rebalance during resampling. The interval conditions on the fixed fitted and balanced predictions, so it does not capture refitting or new-batch uncertainty.

The candidate advances only to further development validation if all three criteria pass: (1) pooled gain at least **0.50 percentage points**; (2) lower equal-tailed 95% interval bound strictly above zero; and (3) no class recall loss greater than **5 percentage points**. Fold scores are diagnostics, not an additional sign gate. The raw and balanced DeBERTa-only scores are component diagnostics; the fixed blend is the sole candidate. Keep all results, including neutral or negative ones, and do not add variants or revise the rule after outcomes. A pass does not warrant an automatic submission or any change to S07 or final selections.

Save only aggregate metrics and verification evidence. Do not publish or log row IDs, prompt IDs, feature rows or per-row probabilities. Do not train a neural encoder, use paid compute, or submit to the competition under this protocol.
