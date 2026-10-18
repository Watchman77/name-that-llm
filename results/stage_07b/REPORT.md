# Stage 07B completed: matched-class-mass data test

**Result: a provisional positive development signal.** The fixed candidate passed its pre-specified development screen with equal per-class weighted training mass. This supports testing the extra examples further; it does not establish a new official score or an improvement to the full S02 ensemble.

| Original held-out-prompt development balanced accuracy | Value |
|---|---:|
| Original-only character/style control | 35.376087% |
| Same classifier plus external examples, class mass matched | 36.561521% |
| Difference | +1.185434 percentage points |
| Paired prompt-bootstrap 95% interval | [+0.296545, +2.074571] points |
| Corrected / broken original passages | 95 / 63 |
| Net additional correct passages | 32 |
| Changed predictions | 382 of 2,699 |
| Fit warnings | 0 |
| Fixed development screen | Passed |

The interval is conditional on the fitted folds. This experiment was designed after seeing Stage 07A on these same official prompt groups. It is therefore **not fresh independent confirmation**, and the interval does not account for the process of choosing this follow-up. Neither the interval nor the pre-execution freeze removes that limitation.

## What changed from 07A

The earlier candidate added 50% more weighted training mass to the five augmented GPT labels. Its +0.925926-point overall difference had an interval spanning zero and could reflect changed class preference as well as additional examples. Its result remains recorded as a failed fixed screen.

For 07B, we normalized the same 1:0.5 original/external ratio so every class has precisely the same total weighted loss mass in control and candidate. For each augmented class, two-thirds of that mass comes from original passages and one-third from external passages. The other thirteen classes are unchanged. All ninety fold/class mass checks passed. Both arms also have the same total weight N per training fold.

Everything else remains fixed: 750 previously admitted direct-OpenAI V4 passages, 150 complete five-GPT prompt blocks, original 2,699 validation passages across eighteen classes, five whole-original-prompt folds, identical fold-local original-only feature fitting, the same 111 style features plus character TF-IDF, and identical LinearSVC settings (C=1, dual=True, seed=20261018, tol=0.0001, max_iter=10000). No fraction, C, feature, threshold or source-subset search was run.

The new candidate still predicts GPT more often than control (831 vs 770 predictions; true GPT count 750), but the shift is much smaller than 07A's 967 predictions. Equal training mass does not force equal prediction counts or guarantee calibration. The positive result supports value from changing the training examples beyond merely increasing the weighted mass of the GPT classes; it is not proof that every possible class-preference effect has disappeared.

## Gains and regressions

| Class group | Control mean recall | Candidate mean recall | Difference |
|---|---:|---:|---:|
| Five augmented GPT labels | 34.13% | 39.33% | +5.20 points |
| Other thirteen labels | 35.85% | 35.50% | −0.36 points |

The GPT group has 63 corrected and 24 broken passages, net +39. The other classes have 32 corrected and 39 broken, net −7. Overall net improvement is +32 correct passages. Balanced accuracy weights classes equally, including the Fable class with 149 original passages, so its percentage-point change is not exactly net_correct/2699.

GPT-5.6-Luna recall rises 9.33 points, GPT-6-Astra 8.00, GPT-5.5 6.00, and GPT-5.6-Sol 5.33. GPT-5.6-Terra falls 2.67. Qwen falls 5.33 and Claude Opus 5 falls 4.67; these regressions matter even though the pooled screen passes. Full per-class results and confusion matrices are retained. Class-group and per-class contrasts are exploratory, not separate multiplicity-adjusted confirmatory tests.

## Execution and verification

The runner, validator, prospective protocol, real preflight and class-mass audit were committed before fitting at **`5f36fc416b061b058372de5a2cf768cb99d927c7`**. Its runner and imported dependencies were checksum-checked before execution; the admitted-row manifest exactly matches 07A. All five folds completed in about 68.0 seconds including deterministic preflight and bootstrap. No trained model was serialized or distributed, no hidden test was used, and no paid inference API or laptop process/browser was accessed.

The separately implemented count-based validator reproduced all scores, fold assignments, the 10,000-draw paired prompt interval, corrected/broken counts and saved artifact hashes. It also independently reconciled every class's expected mass from the ninety-row audit and verified that **all 2,699 refitted control predictions exactly equal 07A's control predictions**. The prior row-prediction hash matches its freeze receipt. This is a separate implementation check, not an independent agent review or a second model-training reproduction.

The source/use boundary remains a private, undistributed classifier on first-party OpenAI responses from [prvsiyan's Extended Name That LLM Dataset V4](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset), source-declared CC BY 4.0; attribution to prvsiyan, October 2026. The pilot's bounded classifier-use assessment does not authorize distributing fitted weights, training a competing generative model or deploying a competition artifact. The user intentionally has the repository public at present; code, audit receipts, public external-data IDs and aggregate results are saved there. Original row-level labels/predictions and native margins stay outside the public repository. Visibility was not changed.

## Interpretation and next gate

Stage 07B passes the fixed rule (positive difference, lower interval endpoint above zero, no fit warnings). Retain it as a **provisional component hypothesis**. No official submission or final-selection change was made. The previously verified official S02 score is 0.549, and this work did not recheck the leaderboard. It would be incorrect to add 1.185434 points to 54.9% or claim that S02 now scores 56.1%.

The next meaningful assessment is whether this information adds value to a sufficiently faithful, aligned ensemble reference, with validation that addresses repeated use of the current folds. The incomplete Stage 06E proxy is not the full S02 ensemble. Do not blend margins from incompatible fold assignments or treat raw SVM margins as calibrated probabilities. A changed representation, an ensemble weight, or broader-route data admission is a separate prospective experiment. No automatic variants or public-score probing were performed.

Original prompt wording remains unavailable; reconstruction/topic and passage-overlap screens cannot certify semantic prompt independence. This and the adaptive follow-up design limit the strength of the current claim. The justified conclusion is that the fixed external-data policy still shows a modest development benefit after matching class mass, with material per-class trade-offs that need further evaluation.
