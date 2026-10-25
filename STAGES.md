# Project stages and current state

Current checkpoint, 24 October 2026: **S03 / Stage 08C V2 succeeded at 0.620 official public balanced accuracy**, +7.1 percentage points above S02. Rank at verification: 377. The S03 watch is complete and disabled. Stage 09 completed and its fixed pretraining candidate was rejected: control29.4444% versus candidate24.2593% on one reused development fold (-5.1852points); no official submission. See `results/stage_09/REPORT.md`. Historical observations below retain their original dates.


## Competition goal

Target first place in the final private evaluation. At an earlier recorded 17 October 2026 check, S02 scored **0.549**, the public leader **0.600** (5.1 percentage points ahead), and tenth place **0.559** (1.0 point ahead). Our rank was **181 of 1,212 teams**. No fresh leaderboard check accompanies this update. S01 remains frozen at 0.354. Public standings are time-specific; the final private result is unknown. The user's request to adopt stronger public methods was completed as one frozen external-baseline reproduction, not an original-method claim or parameter search.

| Stage | Status | Evidence / next decision |
|---|---|---|
| 01 — data checks and fixed baseline | Complete | 30.19% balanced accuracy; private offline Kaggle version 1 ran successfully. |
| 02 — word/character comparison | Complete | Word 26.26%, character 30.86%, combined 30.19%. Character lead remains uncertain; retain both references. |
| 03 — error diagnosis and surface-style features | Complete and Kaggle-verified | Character + style: 35.41%; improved all folds; private offline version 2 succeeded. |
| 04 — offline pretrained encoders / complementary models | Complete; both fixed experiments rejected | Frozen MPNet: 14.59% alone, 34.12% added. Supervised DeBERTa: 22.76% development BA, below the 35.41% reference in every fold; all five folds completed. |
| 05 — candidate freeze and competition submission | S02 succeeded at 54.9% official public BA | S01 remains 35.4%. Unchanged public baseline V1 359332774 reproduced 0.549; rank 177/1204 at observation. Final private evaluation and further original improvements remain pending. |
| 06 — OOF provenance and centering | Audit, lexical test and mixed confirmation complete | Stage 06C mixed-grouping component BA 45.8276% to 46.9518% (+1.1242pp, conditional interval +0.5191 to +1.7179); fixed screen passes. All 2,699 GLM cache rows and five CPU folds complete. Class/context limits remain; official 54.9% unchanged. |
| 06D — fixed available-component ensemble | TECHNICAL_REPRO_FAILURE_NOVALID_ENSEMBLESCORE | V7 359505815 completed 5 folds/15 heads, then failed the strict control gate at 141.6s (151.6s total). Inference paths have zero hard-label changes; 20/32,388 repeated comparisons differ from original C. Call shape ruled out; remaining cause unproven. No valid ensemble/raw screen result; blind refits stopped; official score 54.9%. |
| 06E — canonical CPU evidence | Complete; valid negative development result | V2 359514801 succeeded in 169.5s, all 15 heads/caches verified. Raw proxy 54.8122% to 54.6767% (-0.1356pp), conditional interval -0.5189 to +0.2375; screen not met. Retain original centering; official 54.9% unchanged. Aggregate error analysis identified Gemini 3.7/3.8 as the largest confusion in every context. |
| 06F — fixed Gemini pair diagnostics | Complete; source/replay and independent aggregate validation pass | V4 359521305 succeeded in 20.3s CPU. 275 repeated mutual-confusion observations represent 109 distinct passages; 77 fail in all three contexts. No fits or changed predictions. Original V2 manifest-byte failure preserved; exact-hash constant repair verified by V4. Motivated the fixed G correction; official 54.9% unchanged. |
| 06G — fixed attention-pair complement | Complete; candidate rejected | V2 359531626 succeeded in 22.5s CPU. Raw proxy 54.8122% control to 54.6641% candidate (-0.1481pp; conditional interval -0.6420 to +0.3580), 82 fixed/94 broken repeated decisions. Independent source, numerical and aggregate interval validation pass. Retain control, retire exact rule, inspect representation coverage before a new hypothesis. No official submission; S02 54.9%. |
| Publication assessment | Pending | Review related work and establish a contribution with stronger independent/nested evaluation. Ranking alone is insufficient. |

## How each experiment is tracked

Before a run, record the question, data hash, feature/model changes, folds and primary metric. Afterward, retain every declared result, held-out predictions, per-model recall, warnings, timing and package versions. Interpret uncertainty and regressions before selecting the next experiment.

The same five folds are development data. No supplied prompt is now a new untouched holdout. Baseline 01 remains frozen for reference. Kaggle baseline version 1 is preserved. Private version 2 verified the Stage 03 candidate at 35.41% and succeeded in the official hidden-test rerun as S01, with public score 0.354. Public rank was 651 of 914 ranked teams at the recorded check; final private results remain unknown. See `results/submissions/S01_REPORT.md` for the observation time, precision limits and next decision.

Stage 02 plan: `plans/STAGE_02_FEATURE_COMPARISON.md`.
Stage 02 report: `results/stage_02/REPORT.md`.

Stage 03 plan: `plans/STAGE_03_STYLE_FEATURES.md`.
Stage 03 report: `results/stage_03/REPORT.md`.

Stage 04 plan: `plans/STAGE_04_FROZEN_ENCODER.md`.
Stage 04 report: `results/stage_04/REPORT.md`.
Completed supervised protocol: `plans/STAGE_04B_SUPERVISED_ENCODER.md`; results: `results/stage_04b/REPORT.md` (22.76%, rejected).

Completed public reproduction protocol: `plans/STAGE_05_PUBLIC_REPRODUCTION.md`; official S02 result: `results/submissions/S02_REPORT.md` (0.549). Full source review: `results/stage_05/SOURCE_REVIEW.md`. The separate 55.2% recipe was excluded because its scored version requires a private checkpoint.

Stage 06 audit: `results/stage_06/ADVICE_AUDIT.md`; next protocol: `plans/STAGE_06_OOF_AND_CENTERING.md`. 70% is a stretch goal (+15.1 percentage points), not a forecast.

Stage 06B completed result: `results/stage_06b/REPORT.md`; fixed CPU protocol: `plans/STAGE_06B_CACHED_CENTERING.md`; next prerequisite and confirmation: `plans/STAGE_06C_MIXED_GROUP_VALIDATION.md`.

Stage 06C: `results/stage_06c/KAGGLE_RUN.json`. Next fixed ensemble endpoint: `plans/STAGE_06D_FIXED_ENSEMBLE_PROXY.md`.

Stage 06D preparation: `results/stage_06d/KAGGLE_RUN.json`; current implementation and limits: `results/stage_06d/REPORT.md`. Exact five-cell packaging was committed in 9db86f28cfd763ccbc3dcc048fb35fe649f6445f; the source-verification block was recovered. Actual Version 3 359500151 failed during input metadata validation before new head fitting; the classes-only safe parser repair passed independent review, was frozen before execution in de19608a, and V5 359502730 completed all five folds but failed strict prediction reproduction. The exact-source-call/diagnostic revision passed saved-source verification in V6 359505487 and V7 359505815 subsequently failed strict reproduction after allfivefolds; diagnostics ruleoutcallshape forhardlabels, but establish no ensemble score.

Stage 06E completed: `results/stage_06e/REPORT.md`; terminal validation: `results/stage_06e/INDEPENDENT_TERMINAL_VALIDATION.json`; source/runtime receipt: `results/stage_06e/KAGGLE_RUN.json`. Pre-execution source freeze ac9d9d4b0487d71b3265b1b2b0f2621803a8b701. The frozen plan preserves its original prospective status; actual V2 completed with a failed improvement screen and no official submission.

Stage 06F completed: `results/stage_06f/REPORT.md`; findings: `results/stage_06f/FINDINGS.md`; terminal validation: `results/stage_06f/INDEPENDENT_TERMINAL_VALIDATION.json`; receipt: `results/stage_06f/KAGGLE_RUN.json`. Initial source freeze baee805c080a06c6d797e68c0a0c1a424756d3a1 and controlled repair freeze 1c9f0650f01d2adc43731876f4fe7e8439f34f48 preserve the failed V2 separately. V4's no-fit diagnostic supplies hypotheses only, with no corrective candidate or new official score.

Stage 06G completed: `results/stage_06g/REPORT.md`; terminal validation: `results/stage_06g/INDEPENDENT_TERMINAL_VALIDATION.json`; source/runtime receipt: `results/stage_06g/KAGGLE_RUN.json`; next prerequisite: `results/stage_06g/REPRESENTATION_COVERAGE.md`. Pre-execution freeze cbdd06c11bb62d72c04781bf7c75bceae08a59fb preceded the single actual candidate. The prospective plan remains unchanged; actual performance failed the fixed screen, and no variant ran.


## Stage 07 — external corpus audit (2026-10-18)

Actual V1 short-corpus assets audited: 24,025 answer cells / 24,015 unique response IDs across 18 labels, with hashes and joins verified. No official-training full-response overlaps under three normalizations; bounded character retrieval found no close matches. Licence/source evidence and prompt independence remain under review. The conservative flag-only subset covers ten labels; SpeechMap covers all eighteen but retains output-terms-review flags. One primary-source Fable check matched 352 cells. Training has not started; the prospective matched two-arm protocol is specified, and its runner is not implemented. Best previously verified official S02 remains 0.549. See [report](results/stage_07/REPORT.md), [protocol](results/stage_07/PROTOCOL.md), and [audit](results/stage_07/audit.json). No paid services, laptop access, submissions or final-selection changes.


### Stage 07 source follow-up — 2026-10-18

All 8,658 short SpeechMap cells match response text, prompt, requested ID and returned ID in 25 pinned primary files. However, 66.86% trigger opening-refusal markers versus 0.30% in official train, including every selected Gemini 3.6/3.7/3.8 response. Deprioritize bulk SpeechMap augmentation on domain suitability; this is not an observed accuracy failure. A more relevant candidate was located and audited: prvsiyan extended corpus V4, `new` clean split, 4,068 rows / 226 complete 18-model blocks. All rows match published raw call fields; no normalized full-response overlap with original train. The single refusal regex hit is a manually checked false positive. Gemini punctuation distribution still differs. Top-30 retrieved instruction pairs inspected, without certifying semantic prompt independence. Preferred candidate route admission remains pending; no training, paid calls, laptop access, submission or final-selection change. Primary terms review distinguishes undistributed classifiers from competing generative models and model distribution. See [source review](results/stage_07/SOURCE_REVIEW.md), [use/terms review](results/stage_07/TERMS_REVIEW.md), and [decision](results/stage_07/REVIEW_DECISION.json).


### Stage 07A completed — fixed private augmentation pilot

Pre-execution source freeze d903e688bd7accd86bc343b728451ec8ea258382. Actual CPU run completed all five original-prompt folds: character/style control 35.376087% vs candidate 36.302013% development balanced accuracy, +0.925926 percentage points, paired 10,000-draw prompt-bootstrap 95% interval [-0.074323,+1.930486] points. The fixed screen failed; retain control with no automatic variant. 750 source-admitted direct-OpenAI V4 responses augmented five GPT labels while all eighteen official classes were evaluated. GPT mean recall +7.333333 points; other thirteen classes -1.538462. 110 corrected / 85 broken original passages. No convergence warnings. The fixed policy adds mass to GPT classes, so the result does not isolate extra-example value from changed class preference. Separate count-based validation reproduced folds, scores, interval and artifact hashes. This is not full S02 or a new official public score. Models were not serialized, no paid API or laptop was used, and no submission/final-selection change occurred. Fresh metadata shows the repository PUBLIC, correcting earlier descriptions: only aggregate pilot results are published; official row-level prediction artifacts and margins remain outside it. See [report](results/stage_07a/REPORT.md), [prospective protocol](results/stage_07a/PROTOCOL.md), [terminal validation](results/stage_07a/TERMINAL_VALIDATION.json).


### Stage 07B completed — class mass matched

User-authorized follow-up frozen before fitting at 5f36fc416b061b058372de5a2cf768cb99d927c7. Original-only control 35.376087% versus candidate 36.561521% development balanced accuracy, +1.185434 percentage points, paired 10,000-draw prompt-bootstrap 95% interval [+0.296545,+2.074571] points. Fixed development screen passed without fit warnings. All ninety class/fold objective masses match control; all 2,699 refitted control predictions exactly reproduce 07A. 95 corrected / 63 broken passages, net +32. GPT mean recall +5.20 points, other thirteen labels -0.358630; GPT predictions 831 vs 770 control (967 in 07A). This is provisional: designed after 07A on the same validation folds, not independent confirmation, and not a demonstrated gain to full S02. Separate count-based validation reproduced metrics, intervals, mass checks and artifact hashes. No official submission, final-selection change, trained-model distribution, paid service or laptop access. Public repository receives code and aggregate results only. See [report](results/stage_07b/REPORT.md), [prospective protocol](results/stage_07b/PROTOCOL.md), [validation](results/stage_07b/TERMINAL_VALIDATION.json).


### Stage 07C completed — fixed ensemble diagnostic rejected

Pre-execution freeze `92ba57ed4f4d617b178e71533db14d9073ddb94a`. Correct cloud sign-in restored access to private Stage 06E V2 359514801; downloaded binary matches the pinned SHA exactly. All frozen-input and alignment gates passed. Mean development proxy BA: unchanged base **54.812246%**, base plus original-only 5% vote **54.478001%**, base plus augmented 5% vote **54.527301%**. Candidate versus base **-0.284945 percentage points**, conditional joint-prompt 95% interval **[-0.790123,+0.208383]**; versus the original-only blend **+0.049300 points**, interval **[-0.185268,+0.320738]**. Fixed screen fails. Retain base, retire this exact hard-vote candidate, no automatic weight variants or official submission. Separate source review and 21 independently calculated numeric checks pass. Stage 07B's standalone gain did not establish ensemble value. This is a seven-member development proxy, not full S02; repeated validation and incomplete reference coverage remain. Official S02 remains last verified at 0.549, not freshly rechecked in this run. Next prerequisite is a more faithful reference and differentiated hypothesis; the prepared GLM coverage lead is already an S02 member, not a novel gain. No new reconstruction launched. Code/aggregate results only in public Git; no raw row artifacts, trained models, paid services, laptop computation or final-selection change. See [report](results/stage_07c/REPORT.md), [frozen protocol](results/stage_07c/PROTOCOL.md), [validation](results/stage_07c/VALIDATION.json), [terminal status](results/stage_07c/STATUS.json).



## Stage 08 — public 0.600 recipe reviewed; private copy prepared (2026-10-20)

Verified the author's exact V20 360088024 displays official public score 0.600; the earlier V14 359708344 is 0.557 despite the current title. Captured and independently reviewed its two-stage 8B/14B LoRA + shared-prompt Gaussian/full-stack recipe; no obvious static code blocker. Private watchman777 Quick V1 360471311 preserves exact complete V20 code, with no model execution. Copy inherited latest data: prvsiyan extended V5 instead of source V4, with input size 42.22GB vs 42.19GB. Restored explicit V4 and saved private Quick V2 360473569, source diff +0/-0; no model code run. Read-only CPU input-preflight implemented and all 12 synthetic gates pass, but real mounted inputs/byte identities and pool provenance remain unverified. Next gate is actual aggregate input/overlap/manifest reconciliation, then one frozen private GPU reproduction; no honest OOF claim can use final adapters reportedly fit on all official labels. No model inference, paid APIs, new generation, trained-model publication, submission or final-selection change. S02 remains last verified 0.549, not rechecked. See [review](results/stage_08_review/REPORT.md), [preparation receipt](results/stage_08_review/KAGGLE_PREPARATION.json), [source review](results/stage_08_review/SOURCE_CODE_REVIEW.md), [data review](results/stage_08_review/DATA_REVIEW.md).


## Stage 08 continuation — 24 October 2026

The frozen 0.600 recipe failed the real-input gate: 150/151 checks pass, but the original LoRA24 checkpoint is now private/unavailable; the old extra-data merge also differs (9,983 rows / 689 prompts versus 7,644 / 509). No model execution for that incomplete recipe. See [dated update](results/stage_08_review/UPDATE_20261024.md).

[Stage 08B](results/stage_08b/REPORT.md) reproduces Nishant V2 359887828 without model changes. Our private V1 361730151 completed its three-row preview in 397.7s; no official score.

[Stage 08C](results/stage_08c/REPORT.md) freezes Nishant part-two V1 361311282 (author public 0.620), pins dataset V1, passes 85/85 real CPU input checks, and completed private GPU V2 361732127 in 739.2s. The six model cells are unchanged; an independent validation-only final cell checks probabilities/IDs/parquet. All 22 actual output checks passed. Official S03 is Notebook Running on hidden data, with no score yet; hourly read-only monitoring is enabled. Current public first is 0.700, tenth 0.660; our displayed current official best remains 0.549. No tuning, API purchases or final-selection changes.



## Stage 10 — frozen public 0.639 candidate (24 October 2026)

Kirill V1 361634630 displays source-author official public balanced accuracy 0.639; our best remains S03 at 0.620. CPU V1 361877495 completed with 23/25 input checks and conservatively blocked two Base/chat selectors. Explicit review pinned the original source-first Base paths/configuration hashes without changing the recipe. CPU V2 361879824 passed 25/25 in 120.3 seconds. Attached custom code was statically reviewed; serialized payload behavior and GPU runtime remain unverified. Private candidate V3 361881189 preserves all fourteen source code cells exactly and adds only a separate output guard; all fifteen editor cells passed byte-for-byte readback. Its three-example shortcut emits constant labels and is FORMAT ONLY, never real model inference or a score. Own official evaluation remains pending. No paid services or final-selection changes. See [Stage 10 report](results/stage_10/REPORT.md).


### S04 accepted — 24 October 2026

Private Stage 10 V3 361881189 completed in 18.6 seconds with all 20 FORMAT ONLY checks passing. No model weights loaded; this does not prove GPU/runtime validity or accuracy. Submitted this exact version as S04; the live official row displays Notebook Running and no score. Its hidden rerun will exercise the complete ensemble. Our S03 remains officially 0.620; author 0.639 is not ours. Final selections remain automatic 0/2. Code, aggregate audit receipts and evidence are saved under results/stage_10.


## S04 official completion verified — 25 October 2026

Our S04 / private Stage 10 V3 **361881189** displays **Succeeded** and **0.639 official public balanced accuracy (63.9%)** on the live connected-account Kaggle submissions page. This is our own hidden-test score, separate from the author's 0.639 and the 18.6-second constant-label format preview. It improves on the displayed official **S03 0.620 by 1.9 percentage points** and **S02 0.549 by 9.0 points**; 70% remains 6.1 points away. Decimal arithmetic verifies these differences. Private leaderboard performance and rank are unknown. The official submission-details modal separately confirms Score: 0.639, Succeeded and the exact private V3 link, but exposes no hidden-runtime log link. Hidden-run runtime and detailed output-guard outcomes remain unverified; preview logs are not substituted. Final selections remain automatic 0/2 with boxes unchecked; no watch was created or enabled. No new run, tuning, submission or final-selection change occurred for this check. Prior pending-state entries remain historical checkpoints. See [Stage 10 report](results/stage_10/REPORT.md) and [official result receipt](results/stage_10/OFFICIAL_RESULT_20261025.json).


## Stage 11 — stronger public lead under admission review (25 October 2026)

Preserve S04 official 0.639. Verified source-author 0.642 at Lightning V28 362014737; captured 33 source cells without execution. Its full-text 27B component is a concrete new lead, but 134.1 GB / 28 inputs, 160 mandatory path checks and disabled internal deadlines require an actual CPU input audit and runtime review first. Our candidate score is unknown; no new notebook, GPU run, submission or tuning occurred. Current leaderboard: rank 214 / 0.639; tenth 0.668, leader 0.703. See [prospective Stage 11 plan](results/stage_11/PLAN.md).


## Stage 11 — S05 accepted, official score pending (25 October 2026)

Frozen Lightning V28 362014737 (source-author 0.642), preserving all 33 source code cells and adding only an output guard. Actual CPU V2 362119531 passed 175/175 checks in 686.1s, including all 160 mandatory paths and active NF4 hashes. The initial 174/175 result was an overstrict audit label-order assumption, corrected without modifying inference code or weights. The complete CPU report was collected and hash-verified. Real Qwen-only GPU V3 362125549 succeeded in 418.9s on T4 x2, producing finite ordered 12-by-18 logits within a 900-second process-group deadline. This was execution evidence, not accuracy or complete-ensemble runtime proof.

Private production V6 362129392 matches all 34 saved code cells and 28 source input versions. Its three-row FORMAT ONLY preview succeeded in 21.7s with 25/25 checks; no full models ran in that preview. Actual logs show T4 x2 and the save option explicitly uses GPU for all sessions; the separate notebook document GPU boolean remains false. The exact V6 was submitted as **S05**, now **Notebook Running with no displayed score**. Our confirmed best remains **S04 0.639 official public balanced accuracy**; no gain is claimed from the author's 0.642. Final selections are currently 2/2 and were not changed. Free Kaggle only; no training, tuning, paid API or new watch. Earlier pending statements above are historical checkpoints. See [Stage 11 report](results/stage_11/REPORT.md), [submission receipt](results/stage_11/S05_SUBMISSION_RECEIPT.json), and [notebook/version index](results/stage_11/NOTEBOOK_INDEX.md).

## S05 official completion verified — 25 October 2026

Our S05 / private Stage 11 V6 **362129392** displays **Succeeded** and **0.642 official public balanced accuracy (64.2%)**, confirmed by the fresh official submissions row and the separate Submission Details modal. S04 on the same page is 0.639: the improvement is **+0.3 percentage points**, leaving **5.8 points to 70%**, verified with decimal arithmetic. This matches the existing Lightning V28 362014737 public reproduction; no original modelling gain is established. All 33 source code cells remain unchanged with one output guard. The official hidden evaluation completed, but its exact duration and detailed hidden output-guard results are unavailable on the inspected surface; the 21.7-second format preview and 418.9-second Qwen-only probe are separate milestones. Private leaderboard performance remains unknown. Final selections remain 2/2: S03/S04 selected, S05 unselected, unchanged. No new run, tuning, paid service or watch. Prior pending-state entries and the original submission receipt remain historical evidence. See [Stage 11 report](results/stage_11/REPORT.md) and [official result receipt](results/stage_11/OFFICIAL_RESULT_20261025.json).


## Stage 12 started — 25 October 2026

After S05 official 0.642, the live Code list sorted by Public Score identified Dhruv nlm-v27 V1 362013743 at source-author 0.651. Private source-only Quick V1 362254770 is saved, all six code-cell strings match and 31 copied input versions are frozen. Actual mounted-input audit and executable-asset review precede GPU admission; no model or official submission has run in this stage. Fresh free GPU enable dialog shows 29 hours remaining; cancelled after reading and draft set to CPU / Internet off. The source author's 5,251.1s T4x2 preview is real model execution and is not hidden-runtime evidence. A separate Stage12A fixed two-arm CPU component screen is prepared prospectively; it is source-style SVC versus dormant topic correction under existing prompt-grouped folds, with no full-S05 OOF or official-gain claim. See [Stage12 plan](results/stage_12/PLAN.md) and [Stage12A protocol](results/stage_12a/PROTOCOL.md).
