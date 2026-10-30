# Experiment log

## Current official checkpoint — 30 October 2026

S07 is **Succeeded at 0.651 official public balanced accuracy**, our new best. Exact-decimal comparison: **+0.900 percentage points** over S05 0.642; **4.900 points** below a 0.700 public score. This checkpoint adds no fit, tuning or new submission. Earlier dated entries remain historical. [Final result receipt](results/submissions/S07_FINAL_RESULT.json).

## Baseline 01 — 16 October 2026

- Data: supplied official ZIP, 2,699 labelled passages, 150 prompts, 18 models.
- Checks: no missing required values, duplicate IDs, duplicate exact or whitespace/case-normalised passages, or duplicate prompt/model pairs. 149 prompts cover all 18 models; one covers 17.
- Validation: deterministic `GroupKFold(5)` by `prompt_id`, with fold-local TF-IDF. Same folds should be reused for controlled development comparisons.
- Recipe: word 1–2-grams (80,000 feature cap) + character 3–5-grams (120,000 feature cap), LinearSVC C=1, balanced class weights, seed 20261016, dual=True.
- Pooled balanced accuracy: 0.30187670892368884. Fold values and package versions are in `results/baseline_01/`.
- Recognition extremes: Gemini 3.1 Pro Preview recall 74.67%; Claude Fable 5 recall 6.71%.
- Frequent confusion: Gemini 3.8 Flash → 3.7 Flash, 47/150; reverse, 40/150.
- Execution: sequential code cells in a fresh in-process IPython session; saved figures inspected; metric independently reconciled from the confusion matrix; Parquet round trip passed. Ordinary TCP Jupyter kernel startup was unavailable in the local execution environment.
- No paid API usage or hyperparameter search. No leaderboard score established.

## Next controlled experiments

1. Word-only versus character-only versus combined features, on identical folds.
2. Small, interpretable linguistic feature set, with preprocessing fitted inside each fold.
3. A compatible offline pretrained encoder if simpler features plateau.

Repeated tuning makes these folds development data. Reserve fresh prompts or use nested evaluation for strong research claims. Do not infer prize placement from the first local result.

## Kaggle setup

The notebook is kept private with the competition input attached and internet disabled. Removed the preview sample-ID equality check: the hidden test can have different IDs and row counts. Predictions are checked against the actual supplied test file. Model features, parameters and validation folds are unchanged.

## Stage 02 — fixed feature comparison

- Plan recorded before execution: commit aafa46b8ab9fdef40bb4f88062414524f27a5f8b.
- Same data hash and exact saved folds; only the TF-IDF feature branch changed.
- Word-only: 26.2612%; character-only: 30.8553%; original combined reference: 30.1877%.
- Character lead: +0.6677 percentage points, paired prompt-bootstrap 95% interval −0.5194 to +1.9673. Advantage is not established by this comparison.
- All folds completed with no convergence warnings. Combined OOF predictions were reused. No parameter search or extra candidates were added.
- Full outputs: results/stage_02/. Reproduce with python scripts/feature_ablation.py --data-dir data.
- Decision: keep character-only as a provisional development leader and combined as a reference; next inspect errors and test predeclared style features. The private Kaggle baseline stays unchanged.

## Stage 03 — surface-style features

- Prospective plan commit: 09e2fb41f37fd420caf3184d4d2512b864c0e3e7. Two fixed candidates, exact saved folds, no tuning.
- 111 deterministic features; scaling is fitted inside each training fold.
- Style-only: 29.2637%; character plus style: 35.4131%; references: character-only 30.8553%, combined 30.1877%.
- Gain versus character-only: +4.5578 percentage points; paired 95% prompt-bootstrap interval +2.7783 to +6.3379. All five folds improve. These are development comparisons conditional on fitted predictions.
- Improvement occurs in all three fixed length bins, but some per-model regressions remain. Claude Opus 4.8 recall drops from 24.00% to 16.67%.
- All 20 fold records and 2,699 OOF rows saved; metrics independently reconciled; zero convergence warnings.
- Frozen candidate source: notebooks/Name_That_LLM_Stage_03.ipynb and scripts/stage03_candidate.py. It passed private offline Kaggle verification in version 2, reproducing 35.41% (105.7-second successful run); baseline version 1 is preserved.
- Full plan, results and caveats: results/stage_03/REPORT.md. No paid API usage or competition submission.

## S01 — first official competition submission — 16 October 2026

- Frozen Stage 03 candidate, private Kaggle version 2, scriptVersionId 359158002; output `submission.parquet`. Source SHA256: bee527c28fdc4a163c023d2a560d9a45066c52de8017dd69472f8e9cdbdfe294.
- Hidden-test rerun status: **Succeeded**. Public balanced accuracy, exactly as displayed: **0.354 (35.4%)**. Prompt-grouped development score: 0.35413124533929907. Agreement is at display precision; the small arithmetic difference is below that resolution.
- Public rank observed at 2026-10-16T23:11:20Z: **651 of 914 ranked teams**. Tenth-place public score: 0.549; leader: 0.584. The gap to tenth was about 19.5 percentage points. Rank and cutoffs can change.
- Approximately 40% public / 60% private test split as described on the leaderboard; final private score and ranking are unknown. No hidden per-class labels are available.
- Only one submission made. No recipe changes after seeing the score, no paid API usage, and no manual final-selection change.
- Decision: keep S01 frozen and predeclare a compatible offline pretrained-encoder comparison on the same development folds. Choose trials using development evidence rather than repeated leaderboard feedback.
- Full record and prospective feedback protocol: `results/submissions/`.

## Stage 04 — frozen MPNet comparison — 17 October 2026 UK time

- Protocol committed before embeddings/fits: 9d34e75b0c2b2faff0f995bd441a8a3297bd77c4. Implementation committed before execution; exact original data hash and prompt-grouped folds retained.
- Public checkpoint: sentence-transformers/all-mpnet-base-v2, revision e8c3b32edf5434bc2275fc9bab85f82640a19130, Apache-2.0. Float32 fixed encoder, mask-aware mean pooling and L2 normalization; local-only/offline modelling after resource preparation.
- MPNet alone: **14.5946%**. Character/style plus MPNet: **34.1153%**. Frozen Stage 03 reference: **35.4131%**.
- Combined candidate difference: **−1.2978 percentage points**, conditional paired prompt-bootstrap 95% interval **−2.6324 to −0.0370**. Three folds worsened, one tied, one improved. All three predeclared length groups worsened; mixed class regressions/gains retained.
- Ten new fold fits completed, five reference scores reused; 545.1 seconds CPU; zero fit warnings and zero truncated passages (longest 381 tokens; cap 384). No paid services, extra candidate or official submission.
- Decision: reject both additions and retain Stage 03/S01. This result does not establish that supervised contextual encoders fail; the experiment used a frozen semantic encoder, fixed C=1 and equal normalized block weights.
- Evidence and independently executed evaluation companion: results/stage_04/ and notebooks/Stage_04_Evaluation.ipynb.
- Target clarified: first in final private evaluation; public leader observed at 58.4%, 23.0 points above S01. Plan for a distinct fixed supervised DeBERTa-v3-base trial: plans/STAGE_04B_SUPERVISED_ENCODER.md. That trial is **planned, not executed**.

## Stage 04B — supervised trial launch — 17 October 2026

- Original protocol commit 8ab67d337b139059669455d605cf802eda9d4301; implementation clarifications committed before fitting in 5563d45ef49311afa7ca0cacc184bb2ed46f3ba8.
- Pinned DeBERTa-v3-base, one fixed three-epoch recipe, exact saved prompt folds. Runner commit 8603667afbe1d404aec4f9855ab09be5d3ef412e.
- Exact uneven accumulation gradient check and actual offline model/tokenizer CPU forward passed. No supervised fitting or validation score was produced by these checks.
- Separate private CPU resources version 359317729 succeeded in 35.6 seconds. Includes pinned weights, offline dependency wheels and the exact frozen reference OOF file.
- Kaggle reported 30 included GPU hours before launch. The private training version 359319680 is queued with GPU T4 x2 selected, computation restricted to cuda:0, internet disabled and a three-hour budget. GPU preflight and results are pending.
- No new official submission. See results/stage_04b/KAGGLE_RUN.json for current execution status.

## Stage 05 — public baseline reproduction launch — 17 October 2026

- User asked us to inspect and adopt stronger public methods. Leader observed at 0.593; its private recipe is unknown. The separate public 0.552 V1 requires a private `lora24c` checkpoint and was excluded.
- Selected Lucas Boesen's exact public V3, scriptVersionId 359126546, displayed score 0.549. All 14 inputs are public and attached. Preserved original source, credits, Apache-2.0 licence and exact resource IDs. Prospective plan/clarification committed before execution: dfc009a58c5ca3e0be8b8b6a23044eac600515e8.
- Our private V1 359332774 is running with source diff +0 -0, internet off and T4 x2. Live log confirms two GPUs and completion of stylegram/LightGBM, then GLM feature extraction. No paid APIs, new synthetic data or paid compute.
- Runtime label discrepancy retained: editor requested original environment dated 2026-10-02; Logs labels Latest Container Image. Notebook JSON also says isGpuEnabled=false although the GPU-for-this-session option and live log confirm actual GPU execution. Do not claim bitwise environment reproduction.
- No new accuracy or S02 yet. Source reported 0.549 is not our measured improvement. After a successful preview, verify outputs/parity before one declared official S02 rerun. Public final-fit checkpoints cannot provide independent CV on training folds. Preserve S01 and the separate Stage 04B trial.
- Exact code-cell SHA256 e3d97306f3185ea5f73d7904d81331b27803e3ceb45970596590ec306969762e. Python archive has one extra terminal newline (SHA256 261e27094fbefbfe8e6496feda4a274f8e9a8587a19b304da074a71f49cbc2f3); parsed code is identical. Notebook JSON and format validation passed.
- Evidence: plans/STAGE_05_PUBLIC_REPRODUCTION.md, results/stage_05/, notebooks/Stage_05_Public_Reproduction.ipynb and vendor/NOTICE.md.

## Stage 05 preview completion and S02 — 17 October 2026

- Private V1 359332774 succeeded in 1884.8 seconds (31m 25s), all recipe stages complete. Exact executed code-cell hash matches the declared source. LoRA mean-relative anchor errors were 0 and 0.00017, both below 0.02. Technical in-sample checks passed; they are not independent accuracy.
- Output `submission.parquet` exists (1.76 kB displayed); source construction/row-count/ID-uniqueness/valid-label assertions passed. Parquet UI preview is unsupported, so no independent binary round-trip is claimed. Verification and full rendered stdout/stderr were committed before S02.
- One official S02 was accepted. The competition Submissions page shows Notebook Running for the exact private V1 and output file. No new public/private score yet; S01 remains 0.354. No parameter changes, repeated leaderboard tuning, paid services or manual final-selection changes.
- Stage 04B remains queued with no GPU preflight/training logs. It was not restarted or changed.
- Records: results/stage_05/OUTPUT_VERIFICATION.json, PREVIEW_OUTPUTS.json and results/submissions/S02.json / S02_REPORT.md.

## Live progress — 2026-10-17T13:12:36.877Z

- Official S02 still shows Notebook Running on the fresh competition Submissions page and detail panel. No public score or displayed execution error; S01 remains the only confirmed score (0.354).
- Stage 04B has left the queue. Actual Tesla T4 preflight passed with Transformers 4.56.2, torch 2.11.0+cu128 and one visible GPU. Fold 1 final-epoch balanced accuracy is 0.231481481 versus Stage 03's 0.335185185 on the same fold; fold 2 is 0.218518519 versus 0.331481481. Fold 3 epoch 3 is in progress in the latest visible logs.
- These are partial development results, not official scores or a complete OOF estimate. No exceptions were shown in the inspected logs and recorded completed epochs had zero AMP overflow steps. Preserve the fixed five-fold recipe without restarting or tuning on partial results. See results/stage_04b/PARTIAL_PROGRESS.json.

## Completed outcomes — 2026-10-17T14:02:20.265Z

- Official S02 succeeded with **public score 0.549**, exactly as displayed, versus S01 0.354: **+19.5 percentage points**. This is our measured score, matching the copied public recipe's displayed result; it does not demonstrate an original contribution or identical hidden predictions.
- Current public rank **177/1204**, tenth-place score 0.559, leader 0.593. Gaps are 1.0 and 4.4 percentage points respectively; final private standings remain unknown. No manual final-selection change, additional submission or paid service.
- Stage 04B completed all five folds successfully in 3022.4 platform seconds. Pooled development BA **0.22763857817549094**, versus frozen reference **0.35413124533929907**. Difference **-12.6493 points**, conditional paired 95% interval **-14.6927 to -10.7037 points**. All five folds worsened; zero AMP overflow steps and runner warning list empty. Reject this fixed recipe; no post-result tuning or submission.
- Exact completed DeBERTa metrics and full logs were archived from the visible Download logs link. OOF files, diagnostics and checkpoints remain in the immutable Kaggle output (45 files); independent local recomputation of those full files has not been performed in this status check. Records: results/stage_04b/metrics.json, KAGGLE_COMPLETE.log, completed_fold_comparison.csv and REPORT.md.


## Stage 06A and 06B — verified artifacts and one centering comparison — 17 October 2026

- Stage06A private CPU V1 359394412 succeeded in31.1s. It reconstructed all2,699 public LoRA held-out predictions from outer feature banks/readouts, with max log-probability error1.78e-14. Plain development BA48.16%; fixed50/50 plain/center53.90% at3responses/prompt, with different outcomes at1and6. Not a full S02 score.
- Stage06B scoped protocol committed before outcomes (2a090244c493a6e56990d3427e49f5390df87e4b). Public cached DeBERTa mean24+6 features, archived prompt folds, fixedC0.005 and evidence weights. One candidate: train centered heads on fixed size2/3/6 same-prompt partitions with row weights1/3 instead of full prompt means. Lexical-only inferred groups and fp16-cache approximation explicitly retained.
- Mean paired component BA across contexts1/3/6:46.5685% control vs47.2483% candidate; gain+0.67984pp, paired prompt95% interval+0.08602to+1.29630pp. Scenario gains0,+1.89162,+0.14790pp. Class tradeoffs remain: Qwen recall60%to52% at3responses; GPT-terra40%to34% at6. Six-response oracle diagnostic slightly regresses.
- V1 359399756 succeeded155.8s. Its false screening flag came from classifying15identical SciPy deprecation notices as convergence failures. Original result/log preserved. V2 359401219 changed only warning categorisation and diagnostics/assertions; succeeded170.1s, reproduced all control/candidate scenario scores to1e-12, had zero convergence warnings and passed the numeric screen. No new model setting/candidate was selected.
- Numeric OOF audits: second seed7 LoRA48.04425%; Qwen3.5 token attention39.37808%; DeBERTa token attention41.08153%. Extra Qwen8B attention37.00795% audited but not selected. Object-ID metadata statically inspected without unpickling; numeric arrays read withallow_pickle=False. Twenty NPZ files inventoried.
- Decision: provisional component improvement only. Confirm the same DeBERTa change with S02's actual mixed GLM/lexical grouping before adoption. Missing GLM layer24 passage-feature cache is the next precise prerequisite. No official submission, final-selection change, GPU or paid service used in Stage06B. Full report: results/stage_06b/REPORT.md; next fixed protocol: plans/STAGE_06C_MIXED_GROUP_VALIDATION.md.


## Stage 06C launch — 17 October 2026

- Prospective protocol amendment committed before extraction: 8c10205da8a962d977f144a0b1ebcd3d9da39b79. Reviewed extractor, paired mixed-grouping confirmation and combined runner committed before Kaggle save.
- Private V1 359405443 is queued with GPU T4 x2 requested; 28 included GPU hours were displayed before launch. Internet disabled. Exact UI version export binds competition 170484 and public kernels 358683059 (GLM) / 358745836 (encoder cache).
- Frozen GLM inference saves fixed layers 8, 3 and 24 once, with data/ID/text/model hashes; no supervised neural training. The CPU comparison preserves Stage 06B recipe and changes only group recovery to exact S02 mixed grouping, adding group diagnostics.
- Export reports isGpuEnabled=false despite the GPU-for-this-session save selection and T4 x2 Logs label. Actual CUDA preflight has not yet been observed, nor has any output/cache/result. Preserve this inconsistency.
- No official submission, final-selection change or paid service. S02 remains 0.549. Next decision: drop candidate if mixed-grouping confirmation fails; otherwise test marginal benefit in a fixed available-component ensemble, explicitly not full S02. Protocol: plans/STAGE_06D_FIXED_ENSEMBLE_PROXY.md.


## Stage 06D CPU prerequisite — 17 October 2026

- Prospective clarification f1105cb880b947b7653b1acdb5a09ca3602f179b; reviewed source committed before fitting in 4afba147c4290ae24ddf4f1ea078fe8a8fc729d1.
- Reproduced exact S02 stylegram recipe on original five prompt folds with fold-local vocabularies/IDF, fixed sklearn 1.8.0 defaults and stable score-softmax. No model or weight search.
- All five folds completed in 42.48s CPU; pooled development BA 0.3844842157593836 (38.4484%). Zero warnings. Full class probabilities and scores for all 2,699 rows saved, with no passage text or pickle arrays.
- Independent no-refit checks exactly reconcile IDs/classes/prompts/folds, probabilities/log floor, confusion matrix, per-class recall, fold metrics and original artifact hashes. Largest stylegram confusion is Gemini 3.7/3.8 (80 bidirectional errors), with cross-family errors also present; these are component diagnostics only.
- The GPU Stage 06C confirmation remains queued with no CUDA preflight/output/result at the latest live read. Fusion still awaits common mixed contexts and full DeBERTa class probabilities. No new official score, submission, final selection, neural training or paid service. Report: results/stage_06d/REPORT.md.


## Stage 06C completion — observed 17 October 2026 at 19:39 UTC

- Immutable private V1 359405443 succeeded in 524.8 platform seconds, with 122.82 MB output. Actual CUDA preflight confirms two Tesla T4 devices, resolving execution uncertainty from the export GPU flag. All 2,699 rows extracted in 338 batches; fixed layers [8,3,24], shape (2699,3,4096), float32 output. Cache SHA256 2b41585a9482cef825962a8cc096697933c0b1237e8ede0712d1f8ab361d9414. Live manifest reports zero nonfinite events and zero truncated rows, fp16 backbone throughout.
- Actual S02 mixed GLM24/lexical grouping confirmation: component control mean BA45.8275748% to candidate46.9517773%, gain+1.1242025pp; conditional paired prompt95% interval+0.5190944 to+1.7178971pp. Context gains0,+2.0392742,+1.3333333pp. Predeclared screen passes; all five CPU folds complete, zero convergence warnings and15 SciPy deprecations preserved.
- Independent checks reconcile216 per-class rows with all12 declared scores, group row/fold totals, mean gain, interval endpoints and screen logic. Per-row predictions and bootstrap were not independently recomputed locally. Three/six-context pair precision100% in30-prompt fold batches; larger-batch transfer remains untested. Mean Opus4.8 recall−2.67pp; Terra six-response−7.33pp. Six-response oracle diagnostic slightly regresses.
- Retain candidate for fixed Stage06D available-component ensemble comparison, now with completed frozen GLM cache. Full CPU head probabilities must be retained for fusion; no repeat GPU extraction is needed. This is not full S02 or an official improvement; best confirmed official score remains0.549. No submission, final-selection change, new neural training or paid service. Evidence: results/stage_06c/REPORT.md and original complete logs.


## Stage 06D ensemble preparation — 17 October 2026

- Reviewed fixed comparison and exact frozen style-probability transfer committed in 1ca9bae1f8fb38b0965f4e2e6aba7a366c2a8153. No style refit or neural training is part of transfer. Local decode reproduced every original float64 probability byte; a tampered payload was rejected. Raw fixed rowwise fusion is primary; alpha=0.5 balancing is potentially contaminated by peer supervision across differing fold maps and cannot select the candidate.
- Five-cell packaging committed in 9db86f28cfd763ccbc3dcc048fb35fe649f6445f. All five cells compile below the observed source-viewer truncation boundary; notebook schema, unchanged function ASTs/STYLE constants, exact reconstructed payload and notebook/linear-runner equivalence independently verified. Packaged runner SHA256: 13d4cbd40a563033fcc12d8c8773bc27ab6e112b171a519aa34f0ce2cbc9103b.
- All five private Kaggle editor cells passed complete clipboard roundtrip comparisons. Saved metadata-only QuickSave V1 showed competition 170484 and kernels 359405443 / 358745836 / 358831649 / 358911769 / 358928013 / 358948428. CPU, internet off and private visibility observed.
- Metadata-only QuickSave V2 was requested, but its response and immutable saved five-cell source remain unconfirmed. Repeated browser stalls returned `MCP request timed out`; Version History failed to load. A bounded pre-refresh metadata retry returned `Chat API request failed`; screenshot proof showed Kaggle commit `Failed to fetch`, V1 only, Draft Session off and zero active events. Intentional refresh preserved all five cells and seven input titles. One post-refresh attempt to reopen QuickSave failed while clicking its option, before Save was called. Last browser error: `Chat API request failed`; all retries have ended. **Save & Run was never called and the ensemble comparison has not launched.** No V2 ID is inferred. The error-banner screenshot predates refresh and proves no post-refresh banner state.
- Next: recover access, verify immutable saved source plus seven input bindings, then launch the unchanged fixed CPU comparison. Retain complete logs, probabilities and paired diagnostics, including the required reproduction of all 12 Stage 06C prediction arrays. This preparation generated no ensemble score or official improvement, ran no new fits, used no paid service, made no final-selection change and submitted no candidate. Best confirmed official S02 remains 0.549; leaderboard standings were not refreshed. Evidence: results/stage_06d/KAGGLE_RUN.json, REPORT.md, MULTICELL_SOURCE.json and PACKAGING_REVIEW.json.

- Final preparation state: BLOCKED_AT_SAVED_SOURCE_VERIFICATION. Scratch proof `browser-proof/stage06d-save-error.png` has SHA256 36e0a428368d36868ae9c2e9b1ab83a4ea065a2b6160ed0a2ea764807e65d30a; binary excluded from git/Library. Local fallback is unavailable because required neural numeric caches were not downloaded (fusion-agent inventory). Last verified repository head before these uncommitted state updates is 9db86f28cfd763ccbc3dcc048fb35fe649f6445f. No new fits, ensemble result, submission or official score.


## Stage 06D execution launch — observed 17 October 2026 at 21:47:07 UTC

Historical launch observation; superseded by the terminal V7 outcome below.

- Saved-source access recovered after the historical save/browser failures. QuickSave Version 2 succeeded (`Viewing Version 2`, Quick Version, Oct 17 10:45 PM), named `Verified multi-cell source - no execution`, with zero seconds run. Its full visible saved export matched all five reviewed cells exactly, with zero outputs/null execution counts; all seven input IDs and CPU/internet-off metadata matched. Numeric QuickSave scriptVersionId was not established and is not inferred.
- Unchanged Save & Run All launched actual visible Version 3, named `Fixed raw ensemble comparison`, with advanced selection `Run without an accelerator for this session`. Fresh UI at 2026-10-17T21:47:07Z showed Running and one active event. Numeric execution ID, immutable execution-version metadata and fresh logs are still being obtained.
- Primary raw screen, frozen protocol and all source hashes remain unchanged. This is execution launch only: no ensemble result, measured combined-model gain or official score is available. No official submission or final-selection change. Official S02 remains 0.549. Current receipt: results/stage_06d/KAGGLE_RUN.json.


## Stage 06D Version 3 failure — scriptVersionId 359500151

- Actual Save & Run All version 359500151 is terminal Failed. Live log: https://www.kaggle.com/code/watchman777/name-that-llm-stage-06d-fixed-ensemble/log?scriptVersionId=359500151. Logs reached 20.7 seconds despite the terminal UI displaying zero seconds. The preceding saved-source gate passed; the prior browser/save block is resolved.
- `STYLE_TRANSFER_COMPLETE` verified exact raw probability bytes for all 2,699 rows (SHA256 5e2e05577b7dc632f38f67d47958d7f3d069e8b758ead4664050f0d984803bad). `STAGE06D_COMPONENTS_AUDITED` confirmed five members and 47 files. Own Stage 06C `paired_predictions.npz` metadata then raised `Object arrays cannot be loaded when allow_pickle=False` in the ids/classes/y/groups/fold validation loop. Zero new head fits and zero DeBERTa fold-completion markers; no ensemble result.
- A minimal static expected-string metadata reader is being prepared for independent review. It must keep numeric reads safe, never enable `allow_pickle=True`, and preserve all modelling/fold/grouping/weight/screen settings. No repaired run or new version ID is established yet. This is an input-schema compatibility failure, not evidence for or against centering benefit. No official submission; S02 remains 0.549. Receipt: results/stage_06d/KAGGLE_RUN.json.


## Stage 06D classes-only compatibility repair review

- Successful metadata-only QuickSave V2 numeric ID confirmed as 359499684 from that version's own Open in Viewer href. Failed actual V3 remains 359500151. Original complete V3 log preserved: 5,631 bytes, SHA256 a4a92d501d696289e4d3a2a833933d7ebd3f64fc4ef6c426d9de3d5c36d9b4c3; parsed proofs in V3_FAILURE.json.
- Exact Stage 06C writer uses classes=np.sort(train.model.unique()), yielding object dtype; IDs/groups are Unicode. Classes-only static reader repair SHA256 b27925f978b0c87938c92adc36bcbb52e6536aa79fbe9330217daec7fecbae51 independently passed exact writer/protocol3/4/5/Unicode fixtures and rejected malformed/dangerous metadata, with zero calls to guarded unpickling/construction surfaces. All other algorithm AST nodes and numeric pickle-disabled reads are unchanged; zero fits.
- First four packaging cells remain exact V3 copies; final cell alone changed. New combined SHA256 8d2b999b78cccb48bbed5d223b7c30d1e4b3bc9369abed06c870ba499d6a3b27. Repaired packaging record: CLASS_METADATA_REPAIR_PACKAGING.json; independent review: SAFE_CLASSES_INDEPENDENT_REVIEW.json. Historical pre-V3 source manifest is preserved. No repaired execution or numerical result yet.

- Repair current state: REPAIR_REVIEWED_PENDING_RETRY. Current source hashes are bound to CLASS_METADATA_REPAIR_PACKAGING.json; failed V3 remains tied to the original reviewed source and packaging commit 9db86f28cfd763ccbc3dcc048fb35fe649f6445f. Repaired last editor cell passed a complete 60,610-character roundtrip. Metadata-only QuickSave V4 requested but response/ID/saved source unconfirmed; no CPU retry called and zero new head fits. Frozen protocol and historical pre-V3 review/manifests remain unchanged.


## Stage 06D repaired execution launch — observed 17 October 2026 at 21:55:14 UTC

Historical launch observation; superseded by the terminal V7 outcome below.

- Fourteen-file repair and failure-record freeze committed before retry Save & Run All: de19608a672015fdb84e7921ed2aa1482c495a18. Metadata-only QuickSave V4 scriptVersionId 359502142 passed complete repaired five-cell source matching (77,806 / 156,105 / 156,106 / 146,225 / 60,610 characters), empty outputs/null counts, exact seven IDs and CPU/internet-off metadata.
- Actual Version 5 Save & Run All named `Fixed ensemble - safe class metadata`, with advanced CPU/no-accelerator selection, is Running with one active event at 2026-10-17T21:55:14Z. Numeric V5 scriptVersionId/logs are still being obtained; no new fitting count or result is inferred.
- Prior V3 359500151 failure remains immutable: failure marker20.7s, log end27.7s; UI initially0s later28s. It passed frozen style transfer and component audits but fitted no new heads. The repaired run must still pass metadata/reproduction checks and the unchanged primary raw screen. No official submission or new score; S02 remains0.549.


## Stage 06D V5 reproduction failure and source-path diagnostic revision

- Actual V5 359502730 completed all five fixed DeBERTa folds/15 heads, then failed at164.0s `Frozen 06C argmax mismatch, control` (platformtotal175.5s). Class metadata repair passed actual cache; no ensemble result and primary raw screen NOT evaluated. Full4,864-byte log SHAe6f5896056004ee4c6f8a133f1835c9315f70a888e47a7ba858157050d4fde4e and V5_FAILURE.json preserve the run. Original sourceb27925.../preexeccommitde19608... retained.
- Reviewed exact-C eligible-only inference override and pre-gate diagnostics sourceSHA c6005196a73100676141d18f67eb299c1bf686a547fbb077b91825d2bd2611b3. Call-shape hypothesis unproven; same fitted heads, no changed training/folds/seeds/groups/plain scaling/weights/gates. Old all-row path diagnostic only. All twelve exact argmax and score-atol1e-12 checks remain mandatory; no ensemble output if they fail. Full prediction/margin/runtime/15-head snapshots save before gate. Eight no-fit fixtures and independent AST review passed; new combinedSHA9185a208679ea99c318bb11feb9abd773661886523621e268052ca037e613893.
- State SOURCE_PATH_REPAIR_REVIEWED_PENDING_ONE_RETRY: final69,909-character editor cell paste/copyback and metadata-only V6 gate pending at freeze; no V6 ID guessed and no CPU attempt called. Persist reviewed code before one evidence-based CPU retry. If mismatch remains, retain diagnostics and stop blind refits; no gate relaxation. Official S02 remains0.549.


## Stage06D one source-path diagnostic CPU retry launched —17October2026 at22:05:08UTC

Historical launch observation; superseded by the terminal V7 outcome below.

- Pre-execution fourteen-file freeze15969f9cd696c7a4ce51be92680db16cbb42ea0d completed before Save & Run All. Metadata-only V6 359505487 full immutable export matched all5current cells including69,909-character final cell and all7pinned inputs with CPU/internet off.
- Actual V7 named `Fixed ensemble - source path diagnostics` is Running with1ActiveEvent at2026-10-17T22:05:08Z, Advanced CPU/no accelerator confirmed. NumericV7ID/logs pending. Save&RunAllcomplete UI toast means commit completed only; execution remains Running.
- One evidence-based retry is now used; source-path hypothesis remains unproven, strict12array/score checks unchanged. Preserve diagnostics and stop further blind fits if mismatch persists. Prior V3/V5 failures were technical and did not evaluate primary raw screen. No ensemble score or new official score; S02 remains0.549.


## Stage 06D V7 terminal technical reproduction failure — 359505815

- V7 completed all five CPU folds/15 heads, then failed the strict control argmax gate at 141.6 seconds; platform total 151.6 seconds, output 13.19 MB. The full 28,581-byte log SHA256 is c397e1eafde8fffd8335cd0f3262c6c11a678d967512201cd6d37a0b9c9ed492. No reproduction-pass or ensemble result appeared; the primary raw screen was not evaluated.
- Source-eligible and all-row paths had zero hard-label differences. Both differed from original C in 20 of 32,388 repeated head/context comparisons; this is not a unique passage count. Maximum log-probability delta was 3.774758283725532e-15. Call shape is ruled out for these hard-label differences; the remaining cause is unproven. Oracle one-response mismatches persist without centered evidence.
- Independent summary/log validation reconciles all 12 historical balanced accuracies, 60 fold summaries, original eligible counts, five markers/15 fits, 15 deprecations and zero convergence warnings. Full diagnostic NPZ remains Kaggle-only, reported SHA256 cf0ba543ab468329ee25cc6e01238dd04a8941f522ea61470a6b76f7435e86a5. No raw-array recomputation or unique mismatch identities are claimed.
- Terminal state: TECHNICAL_REPRO_FAILURE_NOVALID_ENSEMBLESCORE. Stop blind refits and retain strict gates. No official submission or new score; S02 remains 0.549. Evidence is preserved in KAGGLE_V7_FAILED.log, reproduction_diagnostics.json, V7_FAILURE.json and INDEPENDENT_V7_VALIDATION.json.

## Stage 06E prospective canonical CPU evidence plan

- Independently reviewed plans/STAGE_06E_CANONICAL_CPU_EVIDENCE.md, SHA256 89dbf531820b965f09fab07157475b199f929f4d297bd268e0f141c765a80f32. No genuine plan errors found and no plan edits made. Review: results/stage_06e/PLAN_REVIEW.json.
- Distinct prospective experiment: instrument the literal original C source without changing its computations; capture existing calls once; freeze a verified cache; consume it without refitting or regrouping. Historical C comparison remains separate and cannot retroactively pass Stage 06D. The fixed raw screen and contaminated-balancing limit remain unchanged.
- Implementation, fitting and model execution have not started. Original-to-instrumented diff, schema/consumer and immutable source/input/runtime checks require pre-execution review and freeze. No new model, GPU/API/paid service, official submission or final-selection change is included.

## Stage 06E implementation and pre-execution source gate — 2026-10-17T22:40:31.759823+00:00

- Literal C output-only capture, static-member probability preparation and pure cache-only ensemble consumer are implemented. Original C/D and frozen plans remain unchanged. No real fitting or inference occurred during implementation.
- Focused producer fixtures (22), consumer fixture groups (8) and independent contract tests (31) passed. Inert five-cell packaging and deterministic rebuild passed. Full review/hash inventory: results/stage_06e/INDEPENDENT_IMPLEMENTATION_REVIEW.json and PACKAGING.json.
- Private metadata-only V1 359514410 exactly matches all five reviewed source strings (77806/156105/156106/146225/149236 characters), seven pinned input IDs, CPU and internet off. All cells have null execution counts and zero outputs. Viewer render 8.3s, zero output files: no head fitting or ensemble evaluation.
- One unchanged CPU execution is planned after guarded remote freeze verification. Primary raw endpoint, weights, prompt bootstrap and screen preserved; balancing remains a contaminated diagnostic. Historical C mismatch remains unresolved. Official S02 remains 0.549. No official submission, final selection change, GPU or paid service.

## Stage 06E CPU execution launch — 2026-10-17T22:42:34.837888+00:00

- Pre-execution freeze ac9d9d4b0487d71b3265b1b2b0f2621803a8b701 verified before Save & Run All. Actual private V2 359514801 is Running with CPU/internet off; complete actual V2 source and seven inputs exactly match the reviewed package. Initial logs page shows 34.2s and 0 B; fitting not yet observed. Source proof: results/stage_06e/SAVED_SOURCE_V2.json. No official submission.


## Stage 06E terminal result — 17 October 2026

- Private V2 359514801 succeeded in 169.5 seconds on CPU, producing 38.58 MB. The pipeline completed at 159.0 log seconds. Five folds/fifteen heads, the same-run cache gate and pure consumer gate passed. Only fifteen deprecation fit warnings were recorded.
- Primary raw mean development BA: control 54.81224625% versus candidate 54.67669235%, change -0.13555390 pp; conditional prompt 95% interval [-0.51886251, +0.23747915] pp. The fixed primary screen was not met. Context 6 corrected 41/broke 52 (net −11); context 3 corrected 40/broke 40. Retain original control centering. The interval includes zero; no population-deterioration claim is made.
- Historical C comparison remains twenty mismatches in 32,388 repeated cells across twelve unique IDs; the cause is unresolved and D stays failed. The pure consumer fitted/inferred/regrouped zero times. Full log SHA256: a267176c4c5e6be764f48381a981fe29acf4c841f6371cc21afe8425d72e3278.
- Independent validation reconciled 216 recalls, sixty fold scores, paired counts and source/cache/runtime gates. Actual output-manifest hashes match local result/metric artifacts and three complete original-control confusion CSVs. Numeric NPZs remain Kaggle-only; no local raw-row/bootstrap-draw audit is claimed.
- Balanced diagnostic mean change +0.02502279 pp is potentially contaminated and cannot select the candidate. Best confirmed official S02 remains 0.549. Next: error analysis of the original control and a prospective complementary hypothesis. No further fits or official submissions were launched.

### Stage 06E aggregate error analysis

Three complete original-control confusion CSVs were recovered through the visible Kaggle preview and matched runtime output-manifest hashes exactly. Gemini 3.7/3.8 is the largest mutual confusion in every context (94 / 87 / 94); GLM 5.3/Flash is second (73 / 69 / 73). These contexts predict the same 2,699 passages, so counts are not additional independent examples. Next inspect saved row probabilities/margins and prompt distribution before choosing a correction. No additional fits were run. Analysis: results/stage_06e/ERROR_ANALYSIS.md and error_pairs.csv.

## Stage 06F fixed pair diagnostics — 2026-10-17 UTC

- Frozen diagnostic source and five-cell CPU/offline package passed independent source/method review and 22 synthetic no-fit fixtures. Saved Quick V1 359519886 matched all five complete cell sources with pins 170484/359514801, null execution counts and empty outputs. Initial Git freeze baee805c080a06c6d797e68c0a0c1a424756d3a1 preceded actual V2 359520233.
- V2 failed after 19.5s at the exact `output_manifest.json` byte-hash gate, before analysis. The local JSON-tree recovery had reversed per-entry field order relative to the original E writer. Its expected hash was wrong despite equal records. Failed source/log/status and original freeze/package remain preserved. No diagnostic or model result came from V2.
- A reviewed single-constant expected-hash repair retained every gate, passed 23 no-fit fixtures and was verified in saved Quick V3 359521070. Repair Git freeze 1c9f0650f01d2adc43731876f4fe7e8439f34f48 preceded the sole controlled retry. Actual V4 359521305 completed in **20.3s CPU**, 1.08MB output. Its saved source matched V3; runtime strict input/replay gates passed, including original manifest SHA a5fcffaef27f4942af8a679370c30c247b8d872b4b6d711eccc0101ea7006dbd. No normalization, allowlist or gate relaxation was used.
- Replayed original-control development mean remains 54.812246%; this is not a new score. Across the same 300 true-pair passages, cross-errors 94/87/94 total 275 repeated observations but 109 distinct passages; 77 fail in all three contexts, 12 in two, 20 in one. At least one component favors truth on 78/79/83 cross-errors; this is not a measured recovery. Exact top-two pair scope includes 68/62/71 cross-errors and 101/113/101 correct pair cases at risk of being broken.
- Independent terminal aggregate validation passed: seven recovered files exactly match runtime manifest hashes, count/support partitions and prior E scores. Numeric/row artifacts remain on immutable V4 output; local review does not independently recompute their row membership/margins/bootstrap draws. Fixed-prediction uncertainty is conditional and pair selection used reused development data.
- Next: one prospective deterministic equal-sum attention pair-choice hypothesis inside the raw-control exact-top-two pair scope, with no training or search. Require all 18-class raw balanced accuracy and corrected/broken counts; no outcome exists yet. Stage 06F performed zero fits, inference calls, regrouping or changed predictions, with no official submission, paid service or final-selection change. S02 remains **0.549 official**. Evidence: results/stage_06f/REPORT.md, FINDINGS.md, KAGGLE_RUN.json and INDEPENDENT_TERMINAL_VALIDATION.json.

## Stage 06G fixed attention-pair comparison — 2026-10-18 UTC

- Implemented the unchanged prospective plan in `scripts/stage06g_pair_complement.py`, SHA256 69b0c3fe996759c6bb37589c64a1257cf5d7794dbbc7a73cf5421cf5e8020af4. Fourteen independent synthetic contract fixtures and source/package review passed. The package reuses the unchanged strict F source gates; one equal-sum attention pair rule, no fitting, new inference, regrouping, balancing or search.
- Private Quick V1 359531282 was saved without execution. Full five-cell source, CPU/offline settings, pins 170484/359514801, null counts and empty outputs matched the reviewed package. Git freeze cbdd06c11bb62d72c04781bf7c75bceae08a59fb preceded the single actual V2 359531626. Its saved source also matched; it completed successfully in **22.5 seconds CPU**, 355.49 kB output, with all source/numerical gates passing.
- Full 18-class raw mean development BA changed **54.812246% to 54.664098%, -0.148148 percentage points**. Context gains were -0.148148, -0.333333 and +0.037037 points. Conditional 95% paired-prompt interval **[-0.641975,+0.358025] points** includes zero. The positive-mean and positive-lower-bound criteria fail; the context floor passes. **Reject this exact rule, retain control, and run no G variant.** This is a valid negative comparison, not a technical failure or proof of population harm.
- The rule fixed 82 and broke 94 repeated context decisions, with 18 nonpair decisions changing between two wrong Gemini labels. Total changes 194; net correct -12. Mean Gemini 3.7 recall rose 6.888889 points while Gemini 3.8 fell 9.555556. All sixteen other recalls, pair-entry masks and outside-scope predictions were unchanged. Contexts reuse the same passages, so these counts are not unique examples.
- Fourteen recovered artifacts exactly match runtime manifest bytes/hashes. Independent validation recalculated scores from all six confusion matrices and reproduced the 3,000-draw seed-20261017 prompt interval from 450 aggregate prompt rows. The saved notebook table/chart were visually inspected. Original binary arrays, chart PNG and status JSON remain on immutable Kaggle output; local review did not independently reread row IDs/native margins from NPZ.
- A separate bounded repository-provenance audit reconciles E's seven evidence branches with sixteen active branches in full S02. Verified own GLM cache is the strongest prepared missing-proxy lead; ELECTRA has less complete cached provenance. Both representations already contribute to S02. Closing proxy coverage would strengthen the evaluation reference and **does not itself demonstrate improvement above 54.9%**. A genuinely new competitive complement remains unidentified and untested; no follow-on experiment plan or candidate was created. Audit: `results/stage_06g/REPRESENTATION_COVERAGE.md`.
- Official S02 remains **0.549**. No official submission, final-selection change, GPU or paid service. Evidence: `results/stage_06g/REPORT.md`, `FINDINGS.md`, `KAGGLE_RUN.json`, `INDEPENDENT_TERMINAL_VALIDATION.json` and full `KAGGLE_V2.log`.


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

[Stage 08C](results/stage_08c/REPORT.md) freezes Nishant part-two V1 361311282 (author public 0.620), pins dataset V1, passes 85/85 real CPU input checks, and launches private GPU V2 361732127. The six model cells are unchanged; an independent validation-only final cell checks probabilities/IDs/parquet. Own preview and official score remain pending. Current public first is 0.700, tenth 0.660; our displayed current official best remains 0.549. No tuning, API purchases or final-selection changes.


### S03 submitted — 24 October 2026

Stage 08C private V2 completed in 739.2s; all 22 actual probability/ID/parquet checks passed. The exact version (361732127) was submitted as S03. The live official row displays Notebook Running, with no own score. S02 remains confirmed at 0.549. Final selections remain automatic at 0/2. The hourly read-only watch is enabled until S03 terminates. See results/stage_08c/EXECUTION_20261024.json and OUTPUT_VALIDATION.json. No model tuning or paid services.


## 24 October 2026 — S03 official completion and Stage 09 resumption

S03 / private Stage 08C V2 361732127 displayed Succeeded and official public balanced accuracy 0.620 in the connected account, independently visible in submission details and the account leaderboard. Improvement versus S02: +7.1 percentage points; rank 377 at verification. The three-row preview remains an execution milestone. Final private score is unknown. The completed watch is disabled.

User resumed work after this milestone. The next experiment is a bounded matched development pilot, declared in `results/stage_09/PROTOCOL.md`: untouched DeBERTa-base, masked-mean/linear head and separate learning rates in both arms; candidate-only two-epoch pretraining on the 750 source-traced direct-OpenAI V4 rows, followed by identical three-epoch official fold-1 training. CPU admission passed. Actual GPU execution status is recorded separately; no Stage 09 result or official submission is implied here.


## 24 October 2026 — Stage 09 final result verified

Private V1 361761811 succeeded in1453.1seconds. Both declared arms completed on the same540held-out official responses/30prompts: official-only control29.444444% development balanced accuracy, candidate24.259259%, difference -5.185185percentage points, conditional paired prompt95%interval[-8.703704,-1.851852]. Corrected37/broken65; all94external and405officialupdates perarm completed withzeroAMPoverflow. The fixed pilot screen failed. Reject this exact candidate; no automatic variants or official submission. S03remains0.620official public balanced accuracy. Aggregate recalculation and complete result saved underresults/stage_09.


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


## Stage 12A complete; Stage 12 input gate — 25 October 2026

The prospectively frozen CPU component screen succeeded in **58.99s**. Source-style four-view SVC development balanced accuracy was **37.8931%** versus **40.0425%** with the fixed topic correction, **+2.1494 percentage points**, conditional paired-prompt 95% interval **[+1.0380,+3.2593]**. All four fixed gates pass; all five folds improve; 146 responses corrected and 88 broken, largest class recall loss 4 points, zero convergence warnings. Independent terminal validation passes 37/37. This admits further ensemble validation, not a measured gain to full S05 or an official score. The pre-fit freeze was **4f08c924c05636b6b06eada719f1a1f517816de6**; row artifacts remain outside public Git. See [Stage12A report](results/stage_12a/REPORT.md).

Actual Stage12 CPU V3 **362258945** succeeded in **373.9s** with **70/71** input checks. Only the required Meta Llama3.1 mount is absent. The model page requires a Meta access/licence/AUP step and subsequent Kaggle verification; no such acceptance or form submission occurred. Four V2 auditor issues were corrected without changing the source recipe, with actual numeric checks passing. All 37 static head schemas and 15 custom-source file hashes were reconciled. No GPU/model run or official submission was made for the source-author 0.651 recipe. See [Stage12 report](results/stage_12/REPORT.md).

## Stage 12B — one fixed correction; S06 accepted — 25 October 2026

Using the established S05 assets, froze one counterfactual before evaluation at **2c778522775d6b34e2efa387bee42fd420775c42**: activate the published 0.62 SVC topic correction once after the decoder/DeBERTa sum and before complement/ATLAS. Other 32 original cells and parent guard34 remain exact; a new guard35 replays the private runtime evidence, reconstructs the uncorrected sum and links the correction into the actual pre-ATLAS mixture. Independent build review passes 23/23; 16 synthetic fixtures pass. Actual saved V3 and V4 each match all 35 reviewed code cells and all 28 parent input-version IDs.

Final private **V4 362263151** succeeded in **20.3s** with actual T4x2 allocation profile and persistent GPU save policy. All 25 parent and 16 new format checks pass, explicitly **FORMAT_ONLY_PREVIEW_INTERVENTION_NOT_EXECUTED**. Submitted this exact V4 as **S06**, now **Notebook Running with no displayed score**. The component gain does not estimate a full-ensemble gain; one blind official experiment is declared, with no automatic variants. Our confirmed best remains **S05 0.642 official public balanced accuracy**. A read-only hourly terminal-result watch is enabled. Final selections remain S03/S04 at 2/2, unchanged; no paid service. See [Stage12B report](results/stage_12b/REPORT.md) and [submission receipt](results/stage_12b/S06_SUBMISSION_RECEIPT.json).


## S06 official completion verified — 26 October 2026

Our S06 / Stage12B V4 **362263151** displays **Succeeded** and **0.641 official public balanced accuracy (64.1%)** on the connected-account Kaggle submissions page. Submission Details separately confirms the exact version and Score: 0.641. The same page displays S05 at **0.642 (64.2%)**. Exact decimal arithmetic gives **-0.1 percentage points** for S06 versus S05; S05 remains the benchmark, with **5.8 points** to the 70% stretch goal.

The fixed SVC correction did not improve the official public score. Retire this exact intervention under its declared one-test policy; no automatic coefficient, placement or blend variants. The small difference alone does not establish statistically certain harm or private-leaderboard performance. Stage12A's +2.1494-point standalone development gain did not establish value in the complete ensemble.

Submission Details exposes no separate hidden-run log link. Hidden runtime, detailed guard counts and replay artifacts remain unavailable; the 20.3-second format preview is not substituted for those logs. The watch is completed and disabled, with the terminal outcome reported. The fresh page shows automatic final selection **0/2**, unlike an earlier recorded 2/2 observation; no selection controls were changed in this check.

The next admitted route is the already frozen Stage12 public-source recipe after resolving Meta Llama access and repeating CPU admission. The model page still displays the Meta access/licence prerequisite on 26 October; no acceptance, access request, GPU run or new submission occurred. Author 0.651 is not our score. See [official receipt](results/stage_12b/OFFICIAL_RESULT_20261026.json) and [next-step decision](plans/NEXT_AFTER_S06_20261026.md).

## Stage12 access and CPU repeat — 29 October 2026

29 October 2026: Hugging Face explicitly displays model access granted. The exact Kaggle Transformers/8b/2 page, signed in as Watchman 777, no longer shows the pending-review warning and enables Download and Code. This clears the account/UI access gate; runtime mounting is not yet established.

The unchanged one-cell CPU auditor was verified against Git, with code-cell SHA256 `413925bdeea88678d22fb0c5c4c868678e28664e221aef9171acbafb2c95b0b2` (96,131 characters), CPU None and Internet off. Save & Run All was requested with intended label **V4: repeat CPU audit after Llama access**. The save dialog closed and Active Events expanded, but the subsequent connected-browser status read timed out. **No immutable V4 ID, queued/running state or new result has been verified.** Last verified actual audit remains V3 70/71.

Recover by inspecting existing saved Versions/Active Events before considering any retry; do not launch a duplicate. Require the new actual complete CPU contract pass, reverify the saved 31 input versions, inspect the Llama configuration and recheck free GPU quota/runtime before admitting the already-reviewed seven-cell private production notebook. No GPU execution, new official submission, final-selection change, paid service or model substitution occurred. S05 remains our best official public balanced accuracy at 0.642; author 0.651 is external.

## Stage12 V4 actual input audit, V5 repeat and production snapshot — 29 October 2026

The actual Llama mount is now verified: V4 selected `models/metaresearch/llama-3.1/transformers/8b/2`, with `model_type=llama`, hidden size4096,32layers, four safetensors shards, tokenizer/index checks and compatible Llama adapter projections. Large weight payloads were not fully hashed or executed.

CPU **V4 363518868** completed in **1012.9seconds**, technically Succeeded, but its contract was **62/71**. Every one of the nine failed late checks was an audit-deadline `TimeoutError`; all53mainselectors and37staticTorchschemas passed. This is an incomplete inspection, not nine invalid assets or an accuracy result.

The only change for CPU **V5 363597773**, now **Running**, is the outer wrapper's inspection deadline argument from900 to1800seconds. The embedded auditor SHA, checks, selectors and31inputversions remain unchanged. The frozen auditor's legacy timeout text still says900; its report records the actual budget. The source was committed before execution in `f58c07cf754fa2a57dc260cc4971b49b01ab71dd`. V5 has no final contract result yet.

Private **Quick V6 363598812** is the prepared production source/settings snapshot: six exact upstream cells plus the separate output guard, all seven byte-exact and all31copiedinputversions unchanged. It has not executed the models. T4x2 is selected with Internet off; free quota was freshly displayed as29hours remaining, and the competition overview still specifies a9-hour maximum. The saved document's historical `isGpuEnabled=false` field is retained as observed; realGPUadmission requires the save-time **Run with GPU for all sessions** setting and actual committed T4x2/device logs.

Require a fresh actual **71/71** complete CPU contract pass before starting the unchanged production run. Inspect all model-group/COMP/ATLAS logs, guard output and real saved-file readback. Source-allowed `lora24_public` absence remains allowed. No GPU inference, new official submission or final-selection change has occurred. Our best official public balanced accuracy remains **S05 0.642**; upstream0.651 is external, and no development or hidden-runtime gain has been established.

## Stage 12 dataset version-pin correction — 29 October 2026

Llama access and the actual mounted model are resolved. CPU **V5 363597773** completed in **670.9 seconds** with **71/71 structural checks**, all 37 static Torch head schemas, all 15 custom source paths matching V3, and 1,680 Llama adapter/base projection checks. This is input inspection, not model inference or an accuracy score.

An independent content comparison found three consumed CSVs had grown since CPU V3: Abdullah 1,033→2,503 rows, Lucas 5,452→7,714, and Prvsiyan 4,926→5,070. Their deterministic full-file hashes changed; the other 247 shared critical-file hashes matched. **The earlier claim that matching saved input IDs established frozen versions was too strong.** All 14 dataset dialogs actually had **Pin to latest version** selected. The V5 structural pass therefore did not admit an unchanged-input reproduction.

All 14 datasets now have explicit version selections verified after reloading the editor. The three changed sources were set to Abdullah V2, Lucas V8 and Prvsiyan V6, selected to recover the V3 corpora. The remaining 11 were pinned to the current versions whose critical content matched V3. Private source-only **Quick V7 363602111** preserves all seven reviewed production code cells. These UI bindings still require actual runtime hash verification.

CPU **V8 363603066 is Running** on accelerator **None**, Internet off, using the same 1,800-second auditor wrapper. Its source and all three critical pins were rechecked after notebook import. Require **71/71 plus equality of all 250 shared V3 critical hashes**, including the restored CSV row counts, before any GPU launch. The auditor embeds the old metadata-ID snapshot; the explicit UI pin manifest and real content hashes control current version reconciliation.

The unchanged GPU recipe remains prepared. Free GPU quota was freshly displayed as 29 hours, and the competition still specifies a 9-hour runtime limit. Actual GPU execution must use T4 x2 and the save-time **Run with GPU for all sessions** policy; then verify the committed accelerator profile and worker `gpus 2`, `smoke False` logs. **No GPU model execution or new official submission has started.** S05 remains our best official public balanced accuracy at **0.642**. The source author's 0.651 is external; original-author input-version parity is unverified.

## Stage 12 CPU V8 admission and GPU preparation — 29 October 2026

The Llama access and dataset-binding issues are resolved at the CPU input/content gate. **CPU V8 363603066 succeeded in 297.9 seconds with 71/71 checks and 20/20 independent review checks.** All 15 custom source paths/seven unique hashes, all 37 static Torch head schemas, and 1,680 Llama adapter/base projection checks pass.

All **250** critical content hashes previously recorded in our V3 audit match after one explicitly reviewed mount relocation: Lucas `gen_passages.csv` moved from `datasets/lucasboesen/ntl-free-proxy-generations/` to `ntl-free-proxy-generations/`. Its full SHA256 and 5,452 rows are identical; the unchanged source recursively selects that filename and there is exactly one match. No broad path normalization or hash exception was used. The only five additional hashes are newly available Llama metadata. All five CSV inventory records match after that mapping.

All 14 dataset bindings were explicitly pinned and verified after reloading. The three grown CSV sources were restored to Abdullah V2 (1,033 rows), Lucas V8 (5,452), and Prvsiyan V6 (4,926). **Earlier matching saved metadata IDs did not establish effective version pins**: the dialogs had selected Pin to latest version. That earlier freeze claim is corrected. The auditor's embedded old UI-ID snapshot is historical; explicit pin controls and actual runtime hashes establish current reconciliation.

Private source-only **Quick V7 363602111** preserves the six reviewed upstream code cells plus the separate output guard. The seven-cell notebook was restored again after V8; the three critical dataset pins still persisted. A final source-only QuickSave was requested with intended Version 9, but browser commands subsequently stalled. Its immutable identity and final source/configuration readback remain **unverified**. **No GPU Save & Run launch was requested and no model inference has started.** Inspect existing versions first; do not create a duplicate snapshot or confuse a QuickSave with inference.

Next: confirm the latest saved production snapshot has seven exact code cells and the explicit pin bindings; select T4 x2, Internet off and **Run with GPU for all sessions** at Save & Run. Verify the committed accelerator profile and worker `gpus 2`, `smoke False` startup. Free quota was last freshly displayed as 29 hours and the competition specifies a 9-hour runtime limit; recheck if the session changes. CPU admission does not prove full vendor payload finiteness, GPU behavior, hidden runtime or accuracy. Original-author input-version/content parity remains unverified. Our best official public balanced accuracy remains **S05 0.642**; upstream 0.651 is external.


## Stage 12 — private unchanged GPU V10 launch — 29 October 2026

The unchanged private GPU execution **V10 363626171** has been launched once from the restored production notebook, using **Save & Run All (Commit)** with **Run with GPU for all sessions** explicitly selected. The authoritative committed viewer reports **Running**, **GPU T4x2**, **Latest Container Image**, and **0 B output** at the first 17.9-second observation. At the latest **95.9-second** observation the job was still Running with the same committed profile and 0 B output. Logs contained initial debugger messages and a later line containing `0`; neither worker `gpus 2` nor `smoke False` had appeared. The saved V10 source was independently read back: all seven code strings/hashes match the canonical notebook and V9, all 31 data-source entries match V9 exactly, and Internet is disabled. Its legacy `isGpuEnabled=false` document field was retained as observed and is not authoritative for actual allocation. This proves that the committed GPU job started; actual worker device counts and model-group execution still require live log confirmation. It is a private notebook preview, not an official submission or an accuracy result.

Browser recovery first established that the earlier source-only save did succeed: **Quick V9 363604858** contains all seven exact reviewed production code cells. Its 31 saved resource metadata entries match pinned Quick V7, with Internet disabled and accelerator configuration `nvidiaTeslaT4`. No duplicate source snapshot was created. CPU **V8 363603066** remains the admitted input/content result: **71/71** checks and **20/20** independent review checks, with all 14 dataset pins verified and all 250 recorded V3 critical hashes matching after the one explicitly reviewed byte-identical mount relocation. Source metadata parity does not itself establish content pinning; the earlier actual pin controls and V8 runtime hashes supply that evidence.

One environment discrepancy is unresolved: the editor displayed the original **2026-10-22** container setting, whereas the actual V10 viewer reports **Latest Container Image**. Record this as observed environment drift; do not describe V10 as an exact environment reproduction. No conclusion about its numerical effect is available.

Next: inspect V10 actual logs for worker `gpus 2` and `smoke False`, each expected model group (`q14v5`, `q8v5`, `ll8v5`, `n14b`, `q25_14`, `deb`), complement/ATLAS loading, failures, final output guard and real saved-parquet readback. Preserve the source-allowed `lora24_public` skip. Free quota before launch displayed **00:51 / 30 hrs** used; the prior whole-hour display was 29 hours remaining. Do not promise a completion time. The competition's recorded GPU limit is nine hours.

**No new official submission or score exists. S05 remains our best official public balanced accuracy at 0.642.** The source author's 0.651 is external. CPU checks and configured/allocated hardware do not establish completed model coverage, full vendor-payload finiteness, hidden-test runtime or accuracy; original-author input-version/content parity remains unverified. No paid service or final-selection change was used.

[V10 live logs](https://www.kaggle.com/code/watchman777/name-that-llm-stage-12-public-651-audit/log?scriptVersionId=363626171) · [verified Quick V9](https://www.kaggle.com/code/watchman777/name-that-llm-stage-12-public-651-audit?scriptVersionId=363604858) · [GPU run receipt](results/stage_12/GPU_V10_RUN_RECEIPT.json)



## S07 — completed unchanged Stage12 reproduction — verified 30 October 2026

**S07 succeeded with our own official public balanced accuracy of 0.651 (65.1%)**, freshly verified on 30 October 2026 from the exact submissions row for private Stage12 **V10 363626171**. This is a new best: **+0.900 percentage points** versus S05 0.642, using exact decimal arithmetic. The public-score gap to 70.0% is **4.900 points**. The matching source-author 0.651 remains a separate external result; our score is now independently displayed by Kaggle.

Fresh public-leaderboard snapshot on **30 October 2026**: Watchman 777 is **713 of 3,967**, score **0.651**, seven entries. The displayed leader is **Lian Matsuo, 0.736** and tenth is **Wilmer E. Henao, 0.692**. Exact-decimal gaps are **8.500 percentage points** to the leader and **4.100 points** to tenth. **70.0% would still be below the current 73.6% leader**; it is not a first-place target at this snapshot. These are current public standings, not final private placement or guaranteed future cutoffs. [Live leaderboard](https://www.kaggle.com/competitions/name-that-llm/leaderboard).

The official hidden-test submission is terminal **Succeeded**. No actual hidden-run duration, final logs or guard count is asserted; **12224.2 seconds and 40/40 checks belong to the three-row private preview**. Final private evaluation and final rank are unknown. The final-selection display remains automatic **0/2**, and no controls, code, inputs, parameters or paid services were changed by this result check. No new run or submission was launched.

Next: preserve S07 as the current best and record this verified outcome. Do not infer a route to 70% from the public-score gain or start a parameter/blend sweep from this result. Any further experiment needs its own prospective rationale and development evidence. The successful public reproduction is not, by itself, an original-method or publication claim.

[Final result receipt](results/submissions/S07_FINAL_RESULT.json)

