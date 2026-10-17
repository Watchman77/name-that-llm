# Experiment log

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
