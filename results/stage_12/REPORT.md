# Stage 12 — private GPU V10 technical preview succeeded

## Current result — 30 October 2026

**Private V10 363626171 succeeded in 12224.2 seconds (3h 23m 44.2s)** on GPU T4 x2, verified after refreshing the authoritative logs page. Output display: 876.62 MB. The three-row execution is real inference. All **40 unique saved-output checks** pass, including actual saved-parquet readback and finite 3×18 complementary logits.

Manual actual-log review confirms all six main groups (`q14v5`, `q8v5`, `ll8v5`, `n14b`, `q25_14`, `deb`) completed, `COMP STACK DONE` and successful loading appeared, and ATLAS ran on 3/3 rows and mixed at 0.1. All 40 adapter completion logs report zero bf16 redo batches. The only missing-checkpoint skip is the predeclared source-allowed `lora24_public`. No inspected traceback, exception, failure, COMP/ATLAS fallback, or nonfinite marker appeared. Known unexpected-weight reports match the intentionally discarded layer tails and unused causal-LM heads in the reviewed source; they are not by themselves a blocker. Debugger and notebook-rendering warnings are recorded separately from model completion.

**Admitted: completed technical preview and saved-output contract.** This does not establish full-main internal finiteness, independent mixture/argmax reconstruction, hidden-test runtime, or accuracy. The small-input sanity checks use seen training rows. Original-author input-content parity and the original/latest-container label discrepancy remain unverified. No recipe changes or paid services were used.

A fresh official submissions-page check shows only **S01–S06**, with **S05 0.642** still best; V10 has not been submitted and has no official score. Automatic final selection remains 0/2 and no selection controls were changed. Next is one unchanged official hidden-test rerun; do not infer its score from the author's 0.651 or from this successful preview.

[V10 final logs](https://www.kaggle.com/code/watchman777/name-that-llm-stage-12-public-651-audit/log?scriptVersionId=363626171) · [actual saved-output guard](GPU_V10_ACTUAL_OUTPUT_VALIDATION.json) · [final technical review](GPU_V10_FINAL_REVIEW.json)


## Historical progress check — 29 October 2026

V10 remains **Running at 3466.1 seconds (about 58 minutes)** on T4 x2. Actual worker logs at 464.0s confirm **`gpus 2` and `smoke False`**. Nine component completion markers have appeared: stylegram, stylometry_lgb, debv3l, electl, desk, lora24_seed7, lora24_claude, lora20_q25_instyle and tokattn_deb. Later weight-loader progress reached 220/220; this does not identify or prove completion of the remaining model groups. No inspected traceback/error/failure or COMP/ATLAS skip marker appeared. The viewer still shows 0 B; no final guard, saved submission or readback is verified. Full ensemble completion remains pending.

The 54-row self-tests explicitly use seen training data. Their displayed 1.00 values are **not validation results or an accuracy gain**. No official submission was made, and no new official score exists; our best confirmed public balanced accuracy remains **S05 0.642**. The recorded container-label discrepancy remains unresolved.

## Historical launch checkpoint — 29 October 2026

The unchanged private GPU execution **V10 363626171** has been launched once from the restored production notebook, using **Save & Run All (Commit)** with **Run with GPU for all sessions** explicitly selected. The authoritative committed viewer reports **Running**, **GPU T4x2**, **Latest Container Image**, and **0 B output** at the first 17.9-second observation. At the latest **95.9-second** observation the job was still Running with the same committed profile and 0 B output. Logs contained initial debugger messages and a later line containing `0`; neither worker `gpus 2` nor `smoke False` had appeared. The saved V10 source was independently read back: all seven code strings/hashes match the canonical notebook and V9, all 31 data-source entries match V9 exactly, and Internet is disabled. Its legacy `isGpuEnabled=false` document field was retained as observed and is not authoritative for actual allocation. This proves that the committed GPU job started; actual worker device counts and model-group execution still require live log confirmation. It is a private notebook preview, not an official submission or an accuracy result.

Browser recovery first established that the earlier source-only save did succeed: **Quick V9 363604858** contains all seven exact reviewed production code cells. Its 31 saved resource metadata entries match pinned Quick V7, with Internet disabled and accelerator configuration `nvidiaTeslaT4`. No duplicate source snapshot was created. CPU **V8 363603066** remains the admitted input/content result: **71/71** checks and **20/20** independent review checks, with all 14 dataset pins verified and all 250 recorded V3 critical hashes matching after the one explicitly reviewed byte-identical mount relocation. Source metadata parity does not itself establish content pinning; the earlier actual pin controls and V8 runtime hashes supply that evidence.

One environment discrepancy is unresolved: the editor displayed the original **2026-10-22** container setting, whereas the actual V10 viewer reports **Latest Container Image**. Record this as observed environment drift; do not describe V10 as an exact environment reproduction. No conclusion about its numerical effect is available.

Next: inspect V10 actual logs for worker `gpus 2` and `smoke False`, each expected model group (`q14v5`, `q8v5`, `ll8v5`, `n14b`, `q25_14`, `deb`), complement/ATLAS loading, failures, final output guard and real saved-parquet readback. Preserve the source-allowed `lora24_public` skip. Free quota before launch displayed **00:51 / 30 hrs** used; the prior whole-hour display was 29 hours remaining. Do not promise a completion time. The competition's recorded GPU limit is nine hours.

**No new official submission or score exists. S05 remains our best official public balanced accuracy at 0.642.** The source author's 0.651 is external. CPU checks and configured/allocated hardware do not establish completed model coverage, full vendor-payload finiteness, hidden-test runtime or accuracy; original-author input-version/content parity remains unverified. No paid service or final-selection change was used.

[V10 live logs](https://www.kaggle.com/code/watchman777/name-that-llm-stage-12-public-651-audit/log?scriptVersionId=363626171) · [verified Quick V9](https://www.kaggle.com/code/watchman777/name-that-llm-stage-12-public-651-audit?scriptVersionId=363604858) · [GPU run receipt](GPU_V10_RUN_RECEIPT.json)

## Historical checkpoint — before browser recovery on 29 October 2026

The Llama access and dataset-binding issues are resolved at the CPU input/content gate. **CPU V8 363603066 succeeded in 297.9 seconds with 71/71 checks and 20/20 independent review checks.** All 15 custom source paths/seven unique hashes, all 37 static Torch head schemas, and 1,680 Llama adapter/base projection checks pass.

All **250** critical content hashes previously recorded in our V3 audit match after one explicitly reviewed mount relocation: Lucas `gen_passages.csv` moved from `datasets/lucasboesen/ntl-free-proxy-generations/` to `ntl-free-proxy-generations/`. Its full SHA256 and 5,452 rows are identical; the unchanged source recursively selects that filename and there is exactly one match. No broad path normalization or hash exception was used. The only five additional hashes are newly available Llama metadata. All five CSV inventory records match after that mapping.

All 14 dataset bindings were explicitly pinned and verified after reloading. The three grown CSV sources were restored to Abdullah V2 (1,033 rows), Lucas V8 (5,452), and Prvsiyan V6 (4,926). **Earlier matching saved metadata IDs did not establish effective version pins**: the dialogs had selected Pin to latest version. That earlier freeze claim is corrected. The auditor's embedded old UI-ID snapshot is historical; explicit pin controls and actual runtime hashes establish current reconciliation.

Private source-only **Quick V7 363602111** preserves the six reviewed upstream code cells plus the separate output guard. The seven-cell notebook was restored again after V8; the three critical dataset pins still persisted. A final source-only QuickSave was requested with intended Version 9, but browser commands subsequently stalled. Its immutable identity and final source/configuration readback remain **unverified**. **No GPU Save & Run launch was requested and no model inference has started.** Inspect existing versions first; do not create a duplicate snapshot or confuse a QuickSave with inference.

Next: confirm the latest saved production snapshot has seven exact code cells and the explicit pin bindings; select T4 x2, Internet off and **Run with GPU for all sessions** at Save & Run. Verify the committed accelerator profile and worker `gpus 2`, `smoke False` startup. Free quota was last freshly displayed as 29 hours and the competition specifies a 9-hour runtime limit; recheck if the session changes. CPU admission does not prove full vendor payload finiteness, GPU behavior, hidden runtime or accuracy. Original-author input-version/content parity remains unverified. Our best official public balanced accuracy remains **S05 0.642**; upstream 0.651 is external.

[CPU V8 result](CPU_V8_RUN_RECEIPT.json) · [independent review](CPU_V8_INDEPENDENT_ACTUAL_REVIEW.json) · [verified pin manifest](DATASET_PIN_RECONCILIATION_20261029.json) · [full compressed audit](REAL_INPUT_AUDIT_V8.json.gz)

## Historical checkpoints — superseded by the current result

### Earlier verification checkpoint

Llama access and the actual mounted model are resolved. CPU **V5 363597773** completed in **670.9 seconds** with **71/71 structural checks**, all 37 static Torch head schemas, all 15 custom source paths matching V3, and 1,680 Llama adapter/base projection checks. This is input inspection, not model inference or an accuracy score.

An independent content comparison found three consumed CSVs had grown since CPU V3: Abdullah 1,033→2,503 rows, Lucas 5,452→7,714, and Prvsiyan 4,926→5,070. Their deterministic full-file hashes changed; the other 247 shared critical-file hashes matched. **The earlier claim that matching saved input IDs established frozen versions was too strong.** All 14 dataset dialogs actually had **Pin to latest version** selected. The V5 structural pass therefore did not admit an unchanged-input reproduction.

All 14 datasets now have explicit version selections verified after reloading the editor. The three changed sources were set to Abdullah V2, Lucas V8 and Prvsiyan V6, selected to recover the V3 corpora. The remaining 11 were pinned to the current versions whose critical content matched V3. Private source-only **Quick V7 363602111** preserves all seven reviewed production code cells. These UI bindings still require actual runtime hash verification.

CPU **V8 363603066 is Running** on accelerator **None**, Internet off, using the same 1,800-second auditor wrapper. Its source and all three critical pins were rechecked after notebook import. Require **71/71 plus equality of all 250 shared V3 critical hashes**, including the restored CSV row counts, before any GPU launch. The auditor embeds the old metadata-ID snapshot; the explicit UI pin manifest and real content hashes control current version reconciliation.

The unchanged GPU recipe remains prepared. Free GPU quota was freshly displayed as 29 hours, and the competition still specifies a 9-hour runtime limit. Actual GPU execution must use T4 x2 and the save-time **Run with GPU for all sessions** policy; then verify the committed accelerator profile and worker `gpus 2`, `smoke False` logs. **No GPU model execution or new official submission has started.** S05 remains our best official public balanced accuracy at **0.642**. The source author's 0.651 is external; original-author input-version parity is unverified.

[CPU V5 result](CPU_V5_RUN_RECEIPT.json) · [independent content review](CPU_V5_INDEPENDENT_ACTUAL_REVIEW.json) · [verified pin manifest](DATASET_PIN_RECONCILIATION_20261029.json) · [CPU V8 run](CPU_V8_RUN_RECEIPT.json)

## Historical checkpoints — earlier 29 October 2026

### Earlier status; superseded above

The actual Llama mount is now verified: V4 selected `models/metaresearch/llama-3.1/transformers/8b/2`, with `model_type=llama`, hidden size4096,32layers, four safetensors shards, tokenizer/index checks and compatible Llama adapter projections. Large weight payloads were not fully hashed or executed.

CPU **V4 363518868** completed in **1012.9seconds**, technically Succeeded, but its contract was **62/71**. Every one of the nine failed late checks was an audit-deadline `TimeoutError`; all53mainselectors and37staticTorchschemas passed. This is an incomplete inspection, not nine invalid assets or an accuracy result.

The only change for CPU **V5 363597773**, now **Running**, is the outer wrapper's inspection deadline argument from900 to1800seconds. The embedded auditor SHA, checks, selectors and31inputversions remain unchanged. The frozen auditor's legacy timeout text still says900; its report records the actual budget. The source was committed before execution in `f58c07cf754fa2a57dc260cc4971b49b01ab71dd`. V5 has no final contract result yet.

Private **Quick V6 363598812** is the prepared production source/settings snapshot: six exact upstream cells plus the separate output guard, all seven byte-exact and all31copiedinputversions unchanged. It has not executed the models. T4x2 is selected with Internet off; free quota was freshly displayed as29hours remaining, and the competition overview still specifies a9-hour maximum. The saved document's historical `isGpuEnabled=false` field is retained as observed; realGPUadmission requires the save-time **Run with GPU for all sessions** setting and actual committed T4x2/device logs.

Require a fresh actual **71/71** complete CPU contract pass before starting the unchanged production run. Inspect all model-group/COMP/ATLAS logs, guard output and real saved-file readback. Source-allowed `lora24_public` absence remains allowed. No GPU inference, new official submission or final-selection change has occurred. Our best official public balanced accuracy remains **S05 0.642**; upstream0.651 is external, and no development or hidden-runtime gain has been established.

[CPU V4 receipt](CPU_V4_RUN_RECEIPT.json) · [V4 verified report](REAL_INPUT_AUDIT_V4.json.gz) · [V5 run receipt](CPU_V5_RUN_RECEIPT.json) · [audit budget amendment](CPU_V5_BUDGET_AMENDMENT.json) · [production Quick V6 parity](PRODUCTION_QUICK_V6_RECEIPT.json)

## Historical access/connection checkpoint — earlier29October2026

### Earlier unresolved status

29 October 2026: Hugging Face explicitly displays model access granted. The exact Kaggle Transformers/8b/2 page, signed in as Watchman 777, no longer shows the pending-review warning and enables Download and Code. This clears the account/UI access gate; runtime mounting is not yet established.

The unchanged one-cell CPU auditor was verified against Git, with code-cell SHA256 `413925bdeea88678d22fb0c5c4c868678e28664e221aef9171acbafb2c95b0b2` (96,131 characters), CPU None and Internet off. Save & Run All was requested with intended label **V4: repeat CPU audit after Llama access**. The save dialog closed and Active Events expanded, but the subsequent connected-browser status read timed out. **No immutable V4 ID, queued/running state or new result has been verified.** Last verified actual audit remains V3 70/71.

Recover by inspecting existing saved Versions/Active Events before considering any retry; do not launch a duplicate. Require the new actual complete CPU contract pass, reverify the saved 31 input versions, inspect the Llama configuration and recheck free GPU quota/runtime before admitting the already-reviewed seven-cell private production notebook. No GPU execution, new official submission, final-selection change, paid service or model substitution occurred. S05 remains our best official public balanced accuracy at 0.642; author 0.651 is external.

[Access verification](ACCESS_VERIFICATION_20261029.json) · [CPU repeat attempt](CPU_V4_ATTEMPT_20261029.json)

## Historical CPU V3 checkpoint — 25 October 2026

The exact Dhruv Tibarewal V1 **362013743** candidate remains blocked before GPU execution. Its author displays **0.651 public balanced accuracy**; our confirmed best remains **S05 0.642**. We have no official score for this candidate.

The real CPU audit **V3 362258945** technically succeeded in **373.9 seconds** with accelerator **None**. Its input contract passed **70/71 checks**. The sole failure is `main_BL`: the source-required Meta Llama3.1 model is not mounted. An account access/licence step remains unresolved. No model substitution, access bypass, legal acceptance, GPU inference or official submission was performed for this candidate.

Four earlier audit constraints are now resolved on the actual mounts. Complete-path string sorting correctly chooses the known **Qwen3-8B Base** config before Chat, matching source semantics. The seed7, Claude and instyle LoRA anchor archives contain unused object-typed `ids` members; only their headers were inventoried. Their numeric `features` arrays and all source-consumed readout arrays passed finite-value, shape and class/layer checks without unpickling. All **37 Torch head archives** received static tensor-schema inspection. This does not prove model execution or numerical payload behavior.

All **15 exported ATLAS/UltraBERT source files**, representing **seven unique hashes**, are unchanged from our actual Stage11 audit. The author preview explicitly skips `lora24_public`, so our identical source-allowed absence is not a new missing component. Earlier V2 evidence and the original model source remain preserved; only the independent auditor was corrected.

The complete V3 report is **448,138 bytes**, verified through **11 compressed transport chunks**, with raw SHA256 `696f25fe7d92682600258c66774e317c45756cdc11e3dfd9b7566997981da979`. Actual schema evidence and its limitations are recorded below. GPU behavior, hidden-test runtime and original-author input-version parity remain unverified.

[Exact CPU V3](https://www.kaggle.com/code/watchman777/name-that-llm-stage-12-public-651-audit?scriptVersionId=362258945) · [pinned author source](https://www.kaggle.com/code/dhruvtibarewal/nlm-v27?scriptVersionId=362013743) · [exact compressed report](REAL_INPUT_AUDIT_V3.json.gz) · [independent actual review](CPU_V3_ACTUAL_REVIEW.json) · [V2 diagnosis](CPU_V2_DIAGNOSIS_REVIEW.json) · [V3 auditor admission](CPU_V3_CORRECTION_REVIEW.json).

The separately prepared Stage12B experiment reuses the established S05 assets; it does not use this blocked 31-input/Llama recipe. Its status and results belong to its own records.

