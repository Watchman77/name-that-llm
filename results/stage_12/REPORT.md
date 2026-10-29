# Stage12 — Llama mounted; longer CPU audit running

## Current checkpoint — 29 October 2026

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
