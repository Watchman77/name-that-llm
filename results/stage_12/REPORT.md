# Stage 12 — access unlocked; repeated CPU audit status unverified

## Current checkpoint — 29 October 2026

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
