# Stage 11 — input and runtime audit before another GPU run

## Decision — 25 October 2026

Preserve S04 at **0.639 official public balanced accuracy**. The next bounded action is an independent CPU audit of the exact assets for [Vladislav Lavrentev V28, 362014737](https://www.kaggle.com/code/lightningv08/0-642-lora-llama-qwen3-8-27b?scriptVersionId=362014737). The live source page displays **0.642**, both as current public score and best score V28. This is the author's result, **not our score**. The difference from our score is only **0.3 percentage points**, and does not isolate any one component's contribution.

The [live public leaderboard](https://www.kaggle.com/competitions/name-that-llm/leaderboard) placed Watchman 777 at **214 / 0.639**, tenth at **0.668**, and first at **0.703**. Thus today's top-ten threshold is 2.9 percentage points above S04 and the leader is 6.4 points above. Rankings and thresholds can change. Public results use approximately 40% of the hidden data; private performance remains unknown.

## Why this candidate

S04 already includes multiple LoRA and encoder branches, style features, prompt grouping, centering, Gaussian heads, ATLAS, spectral readouts and balancing. Generic additions of those same ideas are insufficiently specified. The candidate adds a concrete full-text Qwen3.8-27B NF4 classifier: all 64 decoder layers, rank-16 adapters in the last eight layers, batch size three, and no fixed token truncation. Its nominal final mixture is 81% reference, 9% spectral helper and 10% Qwen before one final Sinkhorn balance. The title alone does not establish a distinct active Llama contribution. The reference is related to S04, not proven byte-identical to S04.

The source was inspected and its 33 code cells captured without execution. The source input page displays **134.1 GB / 28 sources**. Notebook Apache-2.0 attribution is visible; individual asset licences remain a separate gate. The 22.7-second exact-example preview is format-only, not runtime evidence for full inference.

## Gates before one frozen reproduction

1. Make the exact private copy and inspect actual mounts on CPU. Check all 160 mandatory inputs, including 112 paths under the literal `/kaggle/input/ntl-olmo2-native24-final-inference-assets/` root. The source has no alias/discovery fallback. Verify metadata, manifests, hashes, dimensions, class order, adapter configuration, tokenizer assets, NF4 shards, offline wheels and model/input licences. Do not deserialize joblib/Torch payloads merely to inventory them, and do not silently substitute checkpoints.
2. Resolve runtime feasibility on two free T4 GPUs. The source's `RuntimeBudget` is untimed: remaining time is infinity, deadline checks do nothing and parallel execution ignores its apparent maximum. Qwen stages roughly 19.5 GB of NF4 files, requires two GPUs, and evaluates full passages. Available temporary space, RAM, throughput and the complete Kaggle execution limit remain unverified for our copy. An author's successful official score is relevant evidence, but its short preview supplies no throughput measurement.
3. Review executable assets/custom code and adapt a separate output guard for the actual saved files, exact IDs/classes, finite probabilities, fixed mixture and real parquet readback. Preserve the scored source and record any wrapper change explicitly.
4. Only after these gates pass, consider one frozen official evaluation. Keep S04 intact and compare our displayed score only. Do not launch a weight sweep or claim 64.2% before our own result exists.

## If the candidate fails admission

Prepare an honest prompt-grouped development reference from untouched checkpoints before custom modelling. S04's final adapted checkpoints and this candidate's Qwen checkpoint have seen all official training labels. Their training-row predictions cannot become OOF merely by assigning folds afterward. Any custom candidate must keep held-out prompts out of all supervised fitting stages and be compared with a matched control. Stage 09's failed pretraining candidate and Stage 07C's failed style blend remain rejected.

## Current execution status

Discovery and static admission review only. No new private notebook, implemented CPU audit, GPU job, official submission, parameter tuning, paid service or final-selection change occurred in Stage 11. Aggregate discovery and source hashes are recorded in `DISCOVERY_20261025.json`; private capture text is not rehosted with weights or raw competition data.
