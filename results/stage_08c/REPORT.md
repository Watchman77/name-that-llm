# Stage 08C — two-stage part-two reproduction

Prepared and checked live on 24 October 2026 in watchman777. **The real CPU input audit passed all 85 checks. The private GPU reproduction has started; no own preview result or official score is claimed yet.**

## Frozen source and inputs

- Source: Nishant Dahal, V1 **361311282**, https://www.kaggle.com/code/nishantdahal/two-stage-lora-part-two?scriptVersionId=361311282 . Displayed author official public balanced accuracy: **0.620**. Author preview: 653.6s, three example rows.
- Our private GPU V2 **361732127**: https://www.kaggle.com/code/watchman777/name-that-llm-stage-08c-part-two-reproduction?scriptVersionId=361732127 .
- Every one of the six model cells was compared with the complete rendered source. They match apart from one trailing newline per cell. No blend, feature, tokenizer, sampling or model parameter was changed.
- Four inputs: competition; `nishantdahal/ntl-handwriting-blend` explicitly pinned to **V1**; Qwen3-14B-Base V1; Qwen2.5-14B V1. Source environment pin 2026-10-22, Internet off, two T4 GPUs.
- CPU diagnostics were saved in a separate Quick V1, then removed from the GPU version. The CPU session was stopped. A separate validation-only final cell was appended, leaving all six model cells unchanged.

## What differs from the older baseline

The shipped SVC pool has **75,783** rows, compared with 34,940. The author separately describes 83,827 rows used to train the new Qwen3 adapter; that training pool is not inferred from the shipped SVC pool. The ensemble uses new Qwen3-14B LoRA, Qwen2.5-14B LoRA, fully adapted DeBERTa-v3-large, four SVC views, centered twins, and 300-step Sinkhorn.

A material inference fix reads EOS **151643** from training metadata instead of blindly taking the Kaggle Qwen3 tokenizer's end token. The recipe also disables reduced-precision matmul accumulation and retries out-of-memory batches one passage at a time without silently omitting a member. See `STATIC_REVIEW.md` for limits.

## Real CPU gate

`REAL_INPUT_AUDIT_20261024.json` records 85/85 passing checks: known official train SHA; train 2,699 / 150 prompts / 18 labels; unique IDs; shipped pool 75,783; class order; adapter/head shapes; EOS 151643; Qwen3 keep 27 and Qwen2.5 keep 32; base configurations, tokenizer assets and shard completeness; DeBERTa 384-token contract. Zero exact or normalized official full-response overlap and zero full-text duplicates were found. No rows were silently filtered.

Hashes are recorded for pool, adapters, heads, metadata and small configs. Large base/encoder weights are inventoried by name, size and index completeness, not fully hashed. The pinned attachment version is a UI receipt, not an independent byte-for-byte upstream certification.

The standalone input checker has 17 passing bounded synthetic tests. The independent output guard has 6 passing in-memory tests, with the local parquet readback boundary explicitly mocked; real Kaggle readback remains part of execution.

## Required output gate and score boundary

The added final cell validates finite logits/probabilities, N × 18 shape, normalized bounded probabilities, exact unique test-ID order, canonical labels, agreement with probability argmax and actual saved-parquet readback. It writes only aggregate JSON and asserts PASS. This prevents a NaN array from silently producing valid-looking labels.

After preview completion, inspect terminal logs and the guard's actual JSON before an official evaluation. A three-row preview does not validate hidden runtime, clustering or balancing. The author's 0.642 CV is not ours: final adapters reportedly saw official labels, so refitting readouts alone cannot create honest OOF predictions.

No paid services, retraining, generated API data, public checkpoints, row-level predictions, or final-selection change occurred. The 70% target is an objective, not an experimental result.

