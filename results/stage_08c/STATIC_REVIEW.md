# Stage 08C public part-two V1 — bounded static review

Source inspected: captured `upstream_361311282.py`, Kaggle script version **361311282**. No source model code, real data, weights, network, browser action or Git write was executed by this reviewer.

The recipe is materially simpler than the blocked V20 stack: a four-view SVC fitted on the official train plus shipped `pool_v4.parquet`; two truncated Qwen 14B classification adapters and their centred heads; one DeBERTa encoder/head; final Sinkhorn balancing for at least 50 test rows. It has no embedded `exec`, custom-code/trust-remote-code loader, paid API call, or extra-CSV/private-feature-cache branch. Model/data loads point to attached local paths and Hub offline modes are set. The full DeBERTa directory is copied into `/tmp/deberta_v3_large` during inference; this is normal scratch output, separate from the read-only preflight.

## Concrete requirements

- Resolve exactly one intended competition train/test, `pool_v4.parquet`, each LoRA `meta.json`, and DeBERTa `meta.json`. Public code takes the first sorted filename/config match; duplicate plausible attachments can silently select different bytes.
- Verify the shipped SVC pool's author-described **75,783 rows / 18 labels**, distinct from the author's stated **83,827-row adapter-training pool**. The inference source does not recreate or independently authenticate training history.
- Preserve training EOS **151643** for both adapters. The source deliberately overrides the attached Qwen3 tokenizer's chat EOS, which can differ. Use source metadata rather than adapting tokenization to the attached tokenizer.
- Respect the intended two-T4 setup. Both trunks can shard over GPUs; the source estimates memory, retries OOM batches one passage at a time and retries nonfinite hidden states in BF16.

## Runtime checks still needed

No static repair blocker was identified. However, the BF16 retry has no second explicit finiteness assertion, and final source assertions check row count and allowed labels, not numerical probability validity or unique/exact test-ID coverage. **After the frozen source completes, run a separate output audit**: `LOGIT`, `P` and any retained arrays must be finite; `P` must have shape `(len(test),18)`, nonnegative values and row sums near one; candidate IDs must be unique and match every official test ID exactly; parquet labels must be allowed and candidate row count complete. An argmax can otherwise turn NaNs into apparently valid labels. This audit need not change the frozen source cell.

The three-row public preview does not exercise the final Sinkhorn branch and is not hidden-test completion. Its content clusters and speed do not forecast full-test performance. The source supplies no independent grouped OOF evaluation; final adapter features on official training rows cannot provide honest held-out accuracy when supervised adaptation used those same rows.

## CPU checker scope

`scripts/stage08c_input_preflight.py` is standalone, read-only and CPU-only. It imports no Torch/Transformers, executes no model/custom code, reads no pickle and performs no fitting/inference/network operations. It records aggregate receipts, safe tensor headers, selected metadata, model shard presence and pool statistics. **17 synthetic checks pass**. Real input execution remains pending.

The checker enforces the author-described kept depths (Qwen3 27, Qwen2.5 32), exact described Qwen2.5 pool depths, EOS 151643 and DeBERTa max length 384. Qwen3 pooled layers and LoRA token limits are schema-checked and recorded, because exact metadata values have not yet been observed by this reviewer; they are not invented. Large backbone shards are inventoried by name/size/index completeness rather than hashed or deserialized. A successful contract check does not prove equality to the author's immutable attachment versions.

Official-training-response overlaps are counted exactly and under NFKC/case/whitespace normalization. They are reported as a separate admission review, without filtering rows or automatically declaring an inference-contract failure. Such overlap alone does not prove hidden-test leakage. Public pool publication and checkpoint availability also do not independently authenticate upstream provider identity or training provenance.
