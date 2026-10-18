# Representation coverage prerequisite audit

Prepared from existing repository evidence on 18 October 2026, separately from the closed Stage 06G correction. No browser inspection, cache extraction, fitting, candidate evaluation or new experiment plan occurred in this audit. **The existing own GLM cache is the strongest ready lead for closing proxy coverage. GLM already contributes to full S02; reproducing it is baseline reconstruction, not a novel competitive improvement.**

## Current all-18-class coverage

Stage 06E contains seven evidence branches, all with aligned 18-class outputs. Full S02 has sixteen active branches: ten nonzero plain/base contributions, five frozen centered twins and one centered LoRA contribution. Four frozen LLM plain weights are zero, but their centered branches remain active.

| Representation / branch | Stage 06E coverage | Remaining full-S02 gap |
| --- | --- | --- |
| Stylegram | Exact recipe, fold-local OOF, fixed weight 1.516 | Separate stylometric LightGBM branch absent |
| Two adapted LoRA24 models | Both plain held-out outputs | First LoRA centered branch lacks matched mixed-group/context evidence; cross-fold adapted features cannot simply be pooled and centered |
| Qwen and DeBERTa token attention | Both native held-out probability outputs | Outer heads/token-state banks unavailable for independent re-inference; saved-output provenance limits remain |
| Frozen DeBERTa-large | Plain and centered evidence; public fp16 mean L24+L6 cache | Quantization differs from full-S02 fp32 extraction |
| Frozen Qwen3-8B, Qwen2.5-7B, Qwen3-4B | Absent | Required passage caches not located in recorded inventory |
| Frozen GLM | L24 grouping geometry only | L8+L3 centered classifier evidence absent despite available cache |
| ELECTRA / desklib / UltraBERT | Absent | ELECTRA historical cache lead; desklib passage cache not located; UltraBERT held-out provenance unresolved |

Sources: `results/stage_06e/ensemble_result.json` (exact seven weights/classes), `vendor/public_549_v3.py` (`W`, `TWIN_W`, `TWIN_W_LORA`, head definitions), `plans/STAGE_06D_FIXED_ENSEMBLE_PROXY.md`, `results/stage_06b/REPORT.md`. The older `results/stage_06/COMPONENT_INVENTORY.json` predates completed GLM extraction and stylegram reconstruction; its corresponding gaps are historical, not current blockers.

## GLM: verified existing bytes, not yet classifier evidence

Own immutable Stage 06C V1 **359405443** produced `glm_features.npz`, SHA-256 **`2b41585a9482cef825962a8cc096697933c0b1237e8ede0712d1f8ab361d9414`**. Frozen source model version is **358683059**, competition version **170484**. Ordered layers `[8,3,24]` give shape **(2699,3,4096)**, float32 and finite. Layers 8+3 supply the source-defined concatenated **8192-dimensional** classifier representation; layer 24 separately supplies 4096-dimensional grouping geometry.

The extractor uses attention-masked token means, float32 pooling of fp16 hidden states, right padding, 512-token cap, batch size eight and BOS prepending only if defined. The observed tokenizer had no BOS; all 338 batches remained fp16, with zero nonfinite events and zero truncations (44–381 tokens). Ordered-text hash is `6496071874da8f911fafebd9eec6191e56134a97f00226dd8c78805d31f07e36`. Extraction restores official row order and saves IDs/text hashes/train digest; it uses IDs/text without fitting a task classifier.

These are completed C runtime records, subsequently rechecked by E V2 for exact cache hash, official IDs, layer order, shape, dtype and finiteness. This audit did **not** reopen the binary locally or certify pretrained-weight history beyond recorded source/file provenance. Sources: `scripts/stage06c_glm_cache.py`, `results/stage_06c/GLM_CACHE_COMPLETE.json`, `GLM_MANIFEST_OBSERVATION.json`, `scripts/stage06e_capture_original.py`, `scripts/stage06e_cache_checks.py`, and `results/stage_06e/REPORT.md`.

The frozen cache itself has no supervised folds or classifier labels. Existing E DeBERTa fold exclusions do not establish held-out GLM-head predictions. Any future GLM scaler/head must exclude each scoring prompt, preserve class order and common context batches, and document grouping/balancing label dependencies. The source S02 GLM plain weight is zero; its centered contribution is active. A plain-only addition would test a different recipe.

## ELECTRA readiness and decision

The historical inventory located `enc_electl_train.npz` in encoder output **358745836**, with mean/style layers 6/12/18/21/24. Full S02 specifies **L21 mean + L24 style**. No equivalent verified cache digest, ordered-ID/schema/finite audit or prompt-held-out classifier output is recorded here; those bytes were not reverified. Source: `results/stage_06/COMPONENT_INVENTORY.json` and `vendor/public_549_v3.py`.

Choose **GLM** as the single best-prepared coverage lead because extraction, alignment and byte identity already have runtime evidence. Remaining blockers are a separately frozen matched centered-head/evaluation definition, prompt-excluded probability production, source-preserving fixed fusion, and whole-18-class marginal validation with class harms reported. No such trial was launched or specified here. This would make the development reference more faithful for testing future changes; adding an existing S02 member to the incomplete proxy cannot establish improvement over official 54.9%. A genuinely novel complement remains unidentified. Neither cache readiness nor representation diversity establishes a gain or a route to 70%.
