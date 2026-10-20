# Static source review — public 0.600 notebook V20

Reviewed the captured public source of `kirilltym/0-600-teach-the-model-remove-the-prompt`, immutable Kaggle script version **360088024**, and its rendered author description/preview log. No downloaded code was executed, model fitted, checkpoint loaded, browser used by this reviewer, or official submission made. This is static method and dependency review; live input availability is a separate finding.

Captured Python file SHA-256: `456f6c023ce6f997c0c42a5489dda18ee3185142d31782184e1307e0a35f9dd7`. Its embedded `_GAUSSIAN_SOURCE` literal has SHA-256 `b3388879ecf798ad32e42b18b8be3a3157a6ed252e53df7a6c948b8374202d76` (UTF-8 decoded literal). Both parse as Python. The outer file contains 400 lines; one embeds a further 889-line older ensemble as a literal, compiled and executed by the outer recipe at runtime.

## What actually changed

This is a substantially different candidate from our S02 and from the old V14 link. It is not the same weak character/style classifier added at a different weight.

1. A four-view character/word/shape/function-word SVM trains on official train plus `pool_v1.parquet`, with class balancing, C=0.5 and weight 1.021. The author describes a filtered 34,940-passage public pool; this reviewer did not inspect those data bytes.
2. Two externally pretrained, then official-data-finetuned LoRA classifiers use Qwen3-8B and Qwen3-14B. Their plain log-evidence weights are 0.226/0.429; inferred-topic-centered twin weights are 0.079/0.150. Adapters are manually merged into truncated backbones; final RMSNorm is replaced by identity to match their stated training setup.
3. The new 8B features also train an official-data shared-prompt Gaussian head, residual PCA dimension 128, temperatures 3.0, log-evidence weight 0.10. It contributes only to inferred clusters of 2–18 passages.
4. The new branch's complete probability vector is mixed **85%/15%** with an embedded older feature ensemble. That ensemble includes four additional Gaussian group heads on frozen Qwen3-8B, GLM, Qwen3-4B and DeBERTa views, and retains old LoRA, attention, encoder and style branches. It builds family gate scores but does not apply their hard final decisions. This older stack also trains stylegrams on separately located extra CSVs.
5. IDs are checked between branches and class columns explicitly aligned. One final full Sinkhorn balance runs when test length is at least 50, then `submission.parquet` is written.

The new two-stage LoRA representation and shared-prompt Gaussian treatment are the meaningful leads. The author description's historical local CV values and claims about training improvements are author reports, not reproduced evidence from this inference source.

## Required input contracts

| Input | Required source-level files / properties |
| --- | --- |
| Official competition data | `train.csv` with text/model/prompt_id; `test.csv` with id/text; official 18-label names |
| Public pool | Exactly one intended `pool_v1.parquet` with text/model columns. Code selects the lexicographically first recursive filename match; it does not assert row count, source identity, overlap filtering or digest. |
| New 8B checkpoint | `lora8b_v1/meta.json`, `heads.safetensors`, `adapter/adapter_config.json`, `adapter/adapter_model.safetensors`; meta label order must exactly equal the outer `LABELS`; cfg supplies keep/layers/max_len. Heads require per-layer norms and plain/centered weight/bias arrays. |
| New 14B checkpoint | Same tree under `lora14b_v1`. Both new checkpoints are mandatory; neither has an optional skip. |
| New base backbones | Local Qwen3 config matches `(model_type=qwen3, hidden_size=4096, num_hidden_layers=36)` and `(qwen3,5120,40)`, plus tokenizer and weights. Truncated kept depths come from checkpoint metadata. |
| Older frozen decoders | Qwen3-8B, Qwen2.5-7B, Qwen3-4B and GLM-4-9B truncated backbone, with required mean/style layers. GLM layer 24 supplies old-branch topic geometry. |
| Older frozen encoders | Directories `deberta-v3-large`, `electra-large-discriminator`, `desklib-detector`, containing local weights/config/tokenizer. |
| Older LoRA24 public | `**/lora24/final/COMPLETE.json`, adapter, tokenizer, `heads.safetensors`, `readouts.npz`; optional `anchor_features.npz` parity check. |
| Older seed-7 LoRA | `**/name-that-llm-train-lora24-s7/lora_out/job_final/COMPLETE.json`, adapter/tokenizer, `norms.safetensors`, readouts and optional `valid_features.npz` anchor. |
| Older Claude LoRA | `**/ntl-qwen3-8b-lora-claude-head/**/final/COMPLETE.json`, adapter/tokenizer, heads/readouts and optional anchor. |
| Qwen2.5 in-style LoRA | `**/ntl-train-lora-qwen25-instyle/lora_out/job_final/COMPLETE.json`, adapter/tokenizer, norms/readouts and optional anchor. |
| Attention heads | `tokattn_final.pt` and `tokattn_debl23_final.pt`, with matching class order; local Qwen3.5-9B and DeBERTa backbones. |
| UltraBERT | `ultrabert-380m-ntl` root with exactly five appropriate fold directories/configs/weights and its custom Python modules. Loaded with `trust_remote_code=True`. |
| Older extra-data stack | Recursive CSV matches for `ntl_extra_passages_claude.csv`, `gen_passages.csv`, `ntl_generations.csv`, `ntl_proxy_generations.csv`, `ntl_generated*.csv`, `passages_clean.csv`. These alter fitted stylegrams; absent data are not generally fatal. |

The raw 14 source datasets listed in the author description are not all individually necessary for inference when the ready-made pool and both LoRA checkpoints are available. They remain relevant for provenance, usage terms and independent training reproduction.

## Compute and compatibility

The outer branch unconditionally targets CUDA GPU 0 and supports model sharding across multiple GPUs when estimated model weights exceed 13 GiB. It is not a CPU notebook. The visible author execution used two T4s. Qwen3-14B plus output hidden states needs the intended sharding/memory setup; CPU scratch cannot establish inference parity.

The recipe needs compatible Torch, Transformers, Accelerate/device-map support, PEFT, Safetensors, NumPy, Pandas/Parquet, SciPy, scikit-learn and LightGBM, plus local UltraBERT custom code. No exact package lock is embedded. Older LoRA inference monkey-patches TorchAO availability for PEFT compatibility. Outer new-LoRA inference uses BF16 on Ampere-or-newer devices, FP16 otherwise, with a one-batch BF16 retry after nonfinite hidden states. This retry is not followed by a second dedicated finiteness assertion, although final blend checks reject nonfinite mixed probabilities.

Author preview completion in 45m39s and its 2,699-train/3-test log establish successful execution of those inputs in that environment. They do not establish our compatibility, presently available dependency bytes or full hidden-test runtime. Full-test grouping creates dense pairwise distance matrices and hierarchical linkage; its memory/time scale differs from the three-row preview. The public score displayed on V20 is separate evidence of an author's scored submission; it is not our score.

## Reproduction and evidence limits

- Outer `find()` and several model/head selectors take the first sorted recursive match. There is no version/hash pinning and no rejection of duplicate plausible matches. Input membership therefore matters; attach exactly the intended versions and record a complete inventory before execution.
- Most required checkpoints fail explicitly when absent. Older `load_extra_passages()` returns an empty table when no matching CSV exists, and the Claude gate source can be omitted. Since extra stylegram data affect predictions, a run can technically finish after data loss while ceasing to be an unchanged reproduction. Old gate construction is computationally redundant because final hard gates are omitted; do not silently remove it while calling the result byte-identical source.
- New adapter metadata verifies label order, and old readouts verify labels/pool layers. Old anchors check relative feature agreement only when the anchor file exists. New adapters have no saved-anchor inference-parity check in this source. These are useful contracts, not complete checkpoint identity checks.
- Outer code verifies branch test-ID equality and finite normalized mixed probabilities. It does not independently verify unique/complete official IDs or pin original train/test bytes. A separate read-only input/output audit can supply those receipts without changing the recipe.
- The supplied pool's claimed exact-deduplication, formatting filters, model identities, stage-1 source rights and stage-2 training history are not established by inference source. The older CSV loader prefixes only the `passages_clean.csv` prompt bank, so other source prompt IDs are assumed compatible; that assumption merits inspection if recreating the extra-data merge.
- No local grouped-CV evaluation is executed here. Gaussian heads fit on all official features, as appropriate for final inference, and in-sample self-tests are deliberately labeled as such. They are smoke checks, not evidence of new development accuracy. Evaluating the new adapters on original train rows would be in-sample because stage 2 reportedly used all official train data; independent OOF checkpoints or retraining folds are necessary for a genuine local comparison.
- Sinkhorn assumes roughly balanced class totals; Gaussian exact inference prohibits duplicate classes in small inferred prompt groups and approximate larger-group inference repels duplicate assignments. These are explicit distributional assumptions, not uses of hidden labels. Their validity depends on the test construction and grouping quality.
- The captured implementation contains no obvious network-generation API calls or paid inference route. All model loads point to local Kaggle paths. Code and attached model/data notices have separate licensing/provenance requirements; an Apache notebook notice does not by itself license all checkpoints or generated text.

## Recommendation

**An unchanged Kaggle inference reproduction is technically plausible, conditional on live availability of the exact pool, both new LoRA checkpoints and every intended old-stack dependency. No code blocker requiring a repair was identified statically. Do not launch from the V14 link or merely copy its input list.**

Before committing GPU time, freeze the V20 source and all input versions; verify mandatory file/schema/class-order contracts, intended external-row counts and model selection uniqueness. Keep new fitted classifier artifacts private. Preserve our existing S02. After one successful private preview, inspect final logs and the candidate parquet before an official evaluation; preview success alone is not hidden-test completion. Do not claim the author's 0.600 for our account or regard source publication as independent confirmation of training-data provenance.
