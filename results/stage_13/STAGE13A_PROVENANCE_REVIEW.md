# Stage 13A — S07 development-provenance review

Date: 31 October 2026. Status: **static review complete; full-S07 OOF and a common-fold scoring reference are not admitted**.

This review reads frozen source and recorded aggregate evidence. It performs no model fitting/inference, checkpoint loading, browser action, candidate evaluation or fresh mounted-cache validation. S07 remains the verified official public **0.651** reference; this ledger produces no development score.

The exact seven code strings at repository ref `6a73f73602fb7e91dab1224bd60c61989205741f` match source freeze `55cf8622c592690c98f67b65c3e9b6381a726b32`. Upstream source is Dhruv V1 **362013743**; own saved production V10 is **363626171**. Code identity is not training-exclusion proof.

## Active main paths

| Path | Source evidence weights | Development exclusion status |
|---|---|---|
| Q14 v5 | Plain 3.0; centered effective 1.05; Gaussian 0.70 | All five fold-named adapters and the source-declared full-data adapter are averaged for every scoring row. No common held-out prompt manifests established. |
| Q8 v5 | Plain 2.0; centered 0.70; Gaussian 0.44 | Same five-plus-full inference averaging. |
| Llama8 v5 | Plain 1.0; centered 0.35 | Same five-plus-full averaging. |
| Nishant Qwen3-14B | Plain 2.5; centered 0.875 | One fixed adapter and readouts; official/external training-prompt exclusions unresolved. |
| Nishant Qwen2.5-14B | Plain 1.0; centered 0.35 | Same exclusion gap. |
| Fine-tuned DeBERTa | 1.0 | Fixed fine-tuned encoder/head, not the pretrained DeBERTa feature cache. Exclusions unresolved. |
| Four-view SVC | 1.5 | Production fits all official rows plus the public pool, including learned text transforms. External prompt exclusions unresolved. |

Centered weights above equal main weight × `adapt_w=1.0` × `twin=0.35`; these are log-evidence coefficients, not probability-mixture percentages. Source cell 3 explicitly retains the **last full-data adapter's features** for the two Gaussian fits, then fits those heads on all official labels/prompts. Refitting a Gaussian or final linear head alone does not remove upstream adapter supervision.

Source selectors are in code cell 1, inference/readout behavior in cell 3, and active model settings in cell 6. Older legacy adapters are still discovered but are not additional active main-model groups.

## Complement coverage is larger than the ten plain-member log lines

The ten unconditional positive-weight components confirmed by the source are:

| Component | Plain coefficient | Exclusion status in production |
|---|---:|---|
| Stylegram | 1.516 | Vocabulary/IDF and plain/centered SVCs fit all official + external rows. |
| Stylometry LightGBM | 0.063 | Fits all official labels. |
| Frozen-base DeBERTa readout | 0.050 | Base may support new folds; production scaler/readout fits all official labels. |
| ELECTRA readout | 0.032 | Production readout fits all official labels. |
| Desklib readout | 0.132 | Production readout fits all official labels. |
| Seed7 LoRA | 0.113 | Fixed `job_final` adapter/readouts; not the historical outer-fold cache. |
| Claude LoRA | 0.177 | Fixed final adapter/readouts; prompt exclusions unresolved. |
| Qwen2.5 in-style LoRA | 0.170 | Fixed final adapter/readouts; prompt exclusions unresolved. |
| DeBERTa token attention | 0.185 | Fixed final head; separate historical OOF probabilities are not that head. |
| UltraBERT | 0.095 | Averages five checkpoints for every row; fold names are not exclusion manifests. |

Conditional grouped evidence also includes **five frozen centered twins**, **four Gaussian heads**, and the **Claude LoRA centered twin**. The Qwen3-8B, Qwen2.5-7B, Qwen3-4B and GLM plain coefficients are zero, but their grouped branches remain active. Their runtime scaler, classifier and Gaussian fits use all official labels. Source cell 4 defines the exact parts and coefficients in the machine-readable ledger.

The source computes family `GATES` and declares `BALANCE_ALPHA`, but neither is consumed by the final saved `LOGIT` path in this frozen code. They are not counted as active evidence. Activating them would be a changed candidate. The missing `lora24_public` checkpoint is the predeclared allowed skip; `NTL_DROP_Q35=1` intentionally disables Qwen3.5 token attention.

## ATLAS and postprocessing

ATLAS mixes at **0.10** on selected low-margin rows. Its fixed source-conditioned adapter/bundle has no verified common outer-prompt exclusion chain in this review. Source/import hashes and a valid output do not supply that proof.

The main/COMP probability mixture uses COMP weight **0.15**. Sinkhorn with alpha **1.0** operates at **N ≥ 18**, making one hundred column-mass/row-normalization iterations toward **N/18** probability mass per class. This is not a guarantee of equal final argmax counts. Typography allowed-label sets and dense-group Hungarian assignment follow it.

Coupled postprocessing is unsafe on pooled incompatible OOF banks when a model used for one row saw a peer row's scoring prompt during training. Stage06E explicitly excluded its balanced diagnostic from candidate selection for this reason. Every contributing supervised path must exclude **all prompts in the scoring batch**. The three-row S07 preview bypasses the Sinkhorn branch.

## What can support the reduced reference

| Lead | Recorded evidence | Still required |
|---|---|---|
| Own GLM feature cache, Stage06C V1 359405443 | `glm_features.npz`; SHA256 `2b41585a9482cef825962a8cc096697933c0b1237e8ede0712d1f8ab361d9414`; shape (2699,3,4096), float32, layers [8,3,24]; historically aligned and finite. | Fresh mounted-byte/schema/ID/text/fold admission; then separately frozen common-fold readout fits. Feature extraction itself supplied no classifier OOF. |
| Pretrained DeBERTa cache, source 358745836 | `enc_debv3l2_train.npz`; historical exact digest and feature evidence; mean L24+L6. | Safe alignment without unpickling object metadata, source/schema admission and common-fold refits. Do not substitute main fine-tuned DeBERTa. |
| Text/style recipe | Deterministic features and historical fold-local implementation evidence. | Rebuild all learned text transforms and heads inside common folds; external prompt origins need exclusion. |

The separate historical LoRA/attention OOF banks have useful recorded per-row exclusion evidence. They do not admit S07's production final checkpoints, establish a common fold scheme, or make batch-coupled pooling safe. Reused canonical folds are exploratory development data.

GLM already contributes to the competitive pipeline. Reconstructing it improves reference coverage; it is not a new model or evidence of improving 65.1%.

## Admission decision

**Block full-S07 OOF/error-analysis claims. Keep the reduced reference unscored until its common-fold fits and whole-batch exclusions pass.** The next CPU job can admit frozen-base cache inputs only. It must not output a new accuracy, a clean-ensemble claim, or a predicted competition gain.

Per-branch paths, exact weights, source versions where established, unknown mappings, exclusions and gate requirements are in [S07_DEVELOPMENT_PROVENANCE.json](S07_DEVELOPMENT_PROVENANCE.json). UI dataset version values are not equated with notebook metadata source IDs. CPU V8's 71/71 input checks and 20/20 review checks are historical content/schema evidence, not checkpoint training manifests.

References: [Stage12 source review](../stage_12/SOURCE_REVIEW.md), [CPU V8 limits](../stage_12/CPU_V8_INDEPENDENT_ACTUAL_REVIEW.json), [representation coverage](../stage_06g/REPRESENTATION_COVERAGE.md), [Stage06E limits](../stage_06e/REPORT.md), [Stage13 plan](../../plans/STAGE_13_DEVELOPMENT_REFERENCE_AND_BALANCING.md).

