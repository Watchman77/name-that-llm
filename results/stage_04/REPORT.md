# Stage 04 — frozen MPNet comparison

## Decision

**Reject both new candidates and retain Stage 03.** The fixed frozen-encoder experiment did not improve development balanced accuracy. This trial produced no new competition submission; S01's official public score remains 0.354 (35.4%).

| Recipe | Prompt-grouped balanced accuracy | Change versus Stage 03 |
|---|---:|---:|
| Frozen Stage 03 character + style reference | 35.4131% | — |
| Character + style + frozen MPNet | 34.1153% | −1.2978 percentage points |
| Frozen MPNet alone | 14.5946% | −20.8185 percentage points |

The added MPNet block's paired prompt-bootstrap 95% interval is **−2.6324 to −0.0370 percentage points**. The encoder-only interval is −22.7179 to −19.1111 points. These are descriptive intervals conditional on saved development predictions, without refitting, model-selection uncertainty or multiplicity adjustment. The upper bound close to zero is not a basis for a broad claim that contextual encoders are harmful.

## What was run

The prospective protocol was committed before computing embeddings or fitting the two classifiers: **9d34e75b0c2b2faff0f995bd441a8a3297bd77c4**. Implementation was committed before execution. The source, exact saved folds and frozen Stage 03 reference were checked by ID alignment and hashes.

The public Apache-2.0 checkpoint `sentence-transformers/all-mpnet-base-v2`, revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`, was downloaded once. Modelling then used local files with both Hugging Face offline flags set. The encoder remained in float32 eval mode with no gradient updates. Attention-mask mean-pooled embeddings were L2 normalized. Only the predeclared linear SVMs were trained; vocabulary, IDF and style scaling remained fold-local.

- 2,699 labelled rows, 150 prompts, 18 classes; exact original five folds.
- Ten new fold fits completed; five reference fold results reused.
- Zero fit warnings; total encoding/evaluation time **545.1 seconds** on CPU.
- **Zero truncated passages**; longest input was 381 tokens including specials, below the fixed 384-token cap. Input truncation cannot explain this result.
- No paid API calls, paid compute, new competition-labelled data, parameter search, block reweighting or extra candidate.
- Training-data SHA256: `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`.
- Checkpoint safetensors SHA256: `78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658`.
- Execution: 2026-10-16T23:20:56Z to 23:30:02Z (17 October in Europe/London).

## Regressions and limits

The combined candidate worsened three folds, tied one and improved one. It worsened all three predeclared length groups: <=80 words, 27.01% versus 28.06%; 81–150 words, 35.55% versus 36.23%; >150 words, 40.84% versus 43.42%. Every group includes all 18 classes.

Per-model changes are mixed. Claude Opus 4.6 recall fell from 48.00% to 39.33%, while Claude Opus 4.8 improved from 16.67% to 22.67% and Kimi K3 from 20.67% to 25.33%. The full class table is retained; these gains do not outweigh the pooled regression.

This experiment tests a particular **frozen semantic encoder with a fixed linear head and equal block weighting**. It does not test supervised encoder adaptation, style-specific pretraining, different pooling, or other fixed model families. Concatenating a normalized block also changes the classifier's regularization geometry; this result does not isolate a causal content-versus-style explanation. All folds are reused development data, not a fresh independent evaluation.

## Next experiment

The planned next step is a separate, prospectively specified **supervised DeBERTa-v3-base fine-tuning trial**, using the same development folds and fixed final training epoch. A model trained for semantic similarity has no explicit objective to distinguish these 18 generators; supervised adaptation tests that missing capability directly. This is a hypothesis, not a promised gain. The published DeBERTa model card and Transformers classification recipe provide implementation references, not evidence of performance on this competition.

Current competitive target: first place in the final private evaluation. The public leader was 58.4% at this session's check, versus S01's 35.4%, a 23.0-point gap. No local score should be treated as a guaranteed public or final placement.

## Reproduce and inspect

The executed companion `notebooks/Stage_04_Evaluation.ipynb` independently reconciles pooled scores, all 15 fold records, class recalls and paired bootstrap intervals from saved OOF predictions. It executed top-to-bottom in a fresh in-process IPython session, retaining its rich table outputs. A rendered preview was inspected before delivery.

Install `requirements-encoder.txt`, put the official data in `data/`, then use a new output directory:

```bash
python scripts/download_stage04_encoder.py --model-dir models/all-mpnet-base-v2
python scripts/frozen_encoder_comparison.py --data-dir data --model-dir models/all-mpnet-base-v2 --cache-dir encoder-cache --out-dir results/stage_04_reproduction
```

Saved evidence: `metrics.json`, `encoder_provenance.json`, `run_status.json`, `comparison.csv`, `fold_scores.csv`, `oof_predictions.csv`, `paired_differences.csv`, `per_model_recall.csv`, and `length_diagnostics.csv`. Raw passages, weights and embedding caches remain outside version control.

Primary references: [MPNet checkpoint/model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), [DeBERTa-v3-base model card](https://huggingface.co/microsoft/deberta-v3-base), [Transformers 4.56.2 classification documentation](https://huggingface.co/docs/transformers/v4.56.2/en/tasks/sequence_classification).
