# Stage 06C: validate the DeBERTa centering change with S02 mixed grouping
Prepared 2026-10-17 after Stage06B V1; not launched by this document.

## Why this stage
The single predeclared cached DeBERTa test improved the mean lexical-context score by0.67984 percentage points (paired prompt95% interval+0.0860 to+1.2963). This is one component on historically reused development data. It does not establish an improvement to official S02.
The six-response oracle diagnostic slightly favours control, while inferred grouping favours the candidate. Treat the effect as context-dependent; do not generalise to all five centered twins.

## Exact missing prerequisite
S02 mixes content-word TF-IDF distance with GLM-4-9B mean-pooled hidden_states[24] cosine distance, threshold0.60. Its attached GLM source V358683059 supplies truncated weights/tokenizer, but no passage-feature cache was listed. Stage06B therefore used the declared lexical fallback threshold0.85.

Extract GLM layers [8,3,24] together in one forward pass (all are already fixed S02 requirements; layers8/3 support the existing GLM classifier and layer24 supports clustering), without selecting new layers, from the exact attached truncated model. Preserve original train.csv SHA, all2699 ordered IDs and text hashes; store features as float32 plus model/config/tokenizer/source hashes and environment/timing. Use the S02 extraction recipe: model eval, no_grad,25layers,fp16 backbone,512-token limit, prepend tokenizer BOS where defined, add_special_tokens=False,right padding, mean over attention-masked tokens including BOS. Match S02 batch/order settings and record truncation/nonfinite batches. This is frozen inference, not supervised training or API generation. Do not silently sanitise nonfinite features and declare a clean cache.
Use included Kaggle GPU quota and internet off; no paid services. No GPU job is launched merely by committing this plan.

## Fixed validation
Use the same cached DeBERTa mean24+6 representation, same original GroupKFold assignments, C, train size2/3/6 mixture, seeds, plain-head test standardisation, fixed0.05/0.25 evidence weights, and three context simulations as Stage06B. Change only group recovery from lexical fallback to the exact S02 mixed grouping. All scalers and supervised fits remain outer-fold-local; unsupervised batch geometry may use only texts/features supplied in that simulated test batch.
Repeat the same paired primary mean and prompt bootstrap screen; report per-class and oracle diagnostics. Preserve both the earlier lexical result and this mixed-grouping result. Do not choose a clustering threshold or blend weight from validation labels.

Only a consistent benefit supports replacing S02's DeBERTa centered training recipe. The other four frozen centered twins remain untested and must not be changed by extrapolation. Full-ensemble validation still needs its missing representations and appropriately held-out supervised outputs. No leaderboard score is promised; a component result alone does not justify S03.

## Independent review and execution clarification (before extraction)
The Stage06B interval is conditional on fitted folds, one training partition seed and fixed context/clustering outcomes; it does not include refitting or clustering uncertainty. Clustering couples prompts, so a prompt bootstrap is descriptive rather than a complete inferential guarantee. The primary confirmation preserves Stage06B's30prompt-fold batches for a paired comparison, not a claim to reproduce the full hidden-test batch size. Larger pooled grouping can have more accidental merges and remains a separate scale check.
Add group-recovery diagnostics per scenario/fold: pair precision and recall within the supplied batch, eligible-row fraction and cluster-size counts, with oracle grouping labelled diagnostic. Do not count true pairs separated into different simulated batches as cluster-recovery failures.
A single reviewed script performs frozen extraction then the fixed CPU-head comparison. Cache outputs are kept if later CPU fitting fails. Model weights/data come only from public inputs already used by S02. No model training API or paid compute is used. No official submission or final-selection change is part of this run.
