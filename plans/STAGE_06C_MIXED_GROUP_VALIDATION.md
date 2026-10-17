# Stage 06C: validate the DeBERTa centering change with S02 mixed grouping
Prepared 2026-10-17 after Stage06B V1; not launched by this document.

## Why this stage
The single predeclared cached DeBERTa test improved the mean lexical-context score by0.67984 percentage points (paired prompt95% interval+0.0860 to+1.2963). This is one component on historically reused development data. It does not establish an improvement to official S02.
The six-response oracle diagnostic slightly favours control, while inferred grouping favours the candidate. Treat the effect as context-dependent; do not generalise to all five centered twins.

## Exact missing prerequisite
S02 mixes content-word TF-IDF distance with GLM-4-9B mean-pooled hidden_states[24] cosine distance, threshold0.60. Its attached GLM source V358683059 supplies truncated weights/tokenizer, but no passage-feature cache was listed. Stage06B therefore used the declared lexical fallback threshold0.85.

Extract only the required GLM layer24 representation first, from the exact attached truncated model. Preserve original train.csv SHA, all2699 ordered IDs and text hashes; store features as float32 plus model/config/tokenizer/source hashes and environment/timing. Use the S02 extraction recipe: model eval, no_grad,25layers,fp16 backbone,512-token limit, prepend tokenizer BOS where defined, add_special_tokens=False,right padding, mean over attention-masked tokens including BOS. Match S02 batch/order settings and record truncation/nonfinite batches. This is frozen inference, not supervised training or API generation. Do not silently sanitise nonfinite features and declare a clean cache.
Use included Kaggle GPU quota and internet off; no paid services. No GPU job is launched merely by committing this plan.

## Fixed validation
Use the same cached DeBERTa mean24+6 representation, same original GroupKFold assignments, C, train size2/3/6 mixture, seeds, plain-head test standardisation, fixed0.05/0.25 evidence weights, and three context simulations as Stage06B. Change only group recovery from lexical fallback to the exact S02 mixed grouping. All scalers and supervised fits remain outer-fold-local; unsupervised batch geometry may use only texts/features supplied in that simulated test batch.
Repeat the same paired primary mean and prompt bootstrap screen; report per-class and oracle diagnostics. Preserve both the earlier lexical result and this mixed-grouping result. Do not choose a clustering threshold or blend weight from validation labels.

Only a consistent benefit supports replacing S02's DeBERTa centered training recipe. The other four frozen centered twins remain untested and must not be changed by extrapolation. Full-ensemble validation still needs its missing representations and appropriately held-out supervised outputs. No leaderboard score is promised and no S03 is authorised by this plan alone.
