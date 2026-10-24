# Next experiment after the frozen reproduction

This is a proposal, not a run or a 70% performance claim. First validate Stage 08C and obtain its own official result.

The earlier Stage 04B supervised DeBERTa-base already completed five development folds at 22.7639%, below the 35.4131% style reference. Repeating plain fine-tuning is not the next experiment. The new hypothesis is **public-pool authorship pretraining improves subsequent official-data fine-tuning**.

Use one existing development fold for a bounded, matched comparison. Start both arms from the same untouched vendor checkpoint. Control: official training prompts only. Candidate: admitted public-pool pretraining, then the identical official training prompts and fine-tuning schedule. Match architecture, classifier, seed and official training budget. The added pretraining compute remains part of the staged procedure. Never initialize either arm from the shipped Stage 08C checkpoints, which reportedly saw all official labels.

Before training, recover external prompt text and source provenance. Exclude known reconstructions of official prompts, train_regen and other official-prompt derivatives. Check exact/normalized responses and near matches against held-out responses; inspect corresponding prompt pairs. Remove every response in a colliding external prompt group. Exclude rows whose prompt origin cannot be traced. Zero exact response overlap alone does not certify prompt independence. Freeze corpus counts, hashes, all-class support, split and budget before evaluating.

Report the candidate-versus-control difference on that same fixed fold: balanced accuracy, per-class recall, confusion pairs, paired corrected/broken decisions and probabilities. Compare with the matched control on that fold, not pooled Stage 04B. This is reused development data. A promising pilot earns a declared grouped evaluation; it is not an ensemble improvement claim or a clean final test.

The 24 October public leader displayed 0.700, but no public scored 0.700 method was verified in a focused primary-source search. Do not substitute another hard style vote or a leaderboard-driven weight search for an evidence-backed experiment.
