# Stage 06D: fixed available-component ensemble comparison

Prepared 17 October 2026. Planned only; no execution or result is established by this document.

## Decision and scope

The next useful endpoint is marginal ensemble benefit. A centering component's balanced accuracy cannot be added to S02's official 54.9%. First inspect the fixed Stage 06C mixed-grouping confirmation. If that candidate fails its declared screen, drop the change. This protocol can still establish a reproducible available-component reference.

Full S02 held-out reconstruction remains blocked by missing Qwen3-8B, Qwen2.5-7B, Qwen3-4B style and desklib passage caches, verified UltraBERT held-out outputs and common-context LoRA centered evidence. Do not label the available-component proxy as full S02.

## Fixed recipe

Use the original 2,699 IDs, data SHA, 18 class strings, archived prompt folds, seeds and common 1/3/6-response batch definitions. Every supervised prediction must exclude the scored row's entire prompt. Fixed averaging may combine members with different prompt folds if that exclusion is verified for each member. A shared fold number is not sufficient evidence by itself.

Start from audited plain LoRA1 (weight 0.177), plain LoRA2 (0.113), Qwen token attention (0.153), DeBERTa token attention (0.185), and fold-local frozen DeBERTa plain (0.05), plus its eligible-row centered evidence (0.25). Do not use the source LoRA lexical-context centered arrays as though they used our mixed groups. Do not group-center concatenated adapted features from differently trained LoRA folds.

Recompute exact S02 stylegram OOF (weight 1.516) on the archived original prompt folds. Use the vendor's four TF-IDF blocks and exact function-word transform: character 3–5/250,000 features/weight 1; word 1–3/180,000/0.7; shape character 2–6/60,000/0.5; function word 1–3/50,000/0.5. Preserve lowercase=False, sublinear_tf=True, min_df=2, float64 and default per-block L2 normalization. Word/function token pattern is the vendor punctuation-aware pattern. Fit vocabularies/IDF only on training prompts. Stack weighted blocks without another normalization or scaler. LinearSVC uses C=0.5, random_state=42, max_iter=4000 and recorded package defaults. Its score-softmax is not a calibrated probability model.

Retain complete per-class DeBERTa plain/control/candidate log-probabilities by rerunning the identical cached CPU heads once. Stage 06C's saved argmax arrays alone are insufficient for this fusion. Preserve all context-batch and eligibility assignments. This requires no repeat GLM inference after a valid cache exists.

Validate unique and aligned IDs, class-string ordering, once-only row coverage, finite arrays, normalized probabilities and prompt exclusion for every member. Apply S02's 1e-9 probability floor, sum original fixed weights times log-probabilities, and do not search weights or renormalize component weights. Report raw evidence argmax and the S02 alpha=0.5 marginal balancing diagnostic, separately within each simulated batch, using the exact source iteration recipe. Never balance the pooled 2,699 rows across distinct batches.

## Paired endpoint and limits

Compare the same fixed proxy with full-prompt-centered versus size-2/3/6-mixture DeBERTa training, changing only that head. Include the candidate only if Stage 06C meets its predeclared screen; otherwise record the rejection and the control reference. Report all three context balanced accuracies, their predeclared mean, per-class recall, fold scores, changed/corrected/broken predictions and a conditional prompt-bootstrap interval with its fixed-fit and transductive-batch limitations.

S02 weights were fitted using source OOF labels and these prompts have been used repeatedly. Even fixed reuse remains development evidence, not an untouched estimate. No new calibration, component selection or weight fitting is part of this comparison. Learned stacking would require a common nested outer evaluation excluding each scoring prompt from every supervised training step. No leaderboard score or 70% outcome is promised.

No official submission, final-selection change, paid service or new neural training is authorized by this protocol. Commit implementation and provenance checks before execution.
