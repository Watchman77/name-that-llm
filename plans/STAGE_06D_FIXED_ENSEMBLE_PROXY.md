# Stage 06D: fixed available-component ensemble comparison

Prepared 17 October 2026. Planned only; no execution or result is established by this document.

## Decision and scope

The next useful endpoint is marginal ensemble benefit. A centering component's balanced accuracy cannot be added to S02's official 54.9%. The independent stylegram OOF prerequisite may run on CPU while Stage 06C is queued. Candidate ensemble fusion must wait for the fixed Stage 06C mixed-grouping confirmation. If that candidate fails its declared screen, drop the change. This protocol can still establish a reproducible available-component reference.

Full S02 held-out reconstruction remains blocked by missing Qwen3-8B, Qwen2.5-7B, Qwen3-4B style and desklib passage caches, verified UltraBERT held-out outputs and common-context LoRA centered evidence. Do not label the available-component proxy as full S02.

## Fixed recipe

Use the original 2,699 IDs, data SHA, 18 class strings, archived prompt folds, seeds and common 1/3/6-response batch definitions. Every supervised prediction must exclude the scored row's entire prompt. Fixed averaging may combine members with different prompt folds if that exclusion is verified for each member. A shared fold number is not sufficient evidence by itself.

Start from audited plain LoRA1 (weight 0.177), plain LoRA2 (0.113), Qwen token attention (0.153), DeBERTa token attention (0.185), and fold-local frozen DeBERTa plain (0.05), plus its eligible-row centered evidence (0.25). Do not use the source LoRA lexical-context centered arrays as though they used our mixed groups. Do not group-center concatenated adapted features from differently trained LoRA folds.

Recompute exact S02 stylegram OOF (weight 1.516) on the archived original prompt folds. Use the vendor's four TF-IDF blocks and exact function-word transform: character 3–5/250,000 features/weight 1; word 1–3/180,000/0.7; shape character 2–6/60,000/0.5; function word 1–3/50,000/0.5. Preserve lowercase=False, sublinear_tf=True, min_df=2, float64 and default per-block L2 normalization. Word/function token pattern is the vendor punctuation-aware pattern. Fit vocabularies/IDF only on training prompts. Stack weighted blocks without another normalization or scaler. LinearSVC uses C=0.5, random_state=42, max_iter=4000 and recorded package defaults. Its score-softmax is not a calibrated probability model.

Retain complete per-class DeBERTa plain/control/candidate log-probabilities by rerunning the identical cached CPU heads once. Stage 06C's saved argmax arrays alone are insufficient for this fusion. Preserve all context-batch and eligibility assignments. This requires no repeat GLM inference after a valid cache exists.

Validate unique and aligned IDs, class-string ordering, once-only row coverage, finite arrays, normalized probabilities and prompt exclusion for every member. Apply S02's 1e-9 probability floor, sum original fixed weights times log-probabilities, and do not search weights or renormalize component weights. The PRIMARY endpoint is raw fixed rowwise log-evidence argmax. Report the S02 alpha=0.5 marginal balancing result separately within each simulated batch using the exact source recipe, as a potentially contaminated diagnostic only. Different supervised fold maps mean peer predictions used by balancing can have trained on the target prompt; rowwise prompt exclusion does not remove that indirect dependence. Do not use the balanced result for acceptance, clean generalisation claims or a new submission decision. Never balance the pooled 2,699 rows across distinct batches.

## Paired endpoint and limits

Compare the same fixed proxy with full-prompt-centered versus size-2/3/6-mixture DeBERTa training, changing only that head. Include the candidate only if Stage 06C meets its predeclared screen; otherwise record the rejection and the control reference. Report all three context balanced accuracies, their predeclared mean, per-class recall, fold scores, changed/corrected/broken predictions and a conditional prompt-bootstrap interval with its fixed-fit and transductive-batch limitations.

S02 weights were fitted using source OOF labels and these prompts have been used repeatedly. Even fixed reuse remains development evidence, not an untouched estimate. No new calibration, component selection or weight fitting is part of this comparison. Learned stacking would require a common nested outer evaluation excluding each scoring prompt from every supervised training step. No leaderboard score or 70% outcome is promised.

The first CPU prerequisite runs locally on the original official train.csv and archived baseline_oof.csv fold mapping, checking baseline_folds.csv aggregate counts. Record the local Python, NumPy, SciPy and scikit-learn versions; this is a recipe reproduction rather than a claim of bitwise Kaggle environment parity. Retain full float64 class probabilities in NPZ and a text CSV with round-trip precision, without passage text. Independently reconcile the saved confusion matrix and class recalls with the reported macro recall. The prerequisite's component score cannot select another recipe or establish ensemble gain.

No official submission, final-selection change, paid service or new neural training is part of this protocol. Commit implementation and provenance checks before execution.


## Execution clarification committed before the ensemble run

The stylegram prerequisite is already complete. Transfer its exact frozen float64 (2699,18) probabilities into the private Kaggle notebook as bounded zlib/base85 numeric data, with no refitting. Verify original NPZ against the original artifact manifest, original file SHA, canonical little-endian C-order probability-byte SHA (5e2e05577b7dc632f38f67d47958d7f3d069e8b758ead4664050f0d984803bad), exact byte count388656, compressed hash, ordered official IDs/classes and saved fold mapping. Bounded decompression must reject incomplete/trailing data. The derived NPZ file is documented as a transfer copy; its container hash is not claimed equal to the original. Verify the entire inserted source and immutable version-export cell contents, including the substantial payload, rather than only a prefix/suffix.

Only raw fixed fusion determines the predeclared comparison: mean raw balanced-accuracy gain across1/3/6 contexts >0, conditional prompt-bootstrap interval lower endpoint >0, no context regression worse than0.5 percentage points, and no non-deprecation fit warnings. Report per-class regressions regardless. Exact marginal balancing is a contaminated diagnostic because public supervised folds differ; do not silently change batch definitions or choose new weights. The original source weights were learned on historical OOF and are fixed here, so even the primary comparison remains development evidence.

Use seven pinned inputs: official competition170484; encoder cache358745836; LoRA1 source358831649; LoRA2 source358911769; token attention358928013; DeBERTa token attention358948428; our completed GLM cache359405443. CPU only, internet off, no model weights loaded, no API, paid compute, official submission or final-selection change.
