# Stage 07: matched external-data pilot

Status: corpus audit executed; training not started. This protocol is specified before seeing any augmented-model development score. No submission or model-selection change is part of this stage.

## Question and scope

Does a source-reviewed external response subset improve our existing character-plus-style classifier on official passages from held-out prompt groups? This is a component experiment. It is not a reproduction of the 54.9% official S02 ensemble, and a component gain would not establish an ensemble gain.

The corpus is `olegzholobov/name-that-llm-response-corpus`, version 1 (snapshot 2026-10-15). Exact asset checksums are retained in audit.json. Only `external_short_wide.parquet` is a candidate text source. Its all-view and short-view tables overlap; never concatenate them. Exclude the separate Luna quarantine completely.

## Data admission before fitting

1. Resolve source-specific reuse terms, with primary source links and attribution retained. A notebook's Apache licence is not a dataset licence. The corpus-wide licence is Other; a false `license_unknown` flag alone does not establish permissible use. Review SpeechMap first because it records both requested and returned model IDs for all 18 classes. Its repository Apache licence was independently retrieved; its provider-output review flags have not been resolved.
2. Retain only reviewed exact model identities. Dated returned IDs and punctuation aliases require an explicit mapping supported by source records; do not map by model family. The independent Fable source check is evidence for that file only.
3. Require short, text-only, mechanically eligible, nontruncated responses with verified nonempty prompt grouping and no no-training, reasoning-in-answer, or cross-label-shared-text flags. Reject unknown/missing review fields. Treat occurrence evidence and response metadata separately. Deduplicate responses before sampling, while retaining all prompt associations for exclusion.
4. Compare candidate passages against all official training passages using exact, Unicode/case/whitespace, punctuation-normalized, substring, and approximate retrieval checks. Exclude an entire external prompt group when a related response is suspect. Character similarity alone cannot rule out paraphrases.
5. Review candidate prompts against all 150 official prompt groups. The published `pjmathematician/ntl-train-prompts-reconstructed` V1 contains LLM guesses, not actual prompt instructions. It may support conservative exclusions, but must not be called ground truth. Record unresolved prompt independence explicitly. An exploratory score under unresolved overlap cannot promote a candidate.

Save a curated manifest of response IDs, prompt-group IDs, source evidence, mappings and exclusions; do not commit passage text. Resolve duplicate-response links across prompt groups before any fold partition. This is an evidence requirement, not a request for the user to approve an unreviewed corpus.

## Two fixed arms

Both arms use the same five `GroupKFold` splits on official `prompt_id`, the same 18 labels, preprocessing and classifier settings. No random passage split. Every validation fold contains only original official passages, and all passages for a held-out official prompt stay out of official training.

- Control: official fold-training passages only.
- Candidate: the identical official fold-training passages plus the admitted external subset. Before fitting each fold, remove every external group related to any held-out official prompt, and all responses linked to those groups through duplicate IDs. Unknown associations are excluded, not assumed independent.

Use the existing SurfaceStyleFeatures implementation (111 features), standardized and L2 normalized, alongside character TF-IDF (3–5 grams, min_df=3, max_features=120000, sublinear_tf=True). Fit the feature union on official fold-training passages only and reuse that identical mapping in both arms. This isolates added labelled examples from vocabulary/IDF changes. Fit separate LinearSVC heads with C=1, random_state=20261018, dual=True, max_iter=10000, tol=0.0001; record and investigate convergence warnings.

For each fold and class, deterministically retain at most 150 admitted external unique responses, ordered by SHA-256 of `20261018:<response_id>`, after exclusions. Cap one response per external prompt-group/class pair before applying the class cap. Preserve the complete association graph for exclusions even after deduplication.

Keep official example weights n_official_train/(18*n_official_class) in both arms. Where a class has admitted external examples, give its external examples a combined weight of 0.5*n_official_train/18. Do not compensate absent external classes with guessed examples. Report all per-class counts. External weighting is fixed in advance, with no search over caps, weights or C in this pilot.

Both heads use the same optimization settings and an explicit external-data cap. Their row counts and runtimes will differ; this is not a claim of equal wall-clock compute or equal numbers of epochs. Record fit time and convergence for both arms. No transformer, API subscription or paid generation is needed.

## Measurements and decision

Save official IDs, prompt IDs, true classes, fold IDs, both predictions and decision scores. Report pooled official development balanced accuracy across all 18 classes, per-class recall, each fold's score, corrected/broken passages and elapsed time. No hidden-test score is produced by this experiment.

Compute a paired 95% interval for the candidate-minus-control difference by jointly resampling the 150 original prompt groups (10000 draws, seed 20261018), recomputing balanced accuracy on the same resampled rows in both arms. This interval is conditional on the fixed fitted folds and curated dataset; it does not include all training or prompt-overlap uncertainty.

Retain a candidate only for follow-up if the mean difference is positive, the lower interval bound exceeds zero, and source/overlap review has passed. Inspect per-class regressions and source dependence before any promotion. If it fails, report that exact candidate as failed and retain the control; do not automatically try more weights. If it passes, evaluate its marginal contribution against the appropriate existing ensemble predictions before considering an official submission. The incomplete Stage 06E proxy and full S02 remain distinct.

## Execution boundary

The audit script has run on real V1 corpus assets and the checksum-matched official train file. The matched training runner has not been implemented or run, because admission review is unfinished. No development improvement, official score improvement, or 70% forecast is claimed. The laptop was not accessed.
