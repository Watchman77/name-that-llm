# Stage 06G: one deterministic attention-pair complement test

Status: prospective only. No Stage 06G code, candidate evaluation, prediction correction or submission has executed while preparing this plan. Stage 06F remains a diagnostic result with zero changed predictions.

## Hypothesis and selection limits

Stage 06F V4 `359521305` found a fixed Gemini 3.7/3.8 confusion and within-pair disagreements from two existing attention members. The Qwen/DeBERTa attention members prefer truth on 31/32, 31/32 and 35/34 current cross-errors in contexts 1/3/6, but they also oppose many already-correct pair decisions. These marginal counts cannot predict their joint correction accuracy or prove superiority. The hypothesis is that their equal combined pair evidence can improve the existing ensemble inside one precisely defined ambiguous scope.

The pair, members and scope were chosen after inspecting reused-development diagnostics. This experiment is a prospective fixed **development** comparison; it is not untouched validation. It must not train a meta-model on existing OOF score features. Such features can indirectly expose outer validation labels through the base models used for meta-training features. This candidate has no training at all.

## Immutable inputs and pre-execution gates

Attach only official competition version `170484` and own Stage 06E V2 `359514801`; use their already-saved probabilities and evidence. Freeze the plan, implementation, tests and execution configuration before evaluating the candidate. CPU, internet disabled, fresh outputs; no paid API, GPU, model inference, fitting, regrouping or batch balancing.

Reuse all strict Stage 06F source gates: exact hashes for official train, E ensemble arrays/result/output manifest and three raw-control confusion CSVs; exact IDs/classes/labels/prompts/folds; original-order raw evidence and prediction replay; source scores/confusions and known pair counts. The actual manifest hash is now runtime-confirmed as `a5fcffaef27f4942af8a679370c30c247b8d872b4b6d711eccc0101ea7006dbd`. Preserve the historical local recovered JSON representation separately; do not substitute its serialization.

The central source NPZ hash is `9348eeed149ad48bdd828d8055e945f57e384d8917c654125c2ebbe8d4b99519`. No cached classifier coefficients or embeddings are invoked, and no peer predictions are rebalanced. All saved arrays are numeric/Unicode and later loaded with `allow_pickle=False`.

## Exactly one candidate, no search

Let A be `gemini-3.7-flash` and B be `gemini-3.8-flash`. For each unchanged source row:

1. Read `probability_tokattn` and `probability_tokattn_deb`, each shaped `(2699,18)`. Calculate `log(clip(P, 1e-9, 1))` for the whole saved probability matrix, matching the existing clipping recipe. Each member has fixed coefficient **1.0** in this candidate. No source ensemble weight changes.
2. Fix `d = (logP_qwen[A] - logP_qwen[B]) + (logP_deb[A] - logP_deb[B])` in that exact order. These are static saved member probabilities, so `d` is identical across contexts for a given passage. Do not apply calibration, rescaling, class priors or a learned transform.
3. For each context `[1,3,6]`, derive the ordered top two from the **unchanged raw-control log evidence**, using stable descending sort with original class-index tie order. The scope is true only if the two labels are exactly `{A,B}`. There is no margin/confidence threshold or ground-truth-based routing.
4. Copy source raw-control predictions. Inside scope, set A when `d > 0`, B when `d < 0`, and retain the source prediction when `d == 0`. Outside scope, retain the source prediction exactly. No epsilon tie rule or alternative threshold is tested.

Do not compare alternative members, weights, thresholds, gates, combinations or text features in this execution. There is one candidate and one evaluation. Any later revision is a separate experiment with the previous outcome retained.

## Structural invariants before scoring

Require source scope sizes `[193,196,190]`, scoped cross counts `[68,62,71]`, scoped correct pair counts `[101,113,101]` and scoped already-wrong nonpair counts `[24,21,18]`. These are original-source facts, not acceptance targets.

Assert:

- Every outside-scope prediction is unchanged; every scoped source prediction already belongs to `{A,B}`; every candidate change toggles only A/B.
- The pair-entry mask is exactly unchanged in every context, so no new nonpair-to-pair entry is possible.
- Every nonpair row's correct/incorrect indicator is unchanged; each other class's confusion diagonal and recall are exactly unchanged. False-entry destination counts can change and must be reported.
- True-pair rows whose source top one was another class are unchanged. They are outside this candidate's recovery scope.
- Corrected/broken counts reconcile exactly to the total change in correct predictions and to the two affected class diagonals. No assertion restricts results to a favorable sign.
- Probabilities, raw-control evidence, source predictions and scope are not overwritten. Save the single candidate prediction array separately with hashes and exact readback.

## Evaluation and fixed decision rule

Primary endpoint: candidate-minus-control mean **raw rowwise balanced accuracy across all eighteen classes** and all three contexts. Reconcile complete confusion matrices, class supports/recalls, fold scores and per-context correct/broken/changed-both-wrong counts. Report class-specific Gemini regressions and the structural invariance of every other class. Do not score only the target pair, only changed rows or only scoped examples as the primary endpoint.

Use the existing paired prompt bootstrap definition: 3,000 draws, seed `20261017`, jointly resample the original 150 prompts with replacement across contexts, aggregate corrected-minus-broken counts and class supports, average class recall differences and contexts. Keep all probabilities, predictions, fits, folds and routing fixed. The 95% interval is conditional on reused-development models/evidence; it omits refitting, scope-selection and future-data uncertainty.

The inherited development screen is mean gain > 0, conditional 95% interval lower bound > 0, and every context gain >= `-0.005`, with all integrity/numerical gates passed. Class harms are reported, not used as an invented post-hoc veto. No weight, threshold or gate is reselected if the candidate fails. A positive screen supports further matched full-system investigation only; it does not establish an official score or out-of-sample improvement.

Report corrected/broken predictions on true-pair rows, scoped nonpair destination changes, exact-tie counts, scope coverage and each of the two class recall changes. Do not claim the 78/79/83 member-disagreement counts are recoverable gains. Retain an unsuccessful or neutral result.

## Interpretation and stop conditions

Even an oracle correction of every cross-error inside this scope could add at most 2.4815 percentage points to this available-component proxy mean, while the experiment could lose accuracy by breaking currently correct pair decisions. This is a bounded component correction, not a route by itself to 70%.

The proxy remains incomplete relative to S02. If the fixed candidate passes, the next decision is whether to test this exact rule with all required S02 components and inference constraints; no submission or final-selection change is included here. Official S02 remains separately confirmed at 54.9%.

Stop after the one planned no-fit CPU comparison. Preserve any integrity failure for inspection before a retry, and preserve failed candidate performance without automatically trying a variant. No fitting, inference, GPU, paid service, official submission or messages to other people are part of Stage 06G.
