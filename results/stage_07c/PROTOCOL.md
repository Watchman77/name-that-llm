# Stage 07C: one fixed ensemble-complementarity diagnostic

Prospective status: consumer implemented and synthetic checks passed; actual ensemble test not run. It requires a private existing Kaggle output unavailable to the current cloud account. This is a test of the available seven-member development proxy, **not full S02**.

## Why this test

Stage 07B's matched-class-mass candidate showed a provisional standalone development gain. A weak standalone classifier can still help a stronger ensemble if its errors differ, but component accuracy does not establish that. Test the original-only versus augmented classifier contributions at the same fixed vote weight.

## Frozen input contract

- Original official train SHA-256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`.
- Exact Stage 07B row prediction table SHA-256 `72d7d43119367ef100511d0f8174b9483ec86bf28bfc7b7e61d8fb599ee1209d` (kept outside the public repository).
- Existing `ensemble_arrays.npz` from watchman777's immutable Stage 06E V2 **359514801**, SHA-256 `9348eeed149ad48bdd828d8055e945f57e384d8917c654125c2ebbe8d4b99519`.
- Use only `raw_probability_control`, the prior original-control raw posterior. Do not switch to the previous failed centering candidate or to the potentially contaminated batch-balanced posterior.
- Validate all original row IDs, class names/order, labels, prompt IDs, canonical fold IDs, scenario sizes [1,3,6], training digest, cached predictions and per-member prompt-fold consistency. Validate the original raw scores against the recorded 06E scores before comparison.

The source-derived canonical fold map matches all 2,699 Stage 07B rows (zero mismatches). This is a preliminary alignment check against the recorded map, not verification of the inaccessible binary. The complete consumer refuses hash, row, fold or score mismatches.

## Single fixed intervention

For each row/context, let P be the original raw proxy posterior. Use a hard one-hot vote v from the held-out SVM class prediction. Compare:

- Base: P.
- Control addition: 0.95 P + 0.05 one_hot(original-only SVM prediction).
- Candidate addition: 0.95 P + 0.05 one_hot(matched-mass augmented SVM prediction).

The 5% weight is fixed before seeing an ensemble result and will not be searched. This is explicitly a small hard-vote diagnostic; the one-hot term is a voting rule, not a claim of perfect predictive confidence. SVM margins are not calibrated probabilities and are not passed through an arbitrary fitted-on-validation temperature. No fitting, calibration, readout reconstruction, regrouping, inference, context balancing or new model generation occurs.

## Endpoints and fixed screen

Primary contrast: candidate addition minus control addition in mean balanced accuracy over the three contexts. Also require candidate addition to improve over the untouched base. Resample all 150 original prompt groups jointly across contexts, 10,000 draws, seed=20261018. The contexts repeat the same passages; they are not independent samples.

A provisional proxy pass requires positive mean differences and positive lower 95% interval endpoints for **both** contrasts, no candidate-minus-base scenario regression worse than 0.5 percentage points, and all input gates passed. Save per-class recall and corrected/broken counts against both references. If it fails, retain the base and do not try more vote weights. If it passes, retain only a provisional proxy signal requiring a fuller reference and fresh validation. Never add this difference arithmetically to the official 54.9% score.

Repeated validation use, pre-existing development-selected ensemble weights, incomplete branch coverage, mixed member fold schemes and unresolved semantic prompt independence all limit the claim. The potentially contaminated balanced endpoint is excluded. This diagnostic cannot establish full-S02 improvement or justify an official submission by itself.

## Execution and data handling

Freeze code/protocol and create a consumer-checksum receipt before the real cache is read. Keep original row-level predictions, labels and binary caches outside the currently public Git repository. Save code, receipts and aggregate results there. No trained model or raw competition text is published, no paid service is used, no final selections change, and the laptop stays untouched.

Current access finding: the cloud Kaggle session is `bamideleakinwumi`; the required private output belongs to `watchman777`. The direct notebook page was unavailable in that session and an unauthenticated public-output request returned 404. This is an account-access limitation, not evidence the notebook was deleted. Switch the cloud session to the correct account through secure sign-in before downloading the existing artifact; do not make it public or regenerate it under the other account.
