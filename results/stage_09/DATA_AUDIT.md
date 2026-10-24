# Stage 09 data admission audit

Status: **PASS_FROZEN_PILOT**.

The immutable V4 selection contains 750 direct OpenAI rows in 150 five-label prompt blocks. 750 rows / 150 blocks remain after whole-block source and response screens.

All 750 selected published call envelopes match response, requested/served identity, actual request prompt, route and label. All 750 actual prompts contain the mapped prompt-bank body. All selected prompt-bank records declare `own (written for this dataset, host-style protocol)` and `CC BY 4.0`.

Response screening found 0 thresholded rows and excluded 0 prompt blocks. Maximum character cosine to any official response: 0.233093. Checks include exact/normalized/punctuation response equality, full-passage substring, at least three shared distinct word 8-grams and character cosine at least 0.85.

Fold 1 is frozen at 2,159 training responses/120 prompts and 540 validation responses/30 prompts, with all 18 labels in each. Its ordered validation-ID hash matches the actual earlier Stage 07A fold. All five memberships were additionally compared to the saved Stage 07A OOF file in this check.

**Admission is not semantic-independence certification.** Official prompt wording is unavailable. The own-written origin declaration and published call envelopes are evidence from the dataset publisher. The blank source_prompt_id field is documented; actual prompt-body/request association is checked. Unknown origins and explicit official reconstructions/train_regen derivatives are rejected as whole external blocks. The original fold remains reused development data.

This audit does not train a classifier or use an inference API. It admits a bounded private discriminative-classifier experiment; it grants no public trained-weight or generative-model-use approval. Public artifacts contain aggregate results and external identifiers/hashes only.

Self-test fixtures: 13/13 passed. They exercise immutable-input tampering/missing files, unknown and reconstructed origins, train_regen, unknown licensing, invalid splits/empty prompts, and actual-call identity/prompt mismatches.

Corpus, source, manifest and split hashes are in DATA_AUDIT.json and split_receipt.json.
