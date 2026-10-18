# Stage 07 corpus audit — 18 October 2026

Follow-up: [SOURCE_REVIEW.md](SOURCE_REVIEW.md) records the completed all-model SpeechMap source check, its refusal-heavy distribution, and the stronger extended V4 candidate. The initial audit below is preserved as its earlier checkpoint; it is not the latest candidate recommendation.

The public external corpus is real and accessible without a paid API. Its short-answer view has been downloaded and independently audited. It is not yet an admitted training set. No augmented classifier has been fitted, and no new development or official score exists for this stage. Previously verified best official public balanced accuracy remains S02 0.549; this audit did not recheck the leaderboard.

## Verified findings

Source: [Oleg Zholobov's Response Corpus, V1](https://www.kaggle.com/datasets/olegzholobov/name-that-llm-response-corpus), snapshot 15 October 2026. The seven required stable assets match the publisher's SHA-256 manifest. The original official train file was recovered from the user's saved competition download and matches the project's frozen checksum. No laptop browser or local laptop process was accessed.

- 13,642 wide rows; 24,025 occupied answer cells; 24,015 unique response IDs; 12,032 prompt groups; all 18 exact target label strings present.
- All short answers have 40–250 whitespace words. Required row/cell/provenance joins and word counts reconcile. There are 24,110 provenance occurrences; these are not independent examples. One response ID belongs to multiple prompt groups and requires linked-group exclusion during validation.
- No full-response matches against any official training passage under exact, Unicode/case/whitespace, or punctuation normalization, regardless of label. No normalized external response is shared across labels.
- In a bounded retrieval diagnostic, the highest nearest-official character TF-IDF cosine similarity was 0.338630; no candidate reached 0.9. This does not certify semantic or prompt independence. The script did not execute exhaustive substring or semantic checks, and did not independently audit the full all-view response text.
- Coverage is highly uneven: Gemini 3.6, 3.7 and 3.8 have 83, 77 and 71 short-answer cells respectively. Corpus size alone is not evidence of useful complementary signal.
- The source-wide licence is **Other**, with source-specific terms retained. Among short-view provenance occurrences, 9,039 have unknown-licence flags and 10,544 have output-terms-review flags. These are occurrence counts, not distinct responses, and the categories need not be disjoint. A review flag is not a finding that use is forbidden.
- The conservative flag filter yields 4,446 distinct response IDs across only 10 classes. That filter does not independently approve their labels or rights; do not describe this as an approved all-18-model corpus.
- SpeechMap supplies 8,658 distinct short response IDs across all 18 classes, with requested and returned model identity evidence. Its repository Apache-2.0 licence was independently read at commit `476ea8f0adf85b737bbb6e38e838448a8d4f2a9b`. All its 8,658 short provenance occurrences retain provider-output review flags. Repo licensing and provider-output review remain separate facts.
- An independent primary-source check compared all 352 selected Fable cells from `us_hard_anthropic_claude-fable-5.jsonl` to that frozen original source. Response text, exact user prompt and returned model ID all matched. This check covers one file/model, not independent verification of every model.
- [Reconstructed competition prompts, V1](https://www.kaggle.com/datasets/pjmathematician/ntl-train-prompts-reconstructed) were located and read. Their description explicitly identifies them as LLM reconstructions from official answers. They are useful review aids, not ground-truth prompt wording.

The publisher's short-view cell, class, unique-response and prompt-group counts independently reconcile with our calculations. The full competition-rules page did not return readable content through the fresh public lookup; do not treat this audit as a fresh full-rules approval.

## Decision and next execution condition

Hold training on the whole corpus. The immediate review target is a source-restricted SpeechMap pilot: finish source/output-use review and model-ID mapping, conservatively exclude prompts/responses related to official held-out groups, and retain the unresolved limits. Unknown prompt associations cannot silently be treated as independent.

The two-arm design is specified in [PROTOCOL.md](PROTOCOL.md): official-only versus official plus admitted external responses; identical official prompt folds, fold-local fixed feature map, character/style features, fixed SVM settings, fixed external caps and weights; official passages only in validation. It is a cheap component experiment, not an expected 70% solution. The matched training runner is not implemented or running. No request for the user to reopen the laptop or buy API access is required.

Admission review is unfinished; that is the material blocker. The audit's outcome is evidence about data suitability, not an accuracy gain. No external messages, paid services, official submissions or final selections were made.

## Reproduce and inspect

Run `python scripts/stage07_corpus_audit.py --corpus <V1-corpus-directory> --train <official-train.csv> --out <output-directory>` with numpy, pandas, pyarrow and scikit-learn installed. The audit script rejects altered corpus assets or a different official training snapshot. It does not train models or export passages.

Inspect `audit.json`, `source_receipt.json`, class/source coverage tables and `nearest_official_responses.csv`. The last file contains response IDs, official row IDs, prompt IDs and similarity values only. Original competition text, external response text and the raw source spot-check file are deliberately not included in this repository update.
