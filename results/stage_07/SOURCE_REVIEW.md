# Stage 07 follow-up review — 18 October 2026

Execution follow-up: the bounded direct-OpenAI subset was admitted for a private, undistributed SVM pilot and Stage 07A has now completed. Its [result](../stage_07a/REPORT.md) is +0.925926 development percentage points with a confidence interval spanning zero; the fixed improvement screen failed. The all-route admission remains unresolved. A fresh repository metadata check also found the repository public, correcting earlier private-repository descriptions; trained models and official row-level prediction artifacts are not published.

**Decision: deprioritize bulk SpeechMap augmentation; prioritize the independently checked 4,068-row `new` subset of prvsiyan's extended V4 for the next admission review.** No model has been fitted and no accuracy gain has been demonstrated. The reason for changing priority is measured data suitability, not a failed training run. Source/use permissions and prompt independence are qualified below, rather than treated as automatically resolved by a dataset title.

## SpeechMap: provenance stronger, transfer suitability weaker

Every one of the 8,658 selected short responses was checked against all 25 frozen primary JSONL files at commit `476ea8f0adf85b737bbb6e38e838448a8d4f2a9b`. Exact answer text, exact user prompt, requested model ID and returned model ID matched for every cell, with zero mismatches. The two Git LFS objects were downloaded as data and checked against their declared SHA-256 OIDs. All 18 labels are represented. The 32 observed requested/returned/target combinations are recorded separately: thinking/reasoning aliases and dated snapshots are visible, not silently assumed to be generation-equivalent settings.

All finish reasons were `stop` (8,478) or `end_turn` (180). Some older source rows lack a top-level status (640); this is not the same as a failed response. Source inference providers differ, and collection dates range from February to September 2026. Primary files and per-model coverage are recorded in `speechmap_primary_review.json`, `speechmap_model_review.csv` and `speechmap_model_id_mapping.csv`.

The material problem is distribution mismatch. A fixed regex for refusal phrases in the first 500 characters marks **5,789/8,658 (66.86%)** SpeechMap responses versus **8/2,699 (0.30%)** original training responses. This is a diagnostic marker, not an exhaustive or perfectly specific refusal annotation. Manual examples for the Gemini variants are actual refusal openings. Every selected SpeechMap Gemini 3.6 (67), 3.7 (42) and 3.8 (47) response triggers the marker. Removing marked responses leaves no examples for any of these three labels in this source. This weakens the argument for using it to address the earlier Gemini-family confusion in ordinary prose.

The rest of the older corpus does not solve the coverage issue by itself. Its Gemini 3.6 remainder is 14 petergpt benchmark cells and two Artificial Analysis cells; Gemini 3.7 has 24 and 11 respectively; Gemini 3.8 has only 24 Artificial Analysis cells. The former retain output-review flags and the latter unknown-output-licence flags. These are not thousands of verified ordinary-writing examples. See `gemini_source_review.csv`.

Prompt screening covered 1,604 distinct SpeechMap prompt groups against the 150 published reconstructed instructions. There were no exact matches; maximum lexical cosine was 0.12739. The 20 closest retrieved pairs were manually inspected. Four shared-topic groups (CRISPR/space) were marked for conservative exclusion, **not** identified as confirmed duplicate instructions. The remaining reviewed pairs had different instructions in the reconstructions. This does not certify all groups or semantic independence, since the official prompt wording is unknown. Neither this screen nor the previous full-response checks justifies treating random passage splits as valid.

## A more suitable alternative located and checked

Source: https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset , **version 4**, downloaded without generating or purchasing responses.

The complete clean file has 4,584 rows. We isolated its `new` split only: **4,068 passages = 226 complete prompts × 18 models**, exactly 226 rows per class. Older proxy-only batches, the shared-range pilot and Kimi regenerations are excluded. All selected answers have 40–250 whitespace words. Exact and normalized duplicates within this subset are zero. Exact, Unicode/case/whitespace and punctuation-normalized full-response overlaps with official train are zero.

All 4,068 selected rows link uniquely to published API-call envelopes. We parsed the actual response bodies across Anthropic, OpenAI/OpenRouter, Google and proxy schemas. For every row, response text, served model ID, request model ID, exact sent prompt, envelope label and prompt ID match. Every response text was byte-exact as decoded Unicode, so the comparison required no whitespace repairs. These are the publisher's supplied logs, not independent attestations from the providers.

For Gemini 3.7 specifically, all 226 selected rows have Google-route raw `model_version = gemini-3.7-flash`. This is stronger evidence than only a request alias or proxy echo, and matters because the publisher warns of version aliasing on another route. We did not independently query a live model to test that warning.

Only one answer triggers the same refusal regex: `n045__qwen3.8-max__a0`, an explanation of learned helplessness. Manual review shows it uses the phrase “I cannot” while describing the concept; it does not refuse the prompt. Thus this diagnostic found no actual refusal in the selected block set, but all 4,068 passages were not manually annotated for refusal.

The 30 closest retrieved candidate/reconstructed-official instruction pairs were inspected; they concern different instructions, with no confirmed same-prompt pair found. Maximum lexical cosine was 0.10218. The full screen covers all 226 candidate groups; the manual assessment covers the top 30 retrieved pairs, not every possible semantic pairing. Prompt independence remains qualified by guessed official instructions. Exact response non-overlap is a separate, stronger executed check.

## What still needs control

The new corpus is not automatically a solution. Its per-label generation styles differ from official train. In the selected subset, em-dash presence is 20.80% for Gemini 3.6, 27.43% for Gemini 3.7 and 29.65% for Gemini 3.8, versus 0%, 0.67% and 0% in official train respectively. Do not normalize only external punctuation and then call the experiment matched; that would introduce another intervention. The first data-only comparison should preserve the fixed preprocessing and measure the effect honestly.

The selected rows use five routes: Anthropic 1,130; OpenRouter 944; direct OpenAI 750; Google 720; Kaggle proxy 524. The proxy does not expose the same independent provider-version detail. The publisher's CC BY 4.0 declaration, private-classifier permissions and competition-distribution rights must be distinguished. See `TERMS_REVIEW.md` for primary source findings, including OpenAI's explicit undistributed-classifier exception, and the remaining route-specific evidence gaps. Covering fewer external classes is acceptable in principle if official training/validation still covers all 18 and weights are fixed.

The next useful step is a **small source/route-admitted comparison on this V4 candidate**, with original-only control, original plus admitted external responses, and original held-out prompt groups as validation. Freeze the exact candidate IDs, route decisions and exclusions before fitting. A component gain would still need an ensemble contribution test; it cannot be reported as a new S02 score. The earlier matched-run protocol remains prospective, and its runner has not been implemented or executed.

## Evidence and limits

New scripts: `scripts/stage07_speechmap_review.py` and `scripts/stage07_extended_review.py`. Both executed successfully on real saved assets. Counts were reconciled from the row tables; source and model-ID checks passed. The scripts compile. No independent agent review was performed. Detailed JSON receipts, mapping tables and prompt-pair assessments accompany this report. Raw responses and collection headers are not committed; the scripts regenerate their output from the public pinned assets. Full per-cell scratch tables are regenerable and their hashes are retained where relevant.

This review did not access the laptop, use paid APIs, send messages, train models, submit predictions or alter final selections. It produced a better candidate and eliminated an unsuitable bulk-data assumption. It produced no new accuracy result.
