# Stage 11 — S05 official result: 64.2%

## Current verified outcome — 25 October 2026

Our **S05 / private V6 362129392 succeeded with 0.642 official public balanced accuracy (64.2%)**. A fresh signed-in official submissions row and the separate Submission Details modal both display Succeeded, Score: 0.642 and the exact V6 notebook link. S04 on the same page is 0.639: S05 improves our displayed best by **0.3 percentage points**. The verified gap to 70% is **5.8 percentage points**, calculated with decimal arithmetic. Private leaderboard performance remains unknown.

This is a successful reproduction of Lightning / Vladislav Lavrentev's existing public V28 **362014737**, whose author score is also 0.642. All 33 source code cells remain unchanged, with only a separate output-validation cell added; all 34 saved code-cell strings and all 28 pinned input versions were verified. Matching the source result is not evidence of an original modelling contribution.

The official hidden evaluation completed. Its exact execution duration and detailed output-guard results remain unknown: the inspected Submission Details modal exposes the saved candidate link, with no hidden-runtime log or duration link. The displayed submission age is not an execution duration. The older `hidden_runtime_verified: false` flag means measured timing is unavailable, not that evaluation failed.

Earlier evidence remains separate: the CPU audit passed **175/175 checks** in **686.1 seconds**; the real **12-row Qwen-only probe V3 362125549** succeeded in **418.9 seconds** on two Tesla T4 GPUs; the production **V6 three-row FORMAT ONLY preview** passed **25/25 checks in 21.7 seconds** while skipping full models. Those milestones are not the official score or the hidden evaluation duration. See [actual probe](ACTUAL_QWEN_RUNTIME_PROBE.json) and [format preview](FORMAT_PREVIEW_V6.json).

Final selections remain **2/2, with S03 and S04 selected and S05 unselected**; this result check did not change them. No new submission, training or paid service was launched, and no S05 watch was created or enabled. Historical pending-state receipts and checkpoints below remain preserved.

[Official submissions](https://www.kaggle.com/competitions/name-that-llm/submissions) · [exact S05 notebook](https://www.kaggle.com/code/watchman777/name-that-llm-stage-11-public-642-audit?scriptVersionId=362129392) · [official result receipt](OFFICIAL_RESULT_20261025.json) · [arithmetic and consistency review](OFFICIAL_RESULT_INDEPENDENT_REVIEW.json) · [notebook/version index](NOTEBOOK_INDEX.md).

![Official S05 Succeeded and score 0.642](S05_OFFICIAL_SCORE_20261025.png)

## Earlier CPU completion and probe admission checkpoint — 25 October 2026

Private CPU **V2 362119531 succeeded in 686.1 seconds**, with **175/175 audit checks passing**, including **160/160 mandatory source paths**. It verified 333 actual file hashes. All 17 active NF4 files, including seven large weight shards, were checked against the published QWEN38 addition manifest. The final adapter contains 144 tensors and 62 complete LoRA pairs; its native alphabetic label order is transported to the canonical ensemble order by the unchanged source's named permutation. No head, weight, metadata or source inference code was changed.

The complete saved report was 924,007 bytes, exceeding Kaggle's log and JSON-preview display limits. A separate CPU collector **362123509 succeeded in 57.7 seconds** and transported the exact saved bytes as gzip/base64. All 20 chunks, gzip SHA256 and raw-report SHA256 were verified. The full report's SHA256 is `aa39a49d561a11c4954de2fa7b0728b61d431a07332e5d28fc5fb8f29d2e9601`. See [readable aggregate summary](REAL_INPUT_AUDIT_V2_SUMMARY.json), [exact compressed report](REAL_INPUT_AUDIT_V2.json.gz), and [collection receipt](REPORT_COLLECTION_ACTUAL.json).

Independent review of the actual report admits the **bounded Qwen-only runtime probe**. The sole unresolved historical manifest entry refers to a tensor name in an unused mixed-FP8 asset tree; it is explicitly not counted as a verified file and does not affect the active NF4 root. Source byte pins, active schemas and actual NF4 payload checks remain separately verified. CPU `/tmp` had about 1.126 TB available, comfortably above the 19.448 GB staging copy plus 2 GiB reserve. This is CPU evidence; the GPU instance will measure its own space and environment.

The probe uses the five original worker modules unchanged, a real **900-second process-group deadline**, and 12 unlabeled token-length ranks. Only its admission literal has been populated from this verified CPU report. It has not launched at this checkpoint. CPU checks do not prove GPU execution or the complete hidden-test runtime. Our official best remains **S04 0.639**; no Stage 11 score exists.

The snapshots below retain earlier evidence and the corrected audit block.

## Earlier running checkpoint — 25 October 2026

The exact source V28 **362014737** (author public **0.642**) has been copied privately to watchman777. Our confirmed official best remains **S04 at 0.639**. The copy's UI mount for Name That LLM Artifacts is exactly `/kaggle/input/ntl-olmo2-native24-final-inference-assets/`.

The independently written CPU checker covers all **160 source-required paths**, **94 byte pins**, model/tokenizer and NF4 headers, complete adapter matrices, exact active prefit banks, safe numeric arrays, source metadata, offline wheels and manifest checksums. It imports no Torch/Transformers/custom inference modules and deserializes no joblib/pickle/Torch payloads. Synthetic failure tests and independent review passed before launch.

Actual private CPU **V1 362118178** found all 160 paths and passed the base, prefit, style, Night24, wheel, encoder and adapter inventories. Its Qwen native-label check was overstrict: the frozen source preserves native labels and explicitly reorders columns by name before blending. The checker was corrected to verify the canonical unique 18-label set and record that exact permutation; no inference source, weight or metadata changed. V1's code and receipts remain preserved.

Actual private CPU **V2 362119531** uses the corrected checker, with **37/37 synthetic checks** and an independent source-parity review. All 160 actual mandatory paths and the complete Qwen adapter check have passed. Both runs remain in manifest checksum reconciliation at this checkpoint; no overall real CPU pass is claimed yet.

A separate Qwen-only diagnostic is prepared and locked pending actual CPU V2 admission. It uses five byte-exact published worker modules, 12 unlabeled training passages selected across token-length ranks, isolated offline wheels and a **900-second process-group deadline**. Its 25 synthetic tests and independent review passed without loading a model. It will measure the new branch's execution and numerical checks only; the final adapter saw all official labels, so accuracy on these examples would not be validation. Complete hidden-test runtime will remain a separate limitation.

The live competition Code Requirements impose a **9-hour runtime limit for both CPU and GPU submissions**, with Internet disabled. The account enable dialog showed **29 free GPU hours remaining**; it was cancelled after reading. The editor displayed an original-environment pin, while the CPU viewer reports Latest Container Image; actual distribution metadata is being collected before any runtime-parity claim.

No GPU inference, training, new official submission, paid service or final-selection change has occurred in Stage 11.

[CPU V1](https://www.kaggle.com/code/watchman777/name-that-llm-stage-11-public-642-audit?scriptVersionId=362118178) · [CPU V2](https://www.kaggle.com/code/watchman777/name-that-llm-stage-11-public-642-audit?scriptVersionId=362119531) · [Prospective plan](PLAN.md).
