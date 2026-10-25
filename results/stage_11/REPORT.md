# Stage 11 — actual CPU audit in progress

## Current checkpoint — 25 October 2026

The exact source V28 **362014737** (author public **0.642**) has been copied privately to watchman777. Our confirmed official best remains **S04 at 0.639**. The copy's UI mount for Name That LLM Artifacts is exactly `/kaggle/input/ntl-olmo2-native24-final-inference-assets/`.

The independently written CPU checker covers all **160 source-required paths**, **94 byte pins**, model/tokenizer and NF4 headers, complete adapter matrices, exact active prefit banks, safe numeric arrays, source metadata, offline wheels and manifest checksums. It imports no Torch/Transformers/custom inference modules and deserializes no joblib/pickle/Torch payloads. Synthetic failure tests and independent review passed before launch.

Actual private CPU **V1 362118178** found all 160 paths and passed the base, prefit, style, Night24, wheel, encoder and adapter inventories. Its Qwen native-label check was overstrict: the frozen source preserves native labels and explicitly reorders columns by name before blending. The checker was corrected to verify the canonical unique 18-label set and record that exact permutation; no inference source, weight or metadata changed. V1's code and receipts remain preserved.

Actual private CPU **V2 362119531** uses the corrected checker, with **37/37 synthetic checks** and an independent source-parity review. All 160 actual mandatory paths and the complete Qwen adapter check have passed. Both runs remain in manifest checksum reconciliation at this checkpoint; no overall real CPU pass is claimed yet.

A separate Qwen-only diagnostic is prepared and locked pending actual CPU V2 admission. It uses five byte-exact published worker modules, 12 unlabeled training passages selected across token-length ranks, isolated offline wheels and a **900-second process-group deadline**. Its 25 synthetic tests and independent review passed without loading a model. It will measure the new branch's execution and numerical checks only; the final adapter saw all official labels, so accuracy on these examples would not be validation. Complete hidden-test runtime will remain a separate limitation.

The live competition Code Requirements impose a **9-hour runtime limit for both CPU and GPU submissions**, with Internet disabled. The account enable dialog showed **29 free GPU hours remaining**; it was cancelled after reading. The editor displayed an original-environment pin, while the CPU viewer reports Latest Container Image; actual distribution metadata is being collected before any runtime-parity claim.

No GPU inference, training, new official submission, paid service or final-selection change has occurred in Stage 11.

[CPU V1](https://www.kaggle.com/code/watchman777/name-that-llm-stage-11-public-642-audit?scriptVersionId=362118178) · [CPU V2](https://www.kaggle.com/code/watchman777/name-that-llm-stage-11-public-642-audit?scriptVersionId=362119531) · [Prospective plan](PLAN.md).
