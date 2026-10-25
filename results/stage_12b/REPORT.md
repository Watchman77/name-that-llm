# Stage12B — prepared fixed official counterfactual

The candidate is prepared for independent review and source freeze. No full model run, intervention activation or official score is established by this preparation.

Parent S05 is officially **0.642 public balanced accuracy**. This candidate activates one existing fixed SVC topic-correction call after the complete SVC/decoder/DeBERTa reference sum and before complementary mixing. Model weights, model worker code, all 28 input-version IDs and every downstream blend remain unchanged. Upstream orchestrator code cell30 deliberately changes; the other 32 source cells and original output guard34 are byte-for-byte preserved. Guard35 is separate.

The Stage12A component gain was **+2.149391 development percentage points**. Its SVC-only uncertainty gate differs from the complete-reference gate in this official candidate, and full-ensemble gains are unknown. One blind official test is predeclared; no automatic coefficient, threshold, placement or blend variants.

The new guard replays the correction once, reconstructs its complete reference input from saved SVC/decoder/DeBERTa scores, and verifies that the corrected total reaches the saved pre-ATLAS mixture. The inherited Stage11 guard remains unchanged and still contains its older statement that SVC logits are unavailable to its own checks. That limitation applies to the old guard's scope; the separate Stage12B evidence now supplies those margins to the new guard.

The source three-row preview bypasses models and the intervention. Its only valid intervention result is **FORMAT_ONLY_PREVIEW_INTERVENTION_NOT_EXECUTED**. Preparation's 16/16 synthetic checks and all-35-cell static compilation are not a hidden-test result.

See [protocol](PROTOCOL.md), [minimal source patch](SOURCE_PATCH.diff), [build receipt](BUILD_RECEIPT.json), and [synthetic validation](SYNTHETIC_VALIDATION.json).
