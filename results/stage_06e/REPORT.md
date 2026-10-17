# Stage 06E: capture once, compare from verified caches

Status: **IMPLEMENTED_REVIEWED_SAVED_SOURCE_VERIFIED_NOT_EXECUTED**. No Stage 06E head fit or ensemble evaluation has run. Confirmed official S02 remains **54.9%**.

## Implemented experiment

The producer is the literal original Stage 06C script with output-only instrumentation. It keeps all fifteen fits, five prompt folds, seeds, sample weights, grouping and existing inference calls. It captures full probabilities, clipped logs, original component evidence, masks/batches, training partitions and all fitted head/scaler snapshots.

A separate preparation module preserves the existing five static-member audits, including public affine readout reconstruction, and writes probability caches. The ensemble consumer only loads the two verified caches. It cannot fit, invoke estimators, reconstruct affine readouts, regroup passages or extract neural features.

The same-run internal cache gate requires exact readback and reconstruction of all twelve component prediction arrays, reconciled class confusions, complete prompt-disjoint coverage, frozen provenance and no non-deprecation fit warnings. Runtime preflight requires the prospectively fixed Python/package versions and two threads. Original Stage 06C comparisons are retained separately; a historical mismatch does not retroactively pass Stage 06D or select a favorable runtime/reference.

The primary endpoint remains the fixed raw ensemble balanced accuracy averaged across 1/3/6-response contexts. Weights, 3,000-draw paired prompt bootstrap/seed and acceptance screen remain unchanged. Batchwise balancing is a potentially contaminated diagnostic and cannot select the candidate. The complete protocol is [the frozen plan](../../plans/STAGE_06E_CANONICAL_CPU_EVIDENCE.md).

## Review and source verification

- 22 focused producer checks, eight consumer fixture groups and 31 independent contract tests passed without any model fitting or inference.
- Independent review preserves the original C/D scripts and plans byte for byte, checks the literal computational AST/call inventory, verifies cache corruption rejection and independently reconciles fixed fusion/balancing/bootstrap arithmetic.
- The inert package validates with `nbformat`; all modules and five code cells compile. A deterministic rebuild produces identical files.
- First four cells preserve the original frozen style transfer payload exactly. Cell sizes are 77,806 / 156,105 / 156,106 / 146,225 / 149,236 characters, below the preview truncation boundary.
- Private Quick Save V1 [359514410](https://www.kaggle.com/code/watchman777/name-that-llm-stage-06e-canonical-cpu-evidence?scriptVersionId=359514410) passed the complete immutable-source comparison for all five strings, all seven input IDs, CPU accelerator and disabled internet. Each cell has zero outputs and a null execution count. Its 8.3-second successful render is metadata-only, not the planned model experiment.

Authoritative hashes: producer `0445e9c3f8d64f72dfedecdc63881262fc796f00ae8275230a232155b8855ce9`; helper `981e07456a4f47a166eb4678aa497786cf92d2cb89bc8a5f204f57b5b4ee5c1b`; static preparation `eb7cae4b47ef8153a0b60b856319ba95b4d985bdc1277544239a35226c0d2a35`; consumer `a1b1d6f63cf1b9e7305ef866814e82a7ef7900659d0b8d3d1c255b90cd243086`. Full notebook/runner/freeze/cell hashes are in `PACKAGING.json`.

## Execution gate and interpretation

Before the one planned Save & Run All, persist and verify the reviewed bytes in GitHub. Run the unchanged private notebook with fresh output directories, CPU and internet off. Preserve any failure and inspect its evidence before proposing further fitting. No official submission, final-selection change, paid service, GPU execution or new neural training is included.

This is an available-component development proxy. Missing full S02 components, reused development prompts/source weights, quantized feature caches and conditional uncertainty limit the claims. A successful screen supports further matched full-system investigation; it does not establish a new leaderboard score or a route to 70%.

Evidence: `KAGGLE_RUN.json`, `EXECUTION_FREEZE.json`, `PACKAGING.json`, `INDEPENDENT_IMPLEMENTATION_REVIEW.json`, `INDEPENDENT_CONTRACT_FIXTURES.json`, `PRODUCER_FIXTURES.json`, `CONSUMER_NOFIT_FIXTURES.json`, `ORIGINAL_TO_CAPTURE.diff`, `CANONICAL_CACHE_SCHEMA.md`, and `IMPLEMENTATION_NOTES.md`.
