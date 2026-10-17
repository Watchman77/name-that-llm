# Stage 06E implementation and execution boundaries

The generated package is preparation only. Building or validating the notebook
does not execute the five-cell pipeline, fit a classifier, infer probabilities,
or contact Kaggle. The execution freeze must be regenerated after any reviewed
module changes and committed before the one planned CPU execution. The separate
immutable saved-source/settings inspection is still required before launch.

## Five-cell package

`scripts/build_stage06e_package.py` deterministically creates the notebook,
concatenated runner, source freeze and packaging manifest. Its first four cells
are the exact Stage 06D style payload strings. Cell five retains the exact
Stage 06D style transfer functions, followed by reviewed Stage 06E files stored
as inert Python string literals and the pipeline entry point. Each cell remains
below 190,000 characters to avoid the earlier source-preview truncation.

The pipeline materializes and hashes the four Stage 06E modules, literal
original Stage 06C source, prospective plan and execution freeze. The freeze
contains module hashes but does not contain its own hash, notebook hash or
runner hash. The external packaging manifest records those hashes; the runner
passes the exact serialized freeze hash to the consumer. Existing source,
pipeline, style, static, canonical or ensemble directories stop the pipeline
before they can be overwritten, including existing empty directories.

## Three computation phases

1. **Frozen style transfer and static-member audit.** The exact frozen style
   probability bytes are reconstructed without fitting. The static preparation
   retains the already reviewed Stage 06D public-artifact provenance checks,
   including affine readout checks against the archived completion artifacts.
   These checks do evaluate existing public readouts once; they do not train a
   neural model or classifier. They write a separately gated numeric cache.
2. **Original-C canonical evidence producer.** `runpy.run_path` executes the
   reviewed output-instrumented original computation. The cache helper requires
   exact numeric package versions and two-thread limits before its first fit.
   The fifteen planned fits preserve the original arithmetic, grouped splits,
   seeds, batch construction, eligible-only centered inference and parameters.
   Captured values, scaler/head state and failures are archived. A same-run
   replay/readback gate controls whether the cache is usable; historical C
   mismatches remain a separate unresolved comparison.
3. **Pure cache consumer.** The consumer verifies both cache manifests, hashes,
   gates and coordinate arrays and then assembles the fixed paired ensembles
   directly from saved probability/log-probability arrays. It performs no fit,
   affine readout, estimator call, group recovery or batch reconstruction. The
   original fixed weights and raw comparison screen are unchanged. Marginal
   balancing remains a potentially contaminated diagnostic.

The producer wrapper calls `preserve_active_failure` before reraising an error.
The outer pipeline also records its phase, exception and traceback without
restarting or deleting any phase outputs. A failed producer or cache gate blocks
the ensemble consumer and its successful result marker. No stage in this
package submits to the competition or changes a final selection.

## Interpretation

The package produces an available-component development proxy. It does not
establish a full-S02 score, an official public score or progress to 70%. A valid
new canonical cache cannot retrospectively pass Stage 06D's failed historical
reproduction gate. Exact source hashes and package checks establish preparation;
only preserved live execution and final output gates establish a result.
