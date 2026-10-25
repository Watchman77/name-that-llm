# Stage 11 — runtime and output review

Static review of the 33 captured cells from source V28 **362014737**. No source, model or serialized payload was executed. The author's 0.642 remains distinct from our S04 0.639.

## Runtime gate

The internal budget is genuinely untimed: remaining time is infinity, required-duration checks do nothing, the parallel runner ignores `max_seconds`, and ATLAS never uses its `--seconds` argument. The competition-specific limit is **9 hours for CPU and GPU notebooks**, with Internet off. Root verified this in the live competition Overview Code Requirements on **25 October 2026**; generic Kaggle duration limits do not override it. Root also read **29 free GPU hours remaining** in the GPU turn-on modal and cancelled without enabling GPUs. The local capture contains no verified hidden full-run runtime; its 22.7-second preview skips inference.

The new Qwen worker requires exactly two GPUs and runs the portable fp16 path across all 64 decoder layers. Batch size is three with a single-row OOM retry. It never truncates to a fixed length; native-context overflow fails. The explicit NF4 copy totals **19,447,911,132 bytes** and requires free temporary space strictly above **21,595,394,780 bytes**, before other simultaneous temporary data is considered.

All 160 source checks remain mandatory, including historical inactive members and 112 literal-hub paths. The source's PROVEN ledger is provenance evidence, not verification of our actual mounts or bytes.

## Minimal meaningful diagnostic after CPU admission

Use a separate private diagnostic notebook and the unchanged `own_qwen_worker.py`, the same offline wheels and the same staged NF4 files. Append a diagnostic-only wrapper to a byte-exact copy of the source worker materialization. Choose a fixed 12–18 official-text sample across token-length strata, including the long tail, without using labels for selection. Apply a real parent subprocess timeout such as 1200 seconds. Record load/staging time, tokens and seconds per batch, OOM retries, two-GPU peaks, finite ordered logits and declared full-depth/no-truncation metadata. Never submit this diagnostic notebook or describe in-sample accuracy as validation: the final adapter fitted all 2699 official rows and has no holdout.

This measures the new branch only. It cannot prove full-ensemble runtime or hidden-length throughput. A full-stack query override is not trivial: the complement reads its own fixed ledger test path and insists that path is under `/kaggle/input`. Merely changing the parent query path can create axis mismatches. Any diagnostic routing patch must remain separate and explicit.

## Output guard

Saved member and mixture arrays live below `/kaggle/temp/ntl636-fulltext-qwen-query`; the source only persists final parquet and logs to working output. A final validation cell can read the temporary arrays before notebook exit. Required checks include exact ID/class axes, finite and normalized values, selected ATLAS rows, 90/10 ATLAS blending, 90/10 Night24 blending, all-row Qwen 10% blending, the one final 300-step Sinkhorn (for at least 50 rows), and actual parquet readback against the reconstructed argmax. No final balanced probability array is saved by the source, so reconstruct it.

The full pre-ATLAS SVC+decoder+DeBERTa sum cannot be reconstructed solely from the saved files because raw SVC logits are not separately saved. A guard must state that limit instead of claiming complete internal reconstruction. Final finiteness also does not prove every internal activation is finite.

Detailed file names, exact fields, static evidence and limitations are recorded in `RUNTIME_AND_OUTPUT_REVIEW.json`. Export aggregate checks/hashes only to public Git.


## Live competition limit and available quota — 25 October 2026

The authoritative competition Overview, Code Requirements section, states **CPU and GPU notebooks each have a 9-hour runtime limit**, with Internet disabled. A general Kaggle notebook allowance does not override this competition limit. The connected-account GPU enable dialog separately reported **29 hours remaining** out of the weekly 30-hour allocation; the dialog was cancelled after reading, leaving the CPU audit on None. Neither observation measures the candidate's runtime.


## Verified competition boundary — 25 October 2026

The live competition Overview also states 40–250-word train/test passages, unequal model counts, and different public/private model mixtures. Uniform Sinkhorn balancing remains the frozen source heuristic; it is not a known true class-count constraint. This observation does not change the production recipe.
