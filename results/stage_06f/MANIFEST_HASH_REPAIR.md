# Stage 06F source-manifest hash repair

The first actual diagnostic run, private Stage 06F V2 `359520233`, stopped at its strict `output_manifest.json` byte-hash gate. The official training file, ensemble numeric cache and ensemble-result hashes had already passed. No diagnostic result was produced, and the failed source/run remain retained.

The expected hash `626b428135f4ff377d593de29c716042a07593c74678893f874c91b7b71e2bbf` described the locally recovered Stage 06E JSON representation. That representation orders each manifest entry's keys as `bytes`, then `sha256`. The immutable Stage 06E consumer constructs every entry as `dict(sha256=..., bytes=...)` and writes `json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"`. Python preserves insertion order. Reconstructing all twenty unchanged records in the frozen writer's field order yields 2,883 bytes and writer-derived SHA256 `a5fcffaef27f4942af8a679370c30c247b8d872b4b6d711eccc0101ea7006dbd`.

Fresh authoritative Stage 06E UI inspection confirmed the twenty records and the `sha256`, then `bytes` field order. The original raw-file download event timed out, so the proposed SHA256 is **writer-derived and not yet independently verified against downloaded original bytes**. The retry retains exact byte-hash checking; it must match the actually attached immutable file before any diagnosis can proceed. A passing retry gate would confirm this byte hash at runtime. A further mismatch remains a preserved integrity failure, not a reason to relax checking.

The only production-source change replaces this one expected manifest hash. All numeric-cache/result/training/confusion hashes, source alignment, evidence replay, prediction equality, score gates, diagnostic mathematics, bins, bootstrap, no-fit scope and outputs are unchanged. The prospective plan is unchanged. The old local recovered manifest is retained unchanged and is not relabelled as original Kaggle bytes.

Original diagnostic source SHA256: `b264d2f7398fda6127c306057c2e41f3d170cd430bd1d115189f7ddf40fa8d64`.
Repaired diagnostic source SHA256: `a8bb68181087640fb563539870840117531a4d57e529cbeaf69e73d4c0a05be7`.

A separate independent repair review and fresh package/source freeze are required before the single controlled retry. This note does not assert that the retry has run or passed.
