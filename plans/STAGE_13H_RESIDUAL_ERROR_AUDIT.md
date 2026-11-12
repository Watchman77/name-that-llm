# Stage 13H: diagnose errors outside the 12–18 assignment rule

Stage 13G did not justify changing the group-size cutoff. This audit investigates where the frozen Stage 13F predictions still fail. Do not modify predictions, assignment, model weights, or parameters. The existing score benchmark is S07's verified **65.1%** official public balanced accuracy; this audit is on the reduced development proxy and cannot update that benchmark.

Use the exact competition train snapshot 170484, the frozen GLM OOF output 368292220, and the pretrained DeBERTa features 358745836. Reconstruct the canonical five held-out folds, Stage 13F's text-only grouping, and the 75% GLM / 25% DeBERTa balanced evidence. Require the original fold geometry and Stage 13C score anchor. Replay the Stage 13F assignment-on confusion digest `6b3a2cd77f8b7d5c24f343dd685e3be63aaa63b3635d59b6ccefc2528d1ae696` (1,659 correct, 61.461844% balanced accuracy) before reporting the audit. The earlier 10–11 extension is excluded.

Report the **810 unassigned rows** separately from the 1,889 assigned rows. For each, reconcile class supports, recalls and final errors, plus pre-assignment argmax correctness. Differences between these strata are descriptive: they contain different passages. The paired pre/post change on the same assigned rows is a distinct measure.

For unassigned mistakes, count the original score rank of the true class: rank 2, rank 3, or below rank 3. Report top directional errors, their support and fold spread; fixed group-size and held-out fold summaries; and fixed top-two balanced log-score margin bins `[0,.10)`, `[.10,.25)`, `[.25,.50)`, `[.50,1)`, `[1,2)`, and `[2,+∞)`. These diagnostics distinguish close decisions from missing class evidence. Top-k inclusion is an oracle count, not an achievable gain. A proposed targeted fix requires a new protocol and must be tested separately.

Run one private offline CPU notebook, export only aggregates and checksums, and verify the saved report. No hidden test access, training beyond replaying the fixed fold-local readout, GPU, paid service, competition submission, or final-selection change.
