# Stage 12A independent review

**Pass, pending the actual published pre-execution freeze.** No material leakage or arithmetic bug was found. Fourteen checks cover source parity, canonical folds, matched fitting, correction, bootstrap, screen, privacy and resource controls. No real or toy classifier was fitted.

The source correction cell matches the original capture exactly; all five function ASTs and both constant sets match. An independent no-fit preflight reproduced the pinned 2,699 rows, 150 prompts and validation sizes 540/540/540/540/539. The vectorizers and shared SVC train only on outer training rows. Topic grouping receives held-out texts only, while known prompt IDs are restricted to splitting and evaluation.

Independent heterogeneous-cluster correction and unequal-support prompt-bootstrap calculations match the runner. The screen matches the protocol: gain at least 0.5 percentage points, lower 95% interval above zero, maximum class loss at most 5 points and zero convergence warnings. Row outputs remain outside the repository; aggregate outputs contain no original IDs,texts,predictions or margins.

The command-line 40-hex check is an accidental-run guard, not cryptographic Git verification. Before real fitting, the root orchestrator must verify that the published commit contains the exact runner, protocol and check hashes recorded in [INDEPENDENT_REVIEW.json](INDEPENDENT_REVIEW.json). The declared 900-second parent deadline covers the serial child, with group termination on timeout.

The bootstrap conditions on the fixed inferred groups and predictions; it does not recluster in each resample. Any result remains an exploratory component screen on reused folds, not independent confirmation, full S05 OOF or an official score. No further fitting or variant is needed for review admission.
