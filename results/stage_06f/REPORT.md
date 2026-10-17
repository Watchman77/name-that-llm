# Stage 06F: completed Gemini pair diagnostics

Private Kaggle [V4 359521305](https://www.kaggle.com/code/watchman777/name-that-llm-stage-06f-pair-diagnostics?scriptVersionId=359521305) completed successfully in **20.3 seconds on CPU**, with internet off and 1.08 MB of output. The diagnostic used only official train and completed Stage 06E V2 output. It performed **zero fits, inference calls, regrouping or changed predictions**. No corrective candidate or official submission was evaluated. Best confirmed official S02 remains **54.9%**.

## Source verification and retained failure

The original V2 359520233 failed after 19.5 seconds at the exact manifest byte-hash check, before diagnostic analysis. Its expected hash had been computed from a JSON-tree reconstruction with each entry ordered `bytes,sha256`; the original E writer and fresh Kaggle preview used `sha256,bytes`. Every record had the same values. The original failed source, log and receipts remain preserved in Git and immutable Kaggle versions.

A separately reviewed, constant-only repair retained all strict checks. The repair was saved and verified in Quick V3 359521070, then frozen at commit `1c9f0650f01d2adc43731876f4fe7e8439f34f48` before V4 execution. V4's saved five-cell source matched V3 exactly; metadata pinned competition 170484 and E output 359514801, with no GPU or internet. Runtime verified the original manifest SHA256 `a5fcffaef27f4942af8a679370c30c247b8d872b4b6d711eccc0101ea7006dbd`, all other input hashes, ordered IDs/labels/prompts/folds, exact evidence and argmax replay, saved probabilities and all three source confusion matrices. The manifest-byte explanation is now supported by the strict runtime gate.

The fixed control development proxy remains **54.812246% mean balanced accuracy**. This is replayed Stage 06E evidence, not a new result or full S02 evaluation.

## What the fixed-pair diagnosis found

The same 300 Gemini 3.7/3.8 passages recur in three contexts. Counts across contexts are repeated observations.

| Context | Pair correct | Confused with the other Gemini | Predicted another class | Other-class false entries into pair | Cross-errors in exact top-two pair scope | Correct pair rows in that scope |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 135 | 94 | 71 | 65 | 68 | 101 |
| 3 | 146 | 87 | 67 | 54 | 62 | 113 |
| 6 | 139 | 94 | 67 | 49 | 71 | 101 |

The 275 repeated mutual-confusion observations represent **109 distinct passages**: 77 fail in all three contexts, 12 in two and 20 in one. This concentration supports investigating persistent pair separation, without proving a mechanism or a recoverable gain.

At least one component favors the true member of the pair on 78/79/83 of the respective 94/87/94 cross-errors. The same components also disagree on many currently correct cases. Component disagreement is a hypothesis signal, not achieved accuracy. Confidence masses are uncalibrated descriptive scores. Passage-shape associations cannot establish causal explanations.

Even an ideal oracle that corrected every mutual pair cross-error with no harm would add only **3.3951 percentage points** to this available-component proxy. This is an ideal diagnostic upper bound for that restricted error set, not a forecast or a bound on full S02. Fixing this pair alone cannot establish a route to 70%.

## Validation and next decision

Independent review reconciled all seven recovered artifact byte hashes with the runtime manifest, source control counts/scores, overlap patterns, bin partitions, member votes and neutral centers, fold totals, false entries and descriptive feature supports. See [terminal validation](INDEPENDENT_TERMINAL_VALIDATION.json), [runtime receipt](KAGGLE_RUN.json), [findings](FINDINGS.md), [source gate](source_gate.json) and [full log](KAGGLE_V4.log).

`pair_diagnostics.npz`, `pair_rows.csv` and `prompts.csv` remain in the [immutable V4 output](https://www.kaggle.com/code/watchman777/name-that-llm-stage-06f-pair-diagnostics/output?scriptVersionId=359521305). The NPZ numeric/Unicode schema and exact readback passed; row/prompt CSVs were produced and bound by the output manifest. Local independent review did not recompute row membership, margins or bootstrap draws from these binary/row artifacts. Prompt-bootstrap intervals condition on the reused fixed predictions and omit training and future-data uncertainty.

The next single proposed trial is a deterministic attention-based pair choice within the existing exact top-two Gemini scope. It must count both corrected and broken answers and retain the whole 18-class raw balanced-accuracy endpoint. No new classifier, threshold search or weight tuning is part of that proposal; Stage 06F has not executed it. The pair and proposed recipe were selected using reused development evidence, so further evaluation cannot be described as untouched validation. See the [prospective Stage 06G plan](../../plans/STAGE_06G_PAIR_COMPLEMENT.md).
