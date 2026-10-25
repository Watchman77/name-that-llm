# Stage 11 notebook and version index

All main-run versions below belong to the private notebook `watchman777/name-that-llm-stage-11-public-642-audit`. The collector is a separate CPU notebook. Earlier checkpoints are preserved; only V6 is official S05.

| Run | Exact script version | Purpose and verified result |
| --- | --- | --- |
| CPU V1 | 362118178 | Succeeded, 680.2s; 174/175 contract checks. Auditor incorrectly required native Qwen labels to use canonical order. The unchanged source explicitly reorders by name. |
| CPU V2 | 362119531 | Corrected auditor only. Succeeded, 686.1s; 175/175 checks and all active NF4 manifest hashes verified. No model execution. |
| CPU report collector V1 | 362123509 | Separate notebook, succeeded in 57.7s. Retrieved the complete 924,007-byte CPU report; transport and report hashes verified. |
| Qwen probe V3 | 362125549 | Succeeded, 418.9s on T4 x2. Real 12-passage Qwen inference, finite aligned outputs, within a 900s process-group deadline. No accuracy or full-ensemble runtime claim. |
| Production quick snapshot V4 | 362127385 | Source-only snapshot; actual saved JSON confirms 33 unchanged source code cells plus one output guard. |
| Production preview V5 | 362128776 | Succeeded, 21.7s; 25/25 format checks on the three-row fixture. Full models skipped by source preview mode. |
| Production preview V6 / S05 | 362129392 | Same frozen code and inputs, explicit persistent GPU save option. Preview succeeded, 21.7s; 25/25 format checks. Official S05 accepted and Notebook Running, no score at verification. |

The production source is Lightning / Vladislav Lavrentev **V28 362014737**, whose displayed public score is **0.642**. That is not an S05 score. Our confirmed best remains **S04 0.639 official public balanced accuracy**.

The repository notebooks separate the CPU audit, locked/admitted Qwen probe, report collector and production reproduction. Probe and preview results do not substitute for the official hidden-test evaluation. No new weights were trained in Stage 11.

[Official S05 page](https://www.kaggle.com/competitions/name-that-llm/submissions) · [exact S05 notebook](https://www.kaggle.com/code/watchman777/name-that-llm-stage-11-public-642-audit?scriptVersionId=362129392).
