# Stage 13E: Prvsiyan holdout admission blocked

Private CPU audit V1 [368354781](https://www.kaggle.com/code/watchman777/notebook4268adde48/log?scriptVersionId=368354781) completed in **36.1 seconds** on 11 November 2026. It ran offline on the actual mounted Kaggle Version 12 corpus and the exact official train file. The source checksum and saved aggregate report readback passed. **No panel qualified. No model was fitted or scored.**

| Gate | Main `new` split | `new2` split |
| --- | ---: | ---: |
| Complete 18-model groups | 231 | 8 |
| After excluding the 150 prompt groups used by Stage 07A/09 | 81 | 8 |
| After checking prompt origin, declared licence and duplicate prompts | 81 | 0 |
| After checking duplicate response texts | 81 | 0 |
| After checking provider route, model identifiers and quality | 0 | 0 |
| Selected (minimum 30) | 0 | 0 |

The grouped report does not assign a single reason to every group that failed a combined gate. No rows reached raw-envelope matching or official-response overlap comparison, so the zero counters there mean no rows were checked. The mounted clean CSV matches the earlier Stage 12 V5 preflight's drifted 5,070-row file; it differs from the restored 4,926-row file checked for the final S07 lineage. This comparison does not establish prompt-level independence from S07.

The prepared external scorer remains unrun. The separate alternate split stress check is also unrun and would only test robustness on the same previously used prompts. To finish the external comparison, a different source must supply at least 30 verified, complete and previously unused prompt groups; model-feature parity must then pass before scoring. S07's official best remains **65.1%**. No submission or final selection changed.

[Frozen protocol](../../plans/STAGE_13E_EXTERNAL_HOLDOUT.md) · [Aggregate audit](ACTUAL_V1_ADMISSION.json) · [Kaggle receipt](KAGGLE_V1_RECEIPT.json)
