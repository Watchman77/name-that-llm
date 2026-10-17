# S02 — unchanged public baseline reproduction

The exact Lucas Boesen V3 recipe (359126546) completed our private Kaggle preview V1 (359332774) in **1884.8 seconds / 31m 25s**. All stages completed, both LoRA anchor-parity checks passed, and `submission.parquet` (1.76 kB displayed) was created. Source code hash matches the declared frozen candidate.

At 2026-10-17T11:40:05.327Z, one official submission was accepted. The Name That LLM Submissions page shows **Notebook Running** for S02. Kaggle reruns the selected notebook version with hidden test data; this is not a manual upload of three preview predictions. The original author's public score 0.549 remains a reference, not our measured score.

Our latest confirmed score remains S01's **0.354**. S02's public and final private scores are unknown. No model/ensemble parameters were changed after preview, no additional candidate was submitted, and no manual final-selection change was made (0/2 auto-selection candidates displayed).

Verification limits: Kaggle cannot preview this Parquet type, so no independent binary round-trip read was done. Executed source constructs `id,model` directly from the test IDs and passed row-count, ID-uniqueness and valid-label assertions. The official hidden rerun performs its own output/scoring checks. In-sample attention/centering checks are technical checks, not independent accuracy. See `results/stage_05/OUTPUT_VERIFICATION.json` and `PREVIEW_OUTPUTS.json`.

Keep S01 and the independently queued Stage 04B DeBERTa trial unchanged. Record S02's eventual outcome exactly, including failures, before considering another fixed experiment. No paid services were used.
