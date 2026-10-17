# Stage 06A: verified public LoRA component audit
Date: 2026-10-17. Status: complete for one public LoRA component; full S02 OOF inventory remains pending.

## Execution
Private Kaggle V1 359394412 succeeded in 31.1 seconds, accelerator None, internet off, 260.32 kB output. Source copied from lightningv08 V1 solely to retain attribution; all inherited training code was replaced by our CPU script and both the unavailable private helper input and Qwen model input were removed. Only competition data and the public training notebook output remain attached.

- Run: https://www.kaggle.com/code/watchman777/name-that-llm-stage-06a-oof-audit?scriptVersionId=359394412
- Final logs: https://www.kaggle.com/code/watchman777/name-that-llm-stage-06a-oof-audit/log?scriptVersionId=359394412
- Outputs: https://www.kaggle.com/code/watchman777/name-that-llm-stage-06a-oof-audit/output?scriptVersionId=359394412
- Source artifacts: https://www.kaggle.com/code/lightningv08/ntl-lora24-fivefold-training?scriptVersionId=358831649
- Runner: scripts/stage06_oof_audit.py; SHA256 89872b8b773ca40bf0aa3ed21f790e6ba3efc09b1fcf1e254ee756a72e6550fa. Editor clipboard round-trip exactly matched the 11,603-character script before commit. Syntax checked locally, full computation executed on Kaggle.

## Checks passed
The official train digest, 2,699 unique row IDs, 150 prompts, 18 ordered classes and labels match. Five saved outer fits hold out 30 prompts each, with no row or prompt overlap between fit and validation; every row receives exactly one held-out prediction. Fifteen numeric feature/readout/training-report files match their completion-manifest hashes. Three fixed training epochs are recorded for every fold.

We reconstructed plain and centered log probabilities from each fold's held-out feature matrix and classifier coefficients, regenerated all three deterministic lexical context scenarios, and reproduced the public OOF tensors with maximum absolute difference 1.7763568394002505e-14. All recomputed balanced accuracies matched the published component report to 1e-12.

No public attached Python code was executed. No model was retrained. The private Gemma reference and its mixture scores were not loaded or validated. Large adapter binaries were not rehashed; this establishes consistency of saved inference artifacts with source/manifest provenance, not an independent retraining audit.

## Development results (one LoRA component only)
All values below are balanced accuracy on prompt-held-out development predictions. “Responses per prompt” is the source simulation's target batch context, not a measured property of the hidden test. The 50/50 combination is an average of log probabilities.

| Responses per prompt | Plain head | Centered with singleton fallback | 50/50 plain + centered | Combination minus plain, pp |
|---|---:|---:|---:|---:|
| 1 | 48.1564% | 46.9712% | 47.8233% | -0.3331 |
| 3 | 48.1564% | 49.0833% | 53.8993% | +5.7430 |
| 6 | 48.1564% | 55.3473% | 54.9003% | +6.7440 |

These are NOT new official scores, full S02 validation results, or gains to add to S02's 54.9%. Public S02 uses a different downstream blend and mixed GLM/lexical grouping; this component audit reproduces the source's lexical grouping at threshold 0.9.

The gain depends strongly on context. In the one-response simulation, each batch contains at most one response per true prompt, so any inferred multi-row group necessarily merges distinct prompts. Centered evidence hurts that scenario descriptively. In the six-response case the centered head exceeds the fixed 50/50 combination, while the three-response case favours the combination. This motivates robustness checks; it does not justify choosing weights on these scoring labels. No new gate or blend weight was fitted.

## Error signals in the three-response 50/50 component
Lowest recalls among the six printed classes: GLM-5.3 26.67%, Claude-fable-5 28.19% (149 examples), GLM-5.3-flash 34.67%, GPT-5.6-sol 38.00%, Gemini-3.7-flash 41.33%, GPT-5.6-terra 47.33%.

Largest printed directed confusions:
| Actual | Predicted | Count | Fraction of actual class |
|---|---|---:|---:|
| Gemini-3.7-flash | Gemini-3.8-flash | 48 | 32.00% |
| Gemini-3.8-flash | Gemini-3.7-flash | 43 | 28.67% |
| Gemini-3.7-flash | Gemini-3.6-flash | 32 | 21.33% |
| GLM-5.3 | GLM-5.3-flash | 32 | 21.33% |
| Claude-fable-5 | GLM-5.3 | 30 | 20.13% |

The notable cross-family Fable/GLM confusions warn against assuming that all remaining errors are within families. The 91 reciprocal Gemini-3.7/3.8 mistakes are an observed component weakness, not established full-ensemble weakness. Specialists require complementary held-out evidence, not a family-name intuition.

## Artifacts and remaining work
Full per-class recall, directed confusion counts/rates, fold assignments/scores and cluster diagnostics are retained in this exact Kaggle version's outputs. The GitHub log preserves the full provenance result and the printed hardest classes/pairs. LOra_COMPONENT_PROVENANCE.json contains exact metrics and file hashes. The output artifact inventory covers only inputs attached to this audit, not all 14 S02 sources.

Next: inventory the remaining S02 component outputs for compatible OOF and frozen feature caches, then implement the already declared matched-context frozen-head comparison if those feature prerequisites are met. The existing S02 code computes frozen embeddings in memory and does not save them in its three-row preview output; the cache must be found in other public inputs or extracted explicitly. Do not pretend the full S02 validation pipeline is now reconstructed.

Kaggle emitted two package SyntaxWarnings during HTML conversion after AUDIT_COMPLETE; no model or audit warning/failure occurred. No official submission, final-selection change, paid service or GPU training was performed.
