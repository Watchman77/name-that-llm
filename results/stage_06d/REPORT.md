# Stage 06D: stylegram held-out prerequisite

Status: fixed CPU prerequisite complete; available-component ensemble fusion pending. No official submission or ensemble gain has been measured.

## What completed

Reproduced the stylegram recipe from Lucas Boesen's public S02 source V359126546 on the original archived five prompt-held-out folds. All vocabulary and IDF fitting used training prompts only. The four feature blocks, source transforms, classifier defaults and probability construction were fixed before fitting.

- Development balanced accuracy: **0.3844842157593836 (38.4484%)**.
- 2,699 rows, 150 prompts, 18 classes; 1,038 correct and 1,661 errors.
- 42.48 seconds CPU; all five folds completed, zero fit warnings and zero convergence warnings.
- Fold balanced accuracies: 35.7407%, 36.2963%, 40.0000%, 40.9259%, 39.2848%.
- Full float64 class probabilities, floored log probabilities, original decision scores and row/class/fold metadata saved in `stylegram_oof/stylegram_oof.npz`. No pickle/object arrays or passage text.
- Full probability CSV passed exact float round-trip checks. Confusion matrix, per-class recalls, probabilities and fold scores independently reconcile.

This is one component's development score. The best confirmed official S02 public balanced accuracy remains 54.9%. Comparing these as if they were two competing full-system scores would be incorrect.

## Reproducibility and review

Prospective clarification: commit f1105cb880b947b7653b1acdb5a09ca3602f179b. Reviewed implementation committed before fitting: 4afba147c4290ae24ddf4f1ea078fe8a8fc729d1. Script SHA256: 73b8c397733075ea7cbb183964d3b59cbfc24881a71cc043859532edf380acf6.

Official train.csv SHA256: 40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707. All original baseline_oof.csv assignments matched recomputed GroupKFold exactly, with each row validated once and zero prompt overlap. A separate source comparison matched the vendor's function-word set, block specification and transforms on four fixtures. The independent saved-output audit is `stylegram_oof/VALIDATION.json`; original output manifest retained unchanged.

Environment: Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0 and scikit-learn 1.8.0. This is recipe reproduction, not bitwise Kaggle environment parity. SVM score-softmax values are normalized scores, not calibrated probabilities.

Run:

```bash
python scripts/stage06d_stylegram_oof.py --data-dir PATH_TO_OFFICIAL_DATA --reference-dir results/baseline_01 --output-dir NEW_EMPTY_OUTPUT_DIRECTORY
```

## Component errors and next decision

Largest two-direction confusion counts include Gemini 3.7/3.8 (80), GLM/GLM-flash (59), and cross-family Fable/GLM-flash (45). These are stylegram errors only; they do not identify the full ensemble's remaining errors.

Next combine this fixed style member with the audited plain LoRA and token-attention outputs under the original fixed weights, preserving per-member whole-prompt exclusion. The DeBERTa control/candidate comparison needs complete class probabilities and common mixed-context assignments. Stage 06C V1 359405443 has now completed with the GLM cache and passed its mixed-grouping component screen (+1.1242 percentage points). The fixed ensemble comparison still needs complete CPU-head probabilities; its saved cache removes the need for further GPU inference in this step.

Full S02 OOF reconstruction still lacks several frozen representations, verified UltraBERT held-out outputs and common-context LoRA centered evidence. The planned combination is an available-component proxy. Source blend weights and development prompts were already reused; no untouched-evaluation claim, learned reweighting, extra candidate selection, paid service, final-selection change or official submission is made.
