# Stage 06D: fixed available-component ensemble preparation

Status: **BLOCKED_AT_SAVED_SOURCE_VERIFICATION**. Reviewed comparison code is prepared in the private Kaggle CPU editor, but **ensemble execution has not started and Save & Run was never called**. No V2 ID or complete immutable saved five-cell source is confirmed. Bounded save/access recovery attempts ended with `Chat API request failed`. No ensemble result, official submission or new official score is established.

## Prepared comparison and launch limitation

The fixed comparison implementation and exact frozen style-probability transfer were reviewed and committed before any ensemble execution: 1ca9bae1f8fb38b0965f4e2e6aba7a366c2a8153. Five-cell packaging and its independent validation were committed in 9db86f28cfd763ccbc3dcc048fb35fe649f6445f.

The private [Kaggle editor](https://www.kaggle.com/code/watchman777/name-that-llm-stage-06d-fixed-ensemble/edit) contains five reviewed code cells, each verified by a complete clipboard roundtrip against its local source. Saved QuickSave V1 metadata showed all seven required sources: competition 170484; completed own Stage 06C version 359405443; frozen encoder cache 358745836; LoRA1 358831649; LoRA2 358911769; Qwen token attention 358928013; DeBERTa token attention 358948428. CPU, internet off and private visibility were observed.

Metadata-only QuickSave V2 was requested to persist the five-cell layout. Its response and immutable saved source remain unconfirmed. Repeated browser requests stalled, returned `MCP request timed out`, and Version History displayed `Versions failed to load. Try refreshing the page.` A pre-refresh metadata retry returned `Chat API request failed`; screenshot evidence showed `An error occurred while committing kernel: Failed to fetch`, V1 only, Draft Session off and zero active events. An intentional refresh recovered the editor with all five code cells and seven input titles preserved. One post-refresh attempt to reopen QuickSave then failed while clicking the QuickSave option, before its Save action was called. The last browser error is `Chat API request failed`; no retry remains in progress. The pre-refresh screenshot does not establish whether the error banner remained after refresh. These are save/access/verification failures, not an observed ensemble computation failure. No version ID is inferred.

Pre-refresh proof is retained only in scratch as `browser-proof/stage06d-save-error.png`, SHA256 `36e0a428368d36868ae9c2e9b1ab83a4ea065a2b6160ed0a2ea764807e65d30a`; the binary is excluded from the repository and Library. Local execution is currently unavailable because the required neural numeric caches were not downloaded, as independently inventoried by the fusion agent. Prepared code and local style probabilities alone cannot run this fixed comparison.

The five cells contain 77,806 / 156,105 / 156,106 / 146,225 / 53,685 characters. This packaging avoids the observed source-viewer truncation boundary; it changes no function, recipe or probability data. The notebook validates, every cell compiles, all function ASTs and STYLE constants match the reviewed modules, and the 455,182-character payload decodes to the exact original 388,656 probability bytes. Local transfer and independent packaging validation performed zero model fits. Full live editor roundtrips do not establish immutable saved-version parity.

Source and preparation records:

- `KAGGLE_RUN.json`: exact confirmed preparation state and remaining verification.
- `MULTICELL_SOURCE.json` and `PACKAGING_REVIEW.json`: five cell lengths/hashes, notebook and runner hashes, independent packaging checks.
- Combined packaged runner SHA256: `13d4cbd40a563033fcc12d8c8773bc27ab6e112b171a519aa34f0ce2cbc9103b`.
- Exact transferred probability-byte SHA256: `5e2e05577b7dc632f38f67d47958d7f3d069e8b758ead4664050f0d984803bad`.

Once access recovers, inspect the immutable saved source and all seven input bindings before launching unchanged Save & Run All on CPU. Record the actual execution version and fresh logs separately from QuickSave metadata. The upcoming calculation reuses the completed GLM cache, transfers the frozen style probabilities without refitting, reconstructs the audited plain LoRA/attention members and repeats the fixed cached DeBERTa heads while requiring all 12 saved Stage 06C prediction arrays to reproduce.

## Fixed endpoint and scientific limits

The primary comparison is raw fixed rowwise log-evidence fusion, changing only full-prompt versus size-2/3/6-mixture DeBERTa centering training. Weights are fixed; no weight search, learned stacking or new neural training is included. Report all three context scores, their predeclared mean, conditional prompt interval, corrected/broken predictions and class regressions before deciding whether the candidate passes its fixed screen.

Source alpha=0.5 balancing is retained only as a potentially contaminated diagnostic. Members use different prompt fold maps: a peer prediction used in batchwise balancing may have been trained on the scored prompt. Pointwise fixed fusion preserves each member's target-prompt exclusion, while balancing lacks that stronger shared outer exclusion. Balanced scores cannot establish clean validation benefit or select the candidate.

The proxy is incomplete: several S02 frozen representations, verified UltraBERT held-out predictions and common-context LoRA centered evidence are missing. Source weights and development prompts were previously reused; bootstrap intervals condition on fixed fits and group/batch assignments. The calculation will not establish an untouched evaluation or an official leaderboard improvement. There is no evidence yet that this centering change improves the combined model.

## Completed stylegram prerequisite

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

Stage 06C V1 359405443 completed with the GLM cache and passed its mixed-grouping component screen (+1.1242 percentage points). The next measured endpoint remains the fixed ensemble comparison described above. Prepared code retains complete CPU-head probabilities and common mixed-context assignments, but it has not executed. No repeat GPU inference is needed for this comparison.

Full S02 OOF reconstruction still lacks several frozen representations, verified UltraBERT held-out outputs and common-context LoRA centered evidence. The planned combination is an available-component proxy. Source blend weights and development prompts were already reused; no untouched-evaluation claim, learned reweighting, extra candidate selection, paid service, final-selection change or official submission is made.
