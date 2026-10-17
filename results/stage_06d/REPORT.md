# Stage 06D: fixed available-component ensemble comparison

Status: **SOURCE_PATH_REPAIR_REVIEWED_PENDING_ONE_RETRY**. Last actual execution V5 359502730 completed all five fixed CPU folds/15 heads, then failed `Frozen 06C argmax mismatch, control` at 164.0 log seconds (175.5 seconds platform total). This is a technical reproduction-gate failure; the primary raw screen was not evaluated and no ensemble score exists. The class metadata fix passed the actual cache. An independently reviewed exact-source-call and pre-gate diagnostic revision is ready for source verification and one CPU attempt; no new retry has been called. Its hypothesis remains unproven and all strict checks are retained.

## Saved-source gate, launch and historical access failures

The fixed comparison implementation and exact frozen style-probability transfer were reviewed and committed before any ensemble execution: 1ca9bae1f8fb38b0965f4e2e6aba7a366c2a8153. Five-cell packaging and its independent validation were committed in 9db86f28cfd763ccbc3dcc048fb35fe649f6445f.

The private [Kaggle editor](https://www.kaggle.com/code/watchman777/name-that-llm-stage-06d-fixed-ensemble/edit) contains five reviewed code cells, each verified by a complete clipboard roundtrip against its local source. Saved QuickSave V1 metadata showed all seven required sources: competition 170484; completed own Stage 06C version 359405443; frozen encoder cache 358745836; LoRA1 358831649; LoRA2 358911769; Qwen token attention 358928013; DeBERTa token attention 358948428. CPU, internet off and private visibility were observed.

Earlier preparation attempts encountered browser timeouts, Version History failure and a pre-refresh Kaggle `Failed to fetch` commit error. An intentional refresh preserved five code cells and seven input titles, but the first post-refresh attempt to reopen QuickSave returned `Chat API request failed` before Save was called. These failures are retained in `KAGGLE_RUN.json` under historical failure records. Access subsequently recovered; they are not the current execution state.

Historical pre-refresh proof remains only in scratch as `browser-proof/stage06d-save-error.png`, SHA256 `36e0a428368d36868ae9c2e9b1ab83a4ea065a2b6160ed0a2ea764807e65d30a`; the binary is excluded from the repository and Library. It does not establish the post-refresh banner state. Local execution fallback was unavailable because the required neural numeric caches were not downloaded; the recovered Kaggle session now supplies the intended execution path.

QuickSave Version 2 subsequently succeeded, visible as `Viewing Version 2`, `Quick Version`, `Oct 17, 10:45 PM`, named `Verified multi-cell source - no execution`, with zero seconds run. Its complete visible saved export matched every reviewed source string across all five code cells, each with empty outputs and null execution count. Metadata exactly matched the seven pinned input IDs, accelerator `none`, `isGpuEnabled=false` and `isInternetEnabled=false`. Its numeric scriptVersionId was subsequently confirmed as **359499684** from that version's own Open in Viewer link. This passed the saved-source gate.

The unchanged Save & Run All launched visible Version 3, `Fixed raw ensemble comparison`; fresh UI at 2026-10-17T21:47:07Z showed Running and one active event with CPU selected. Its numeric scriptVersionId is 359500151. It subsequently failed at 20.7 seconds in the actual logs; the terminal UI displayed zero seconds, which does not represent its observed execution duration. This computation failure occurred after the saved-source gate had passed and is distinct from the earlier recovered browser/save errors.

The five cells contain 77,806 / 156,105 / 156,106 / 146,225 / 53,685 characters. This packaging avoids the observed source-viewer truncation boundary; it changes no function, recipe or probability data. The notebook validates, every cell compiles, all function ASTs and STYLE constants match the reviewed modules, and the 455,182-character payload decodes to the exact original 388,656 probability bytes. Local transfer and independent packaging validation performed zero model fits. Full live editor roundtrips do not establish immutable saved-version parity.

Source and preparation records:

- `KAGGLE_RUN.json`: exact confirmed preparation state and remaining verification.
- `MULTICELL_SOURCE.json` and `PACKAGING_REVIEW.json`: five cell lengths/hashes, notebook and runner hashes, independent packaging checks.
- Combined packaged runner SHA256: `13d4cbd40a563033fcc12d8c8773bc27ab6e112b171a519aa34f0ce2cbc9103b`.
- Exact transferred probability-byte SHA256: `5e2e05577b7dc632f38f67d47958d7f3d069e8b758ead4664050f0d984803bad`.

Current next checks are to persist the reviewed source-path/diagnostic revision, verify immutable source/input bindings and launch only the one approved CPU attempt. Its preserved diagnostics must establish whether call shape explains the mismatch; strict reproduction still gates every combined-model result. No new GPU extraction, model search or tolerance relaxation is part of this work.

## Version 3 failure and bounded repair

`STYLE_TRANSFER_COMPLETE` verified all 2,699 rows and exact original probability bytes, SHA256 `5e2e05577b7dc632f38f67d47958d7f3d069e8b758ead4664050f0d984803bad`. `STAGE06D_COMPONENTS_AUDITED` then confirmed five members and 47 files. Loading the own Stage 06C `paired_predictions.npz` metadata inside the `ids/classes/y/groups/fold` validation loop raised `Object arrays cannot be loaded when allow_pickle=False`. No `STAGE06D_DEBERTA_FOLD_COMPLETE` marker occurred; root confirmed the failure preceded new head fitting.

This is an input-schema compatibility failure, not a negative ensemble result. The fusion implementation now has an independently reviewed classes-only static serialization reader without `allow_pickle=True`. Stage 06C's `classes=np.sort(train.model.unique())` produced the object array; IDs/groups were explicitly Unicode. All numeric arrays remain numeric-only. The repair still requires saved-source verification and a separately identified rerun. Fixed weights, folds, grouping, protocol and primary raw acceptance rule remain unchanged. No repaired run or numerical result is claimed.

## Classes-only repair verification

New fusion SHA256: `b27925f978b0c87938c92adc36bcbb52e6536aa79fbe9330217daec7fecbae51`. Independent review against the unchanged original fusion SHA `61cf698548c1352caa2e81eea937753c607a554cffea2e2b5d32c678c7101163` proves only the new helper and classes metadata handoff changed. All remaining fitting, fold, grouping, weighting, twelve-prediction reproduction, raw acceptance and balancing AST nodes match exactly. Every `np.load` call retains `allow_pickle=False`.

The exact official-CSV class-writer expression reproduced the object-array failure. The static reader accepts the expected eighteen classes in Unicode and NumPy object serialization protocols 3/4/5. Wrong order/labels, duplicates, extra/missing values, numeric/mapping/object-bytes content, dangerous reduction and trailing serialized bytes reject. During independent fixtures, guarded `pickle.load`, `pickle.loads`, NumPy reconstruction and a dangerous callable were never invoked. This is a bounded inert-symbol grammar, not an unpickler. Zero model fits were performed.

Repaired packaging retains the first four cells byte-for-byte; only the final cell changed. Schema, five-cell compilation, exact module embedding and notebook/linear AST equivalence pass. New combined SHA256: `8d2b999b78cccb48bbed5d223b7c30d1e4b3bc9369abed06c870ba499d6a3b27`. `CLASS_METADATA_REPAIR_PACKAGING.json` describes repaired artifacts; `MULTICELL_SOURCE.json` remains the historical pre-V3 snapshot. Full independent findings: `SAFE_CLASSES_INDEPENDENT_REVIEW.json`; repair-author fixtures: `CLASS_METADATA_REPAIR_FIXTURES.json`. Actual Kaggle cache bytes and an end-to-end repaired run remain unverified locally. Root replaced the final editor cell and verified all 60,610 characters exactly. Metadata-only QuickSave V4 succeeded with scriptVersionId 359502142. Its complete five-cell source matched the reviewed repaired strings, each with empty outputs/null counts; all seven pinned input IDs and CPU/internet-off metadata matched. The fourteen-file repair freeze commit de19608a672015fdb84e7921ed2aa1482c495a18 preceded actual Version 5 Save & Run All. Fresh UI at 2026-10-17T21:55:14Z showed Version 5 Running with one active event; its numeric execution ID/logs remain pending. Current active source hashes are in the repair packaging manifest, while V3 failure evidence remains bound to the historical original hashes/packaging commit 9db86f28cfd763ccbc3dcc048fb35fe649f6445f.

## V5 strict reproduction failure and one diagnostic retry

V5 actual execution 359502730 passed the classes-only metadata reader, exact frozen style transfer and five-member/47-file audit. All five DeBERTa fold markers appeared (66.0 / 88.8 / 115.6 / 140.4 / 164.0 seconds), establishing 15 fixed head fits. It then failed the first strict frozen-control argmax comparison before `ENSEMBLE_RESULT` or the raw improvement screen. Full original log is `KAGGLE_V5_FAILED.log`, 4,864 bytes, SHA256 `e6f5896056004ee4c6f8a133f1835c9315f70a888e47a7ba858157050d4fde4e`; parsed facts are `V5_FAILURE.json`. Its source remains bound to b27925f978b0c87938c92adc36bcbb52e6536aa79fbe9330217daec7fecbae51 and pre-execution commit de19608a672015fdb84e7921ed2aa1482c495a18. Mismatch counts, margins and coefficients were not retained before this assertion and cannot be inferred.

Independent C/D source review found C predicts centered probabilities on eligible rows only, whereas D predicted all rows then selected eligible ones. This changes inference call shape while preserving the formula. It is a concrete implementation difference, not a demonstrated root cause; runtime/solver differences remain possible.

Reviewed source c6005196a73100676141d18f67eb299c1bf686a547fbb077b91825d2bd2611b3 keeps the same fitted heads, computes the original all-row path for diagnosis, then overwrites eligible rows using exactly C's residual[eligible] call. The original all-row predictions never select the candidate or enter fusion. Before strict checks, it saves source/all-row/frozen predictions, full probabilities, margins, batch/group/eligibility assignments, runtime/BLAS versions and all15 head scaler/coefficient snapshots. The unchanged twelve exact argmax-array and score-atol1e-12 gates still precede fusion; if they fail, no ensemble score/result is computed.

Training, partitions/seeds, grouping, plain validation scaling, weights and primary raw criteria are unchanged by AST checks; eight no-fit fixtures cover no/all/subset-active masks, shape/finiteness rejection and independent copies. Findings: `INFERENCE_PATH_INDEPENDENT_REVIEW.json`; author fixtures: `SOURCE_PATH_REPAIR_FIXTURES.json`; packaging: `SOURCE_PATH_REPAIR_PACKAGING.json`. First four cells are unchanged; the final cell is69,909characters. New combined SHA256: `9185a208679ea99c318bb11feb9abd773661886523621e268052ca037e613893`.

One evidence-based CPU attempt is planned only after source persistence and complete immutable saved-source verification. No new CPU retry has been called. If strict mismatch persists, retain diagnostics and stop blind refits rather than weakening the gate. Historical protocols, source manifests and prior reviews remain intact. This is not a failed primary improvement screen and not evidence of any official gain.

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

Stage 06C V1 359405443 completed with the GLM cache and passed its mixed-grouping component screen (+1.1242 percentage points). The next measured endpoint remains the fixed ensemble comparison described above. The prepared fixed comparison is designed to retain complete CPU-head probabilities and common mixed-context assignments; Version 3 failed before fitting those heads. No repeat GPU inference is needed for this comparison.

Full S02 OOF reconstruction still lacks several frozen representations, verified UltraBERT held-out outputs and common-context LoRA centered evidence. The planned combination is an available-component proxy. Source blend weights and development prompts were already reused; no untouched-evaluation claim, learned reweighting, extra candidate selection, paid service, final-selection change or official submission is made.
