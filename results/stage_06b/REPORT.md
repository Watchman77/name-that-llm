# Stage 06B: cached DeBERTa centering comparison
Date: 2026-10-17. Scope: one frozen centered component, not a new official competition score.

## Main result
A fixed change to centered-head training improved the mean balanced accuracy across three lexical context simulations from 46.5685% to 47.2483%: +0.67984 percentage points. Paired 3,000-draw prompt-bootstrap 95% interval: +0.08602 to +1.29630 percentage points. No scenario regressed.

| Responses per prompt in simulation | Full-group training control | Small-group mixture candidate | Difference, pp |
|---|---:|---:|---:|
|1|43.5675%|43.5675%|0.0000|
|3|44.9744%|46.8660%|+1.8916|
|6|51.1636%|51.3115%|+0.1479|

The two heads are evaluated as the fixed S02 DeBERTa evidence contribution (plain weight0.05, centered weight0.25, centered evidence on inferred size2..18 groups only). This is not a standalone end-to-end S02 score. The official S02 public BA remains0.549 at its verified submission record. Do not add the component gain to0.549.

## Controlled recipe
Prospective scoped protocol commit 2a090244c493a6e56990d3427e49f5390df87e4b. Frozen public encoder cache V358745836, mean layers24+6 of microsoft/deberta-v3-large;2,699 rows, 2,048 features. IDs in the archive match official competition order; cached features are float16, converted to float32. Numeric cache digest11eed58762ad128f97457dde9cc4ab03c7932e1b4b04845ec6664c834b1b1dda.
GroupKFold5 prompt assignments match our archived original mapping. All scalers and classifiers are fitted inside training prompts. Both use LogisticRegression C 0.005 and 5,000 iteration cap. Candidate residuals use a fixed mixture of same-prompt partitions of sizes2,3,6, with equal total weight per original row. No layer, model, weight or threshold search occurred.
Lexical test grouping uses TF-IDF average linkage with threshold0.85. Plain test standardisation and fixed evidence weights are unchanged. Target contexts1,3,6 are evaluated on paired rows and fixed seeds. True groups are diagnostics only. Raw passage text is excluded from GitHub.

## Numeric artifact audit
- Second seed7 LoRA reconstructed pooled development BA: 48.04425%. Each fold's reconstructed ordinary accuracy exactly matches its completion report despite float16 serialization;360 steps and 0 skipped updates are recorded for every outer job. These manifests contain no file hashes, so this provenance is weaker than Stage06A.
- Qwen3.5 token attention: 39.37808% development BA.
- DeBERTa token attention: 41.08153%.
- Additional Qwen3-8B token-attention file (not an S02 member): 37.00795%, inspected for inventory, not selected as a candidate.
- ID/class/order/fold checks and probability normalisation passed. Object-dtype ID metadata was inspected as bounded NPY/pickle bytecode without unpickling; numeric arrays always used allow_pickle=False. Attached source and torch pickle model files were not executed or loaded.
- Twenty numeric NPZ files were inventoried across these attached outputs. Full S02 validation is still incomplete: missing GLM and other frozen LLM passage features, desklib features and adequately verified UltraBERT held-out provenance. UltraBERT dataset description reports five GroupKFold prompt fits and0.408OOF, but those numbers have not been independently recomputed here.

## Warning-gate correction and repeat verification
V1 private run359399756 succeeded in155.8seconds on CPU, internet off. Its numeric screen criteria passed, but its recorded screen_pass was false because the runner incorrectly categorised every caught warning as a convergence warning. All 15 messages were the same SciPy deprecation of L-BFGS-B disp/iprint options. Keep the original false flag and log; WARNING_REVIEW.json documents the explicit review.
V2 private run359401219 uses the identical model recipe and adds warning categories, full class diagnostics and assertions that the original control/candidate scores reproduce. It is a reporting/verification correction, not another tuned candidate.
V2 succeeded in 170.1 seconds on CPU with internet off and 100.7 kB output. Its assertions reproduced all six original control/candidate scenario scores to 1e-12. The corrected screen_pass is true, with no convergence warnings; all 15 recorded notices have category DeprecationWarning. Numerical criteria and model settings were not changed. Debugger notices and post-execution HTML-conversion SyntaxWarnings are retained in the logs. Exact per-row equality of the two saved NPZ files was not independently checked.

## Class tradeoffs
All 216 per-class recall records (including oracle diagnostics) are saved. Their macro averages independently reconcile with the reported scenario scores. The largest inferred-context class regression is Qwen3.8-max with 3 responses: recall 60% to 52% (-8 percentage points). GPT-5.6-terra falls 40% to 34% with 6 responses (-6 points). Across the three scenarios, Qwen3.8-max has the largest mean regression (-3.11 points), followed by Claude-opus-4-8 (-2.44 points). The average improvement therefore includes meaningful class tradeoffs. No per-class no-regression claim is made.

## Limits and decision
The statistical development screen is promising but narrow. The confidence interval's lower bound is close to zero. Context scenarios are fixed simulations, not an inferred hidden-test distribution, and the same historical development dataset has been reused.
Oracle scores (control/candidate):1response43.6416/43.6416%;3responses47.2722/48.4596%;6responses54.0159/53.8665%. The small six-response oracle regression shows that the change is not uniformly superior even with true groups.
Only one of five frozen centered twins has been tested. Cached fp16 features and lexical grouping approximate S02, which uses in-memory fp32 frozen features and mixed GLM/lexical clustering at0.60. No direct full-ensemble or official improvement is established.

Advance to a fixed confirmation with S02's actual mixed grouping before considering adoption. That requires a new frozen GLM layer24 feature cache; see plans/STAGE_06C_MIXED_GROUP_VALIDATION.md. Do not change other twins by extrapolation or submit a candidate on this component result alone. No GPU or paid service was used in Stage06B. All raw outputs and paired predictions remain in the immutable Kaggle versions.

Run links:
- V1: https://www.kaggle.com/code/watchman777/name-that-llm-stage-06b-cached-centering?scriptVersionId=359399756
- V2: https://www.kaggle.com/code/watchman777/name-that-llm-stage-06b-cached-centering?scriptVersionId=359401219
