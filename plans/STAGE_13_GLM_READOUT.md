# Stage 13 — fixed GLM classifier on held-out prompts

Date: 11 November 2026. Prospective protocol. Freeze the exact source, input contract, notebook and tests in one immutable commit before the real fit. S07 stays unchanged at 0.651 official public balanced accuracy.

## Question and scope

What is the exploratory development balanced accuracy of a fixed, pointwise classifier on the admitted GLM representation when every evaluated prompt is excluded from its fit? This supplies one aligned held-out probability cache for later controlled comparisons. The canonical folds have been used in earlier experiments; this is not a fresh confirmation set.

Stage13A V3 364509756 admitted the cache and established its row alignment, finite values and fold identity. It performed no fitting. This experiment trains one reduced GLM-only reference. S07's plain GLM branch has zero weight and is not fitted; its active centered and Gaussian branches are distinct. The new reference is neither a reconstruction of S07 nor an estimate of a gain in its official score.

## Inputs and execution budget

Use one new private Kaggle notebook, CPU (`None`), Internet off, with exactly two saved bindings: competition **170484** and own Stage06C V1 output **359405443**. Verify visibility, accelerator, network setting and both source versions in the saved editor separately from filesystem evidence. Do not edit the successful Stage13A audit notebook.

The reviewed Stage13A runner repeats its seven production checks before the first fit. The only train table and feature cache must match the following SHA256 values:

| Artifact | SHA256 |
|---|---|
| `train.csv` | `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707` |
| `glm_features.npz` | `2b41585a9482cef825962a8cc096697933c0b1237e8ede0712d1f8ab361d9414` |
| Ordered text-hash list | `6496071874da8f911fafebd9eec6191e56134a97f00226dd8c78805d31f07e36` |
| Canonical prompt-to-fold map | `a55b035686cb7b31f6fbe852ac990668709aec4b913c5d0928d89d843bdbfe62` |

Recheck the input file hashes and exact feature/ID bindings before loading the readout arrays. Require 2,699 unique aligned official rows, 150 prompts, the 18 canonical classes, the safe six-key NPZ schema, and finite float32 source features shaped `(2699, 3, 4096)` with ordered layers `[8, 3, 24]`. Use `allow_pickle=False`. No attached model code or weights are loaded. Require scikit-learn **1.6.1** before fitting and record the Python, NumPy, pandas and scikit-learn versions. Cap numerical thread pools at two via `threadpoolctl`. Run a single private CPU Save & Run All; the notebook gives the runner 1,800 seconds. On any gate or timeout failure, preserve the evidence and inspect it before considering another execution.

## Frozen classifier and fold boundaries

1. Concatenate L8 followed by L3 to 8,192 columns in C-contiguous float64. L24 is not a classifier input. Use the official ordered row and class axes.
2. Recreate the five canonical `GroupKFold` partitions and require the admitted map digest and fold counts. Every fold must have 120 fit prompts, 30 held-out prompts, zero prompt overlap and all 18 classes on both sides. Folds 1–4 hold out 540 rows each; fold 5 holds out 539.
3. Fit a default `StandardScaler` on the fit rows alone, then fit multinomial `LogisticRegression` on the transformed fit rows and their labels: L2 default, `solver='lbfgs'`, `C=0.0025`, `tol=1e-4`, `max_iter=5000`, `fit_intercept=True`, `random_state=0`, no sample or class weights. Do not tune or select a fold. The scaler and classifier are recreated for each fold.
4. Transform each held-out row using its own fold's training scaler and predict its probabilities. The held-out labels are used only for validation checks and scoring, and no held-out peer affects the prediction for another row. No evaluation-side normalisation, prompt centering, calibration, grouping, model ensemble or balancing is used.

## Technical admission and reporting

Require all five fits to converge without convergence warnings and below the iteration cap. Require the exact class order, one held-out probability vector per original row, finite positive float64 probabilities no greater than one, and a maximum absolute row-sum error of `1e-12`. Require the held-out shape `(2699, 18)`. Reconcile the support and confusion counts against the official axes: `claude-fable-5` has 149 rows and each other class has 150.

The primary metric is pooled balanced accuracy: the unweighted mean of the 18 recalls computed from the integer confusion matrix with true classes on rows and predicted classes on columns. Predict by the maximum probability in canonical class order. Also report per-fold balanced accuracy, validation supports, iterations and timings, and pooled class recalls and supports. There is no accuracy threshold for this reference; the gate is technical validity. A successful result must report `COMPLETE_GLM_PLAIN_GROUPED_DEVELOPMENT_REFERENCE` and pass the saved-report readback.

Keep the aligned `heldout_probabilities.npz` in private Kaggle output. Reread its schema, dtypes and exact arrays with `allow_pickle=False` before recording success. Record its file SHA256 and the canonical float64 probability-array SHA256 in the aggregate report. The notebook separately rereads the saved report and verifies the file digest. Only code, aggregate evidence and digests may be published; no passages, raw row or prompt IDs, features, per-row probabilities or model weights go to GitHub or logs.

Freeze these implementation files with the protocol and source hash manifest:

- `scripts/stage13_glm_readout.py` and the unchanged `scripts/stage13a_cache_admission.py`.
- `scripts/build_stage13_readout_notebook.py` and `notebooks/Stage_13_GLM_Readout.ipynb`.
- `tests/test_stage13_glm_readout.py` for held-out label and peer isolation, split-prompt rejection, metrics and cache readback, and missing-input failure.

Verify the saved Kaggle code against that freeze before running. A source-only save or synthetic fixture pass does not establish real-data training or accuracy.

## Downstream decision

The pointwise probabilities are the same inputs for any future context size 1, 3 or 6. Batch construction, stress masks and balancing must have their own prospective freeze and must consume only predictions from heads that excluded every scoring prompt in the corresponding batch. This job admits no full ensemble, batch-coupled comparison, GPU run, new official submission or final-selection change. The exploratory score cannot be added to 65.1% or used as a forecast of private performance.
