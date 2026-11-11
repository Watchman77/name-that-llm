# Stage 13D — fixed DeBERTa swap in a broader held-out ensemble

Date: 11 November 2026. Prospective protocol. Freeze this plan, the runner, the notebook, the fixtures, and the input bindings before scoring the comparison. S07 remains the best verified official public balanced accuracy at 65.1%.

## Question and scope

Does the fixed Stage 13C DeBERTa swap still improve the development reference after adding a character TF-IDF branch? First test whether that extra branch produces a stronger control. Then test the marginal contribution of DeBERTa without changing the character branch's weight. This is a reduced proxy on the same 150 previously used development prompts, not a full S07 ablation or a blind confirmation set.

## Inputs and admission

Use one new private Kaggle notebook, CPU (accelerator `None`), Internet off, and a 900-second runner timeout. Verify the saved bindings and settings before running. The three pinned inputs are competition **170484**, the private GLM V3 output `kernelVersion` **368292220**, and the frozen pretrained DeBERTa extractor output `kernelVersion` **358745836**. Preserve the source notebooks unchanged.

- Require the unique competition `train.csv`, SHA256 `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`. Read `id`, `prompt_id`, `model`, and `text` as strings. Check nonempty text and exact ordered agreement of IDs, prompt IDs, and canonical class targets with the verified GLM arrays before fitting.
- Require the unique `heldout_probabilities.npz` and its matching `readout_report.json` from GLM V3. Its cache SHA256 is `565df6f8e652a9850ff11cf4e8b6a81fcccbc7db593186ff7f53358e7190bd16`. Reuse the Stage 13B admission contract: 2,699 unique rows, 150 prompts, the exact ordered 18-class axis, canonical prompt-to-fold digest `a55b035686cb7b31f6fbe852ac990668709aec4b913c5d0928d89d843bdbfe62`, finite positive float64 probabilities, and documented held-out fold exclusion. Do not refit GLM.
- Require the unique pretrained `enc_debv3l2_train.npz`, SHA256 `11eed58762ad128f97457dde9cc4ab03c7932e1b4b04845ec6664c834b1b1dda`. Use unchanged Stage 13C admission: exact bytes and ordered row IDs; finite `(2699, 9, 1024)` float16 embeddings; numeric loading without pickle; bounded static ID-literal inspection if necessary. Do not use a supervised checkpoint or earlier mixed-fold predictions.

## Frozen fits and comparison

Require scikit-learn **1.6.1** and limit numerical thread pools to two threads. Each of five folds fits only on its 120 training prompts and scores its 30 held-out prompts. Folds 1–4 each score 540 rows; fold 5 scores 539. Require no prompt overlap, complete coverage, the exact class order, finite positive normalized probabilities, and no convergence warnings.

- **DeBERTa:** reuse the unchanged Stage 13C fit function. Concatenate layer 24 then layer 6 into float64, then fit `StandardScaler` and multinomial `LogisticRegression(C=0.005, solver='lbfgs', tol=1e-4, max_iter=5000, random_state=0)` within each fold.
- **Character branch:** fit `TfidfVectorizer(analyzer='char', lowercase=False, ngram_range=(3, 5), min_df=2, max_features=50000, sublinear_tf=True, norm='l2', dtype=float64)` on that fold's training texts only. Transform held-out texts with that same vocabulary and IDF. Fit multinomial `LogisticRegression(C=10, solver='lbfgs', tol=1e-4, max_iter=1000, random_state=0, class_weight=None)` on the training vectors only.
- **Reference:** the frozen GLM probabilities.
- **Broader control:** `0.75 * GLM + 0.25 * character`.
- **Candidate:** `0.50 * GLM + 0.25 * character + 0.25 * DeBERTa`. The character share stays at 25%; the intervention transfers 25 points of probability weight from GLM to DeBERTa, as in Stage 13C.

All mixtures are arithmetic on held-out probabilities before balancing. Apply the unchanged Stage 13B operation to each arm separately within each complete held-out fold (`alpha=1`, 100 column-mass/row-normalization iterations, `log(Q + 1e-12)`, canonical first maximizer). Never pool folds before balancing. No other weighting, fitted calibration, centering, stress scenarios, or tuning is in scope.

Replay the prior GLM raw and balanced control confusion digests `fa3ced3a896ce9389bf0427a42a30d8e7a30089c792cb92b8988953204b98747` and `e81a80974d67dc60dc93faf3144dff56ab2055a327abcf7805dbb3556f364b0b`, respectively. Confirm the fixed Stage 13C balanced GLM/DeBERTa pair confusion digest `c076ee1a507753660bd7356baa3083d277c8fc8e61bedc81bae9192986caad9b` after repeating the DeBERTa fits. Stop without a comparison decision if a binding, provenance, replay, numerical, or saved-report readback gate fails.

## Metrics and fixed decision

Report pooled 18-class balanced accuracy and paired differences in percentage points: **broader control minus GLM** and **candidate minus broader control**. Report class supports, recalls and deltas, fold scores, aggregate confusion counts, changed/corrected/broken decisions, and fit and numerical checks. For both comparisons, use the same 5,000 paired within-fold prompt bootstrap draws, seed **20261111**. In each draw sample 30 prompts with replacement per fold, use shared multiplicities across arms, pool class denominators, and recompute the difference. Fail on a class missing from a draw rather than dropping the draw. Report equal-tailed 95% intervals. They condition on the fixed fitted and balanced predictions and omit refitting and new-batch uncertainty.

Admit the broader control only if its pooled gain over GLM is at least **0.50 percentage points** and its interval lower bound is strictly above zero. Admit the marginal DeBERTa result only if its gain over the broader control is at least **0.50 percentage points**, its interval lower bound is strictly above zero, and no class recall declines by more than **5 percentage points**. Report both comparisons even if the broader control is not admitted. The overall decision passes only if both sets of gates pass. Fold signs are diagnostics. Preserve negative and neutral results; do not add variants or revise the rule after seeing outcomes.

Save only aggregate metrics and verification evidence. Do not publish or log text, row or prompt IDs, features, fitted weights, or row-level probabilities. A pass warrants further validation only; do not change S07, submit, change final selections, train a neural encoder, or use paid compute under this protocol.
