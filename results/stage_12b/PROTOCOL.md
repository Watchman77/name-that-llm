# Stage12B — one fixed official SVC topic-correction counterfactual

## Parent, evidence and decision

Our working parent is **S05 / private Stage11 V6 362129392**, officially **0.642 public balanced accuracy**. Its source is Lightning / Vladislav Lavrentev V28 **362014737**, with retained Apache-2.0 credits. This candidate changes that working parent; it does not use the separate public 0.651 lead or introduce any new input assets.

Stage12A's frozen five-fold prompt-disjoint **component** screen produced control **0.3789311459110117** versus candidate **0.40042505592841166** development balanced accuracy: **+2.149391 percentage points**, conditional paired-prompt 95% interval **[+1.038025, +3.259259]**. Its fixed screen passed. The folds were reused, its four-view SVC used fixed diagnostic C=1 rather than verified S05 fit settings, and it did not evaluate the full ensemble. These are supporting grounds for one prospective blind official test, not an estimate of the official gain.

Crucially, Stage12A gated using SVC-only scores. Stage12B uses the same existing function with the **complete SVC/decoder/DeBERTa reference log-vote sum** as `total`; the uncertainty gate can therefore differ. No full-S05 OOF gain, 70% result, independent development confirmation or scientific novelty is claimed.

## Exact single intervention

The existing source helper `adaptive_svc_topic_correction(total, svc_scores, groups)` is unchanged. Activate it **exactly once**, after the complete DeBERTa contribution and before the complementary probability mixture:

```python
total += .25 * clipped_logp(aligned(work/'deberta.npz', ids))
# Existing family if/else is complete here.
svc_topic_before_total = total.copy()
total = adaptive_svc_topic_correction(total, svc_scores, groups)
# Save finite before/after evidence here, then enter complementary mixing.
```

Use its original coefficient **0.62**, uncertainty gate, eligibility for inferred clusters of **8–18** rows, and original word-TF-IDF average-linkage topic groups at threshold **0.85**. Do not use true hidden prompt IDs or labels. Preserve all model weights, fitting states, tokenization, natural model inputs, mixture weights, ATLAS selection, Night24 helper, all-row Qwen share and the single final Sinkhorn operation.

Only upstream **code cell30** changes. All other **32/33 upstream source code cells** remain byte-for-byte unchanged. The original independent Stage11 output guard, **code cell34**, also remains byte-for-byte unchanged. Append a separate intervention guard as **code cell35**. See `SOURCE_PATCH.diff` and hash-bound `BUILD_RECEIPT.json`.

Preservation refers to model weights and model worker/component code. The orchestrator in cell30 is deliberately changed; broad notebook-header wording about preserving model cells must be read with this explicit exception.

## Inputs and intended runtime

Preserve the exact **28 S05 V6 source-version IDs**, copied from the actual saved V6 metadata receipt. No dataset/model/notebook input version is upgraded. Their full list and equality check are in the build receipt.

Intend a **private** Kaggle notebook with **GPU T4 x2**, Internet disabled and free resources only. The built document's GPU boolean is true as an intended configuration; the parent V6 document's legacy false value is recorded separately. A build cannot verify actual GPU allocation. Root must verify the saved Kaggle version and actual allocation profile. No run is launched by the builder.

The successful S05 establishes parent execution feasibility; it does not guarantee the new candidate's outcome. There is no additional model pass. The intervention adds CPU score arithmetic and private numeric evidence. Competition execution limits continue to apply.

## Evidence and validation

The changed orchestrator privately saves `svc-topic-correction-evidence.npz` under its existing `/kaggle/temp` work directory. It contains exact ordered IDs/classes, inferred groups, SVC margins and the before/after complete reference totals. It saves no response text. Before saving, all score arrays must be finite.

The separate intervention guard reads safe NumPy numeric/string arrays with `allow_pickle=False`; it loads no model, Torch object, joblib or pickle payload. It checks:

1. Hash-bound generated orchestrator and unchanged source-helper code, exactly one correction call, original arguments and exact placement after the complete family branch and before complementary mixing.
2. Complete ordered ID and class axes, finite shapes and bounded integer group IDs.
3. The saved `before_total` reconstructs from SVC log probabilities plus the saved decoder logits and **0.25** times DeBERTa log probabilities.
4. Replaying the unchanged correction **once** reproduces `after_total`; ineligible rows are exactly unchanged.
5. The corrected total feeds the actual saved **0.85/0.15** pre-ATLAS probability mixture. The unchanged original Stage11 guard checks the downstream output chain separately.

Report eligible-row/cluster coverage and actual score changes. Zero eligible rows or a numerically null effect must be reported as such; array replay is not accuracy. No hidden IDs, original text or row logits enter the public report.

The unchanged Stage11 guard retains its historical limitation that saved SVC logits are unavailable to that guard. Stage12B now saves those margins in a separate evidence file, and its new guard independently reconstructs the pre-complement reference sum. The original guard's inherited limitation describes its own scope; it does not negate the new guard's distinct evidence check.

The exact source three-row public fixture skips the full inference branch. Its guard result must explicitly be **FORMAT_ONLY_PREVIEW_INTERVENTION_NOT_EXECUTED**. A successful preview verifies syntax/format preparation and source shape; it cannot verify activation or hidden-test accuracy.

Synthetic checks use a nontrivial analytic golden case: SVC margin 2 but complete-reference margin 0.4, so the original gate is 1 and the corrected class0/class1 margin is **-0.84**. They also reject duplicate application, wrong axes, nonfinite arrays, illegal groups, changed ineligible rows, wrong before sums and a prior that ignores the corrected total. Fixtures are written and read through real NPZ files. No upstream model is executed by preparation.

## Predeclared official test and stopping rule

After independent review, a published source freeze, exact saved-code/input parity and a format preview, allow **one blind official submission of this fixed candidate**. There is no automatic coefficient, threshold, placement, feature, blend-weight or selection variant and no batch of public-score probes.

The only public comparison is the displayed candidate official public balanced accuracy versus parent **0.642**, in percentage points. If the candidate is no better, retain S05 as the better measured public result and retire this exact intervention. A higher public score remains conditional public evidence; private performance remains unknown. Execution failure is not a modelling result. Do not silently alter the hypothesis or submit a repaired candidate as another blind test without a separately recorded execution decision.

Do not change final selections during this experiment. No new training, paid API calls, external data generation or trained-weight publication is authorized by this protocol. Code and aggregate receipts may be committed; private test evidence stays on private Kaggle.
