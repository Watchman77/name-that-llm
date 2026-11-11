# Stage 13B — fixed GLM balancing stress comparison

Date: 11 November 2026. Prospective protocol. Freeze the source, notebook, fixtures, input binding and this protocol before the real comparison. The best verified official submission remains S07 at 0.651 public balanced accuracy.

## Question and scope

Does switching off the fixed uniform class-marginal balancing step improve the existing standalone GLM reference under unequal class mixtures without material harm on balanced mixtures? Compare the same frozen probabilities, held-out rows and scoring batches in both arms. Stage 13 GLM V3 completed the five prompt-held-out fits, so this job consumes their saved predictions and fits nothing.

This is a reduced development stress test, not an S07 reproduction. It tests the source balancing arithmetic at the probability input, with no typography guard, Hungarian assignment, contextual features, prompt centering, fitted calibration or extra ensemble branches. The class mixtures are synthetic scenarios for testing robustness; they are not estimates of hidden test priors. This experiment does not evaluate the earlier context sizes 1, 3 or 6. Reused development prompts are not a new confirmation set.

## Input and admission

Run once in a new private Kaggle notebook, CPU accelerator `None`, Internet off, with only the pinned output of private Stage 13 GLM V3, `kernelVersion` **368292220**. Require the unique `heldout_probabilities.npz` and its `readout_report.json` from that version. Use a 180-second execution limit and stop if input, source, numerical or readback checks fail. Verify the exact saved input binding and settings before running. Do not edit or rerun the source notebook. No new training, paid services, official submission or final-selection change is in scope.

The expected cache SHA256 is `565df6f8e652a9850ff11cf4e8b6a81fcccbc7db593186ff7f53358e7190bd16`. Read the NPZ with `allow_pickle=False` and require the six expected keys: `ids`, `prompt_ids`, `targets`, `folds`, `classes`, `probabilities`. Require 2,699 unique rows, 150 prompts, the exact ordered 18-class axis, canonical prompt-to-fold mapping digest `a55b035686cb7b31f6fbe852ac990668709aec4b913c5d0928d89d843bdbfe62`, and finite positive float64 probabilities of shape `(2699, 18)` with row sums within `1e-12` of one. Check the original report’s completion, probability-cache digest, held-out provenance and folds. Folds 1–4 contain 540 validation rows, fold 5 contains 539, and each contains 30 validation prompts; no prompt may appear in multiple folds. Run the balancing operations separately within each fold, because only the corresponding classifier excludes every scoring prompt in that batch. Never balance a batch pooled across folds.

## Frozen arms

Use two arms only. For `off`, predict the first maximizer in canonical class order of `log(P + 1e-12)` from the input probabilities `P`. For `uniform`, copy `P` to `Q`. When the batch has at least 18 rows, apply exactly 100 iterations of the S07 source operations, in order:

```python
Q *= (len(Q) / 18) / Q.sum(0, keepdims=True)
Q /= Q.sum(1, keepdims=True)
```

Predict from `log(Q + 1e-12)`, equivalent to the source’s fixed `alpha=1` arm. Below 18 rows bypass balancing, returning the same scoring array as `off`. Do not tune the coefficient, iteration count, temperature, thresholds or masks. Verify that the original probabilities are unchanged, the transformed matrix stays finite and positive, and row sums are within `1e-12` of one. Record the final column-mass residual; uniform probability mass does not require equal hard-label counts. Before using the real cache, run 17- and 18-row synthetic fixtures that prove the bypass and applied branches, with a nonuniform positive input and an independent numerical check of the expected result.

## Frozen scoring batches

Build four scenarios per fold, retaining all 18 classes and original row order when scoring. Scenario construction may use the held-out labels to simulate class mixtures but must never feed them into the arms. Use no outcome, probability or margin in mask construction.

| Scenario | Rows per class per fold | Rows per fold |
|---|---|---:|
| `natural_full` | All available rows | 540, or 539 for fold 5 |
| `matched_balanced` | 20 for every class | 360 |
| `skew_a` | 27 for the first nine ranked classes, 13 for the other nine | 360 |
| `skew_b` | 13 for the first nine ranked classes, 27 for the other nine | 360 |

Rank classes by ascending SHA256 digest of `b"stage13b-class-v1\x00" + label.encode("utf-8")`. Within each fold and class, rank rows by ascending SHA256 digest of `b"stage13b-row-v1\x00" + row_id.encode("utf-8")`. For a requested count take the first ranked rows. Break any digest ties by ascending original label or row ID UTF-8 bytes. Both arms use identical rows in a scenario, and the three matched scenarios use the same row rankings, so the skew comparisons keep the batch size fixed and control selection as far as possible. Different masks can still change which prompts and passages are represented; they do not isolate a real hidden class-prior change.

## Metrics and fixed decision

For each scenario, pool predictions across the five separate scoring batches only after applying the two arms. Report pooled 18-class balanced accuracy as the mean of class recalls with the scenario’s own support counts; also report fold scores, class recalls and their differences, confusion counts, prediction changes, corrected and broken decisions, and the numerical diagnostics. Compare `off − uniform` in percentage points. As an admission check, the `natural_full` off arm must reproduce the source GLM report’s full confusion counts, 1,239 correct predictions and exact macro recall `61547/134100` (45.89634601044%).

Use 5,000 paired prompt bootstrap draws with seed **20261111**. Within each fold, resample 30 prompt clusters with replacement from its canonical 30 held-out prompts; apply the same multiplicities to both arms and all scenarios. Recompute the pooled macro recall and arm difference using each draw’s class denominators. Report equal-tailed percentile 95% intervals for each scenario. If any draw lacks a class, fail the run without reporting scores rather than discarding the draw. These are conditional intervals on the already transformed predictions; do not refit or rebalance during bootstrap, and do not interpret them as uncertainty across training runs, hidden class mixtures or batches.

The `off` arm advances only to further validation if all gates pass:

- Both `skew_a` and `skew_b` improve by at least **0.5 percentage points**, and each paired 95% interval has a lower bound strictly above zero.
- The `natural_full` and `matched_balanced` deltas are each at least **−0.5 percentage points**.
- No class loses more than **5 percentage points** of recall in any scenario.

Keep all results, including neutral or negative ones. Do not change the rule after seeing outcomes. Even a pass does not establish benefit to S07 or justify a competition submission. If the screen fails, retain the source’s balancing setting in S07 and stop this fixed ablation; do not begin an automatic sweep.

Publish only source, aggregate metrics and verification evidence. Do not copy the private probabilities, row IDs, prompt IDs or masks into logs or public version control.
