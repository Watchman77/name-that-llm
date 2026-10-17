# Stage 06F findings and the next single hypothesis

[Stage 06F V4, immutable version 359521305](https://www.kaggle.com/code/watchman777/name-that-llm-stage-06f-pair-diagnostics?scriptVersionId=359521305) completed its CPU diagnostic in 20.3 seconds; the diagnostic function took approximately 1.57 seconds. All strict source gates passed, including the actual attached manifest hash `a5fcffaef27f4942af8a679370c30c247b8d872b4b6d711eccc0101ea7006dbd`. This confirms the writer-derived repair at runtime. The prior V2 integrity failure remains retained. No model was fitted or invoked, no prediction changed, and no corrective candidate was evaluated.

The source is the same available-component raw-control development proxy: contexts 1/3/6 score 52.6381%, 55.3067% and 56.4919%, mean 54.8122%. These are replayed Stage 06E scores, not a gain from this diagnostic. Official S02 remains a separate 54.9% result.

## Persistent pair confusion, not 900 independent examples

| Diagnostic | Context 1 | Context 3 | Context 6 |
| --- | ---: | ---: | ---: |
| True Gemini 3.7/3.8 passages | 300 | 300 | 300 |
| Correct global prediction | 135 | 146 | 139 |
| Cross-confused within the pair | 94 | 87 | 94 |
| True pair predicted as another class | 71 | 67 | 67 |
| Other classes falsely predicted as the pair | 65 | 54 | 49 |

The three contexts repeat the same 300 true-pair passages. There are **109 unique passages cross-confused at least once; 77 fail in all three contexts**, 12 in two and 20 in one. Thus 275 repeated cross-error observations must not be reported as 275 unique errors. Twenty-five point seven percent of the 300 passages cross-fail consistently. More context alone has not removed this persistent confusion.

At least one existing member prefers the true member of the pair on 78/94, 79/87 and 83/94 cross-errors. That demonstrates disagreement in the saved evidence; it does not tell us how to identify the right member without labels. There are 16, 8 and 11 cross-errors on which no member prefers the true pair label.

## Complementary evidence comes with substantial harm risk

The following counts use each member's **within-pair preference**, not its global top-one accuracy. A preference for the true pair label can coexist with an incorrect global prediction. Entries are repeated context observations, not independent passages.

| Member | Supports truth on current cross-errors, contexts 1 / 3 / 6 | Opposes truth on currently correct pair rows, contexts 1 / 3 / 6 |
| --- | --- | --- |
| Stylegram | 26 / 26 / 28 | 25 / 32 / 28 |
| Plain LoRA1 | 25 / 19 / 25 | 29 / 31 / 29 |
| Plain LoRA2 | 18 / 16 / 22 | 13 / 18 / 16 |
| Qwen token attention | 31 / 31 / 35 | 40 / 48 / 44 |
| DeBERTa token attention | 32 / 32 / 34 | 33 / 41 / 36 |
| Frozen DeBERTa plain | 30 / 29 / 31 | 31 / 31 / 25 |
| Frozen DeBERTa centered | 0 / 26 / 29 | 0 / 20 / 24 |

The two attention members offer a plausible complementary pair signal. They also oppose many correct decisions. Neither independently prefers truth on a majority of current cross-errors, and its mean signed contribution remains adverse on cross-errors in every context. These marginal counts do not establish which rows both members share, their joint ranking or a positive ensemble gain. Increasing an existing attention weight blindly is unsupported.

Centered DeBERTa is inactive on all 300 true-pair rows in context 1. Its zero votes there are inactive evidence, not failed predictions or 50/50 probabilities.

## A narrow test can bound collateral damage

| Existing raw top-two labels are exactly Gemini 3.7 and 3.8 | Context 1 | Context 3 | Context 6 |
| --- | ---: | ---: | ---: |
| Total scoped rows | 193 | 196 | 190 |
| Current pair cross-errors inside scope | 68 | 62 | 71 |
| Current correct pair rows inside scope | 101 | 113 | 101 |
| Already-wrong nonpair false entries inside scope | 24 | 21 | 18 |

This scope includes 201 of the 275 repeated pair cross-errors, but it also exposes 315 correct pair decisions to reversal. Its 63 nonpair observations are already falsely classified into the pair. A correction that only toggles these two labels inside this exact scope cannot create a new pair entry or alter any other class's recall; those other-class examples remain wrong. It can still break correct Gemini predictions and change which false label receives other-class passages. All these invariants must be asserted in a future run.

For scale only, perfectly correcting every currently cross-confused pair prediction while preserving everything else would add at most **3.3951 percentage points** to this proxy's mean balanced accuracy. Restricting to the top-two scope reduces that arithmetic ceiling to **2.4815 points**. These are label-informed ceilings, not candidate predictions, attainable gains or forecasts. This pair correction alone cannot bridge the current proxy to 70%.

## No useful formatting shortcut was established

Pair-correct, pair-cross and pair-other groups all have one line per passage and zero observed blank-line, heading, bullet or code-fence features in these summaries. Cross-confused passages are somewhat longer on average, but their distributions overlap broadly. The feature summaries do not demonstrate a label-separating rule or causal mechanism. Confidence bins likewise include correct and incorrect predictions together; no confidence threshold has been selected.

The conditional prompt bootstrap places mean pair-correct rate around a 41.78–51.89% interval and mean cross-error rate around 26.33–34.78%. These intervals fix every fitted model and prediction and reuse development prompts. They are descriptive uncertainty, not untouched validation or an estimate of a future candidate gain.

## One prospective next experiment

Test **one deterministic attention-pair correction**: only where the existing raw-control top two are exactly the fixed Gemini pair, choose the pair label favored by the equal sum of the cached Qwen and DeBERTa attention log-odds; preserve the control on an exact tie and preserve every prediction outside scope.

This is a cheap, falsifiable use of the observed complementary signal. It introduces no fitted meta-model, no nested-OOF training leakage, no new inference and no tuning sweep. It has a clear failure mode: correcting some existing pair errors may cost more currently correct decisions. The separate prospective Stage 06G plan fixes arithmetic, routing, endpoints and stop rules before any candidate evaluation.

Stage 06G has not run. The diagnostic binary arrays and row/prompt tables remain on the immutable Kaggle output unless separately retrieved; local source/result/aggregate table integrity does not amount to a second-machine row-level binary audit.
