# Stage 09 completed — fixed pretraining candidate rejected

**The candidate underperformed the matched control.** Private Kaggle V1 **361761811** succeeded in **1,453.1 seconds** on 24 October 2026. This is a completed negative development result. Our official best remains **S03: 0.620 public balanced accuracy**.

| Fixed fold-1 development balanced accuracy | Result |
|---|---:|
| Official-only control | 29.444444% |
| External pretraining then identical official training | 24.259259% |
| Candidate minus control | -5.185185 percentage points |
| Conditional paired prompt-bootstrap 95% interval | [-8.703704, -1.851852] points |
| Corrected / broken held-out decisions | 37 / 65 |
| Correct control / candidate responses | 159 / 131 of 540 |
| Changed predictions | 250 of 540 |
| Frozen pilot screen | Failed |

Both arms used the same 2,159 official training responses across 120 prompts, identical initial linear heads, official loading order, weighted loss and fixed three-epoch schedule. Validation contained 540 responses from 30 held-out prompt groups, with 30 examples for each of 18 labels. The candidate alone first received two full epochs of the 750 audited direct-OpenAI V4 examples: 150 examples for each of five GPT labels per epoch. Its head was reset before official training.

All phases completed: 94 pretraining updates and 405 updates in each official arm, with zero reported AMP overflows. No passage was truncated. Real CUDA preflight and saved probability/logit readback guards passed. Conservative allocated-device consumption was 2,867.15 GPU-seconds, below the 10,800-second cap. Kaggle's total wall time was about 24 minutes 13 seconds.

Reject this exact candidate. Even the five GPT labels receiving external pretraining lost mean recall overall, from 29.33% to 23.33%. Six classes improved, ten worsened and two tied. The largest regressions were Gemini 3.1 Pro (-30 points), Fable 5 (-20) and Opus 5 (-20); Gemini 3.6 Flash (+23.33) and Kimi K3 (+10) improved. RESULT.json preserves every class recall and both confusion matrices.

This reused development fold tests one narrow procedure on a smaller encoder, with extra compute. It does not establish that all public-data pretraining is ineffective or measure a contribution to the S03 ensemble. Official prompt wording is unavailable, so source declarations and response-overlap checks do not certify semantic prompt independence. The paired interval is conditional on the fitted models and excludes refitting and adaptive-selection uncertainty.

Source and prospective protocol were committed at `232587c3c1331e44c7e88cb02e71d64ad65e5148` before fitting. All three imported Kaggle code cells exactly matched the committed notebook. The terminal freeze hash matches SOURCE_FREEZE.json. CPU admission passed 13 contract checks and the runner passed 24 algebra/alignment/schema checks.

TERMINAL_VALIDATION.json independently recalculates both balanced accuracies and per-class recalls from the confusion matrices, confirms all supports, reconciles the net 37 - 65 = -28 correct decisions, and checks update counts and zero overflows. A separate reviewer reached the same aggregate conclusion. Private-row probabilities, exact paired identities and bootstrap draws were not independently reloaded; this validation is not an independent model refit.

Execution stayed private, offline and on included T4 compute. Original passages, row identifiers/probabilities and weights remain private. No paid API, new generation, automatic variant, new official submission or final-selection change followed this failed screen. The completed S03 watch remains disabled, and no scheduled Stage 09 watch was created.

[Immutable Kaggle run](https://www.kaggle.com/code/watchman777/name-that-llm-stage-09-pretraining-pilot?scriptVersionId=361761811). See PROTOCOL.md, DATA_AUDIT.json, GPU_PREFLIGHT.json, RESULT.json and TERMINAL_VALIDATION.json.
