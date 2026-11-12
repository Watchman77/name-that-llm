# Stage 13I: Gemini specialist reduced development accuracy

Private offline CPU Kaggle [V1 368935266](https://www.kaggle.com/code/watchman777/notebook87183f1de7/log?scriptVersionId=368935266) succeeded in **59.6 seconds** on 12 November 2026. The saved version matched the [frozen source](https://github.com/Watchman77/name-that-llm/commit/b28a0dd0f6e0ca990a3eba3e3f35e0f0ac5dd5a1), CPU, Internet off and all three input versions. The original Stage 13F geometry and confusion and the exact Stage 13H eligible cohort replayed before the specialist was scored.

**Decision: keep the frozen baseline.** The specialist changed only variants on 223 unassigned passages already predicted as Gemini. It fixed **14** decisions and broke **30** (net −16; 64 changed). It left all other predictions and all Gemini-versus-non-Gemini decisions intact.

| Arm | Development balanced accuracy | Correct / 2,699 |
| --- | ---: | ---: |
| Frozen Stage 13F assignment-on | 61.461844% | 1,659 |
| Add the Gemini specialist | 60.869252% | 1,643 |

The change was **−0.592593 percentage points**, exactly -16/27 pp; conditional 95% interval **[−1.037037, −0.148148] pp**. All three prespecified gates failed. Gemini 3.6 Flash recall fell **6.667 points**. The largest original confusion (Gemini 3.7 Flash → Gemini 3.8 Flash) remained at 18; the reverse confusion rose from 8 to 15. Within-Gemini errors in the gated cohort rose from **91 to 107**. This rejects this specific DeBERTa-only replacement on the reused development proxy; it does not rule out a discriminator with new evidence.

The [live public leaderboard](https://www.kaggle.com/competitions/name-that-llm/leaderboard) on **12 November 2026** showed Watchman 777 at **65.1%, rank 2,117**. First was **76.9%** and tenth was **74.7%**, gaps of **11.8** and **9.6** points respectively. The public board uses about 40% of the test data and final standings use the remaining 60%. No new official submission or final-selection change was made.

[Protocol](../../plans/STAGE_13I_GEMINI_SPECIALIST.md) · [Aggregate report](ACTUAL_V1_REPORT.json) · [Run receipt](KAGGLE_V1_RECEIPT.json) · [Reconciliation](AGGREGATE_RECONCILIATION.json) · [Leaderboard snapshot](../../results/submissions/PUBLIC_LEADERBOARD_2026-11-12.json)
