# Stage 13D: DeBERTa with a character branch

Private Kaggle CPU V2 [368344717](https://www.kaggle.com/code/watchman777/notebook0770477e92?scriptVersionId=368344717) succeeded in **149.1 seconds** on 11 November 2026. The saved run matched the [prospective source freeze](https://github.com/Watchman77/name-that-llm/commit/8766a6704f49515f91a7806f16f643366b804a5e) and all three pinned inputs, with Internet off. The run exactly replayed Stage 13C, completed all five character-model folds, and reloaded its saved aggregate report.

**Decision: the overall screen failed.** The character branch did not strengthen the GLM control. DeBERTa improved the character blend in every fold, but this did not validate its gain against a stronger comparator.

| Arm | Development balanced accuracy | Correct / 2,699 |
| --- | ---: | ---: |
| GLM control | 46.082525% | 1,244 |
| 75% GLM + 25% character TF-IDF | 45.971414% | 1,241 |
| 50% GLM + 25% character TF-IDF + 25% DeBERTa | 49.120805% | 1,326 |

The character blend changed balanced accuracy by **−0.111111 percentage points** (exactly −1/9 pp; conditional 95% interval **[−0.668168, +0.444202] pp**) and failed both admission requirements: at least +0.50 pp and a positive lower interval bound. It corrected 33 decisions and broke 36.

With character weight fixed, the GLM-to-DeBERTa swap added **+3.149391 pp** (exactly 12670/4023 pp; conditional 95% interval **[+2.074074, +4.259259] pp**). It corrected 155 decisions and broke 70. All 18 class recalls rose against that comparator; the smallest increase was +0.667 pp (Gemini 3.6 Flash). The marginal criteria passed, but both admission and marginal criteria were required for an overall pass.

The three-branch blend is the highest point score observed on these reused prompts. It is **+1.555804 pp** above the Stage 13C pair as an exploratory point comparison; no paired interval or promotion decision was prespecified for that contrast. The reported intervals resampled held-out prompts within folds on fixed predictions. Neither result estimates the official ensemble or an unseen leaderboard gain.

No official submission or final selection was changed. S07 remains the best verified official public balanced accuracy at **65.1%**.

[Protocol](../../plans/STAGE_13D_BROADER_COMPLEMENT.md) · [Kaggle logs](https://www.kaggle.com/code/watchman777/notebook0770477e92/log?scriptVersionId=368344717) · [Saved aggregate](ACTUAL_V2_REPORT.json) · [Run receipt](KAGGLE_V2_RECEIPT.json) · [Reconciliation](AGGREGATE_RECONCILIATION.json)
