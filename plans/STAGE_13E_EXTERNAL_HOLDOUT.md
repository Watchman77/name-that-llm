# Stage 13E: external prompt validation

Stage 13D's three-branch blend had the highest point score on the reused official development prompts. The character-only addition had weakened the GLM comparator, so the overall Stage 13D screen failed. The stronger observed comparator for Stage 13E is the fixed Stage 13C pair (47.565001% on those prompts). The 13D-minus-13C difference of +1.555804 points is retrospective and motivates the test without validating it.

## Admission before fitting or scoring

The [official Kaggle files](https://www.kaggle.com/competitions/name-that-llm/data) supply 150 labelled training prompts, already used across the previous experiments. The hidden competition labels are unavailable. Kaggle [allows freely and publicly available external data](https://www.kaggle.com/competitions/name-that-llm/overview/abstract). Use an exact, freely available dataset version and review its stated source and conditions; public availability alone does not resolve model identity or every upstream use condition.

First audit [Prvsiyan's extended corpus](https://www.kaggle.com/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset) on CPU with Internet off. Require clean complete `new` or `new2` groups, a publisher raw response matching text and exact requested/served identity, an attributable provider route, and one passage from each of the 18 labels. Exclude Kaggle Proxy aliases without a provider-side version string, `train_regen`, the 150 prior Stage 07A/09 prompts, and groups with exact, normalized, substring, shared 8-gram or high character similarity to official responses. The original official prompt wording is unavailable: this bounds known overlaps and does not prove absence of every semantic paraphrase. Fix the outcome-blind order by salted prompt hash. Require at least 30 whole groups, take whole blocks of 30 up to 90, and freeze the panel digest before feature inference. Report admission counts even when blocked. No response text, IDs, or prediction vectors leave the private notebook.

Oleg Zholobov's [Response Corpus](https://www.kaggle.com/datasets/olegzholobov/name-that-llm-response-corpus) is a separate lead, not a fallback panel ready for scoring. Its full-corpus counts do not establish clean short complete blocks, and the underlying sources have mixed terms and varying identity evidence. Qualifying it would require a separately frozen source/route and deduplication audit; never combine partially validated rows across corpora to force a pass.

## Frozen model comparison

Use only official training rows for all learned heads, scalers and the character vocabulary. Preserve the Stage 13C/13D estimators and preprocessing. Fit once per branch; compare on identical external rows:

| Arm | GLM | DeBERTa | Character TF-IDF |
| --- | ---: | ---: | ---: |
| G: reference | 100% | 0% | 0% |
| C: fixed Stage 13C comparator | 75% | 25% | 0% |
| D: fixed Stage 13D candidate | 50% | 25% | 25% |

The D–C contrast measures the replacement of GLM weight by character evidence while DeBERTa remains fixed. It is not a test of adding DeBERTa to a different strong production ensemble. Validate C–G on the external panel before calling C stronger there.

The existing embeddings contain official rows only. For an admitted external panel, pin the original GLM 358683059 and pretrained DeBERTa 358745836 extraction inputs; verify exact source and preprocessing, replay original official token batches against cached feature values, and stop if the frozen recipes do not reproduce within predeclared tolerances. The scoring code must only receive aligned row-bound features. Use the same fixed uniform Sinkhorn policy per 30-prompt block for every arm. Resample shared whole prompts within those blocks in 5,000 fixed-seed bootstrap draws. Both comparisons must gain at least 0.50 points with a strictly positive conditional 95% lower endpoint; D must also lose no more than 5 points of recall for any class versus C. Always report both comparisons if scored.

The external corpus can test generalization of these reduced official-only recipes to independently published responses. Different providers, settings and generation dates can shift style. It cannot validate or estimate S07's official score; older versions of Prvsiyan's corpus were used by S07. No new official submission or final-selection change is in scope.

## If external admission fails

The separate alternate-partition stress check uses one prospectively fixed SHA-256 ranking of the original 150 prompts. Refit each head inside the alternate five folds on 120 prompts and hold out the other 30, applying the same three arms and the same decision gates. Never reuse canonical out-of-fold probabilities for the new partition. A successful check would establish split robustness on reused development data; it would leave the fresh-prompt question open. Do not use either a blocked audit or an exploratory pass to justify promotion to the leaderboard.
