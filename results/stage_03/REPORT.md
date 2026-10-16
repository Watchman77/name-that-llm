# Stage 03 — surface-style features

## Decision

The predeclared **character-plus-style** candidate is the new provisional development leader: **35.4131% balanced accuracy**. It improves over character-only (30.8553%) by **4.5578 percentage points**, with gains in all five fixed prompt-grouped folds. Freeze this recipe for private offline Kaggle verification. Preserve both earlier references.

## All candidates and references

| Variant | Balanced accuracy | Role |
|---|---:|---|
| Character TF-IDF + 111 style features | 35.4131% | New candidate |
| Character TF-IDF | 30.8553% | Frozen Stage 02 reference |
| Word + character TF-IDF | 30.1877% | Frozen baseline 01 |
| Style features alone | 29.2637% | New candidate; weaker than both references |

| New candidate | Reference | Difference in percentage points | Paired 95% prompt-bootstrap interval |
|---|---|---:|---:|
| Character + style | Character-only | +4.5578 | +2.7783 to +6.3379 |
| Character + style | Combined baseline | +5.2255 | +3.3733 to +7.1569 |
| Style-only | Character-only | −1.5916 | −3.4831 to +0.4087 |
| Style-only | Combined baseline | −0.9239 | −2.8526 to +1.0779 |

Intervals use 2,000 paired resamples of 150 prompt clusters with seed 20261016. They are descriptive development intervals conditional on fitted OOF predictions, excluding refitting/selection uncertainty and multiplicity correction. They are not a private-leaderboard guarantee or confirmatory publication result.

## What changed

The plan was committed before execution in `09e2fb41f37fd420caf3184d4d2512b864c0e3e7`. Two candidates were run, with no additional trials or parameter tuning. The 111 fixed features comprise 15 surface/format statistics, 39 punctuation rates and 57 function-word rates. Style features are standardised and L2-normalised inside the fold; the character block uses its existing TF-IDF recipe. Both blocks are concatenated with no tuned weights. LinearSVC settings remain C=1, balanced class weights, dual=True, max_iter=10000 and seed 20261016.

This supports the combined *recipe* on these development prompts. It does not isolate a causal contribution of punctuation or prove complete removal of topic/length effects. Adding a normalised block changes the representation and its effective regularisation geometry as well as adding information.

## Remaining weaknesses

Several labels remain difficult: GLM 5.3 Flash recall is 13.33%, GLM 5.3 is 15.33%, and GPT 5.6 Sol is 16.67%. Claude Opus 4.8 recall fell from 24.00% to 16.67% despite the aggregate gain; retain this regression in the record.

| Passage length | Rows | Character-only balanced accuracy | Character + style |
|---|---:|---:|---:|
| <=80 words | 1,019 | 23.99% | 28.06% |
| 81–150 words | 929 | 31.58% | 36.23% |
| >150 words | 751 | 38.34% | 43.42% |

All 18 labels appear in every length bin. The candidate improves in all three bins, but length remains a strong descriptive correlate; this table does not establish why.

## Checks and reproducibility

Source hash and all saved row/fold/label assignments matched. Prompt groups do not cross a split; scaling, vocabulary and IDF are fold-local. All 20 fold records and 2,699 OOF rows are retained. Scores were independently reconciled from per-model recalls. There were zero convergence warnings. Extractor checks verified simple manually counted punctuation/case/function-word examples and finite empty-input handling.

Run from the project root:

```bash
python scripts/style_comparison.py --data-dir data
```

Raw passage text is excluded from the outputs. No paid APIs, new data or competition submissions were used. All supplied prompts are development data; stronger research claims require genuinely new prompts or nested evaluation.
