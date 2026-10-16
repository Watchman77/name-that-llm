# Stage 03 private Kaggle verification

Saved version 2: https://www.kaggle.com/code/watchman777/name-that-llm-baseline-01?scriptVersionId=359158002

The frozen character-plus-style candidate completed successfully in 105.7 seconds, with CPU (no accelerator), internet disabled and private visibility. It reproduced 0.3541 pooled balanced accuracy, with fold values 0.3352, 0.3315, 0.3759, 0.3926 and 0.3355, and zero convergence warnings. Eight output files were saved.

This verifies execution and reproducibility in Kaggle's runtime on the same development data; it is not independent scientific validation. The supplied test has three preview rows; no official hidden-test score exists yet. Baseline version 1 remains accessible. No competition submission or paid API usage occurred.

Environment: Python 3.13.15, numpy 2.1.3, pandas 2.3.3, scipy 1.16.3, scikit-learn 1.6.1 and pyarrow 23.0.1. Candidate code is scripts/stage03_candidate.py; the reusable experiment runner is scripts/style_comparison.py with scripts/style_features.py.
