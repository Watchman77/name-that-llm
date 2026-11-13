# Stage 14B: Paired prompt training result

The corrected private [Kaggle CPU V4, 369084633](https://www.kaggle.com/code/watchman777/notebookb97e20c6ce/log?scriptVersionId=369084633) completed in 247.8 seconds on 13 November 2026. All five folds completed, covering 2,699 rows from 150 previously used training prompts. Input admission passed the official data, frozen feature hashes and exact external split declarations. No native adapter or native cached score was used. See the [prospective protocol](CONTRASTIVE_PROTOCOL.md) and [verified log aggregates](CONTRASTIVE_KAGGLE_V4.json).

| Arm | Primary: independent row balanced accuracy | Secondary: same balancing and assignment |
| --- | ---: | ---: |
| Refit GLM/DeBERTa control | 46.75% | 60.68% |
| Control + 20% cross-entropy head | 47.38% | 61.64% |
| Control + 20% prompt-contrastive head | 47.34% | 61.38% |

The contrastive mixture gained 0.59 percentage points over the control (paired 95% interval +0.19 to +1.04), below the +1.00 point threshold. It lost 0.04 points to the cross-entropy mixture (interval −0.63 to +0.56). It fixed 25 control errors and broke 9 correct answers on the primary comparison; the CE mixture fixed 45 and broke 28. Neither arm corrected any primary errors where an unassigned passage's true class had been outside the control's top three. Class regression caps passed for the contrastive arm, but the overall gate failed. The secondary contrastive gain was smaller than the CE gain as well. These intervals resample whole prompts on the fixed predictions; the 150 prompts are not an untouched holdout.

**Decision:** do not promote or spend a submission on either small head. The new loss failed to improve over the matched simpler control. Leave S07 as the best verified official submission, at 65.1%. For the next material attempt, qualify the exact Alex Chilton V25 68.4% public pipeline and its consumed inputs, rather than retuning this projection on the same 150 prompts. The separate Stage 14A GPU feasibility run only proves that the isolated native branch can run.
