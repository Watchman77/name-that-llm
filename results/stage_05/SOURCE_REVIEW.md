# Public method review and launch record

Observed 17 October 2026. This independently tracked reproduction has now succeeded as official S02 with displayed public balanced accuracy 0.549.

## What we learned

Our S01 uses character n-grams plus 111 surface-style features and scores 0.354 publicly. Stronger public methods add classifiers trained on large model representations and combine complementary predictions. They also attempt to remove subject-matter variation: test prompt IDs are unavailable, so the chosen recipe infers topic clusters and adds cluster-centered evidence. The selected source also adjusts class totals using unlabelled test predictions. These are concrete methods to reproduce, not evidence of what the leaderboard leader privately used.

The leader displayed 0.593 at this check. Zbigniew Kirchner's public notebook displayed 0.552 for exact V1 `359152926`, but its Input tab included a private datasource, and its source requires a third LoRA checkpoint (`lora24c/final/COMPLETE.json`). Latest V4 is a different version. We excluded that exact recipe rather than silently omitting a member or claiming its score.

Selected: Lucas Boesen's **LoRA x Adaptive Centering**, exact scored V3 `359126546`, displayed public score **0.549**, preview runtime 34m 8s. Its 14 inputs are public and attached in our private fork. All model settings and prediction code are unchanged, with Kaggle displaying `+0 -0`. Original markdown credits and Apache-2.0 attribution remain.

## Completed checks

- Read the selected Python source and parsed its syntax. No subprocess, shell, paid generation API, outbound HTTP call, credential handling or unrelated private input appears in the main implementation.
- Confirmed UltraBERT dataset V1 displays Apache-2.0 and attributes its model to dmis-lab. Inspected its configuration/model Python, including imports. Custom model loading and third-party checkpoints remain dependencies, not proofs of byte-level supply-chain integrity.
- Saved exact notebook JSON from Kaggle's rendered version viewer, including all 14 resource version IDs. The JSON's executable code cell matches the source page exactly.
- Actual code-cell SHA256: `e3d97306f3185ea5f73d7904d81331b27803e3ceb45970596590ec306969762e`.
- Python archive SHA256: `261e27094fbefbfe8e6496feda4a274f8e9a8587a19b304da074a71f49cbc2f3`. Its capture adds one terminal newline; AST and all non-terminal-newline text are identical. This capture distinction was established after launch and does not change the experiment.
- Verified fork is private, Internet off, GPU T4 x2 requested, and included quota displayed 00:00 / 30 hrs. No paid services used.
- Launched our V1 `359332774` after protocol commit `dfc009a58c5ca3e0be8b8b6a23044eac600515e8`. Live logs confirm `train (2699, 4) test (3, 2) gpus 2`, then completed stylegram and LightGBM stages and began GLM feature extraction.

## Runtime discrepancy to preserve

The editor showed **Pin to original environment (2026-10-02)**, but the running Logs page labels the environment **Latest Container Image**. Also, the version JSON's `isGpuEnabled=false` conflicts with its T4 accelerator setting and the explicit GPU-for-this-session launch option. Actual logs prove two visible GPUs. Record the actual runtime labels; do not claim bitwise environment reproduction. No restart or parameter change was made to conceal these discrepancies.

## Interpretation and next action

The private preview has now succeeded in 31m 25s. All stages and LoRA anchor checks passed; the exact code hash matches and submission.parquet exists. Source construction and assertions verify its intended IDs/columns, unique IDs, row count and labels; the Parquet UI preview is unsupported and no independent binary read is claimed. The declared official S02 has now succeeded at 0.549 public balanced accuracy, matching the source's displayed score and improving S01's 0.354 by 19.5 percentage points. The separate 0.552 recipe was not run. The three-row preview proves technical execution only. Preserve failures and diagnose before retrying. See OUTPUT_VERIFICATION.json and results/submissions/S02_REPORT.md.

The pretrained competition classifiers were fitted on the training set. Reusing those final checkpoints to score our saved training folds would leak labels. Independent development evaluation of a new method requires fitting within each held-out-prompt split. This copied recipe is a competitive reference; an original publication needs an attributable contribution, full upstream licence/provenance review and stronger independent evaluation.

## Final official outcome

Observed 2026-10-17T14:02:20.265Z: S02 Succeeded, public score **0.549**, rank **177/1204**. Tenth place 0.559, leader 0.593. The score matches the copied recipe at displayed precision; the final private score is unknown and no original contribution is claimed. The separate own DeBERTa trial completed at 22.76% development BA and is rejected. See results/submissions/S02_REPORT.md and results/stage_04b/REPORT.md.
