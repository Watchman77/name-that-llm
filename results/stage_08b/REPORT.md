# Stage 08B — frozen two-stage baseline

Checked live on 24 October 2026. Our private V1 completed its three-row preview successfully in **397.7 seconds** on two T4 GPUs, producing a **1.76 kB** output. This is an execution result, not development accuracy or an official public score.

- Our V1: https://www.kaggle.com/code/watchman777/name-that-llm-stage-08b-two-stage-reproduction?scriptVersionId=361730151
- Upstream: Nishant Dahal, V2 359887828, https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool?scriptVersionId=359887828 . Its displayed author public balanced accuracy is **0.592**.
- All five model code cells matched the complete rendered upstream cells exactly; saved diff +0/-0. No tuning or algorithm change.
- Four attachments: competition, `nishantdahal/ntl-two-stage-lora`, Qwen3-8B-Base V1 and Qwen3-14B-Base V1. Their pool/adapter/base contracts were among the real Stage 08 input audit's passing checks.
- Log milestones: train 2,699 / test 3; SVC 199s; 8B 277s; 14B 369s; candidate written. Full logs are retained in `PREVIEW_LOG.txt`. Author preview runtime and our preview runtime are distinct.

The original pipeline's training claims and reported grouped CV are not independently reproduced here. Three singleton examples do not exercise test-group centering or full-test balancing. No hidden test has run for this candidate. No official submission or final selection was changed.

We keep this completed baseline as a fallback while evaluating the newer Stage 08C recipe; avoid spending an official evaluation on the older candidate solely because its preview succeeded.

