# Stage 09 — matched pretraining pilot launched

Private Kaggle V1 **361761811** is Running. The real pinned-model CUDA preflight passed at job time 68.4 seconds, including finite AMP outputs, finite encoder/head backward gradients and no fixture optimizer update. The live log verified both full external pretraining epochs completed at job time **207.9 seconds**: each used all 750 rows, exactly 150 examples per each of the five GPT classes; 94 optimizer updates total. The official-only control then started and was verified at epoch 1/3, update 25/405 (job time 248.6 seconds). Both final evaluations remain pending. **No development accuracy is available yet.**

Both arms use the same untouched DeBERTa-v3-base revision and identical seeded linear head before the official phase, identical 2,159 official training responses and a fixed three-epoch schedule. The candidate alone first receives two full epochs of the 750 audited, own-prompt direct-OpenAI V4 examples. Validation is fixed at 540 official responses from 30 held-out prompt groups, all 18 labels. See PROTOCOL.md for settings and decision boundaries.

CPU data admission: 750/750 calls and prompt associations matched; no response-overlap threshold fired; 13 audit failure/contract checks passed. Runner: 24 CPU algebra/alignment/schema checks passed. All three imported Kaggle code cells were copied through the visible editor and exactly matched the frozen notebook before execution. Source and protocol were committed at `232587c3c1331e44c7e88cb02e71d64ad65e5148` before fitting.

Runtime is private, Internet off, T4 x2, using the existing unmodified vendor resource output V1 359317729 and the external dataset explicitly pinned V4. The editor environment preference and saved viewer container label differ; actual Python/package/model hashes are enforced and the GPU preflight records the verified environment. No equality of unspecified container components is claimed. The conservative budget counts both allocated GPUs and is capped at three GPU-hours.

This is a narrow development comparison, including extra pretraining compute. It is not the upstream larger-corpus reproduction, new untouched evaluation or evidence of a gain above S03. Original text, original row identifiers/probabilities and fitted checkpoints remain private. No competition submission, final-selection change or paid API was made.

Our official best remains **S03: 0.620 public balanced accuracy**, +7.1 percentage points versus S02. The completed S03 watch remains disabled; this launch does not imply an active scheduled Stage 09 watch.

Live notebook: https://www.kaggle.com/code/watchman777/name-that-llm-stage-09-pretraining-pilot?scriptVersionId=361761811
