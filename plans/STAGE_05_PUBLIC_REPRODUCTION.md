# Stage 05 — public baseline reproduction, trial 01

Declared 17 October 2026, before executing our fork or receiving its competition feedback.

## Question and scope

Can our account run the unchanged public **LoRA x Adaptive Centering**, Lucas Boesen, scored V3 (`359126546`, displayed public balanced accuracy **0.549**), using its publicly attached inputs? The user explicitly asked us to investigate and adopt stronger public methods. This is a technical reproduction and candidate reference, not a claim of a new method or independent validation.

Source: https://www.kaggle.com/code/lucasboesen/lora-x-adaptive-centering?scriptVersionId=359126546

Our private fork: https://www.kaggle.com/code/watchman777/name-that-llm-stage-05-public-reproduction

The leading account's private recipe remains unknown. At the 17 October browser observation, the leaderboard leader scored 0.593. A separate public 0.552 notebook (Zbigniew Kirchner V1, `359152926`) requires a private datasource and a third `lora24c` checkpoint. We do not substitute missing weights or describe that notebook as reproducible. Its captured code hash was `00d652612d7465c355a551fc67cdbbdd354b518961f3503ef046099fff502862`.

## Frozen candidate

- Source code: `vendor/public_549_v3.py`, extracted from the rendered code cell of the exact scored version. SHA256: `261e27094fbefbfe8e6496feda4a274f8e9a8587a19b304da074a71f49cbc2f3`.
- Preserve original source, markdown credits, Apache-2.0 licence and fork lineage. Change only our notebook title and run metadata. No model weights, blend weights, clustering thresholds or decision rules are tuned.
- Official train SHA256: `40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707`.
- GPU T4 x2; internet off; inherited environment pinned to 2026-10-02. Included Kaggle quota only. No API calls, paid subscriptions, new synthetic data or paid compute.
- Source preview ran in 34m 8s. The author's markdown estimates about 1.5 hours for a full test rerun. These are estimates for our run, not guarantees. Prelaunch clarification: Save Version's advanced settings expose only accelerator use, with no configurable execution timeout. Do not claim a three-hour hard stop. Run the unchanged recipe under the included platform quota (00:00 / 30 hrs displayed); competition hidden-test execution is subject to its nine-hour limit. The preview session's exact hard limit is not exposed in this dialog. No continuous monitoring after this conversation turn is promised.
- Leave the separate Stage 04B DeBERTa V1 (`359319680`) unchanged. It remains an independently declared trial.

## Inputs inherited from the exact V3 fork

All 14 sources are visible and attached; there is no private/missing datasource in this version. Record exact inherited versions through the saved Kaggle version rather than replacing them with latest resources.

1. Name That LLM competition data.
2. `bbucxi/qwen35-9b-safetensors`.
3. `pjmathematician/ultrabert-380m-ntl`, dataset V1.
4. QwenLM Qwen3 4b-base, Transformers V1.
5. QwenLM Qwen3 8b-base, Transformers V1.
6. QwenLM Qwen2.5 7b, Transformers V1.
7. `lucasboesen/name-that-llm-glm4-9b-trunc25`.
8. `lucasboesen/name-that-llm-extract-encoders2`.
9. `lucasboesen/name-that-llm-electra-large-weights`.
10. `lucasboesen/name-that-llm-desklib-weights`.
11. `lightningv08/ntl-lora24-fivefold-training`.
12. `lucasboesen/name-that-llm-train-lora24-s7`.
13. `lucasboesen/name-that-llm-train-tokattn`.
14. `lucasboesen/name-that-llm-train-tokattn2`.

The recipe combines style-aware n-grams, LightGBM style features, frozen large-model linear heads, two Qwen LoRA classifiers, two token-attention heads and an UltraBERT fold ensemble. It infers test topic clusters from text; test data has no prompt_id. Centered heads add evidence only for clusters of 2–18 passages. Test standardization and partial Sinkhorn balancing use unlabelled test features, never hidden labels.

## Verification and decision rules

1. Confirm fork privacy, all 14 inputs, internet off, T4 x2 and environment pin before Save & Run All.
2. Retain syntax-check result, source hash, saved run ID, runtime settings and every execution outcome. Check compatibility assertions, LoRA anchor parity, non-finite messages and errors. Source in-sample checks are only technical checks.
3. A successful preview must create `submission.parquet` with exactly the preview IDs, valid model labels, no duplicate/missing IDs and one prediction per row. The three preview rows cannot measure accuracy.
4. Only after a successful technical run, use one official hidden-test rerun as S02 to measure reproduction. Record the displayed score at its actual precision. Do not treat the source's 0.549 as our score. Do not silently alter parameters and resubmit on leaderboard feedback.
5. Preserve S01 and all negative/failed experiments. Any failure is a technical outcome, not a score; diagnose before proposing a separately recorded retry.
6. Public final-fit checkpoints were trained on supplied training data. Evaluating those final checkpoints on our saved train folds would leak labels. Do not report an independent CV improvement from this reproduction. Later original changes require retraining within prompt-grouped folds and honest separation of model/ensemble selection.
7. If S02 reproduces the public baseline and improves on S01, retain it as a competitive reference. First place and a publishable original contribution remain separate, unproven goals.

## Source and licence review limits

The notebook is Apache-2.0; the UltraBERT dataset card also displays Apache-2.0 and attributes model code to dmis-lab. Original attribution is preserved in `vendor/NOTICE.md` and the fork. The main code was read and parsed; it uses attached local resources and writes the submission, with no shell, API-generation or outbound upload code. UltraBERT's attached configuration/model Python was inspected for its model implementation and imports. It uses `trust_remote_code=True` for that custom architecture; token-attention weights use `torch.load` without disabling the runtime's default protection. These are not guarantees about every byte of every third-party weight. Keep execution in Kaggle with internet off and do not attach secrets or unrelated private files. Upstream model terms still apply; this record does not grant new redistribution rights. A publication/release requires the complete upstream licence/provenance review.
