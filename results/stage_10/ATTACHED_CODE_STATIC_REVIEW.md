# Attached custom-code static review

Reviewed the actual CPU export for Kirill V1 **361634630** on **24 October 2026**: 27 records deduplicated to 11 files. Text/AST/JSON inspection found **no concrete static blocker**. No custom code, model, joblib or Torch checkpoint was imported or executed. This report contains no custom source or weights.

- Night24 readouts uses the supplied fitted bank, features and texts; it does not indirectly open the quality-selected pool. Its expected fitted objects are standard sklearn estimators.
- UltraBERT uses local custom-model mappings and 18 canonical labels. Its config declares Transformers **5.16.1**; verify actual Kaggle imports and SDPA compatibility before inference. Do not apply earlier 4.56 pins. Hub-kernel decorators remain subject to the outer offline flags and `USE_HUB_KERNELS=NO`.
- ATLAS/rlcd contains local numerical/model code, safetensors key/shape/finite checks and caller-supplied model paths. No additional runtime asset dependency was found. ATLAS revision metadata alone does not prove local weight equivalence.
- No direct network, subprocess, credential/environment access or file writes were found in the reviewed custom modules.

The joblib bank and Torch token-attention files remain separate serialized execution boundaries: **their embedded pickle globals/reducers were not inspected**. Expected estimator classes from source do not establish the payload's complete behavior. Runtime compatibility, GPU execution and hidden-test success remain unverified.

Receipt correction: `modeling_ultrabert.py` has **30,581 UTF-8 bytes / 30,293 Unicode characters**. The index mislabeled its character count as bytes; its captured SHA-256 matches.

The adjacent JSON records all 11 SHA-256 receipts and 58,684 reviewed UTF-8 bytes. The private export matches the live CPU audit at SHA-256 `543be2bdafb87a7f558f397669c91c7ed2e31d303849d7b696cbb3a337633c6c`. The aggregate file-receipt SHA-256 is `f3dc56895f18c1f597aac5928c5a6b5b3c6c114f8f0451903260c41fb1767021`.
