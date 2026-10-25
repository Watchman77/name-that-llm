"""Independent CPU inventory/schema/hash audit for frozen Lightning V28 362014737.
Never imports Torch/Transformers, upstream code, pickle/joblib or fitted estimators.
Only stdlib, NumPy and pandas; no training, inference, installations or network.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, importlib.metadata, io, json, math, os, platform, re, shutil, struct, sys, tempfile, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
SOURCE_VERSION=362014737
CONTRACT = {'source_version': 362014737, 'capture_json_sha256': '143351168ff87d0cd0f2712c29778dfad8acd34c1527088a19d4b42141869413', 'required_input_checks': [{'source': 'legacy.paths.train_csv', 'path': '/kaggle/input/competitions/name-that-llm/train.csv', 'kind': 'file'}, {'source': 'legacy.paths.test_csv', 'path': '/kaggle/input/competitions/name-that-llm/test.csv', 'kind': 'file'}, {'source': 'legacy.paths.pool', 'path': '/kaggle/input/datasets/nishantdahal/ntl-two-stage-lora/pool_v1.parquet', 'kind': 'file'}, {'source': 'legacy.paths.main_lora8', 'path': '/kaggle/input/datasets/nishantdahal/ntl-two-stage-lora/lora8b_v1', 'kind': 'directory'}, {'source': 'legacy.paths.main_lora14', 'path': '/kaggle/input/datasets/nishantdahal/ntl-two-stage-lora/lora14b_v1', 'kind': 'directory'}, {'source': 'legacy.paths.qwen3_8b_base', 'path': '/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1', 'kind': 'directory'}, {'source': 'legacy.paths.qwen3_14b_base', 'path': '/kaggle/input/models/qwen-lm/qwen-3/transformers/14b-base/1', 'kind': 'directory'}, {'source': 'legacy.paths.qwen2_7b_base', 'path': '/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b/1', 'kind': 'directory'}, {'source': 'legacy.paths.qwen3_4b_base', 'path': '/kaggle/input/models/qwen-lm/qwen-3/transformers/4b-base/1', 'kind': 'directory'}, {'source': 'legacy.paths.glm_base', 'path': '/kaggle/input/notebooks/lucasboesen/name-that-llm-glm4-9b-trunc25/glm4_9b_trunc25', 'kind': 'directory'}, {'source': 'legacy.paths.deberta_base', 'path': '/kaggle/input/notebooks/lucasboesen/name-that-llm-extract-encoders2/deberta-v3-large', 'kind': 'directory'}, {'source': 'legacy.paths.electra_base', 'path': '/kaggle/input/notebooks/lucasboesen/name-that-llm-electra-large-weights/electra-large-discriminator', 'kind': 'directory'}, {'source': 'legacy.paths.desklib_base', 'path': '/kaggle/input/notebooks/lucasboesen/name-that-llm-desklib-weights/desklib-detector', 'kind': 'directory'}, {'source': 'legacy.paths.lora24_seed7', 'path': '/kaggle/input/notebooks/lucasboesen/name-that-llm-train-lora24-s7/lora_out/job_final', 'kind': 'directory'}, {'source': 'legacy.paths.lora24_claude', 'path': '/kaggle/input/datasets/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head/final', 'kind': 'directory'}, {'source': 'legacy.paths.lora20_q25_instyle', 'path': '/kaggle/input/notebooks/vnton37nguyn/ntl-train-lora-qwen25-instyle/lora_out/job_final', 'kind': 'directory'}, {'source': 'legacy.paths.tokattn_q35_head', 'path': '/kaggle/input/notebooks/lucasboesen/name-that-llm-train-tokattn/tokattn_final.pt', 'kind': 'file'}, {'source': 'legacy.paths.tokattn_deb_head', 'path': '/kaggle/input/notebooks/lucasboesen/name-that-llm-train-tokattn2/tokattn_debl23_final.pt', 'kind': 'file'}, {'source': 'legacy.paths.q35_base', 'path': '/kaggle/input/datasets/bbucxi/qwen35-9b-safetensors', 'kind': 'directory'}, {'source': 'legacy.paths.hub_root', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts', 'kind': 'directory'}, {'source': 'legacy.paths.hub_manifest', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/QWEN38_ADDITION_MANIFEST.json', 'kind': 'file'}, {'source': 'legacy.paths.lora24_public', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/notebook-inputs/notebook-inputs/ntl-lora24-fivefold-training/lora24/final', 'kind': 'directory'}, {'source': 'legacy.paths.olmo_manifest', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/ntl-olmo2-native24-final-inference-assets/asset_manifest.json', 'kind': 'file'}, {'source': 'legacy.paths.olmo_base', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/notebook-inputs/notebook-inputs/olmo2-7b-fp16-offline-weights/olmo2-fp16-base', 'kind': 'directory'}, {'source': 'legacy.paths.smol_manifest', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/ntl-smollm3-two-stage-final-inference-assets/final_inference_manifest.json', 'kind': 'file'}, {'source': 'legacy.paths.smol_base_marker', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/notebook-inputs/notebook-inputs/smollm3-3b-base-offline-weights/smollm3_staging_complete.json', 'kind': 'file'}, {'source': 'legacy.paths.smol_base', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/notebook-inputs/notebook-inputs/smollm3-3b-base-offline-weights/smollm3-base', 'kind': 'directory'}, {'source': 'legacy.paths.mixed_manifest', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/manifest.json', 'kind': 'file'}, {'source': 'legacy.paths.qwen_adapter', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-27b-lora-tail8/final.safetensors', 'kind': 'file'}, {'source': 'legacy.paths.atlas_manifest', 'path': '/kaggle/input/datasets/josephadamski91/ntl-atlas-qwen3-8b-v1/atlas_adapter/atlas_adapter/atlas_bundle.json', 'kind': 'file'}, {'source': 'legacy.paths.main_adapters', 'path': '/kaggle/input/nlm-l24-adapters'}, {'source': 'legacy.paths.llama31_8b_base', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/llama31-prefix24'}, {'source': 'legacy.paths.prefit', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/ntl611-prefit'}, {'source': 'legacy.lists.qwen_wheels[0]', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/accelerate-1.15.0-py3-none-any.whl'}, {'source': 'legacy.lists.qwen_wheels[1]', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/bitsandbytes-0.50.2-py3-none-manylinux_2_24_x86_64.whl'}, {'source': 'legacy.lists.qwen_wheels[2]', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/huggingface_hub-1.32.0-py3-none-any.whl'}, {'source': 'legacy.lists.qwen_wheels[3]', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'}, {'source': 'legacy.lists.qwen_wheels[4]', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'}, {'source': 'legacy.lists.qwen_wheels[5]', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/transformers-5.14.1-py3-none-any.whl'}, {'source': 'legacy.lists.ultrabert_folds[0]', 'path': '/kaggle/input/datasets/pjmathematician/ultrabert-380m-ntl/fold0'}, {'source': 'legacy.lists.ultrabert_folds[1]', 'path': '/kaggle/input/datasets/pjmathematician/ultrabert-380m-ntl/fold1'}, {'source': 'legacy.lists.ultrabert_folds[2]', 'path': '/kaggle/input/datasets/pjmathematician/ultrabert-380m-ntl/fold2'}, {'source': 'legacy.lists.ultrabert_folds[3]', 'path': '/kaggle/input/datasets/pjmathematician/ultrabert-380m-ntl/fold3'}, {'source': 'legacy.lists.ultrabert_folds[4]', 'path': '/kaggle/input/datasets/pjmathematician/ultrabert-380m-ntl/fold4'}, {'source': 'legacy.lists.extra_claude[0]', 'path': '/kaggle/input/datasets/seyitkaangunes/ntl-extra-api-passages/ntl_extra_passages_claude.csv'}, {'source': 'legacy.lists.extra_gen[0]', 'path': '/kaggle/input/datasets/lucasboesen/ntl-free-proxy-generations/gen_passages.csv'}, {'source': 'legacy.lists.extra_generations[0]', 'path': '/kaggle/input/datasets/abdullah1afify/ntl-generations/ntl_generations.csv'}, {'source': 'legacy.lists.extra_proxy[0]', 'path': '/kaggle/input/datasets/dophuchung/ntl-proxy-generations/ntl_proxy_generations.csv'}, {'source': 'legacy.lists.extra_clean[0]', 'path': '/kaggle/input/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset/passages_clean.csv'}, {'source': 'legacy.staging.root', 'path': '/kaggle/input/datasets/bbucxi/qwen35-9b-safetensors', 'kind': 'directory'}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/datasets/bbucxi/qwen35-9b-safetensors/config.json', 'kind': 'file', 'expected_bytes': 3126}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/datasets/bbucxi/qwen35-9b-safetensors/merges.txt', 'kind': 'file', 'expected_bytes': 3353259}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/datasets/bbucxi/qwen35-9b-safetensors/model.safetensors-00001-of-00004.safetensors', 'kind': 'file', 'expected_bytes': 5276436216}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/datasets/bbucxi/qwen35-9b-safetensors/model.safetensors-00002-of-00004.safetensors', 'kind': 'file', 'expected_bytes': 5335161512}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/datasets/bbucxi/qwen35-9b-safetensors/model.safetensors-00003-of-00004.safetensors', 'kind': 'file', 'expected_bytes': 5368717440}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/datasets/bbucxi/qwen35-9b-safetensors/model.safetensors-00004-of-00004.safetensors', 'kind': 'file', 'expected_bytes': 3325995712}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/datasets/bbucxi/qwen35-9b-safetensors/model.safetensors.index.json', 'kind': 'file', 'expected_bytes': 79657}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/datasets/bbucxi/qwen35-9b-safetensors/tokenizer.json', 'kind': 'file', 'expected_bytes': 12807982}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/datasets/bbucxi/qwen35-9b-safetensors/tokenizer_config.json', 'kind': 'file', 'expected_bytes': 16710}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/datasets/bbucxi/qwen35-9b-safetensors/vocab.json', 'kind': 'file', 'expected_bytes': 6722759}, {'source': 'legacy.staging.root', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text', 'kind': 'directory'}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/LICENSE', 'kind': 'file', 'expected_bytes': 11544}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/README.md', 'kind': 'file', 'expected_bytes': 65249}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/chat_template.jinja', 'kind': 'file', 'expected_bytes': 8952}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/config.json', 'kind': 'file', 'expected_bytes': 51350}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/generation_config.json', 'kind': 'file', 'expected_bytes': 202}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-0.safetensors', 'kind': 'file', 'expected_bytes': 383865448}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-1.safetensors', 'kind': 'file', 'expected_bytes': 383865448}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-10.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-11.safetensors', 'kind': 'file', 'expected_bytes': 372313760}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-12.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-13.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-14.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-15.safetensors', 'kind': 'file', 'expected_bytes': 372313760}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-16.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-17.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-18.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-19.safetensors', 'kind': 'file', 'expected_bytes': 372313760}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-2.safetensors', 'kind': 'file', 'expected_bytes': 383865448}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-20.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-21.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-22.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-23.safetensors', 'kind': 'file', 'expected_bytes': 372313760}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-24.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-25.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-26.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-27.safetensors', 'kind': 'file', 'expected_bytes': 372313760}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-28.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-29.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-3.safetensors', 'kind': 'file', 'expected_bytes': 372313744}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-30.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-31.safetensors', 'kind': 'file', 'expected_bytes': 372313760}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-32.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-33.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-34.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-35.safetensors', 'kind': 'file', 'expected_bytes': 372313760}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-36.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-37.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-38.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-39.safetensors', 'kind': 'file', 'expected_bytes': 372313760}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-4.safetensors', 'kind': 'file', 'expected_bytes': 383865448}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-40.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-41.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-42.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-43.safetensors', 'kind': 'file', 'expected_bytes': 372313760}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-44.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-45.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-46.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-47.safetensors', 'kind': 'file', 'expected_bytes': 372313760}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-48.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-49.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-5.safetensors', 'kind': 'file', 'expected_bytes': 383865448}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-50.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-51.safetensors', 'kind': 'file', 'expected_bytes': 372313760}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-52.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-53.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-54.safetensors', 'kind': 'file', 'expected_bytes': 383865472}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-55.safetensors', 'kind': 'file', 'expected_bytes': 372313760}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-6.safetensors', 'kind': 'file', 'expected_bytes': 383865448}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-7.safetensors', 'kind': 'file', 'expected_bytes': 372313744}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-8.safetensors', 'kind': 'file', 'expected_bytes': 383865448}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/layers-9.safetensors', 'kind': 'file', 'expected_bytes': 383865448}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/manifest.json', 'kind': 'file', 'expected_bytes': 151231}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/merges.txt', 'kind': 'file', 'expected_bytes': 3353259}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/model-00016-of-00018.safetensors', 'kind': 'file', 'expected_bytes': 3979564040}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/model-00017-of-00018.safetensors', 'kind': 'file', 'expected_bytes': 2108759344}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/model.safetensors.index.json', 'kind': 'file', 'expected_bytes': 110564}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/text-outside.receipt.json', 'kind': 'file', 'expected_bytes': 707}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/text-outside.safetensors', 'kind': 'file', 'expected_bytes': 2542797120}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/tokenizer.json', 'kind': 'file', 'expected_bytes': 12809320}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/tokenizer_config.json', 'kind': 'file', 'expected_bytes': 17928}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/vocab.json', 'kind': 'file', 'expected_bytes': 6722759}, {'source': 'legacy.staging.root', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/notebook-inputs/notebook-inputs/smollm3-3b-base-offline-weights/smollm3-base', 'kind': 'directory'}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/notebook-inputs/notebook-inputs/smollm3-3b-base-offline-weights/smollm3-base/README.md', 'kind': 'file', 'expected_bytes': 12340}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/notebook-inputs/notebook-inputs/smollm3-3b-base-offline-weights/smollm3-base/config.json', 'kind': 'file', 'expected_bytes': 943}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/notebook-inputs/notebook-inputs/smollm3-3b-base-offline-weights/smollm3-base/generation_config.json', 'kind': 'file', 'expected_bytes': 126}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/notebook-inputs/notebook-inputs/smollm3-3b-base-offline-weights/smollm3-base/model-00001-of-00002.safetensors', 'kind': 'file', 'expected_bytes': 4966315264}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/notebook-inputs/notebook-inputs/smollm3-3b-base-offline-weights/smollm3-base/model-00002-of-00002.safetensors', 'kind': 'file', 'expected_bytes': 1183919744}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/notebook-inputs/notebook-inputs/smollm3-3b-base-offline-weights/smollm3-base/model.safetensors.index.json', 'kind': 'file', 'expected_bytes': 26899}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/notebook-inputs/notebook-inputs/smollm3-3b-base-offline-weights/smollm3-base/special_tokens_map.json', 'kind': 'file', 'expected_bytes': 151}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/notebook-inputs/notebook-inputs/smollm3-3b-base-offline-weights/smollm3-base/tokenizer.json', 'kind': 'file', 'expected_bytes': 17208861}, {'source': 'legacy.staging.file', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/notebook-inputs/notebook-inputs/smollm3-3b-base-offline-weights/smollm3-base/tokenizer_config.json', 'kind': 'file', 'expected_bytes': 50405}, {'source': 'v17.paths.test_csv', 'path': '/kaggle/input/competitions/name-that-llm/test.csv'}, {'source': 'v17.paths.new_two_stage_root', 'path': '/kaggle/input/datasets/nishantdahal/ntl-handwriting-blend'}, {'source': 'v17.paths.qwen3_14b', 'path': '/kaggle/input/models/qwen-lm/qwen-3/transformers/14b-base/1'}, {'source': 'v17.paths.qwen25_14b', 'path': '/kaggle/input/models/qwen-lm/qwen2.5/transformers/14b/1'}, {'source': 'v17.paths.svc_prefit', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/ntl627-svc-prefit'}, {'source': 'v17.paths.qwen_wheels[0]', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/accelerate-1.15.0-py3-none-any.whl'}, {'source': 'v17.paths.qwen_wheels[1]', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/bitsandbytes-0.50.2-py3-none-manylinux_2_24_x86_64.whl'}, {'source': 'v17.paths.qwen_wheels[2]', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/huggingface_hub-1.32.0-py3-none-any.whl'}, {'source': 'v17.paths.qwen_wheels[3]', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'}, {'source': 'v17.paths.qwen_wheels[4]', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'}, {'source': 'v17.paths.qwen_wheels[5]', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/transformers-5.14.1-py3-none-any.whl'}, {'source': 'v17.paths.legacy_prefit', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/ntl611-prefit'}, {'source': 'v17.paths.complement_style', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/ntl636-updated-style'}, {'source': 'v17.paths.own_nf4_base', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-27b-nf4'}, {'source': 'production.final.adapter', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/ntl627-own-fulltext/ntl627-own-fulltext/final.safetensors', 'kind': 'file', 'expected_bytes': 64605840}, {'source': 'production.final.metadata', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/ntl627-own-fulltext/ntl627-own-fulltext/final.json', 'kind': 'file', 'expected_bytes': 1377}, {'source': 'night24.synthetic_bank', 'path': '/kaggle/input/datasets/kirilltym/ntl-spectral-style-heads-paired-4404/night24_readout_banks/synthetic4404.joblib', 'kind': 'file', 'expected_bytes': 22110733}, {'source': 'night24.complete_metadata', 'path': '/kaggle/input/datasets/kirilltym/ntl-spectral-style-heads-paired-4404/night24_readout_banks/COMPLETE.json', 'kind': 'file', 'expected_bytes': 4243}], 'paths': {'test_csv': '/kaggle/input/competitions/name-that-llm/test.csv', 'new_two_stage_root': '/kaggle/input/datasets/nishantdahal/ntl-handwriting-blend', 'qwen3_14b': '/kaggle/input/models/qwen-lm/qwen-3/transformers/14b-base/1', 'qwen25_14b': '/kaggle/input/models/qwen-lm/qwen2.5/transformers/14b/1', 'svc_prefit': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/ntl627-svc-prefit', 'qwen_wheels': ['/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/accelerate-1.15.0-py3-none-any.whl', '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/bitsandbytes-0.50.2-py3-none-manylinux_2_24_x86_64.whl', '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/huggingface_hub-1.32.0-py3-none-any.whl', '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl', '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl', '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/transformers-5.14.1-py3-none-any.whl'], 'legacy_runtime': '/kaggle/temp/ntl636-fulltext-qwen-external3200-runtime/legacy', 'legacy_prefit': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/ntl611-prefit', 'complement_style': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/ntl636-updated-style', 'own_nf4_base': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-27b-nf4', 'own_fulltext_adapter': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/ntl627-own-fulltext/ntl627-own-fulltext/final.safetensors', 'night24_bank': '/kaggle/input/datasets/kirilltym/ntl-spectral-style-heads-paired-4404/night24_readout_banks/synthetic4404.joblib', 'night24_metadata': '/kaggle/input/datasets/kirilltym/ntl-spectral-style-heads-paired-4404/night24_readout_banks/COMPLETE.json'}, 'recipe': {'status': 'FROZEN_ALLROWS_QWEN10_KIRILL10_CANDIDATE', 'reference': 'fleongg/ntl-part-two-complement', 'reference_scriptVersionId': 361351359, 'reference_family': 'part_two_complement_atlas_0636', 'reference_score': 0.636, 'qwen_probability_share': 0.1, 'qwen_query_fraction': 1.0, 'complement_cache_source_audited': True, 'complement_share': 0.15, 'ATLAS': True, 'ATLAS_share': 0.1, 'checkpoint_stage': 'final', 'checkpoint_source_stage': 'official', 'checkpoint_source_update': 300, 'official_fit_rows': 2699, 'official_holdout_rows': 0, 'note': 'New unscored combination: published Kirill10% augmented readouts after ATLAS, then own final full-text Qwen10% on all rows before unchanged single Sinkhorn; ready readouts, no additional training.', 'helper_probability_share': 0.1, 'helper_reference': 'kirilltym/0-639-teach-more-keep-the-signature', 'helper_reference_script_version': 361634630, 'helper_dataset': 'kirilltym/ntl-spectral-style-heads-paired-4404/1'}, 'explicit_paths': {'status': 'ALL_FIXED_BINDINGS_PROVEN', 'source_v11_saved_cells_proof': 'research/large-qwen-20oct/candidate-v11-source-proof.json', 'paths': {'train_csv': {'path': '/kaggle/input/competitions/name-that-llm/train.csv', 'kind': 'file', 'status': 'PROVEN', 'evidence': 'research/large-qwen-20oct/explicit-paths-editor-root-copy-proof.json'}, 'test_csv': {'path': '/kaggle/input/competitions/name-that-llm/test.csv', 'kind': 'file', 'status': 'PROVEN', 'evidence': 'research/large-qwen-20oct/explicit-paths-editor-root-copy-proof.json'}, 'pool': {'path': '/kaggle/input/datasets/nishantdahal/ntl-two-stage-lora/pool_v1.parquet', 'kind': 'file', 'status': 'PROVEN', 'evidence': 'research/external-volume-20oct/diag-v2-reports/mount-preflight.json'}, 'main_lora8': {'path': '/kaggle/input/datasets/nishantdahal/ntl-two-stage-lora/lora8b_v1', 'kind': 'directory', 'status': 'PROVEN', 'evidence': 'research/external-volume-20oct/diag-v2-reports/mount-preflight.json'}, 'main_lora14': {'path': '/kaggle/input/datasets/nishantdahal/ntl-two-stage-lora/lora14b_v1', 'kind': 'directory', 'status': 'PROVEN', 'evidence': 'research/external-volume-20oct/diag-v2-reports/mount-preflight.json'}, 'qwen3_8b_base': {'path': '/kaggle/input/models/qwen-lm/qwen-3/transformers/8b-base/1', 'kind': 'directory', 'status': 'PROVEN', 'evidence': 'research/external-volume-20oct/diag-v2-reports/mount-preflight.json'}, 'qwen3_14b_base': {'path': '/kaggle/input/models/qwen-lm/qwen-3/transformers/14b-base/1', 'kind': 'directory', 'status': 'PROVEN', 'evidence': 'research/external-volume-20oct/diag-v2-reports/mount-preflight.json'}, 'qwen2_7b_base': {'path': '/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b/1', 'kind': 'directory', 'status': 'PROVEN', 'evidence': 'research/external-volume-20oct/diag-v2-reports/mount-preflight.json'}, 'qwen3_4b_base': {'path': '/kaggle/input/models/qwen-lm/qwen-3/transformers/4b-base/1', 'kind': 'directory', 'status': 'PROVEN', 'evidence': 'research/external-volume-20oct/diag-v2-reports/mount-preflight.json'}, 'glm_base': {'path': '/kaggle/input/notebooks/lucasboesen/name-that-llm-glm4-9b-trunc25/glm4_9b_trunc25', 'kind': 'directory', 'status': 'PROVEN', 'evidence': 'research/external-volume-20oct/diag-v2-reports/mount-preflight.json'}, 'deberta_base': {'path': '/kaggle/input/notebooks/lucasboesen/name-that-llm-extract-encoders2/deberta-v3-large', 'kind': 'directory', 'status': 'PROVEN', 'evidence': 'research/external-volume-20oct/diag-v2-reports/mount-preflight.json'}, 'electra_base': {'path': '/kaggle/input/notebooks/lucasboesen/name-that-llm-electra-large-weights/electra-large-discriminator', 'kind': 'directory', 'status': 'PROVEN', 'evidence': 'research/external-volume-20oct/diag-v2-reports/mount-preflight.json'}, 'desklib_base': {'path': '/kaggle/input/notebooks/lucasboesen/name-that-llm-desklib-weights/desklib-detector', 'kind': 'directory', 'status': 'PROVEN', 'evidence': 'research/external-volume-20oct/diag-v2-reports/mount-preflight.json'}, 'lora24_seed7': {'path': '/kaggle/input/notebooks/lucasboesen/name-that-llm-train-lora24-s7/lora_out/job_final', 'kind': 'directory', 'status': 'PROVEN', 'evidence': 'research/external-volume-20oct/diag-v2-reports/mount-preflight.json'}, 'lora24_claude': {'path': '/kaggle/input/datasets/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head/final', 'kind': 'directory', 'status': 'PROVEN', 'evidence': 'research/external-volume-20oct/diag-v2-reports/mount-preflight.json'}, 'lora20_q25_instyle': {'path': '/kaggle/input/notebooks/vnton37nguyn/ntl-train-lora-qwen25-instyle/lora_out/job_final', 'kind': 'directory', 'status': 'PROVEN', 'evidence': 'research/external-volume-20oct/diag-v2-reports/mount-preflight.json'}, 'tokattn_q35_head': {'path': '/kaggle/input/notebooks/lucasboesen/name-that-llm-train-tokattn/tokattn_final.pt', 'kind': 'file', 'status': 'PROVEN', 'evidence': 'research/external-volume-20oct/diag-v2-reports/mount-preflight.json'}, 'tokattn_deb_head': {'path': '/kaggle/input/notebooks/lucasboesen/name-that-llm-train-tokattn2/tokattn_debl23_final.pt', 'kind': 'file', 'status': 'PROVEN', 'evidence': 'research/external-volume-20oct/diag-v2-reports/mount-preflight.json'}, 'q35_base': {'path': '/kaggle/input/datasets/bbucxi/qwen35-9b-safetensors', 'kind': 'directory', 'status': 'PROVEN', 'evidence': 'research/large-qwen-20oct/q35-private-gpu-diagnostic/actual-v6-report.json'}, 'hub_root': {'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts', 'kind': 'directory', 'status': 'PROVEN', 'evidence': ['research/large-qwen-20oct/explicit-paths-editor-root-copy-proof.json current production editor exact flat hub Copy path', 'research\\large-qwen-20oct\\explicit-paths-live-dataset-file-inventory.json @hub/7']}, 'hub_manifest': {'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/QWEN38_ADDITION_MANIFEST.json', 'kind': 'file', 'status': 'PROVEN', 'evidence': ['research/large-qwen-20oct/explicit-paths-editor-root-copy-proof.json current production editor exact flat hub Copy path', 'research\\large-qwen-20oct\\explicit-paths-live-dataset-file-inventory.json @hub/7']}, 'lora24_public': {'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/notebook-inputs/notebook-inputs/ntl-lora24-fivefold-training/lora24/final', 'kind': 'directory', 'status': 'PROVEN', 'evidence': ['research/large-qwen-20oct/explicit-paths-editor-root-copy-proof.json current production editor exact flat hub Copy path', 'research\\large-qwen-20oct\\explicit-paths-live-dataset-file-inventory.json @hub/7']}, 'olmo_manifest': {'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/ntl-olmo2-native24-final-inference-assets/asset_manifest.json', 'kind': 'file', 'status': 'PROVEN', 'evidence': ['research/large-qwen-20oct/explicit-paths-editor-root-copy-proof.json current production editor exact flat hub Copy path', 'research\\large-qwen-20oct\\explicit-paths-live-dataset-file-inventory.json @hub/7']}, 'olmo_base': {'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/notebook-inputs/notebook-inputs/olmo2-7b-fp16-offline-weights/olmo2-fp16-base', 'kind': 'directory', 'status': 'PROVEN', 'evidence': ['research/large-qwen-20oct/explicit-paths-editor-root-copy-proof.json current production editor exact flat hub Copy path', 'research\\large-qwen-20oct\\explicit-paths-live-dataset-file-inventory.json @hub/7']}, 'smol_manifest': {'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/ntl-smollm3-two-stage-final-inference-assets/final_inference_manifest.json', 'kind': 'file', 'status': 'PROVEN', 'evidence': ['research/large-qwen-20oct/explicit-paths-editor-root-copy-proof.json current production editor exact flat hub Copy path', 'research\\large-qwen-20oct\\explicit-paths-live-dataset-file-inventory.json @hub/7']}, 'smol_base_marker': {'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/notebook-inputs/notebook-inputs/smollm3-3b-base-offline-weights/smollm3_staging_complete.json', 'kind': 'file', 'status': 'PROVEN', 'evidence': ['research/large-qwen-20oct/explicit-paths-editor-root-copy-proof.json current production editor exact flat hub Copy path', 'research\\large-qwen-20oct\\explicit-paths-live-dataset-file-inventory.json @hub/7']}, 'smol_base': {'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/notebook-inputs/notebook-inputs/smollm3-3b-base-offline-weights/smollm3-base', 'kind': 'directory', 'status': 'PROVEN', 'evidence': ['research/large-qwen-20oct/explicit-paths-editor-root-copy-proof.json current production editor exact flat hub Copy path', 'research\\large-qwen-20oct\\explicit-paths-live-dataset-file-inventory.json @hub/7']}, 'mixed_manifest': {'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text/manifest.json', 'kind': 'file', 'status': 'PROVEN', 'evidence': ['research/large-qwen-20oct/explicit-paths-editor-root-copy-proof.json current production editor exact flat hub Copy path', 'research\\large-qwen-20oct\\explicit-paths-live-dataset-file-inventory.json @hub/7']}, 'qwen_adapter': {'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-27b-lora-tail8/final.safetensors', 'kind': 'file', 'status': 'PROVEN', 'evidence': ['research/large-qwen-20oct/explicit-paths-editor-root-copy-proof.json current production editor exact flat hub Copy path', 'research\\large-qwen-20oct\\explicit-paths-live-dataset-file-inventory.json @hub/7']}, 'atlas_manifest': {'path': '/kaggle/input/datasets/josephadamski91/ntl-atlas-qwen3-8b-v1/atlas_adapter/atlas_adapter/atlas_bundle.json', 'kind': 'file', 'status': 'PROVEN', 'evidence': ['research/large-qwen-20oct/explicit-paths-editor-root-copy-proof.json', 'research\\large-qwen-20oct\\explicit-paths-live-dataset-file-inventory.json @ATLAS/3']}, 'main_adapters': {'status': 'PROVEN', 'path': '/kaggle/input/nlm-l24-adapters', 'evidence': 'Own private diagnostic editor Copy path verified 23Oct; dataset pinned version 6'}, 'llama31_8b_base': {'status': 'PROVEN', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/llama31-prefix24', 'evidence': 'Fixed new archive name llama31-prefix24.zip; CPU mount verification required before candidate dispatch'}, 'prefit': {'status': 'PROVEN', 'path': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/ntl611-prefit', 'evidence': 'Hub v9 CPU audit v17 verified all 73 listed files and numeric states'}}, 'lists': {'qwen_wheels': {'paths': ['/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/accelerate-1.15.0-py3-none-any.whl', '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/bitsandbytes-0.50.2-py3-none-manylinux_2_24_x86_64.whl', '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/huggingface_hub-1.32.0-py3-none-any.whl', '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl', '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl', '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/transformers-5.14.1-py3-none-any.whl'], 'status': 'PROVEN', 'evidence': 'research\\large-qwen-20oct\\explicit-paths-live-dataset-file-inventory.json @hub/7'}, 'ultrabert_folds': {'paths': ['/kaggle/input/datasets/pjmathematician/ultrabert-380m-ntl/fold0', '/kaggle/input/datasets/pjmathematician/ultrabert-380m-ntl/fold1', '/kaggle/input/datasets/pjmathematician/ultrabert-380m-ntl/fold2', '/kaggle/input/datasets/pjmathematician/ultrabert-380m-ntl/fold3', '/kaggle/input/datasets/pjmathematician/ultrabert-380m-ntl/fold4'], 'status': 'PROVEN', 'evidence': ['research/large-qwen-20oct/explicit-paths-editor-root-copy-proof.json', 'research\\large-qwen-20oct\\explicit-paths-live-dataset-file-inventory.json']}, 'extra_claude': {'paths': ['/kaggle/input/datasets/seyitkaangunes/ntl-extra-api-passages/ntl_extra_passages_claude.csv'], 'status': 'PROVEN', 'evidence': 'research\\large-qwen-20oct\\explicit-paths-live-dataset-file-inventory.json + canonical dataset attachment root'}, 'extra_gen': {'paths': ['/kaggle/input/datasets/lucasboesen/ntl-free-proxy-generations/gen_passages.csv'], 'status': 'PROVEN', 'evidence': 'research\\large-qwen-20oct\\explicit-paths-live-dataset-file-inventory.json + canonical dataset attachment root'}, 'extra_generations': {'paths': ['/kaggle/input/datasets/abdullah1afify/ntl-generations/ntl_generations.csv'], 'status': 'PROVEN', 'evidence': 'research\\large-qwen-20oct\\explicit-paths-live-dataset-file-inventory.json + canonical dataset attachment root'}, 'extra_proxy': {'paths': ['/kaggle/input/datasets/dophuchung/ntl-proxy-generations/ntl_proxy_generations.csv'], 'status': 'PROVEN', 'evidence': 'research\\large-qwen-20oct\\explicit-paths-live-dataset-file-inventory.json + canonical dataset attachment root'}, 'extra_clean': {'paths': ['/kaggle/input/datasets/prvsiyan/prvsiyans-extended-name-that-llm-dataset/passages_clean.csv'], 'status': 'PROVEN', 'evidence': 'research\\large-qwen-20oct\\explicit-paths-live-dataset-file-inventory.json + canonical dataset attachment root'}, 'extra_generated': {'paths': [], 'status': 'PROVEN', 'evidence': 'research\\large-qwen-20oct\\explicit-paths-live-dataset-file-inventory.json proves original wildcard contributes no files'}}, 'staging': {'/kaggle/input/datasets/bbucxi/qwen35-9b-safetensors': {'status': 'PROVEN', 'evidence': 'research/prvs-preflight-v2-real-layout/q35-pinned1-file-list.json', 'files': [{'name': 'config.json', 'bytes': 3126}, {'name': 'merges.txt', 'bytes': 3353259}, {'name': 'model.safetensors-00001-of-00004.safetensors', 'bytes': 5276436216}, {'name': 'model.safetensors-00002-of-00004.safetensors', 'bytes': 5335161512}, {'name': 'model.safetensors-00003-of-00004.safetensors', 'bytes': 5368717440}, {'name': 'model.safetensors-00004-of-00004.safetensors', 'bytes': 3325995712}, {'name': 'model.safetensors.index.json', 'bytes': 79657}, {'name': 'tokenizer.json', 'bytes': 12807982}, {'name': 'tokenizer_config.json', 'bytes': 16710}, {'name': 'vocab.json', 'bytes': 6722759}]}, '/kaggle/input/ntl-olmo2-native24-final-inference-assets/qwen38-mixed-tail8/name-that-llm-artifacts/qwen38-27b-mixed-tail8-text': {'status': 'PROVEN', 'evidence': 'research\\large-qwen-20oct\\explicit-paths-live-dataset-file-inventory.json @hub/7', 'files': [{'name': 'LICENSE', 'bytes': 11544}, {'name': 'README.md', 'bytes': 65249}, {'name': 'chat_template.jinja', 'bytes': 8952}, {'name': 'config.json', 'bytes': 51350}, {'name': 'generation_config.json', 'bytes': 202}, {'name': 'layers-0.safetensors', 'bytes': 383865448}, {'name': 'layers-1.safetensors', 'bytes': 383865448}, {'name': 'layers-10.safetensors', 'bytes': 383865472}, {'name': 'layers-11.safetensors', 'bytes': 372313760}, {'name': 'layers-12.safetensors', 'bytes': 383865472}, {'name': 'layers-13.safetensors', 'bytes': 383865472}, {'name': 'layers-14.safetensors', 'bytes': 383865472}, {'name': 'layers-15.safetensors', 'bytes': 372313760}, {'name': 'layers-16.safetensors', 'bytes': 383865472}, {'name': 'layers-17.safetensors', 'bytes': 383865472}, {'name': 'layers-18.safetensors', 'bytes': 383865472}, {'name': 'layers-19.safetensors', 'bytes': 372313760}, {'name': 'layers-2.safetensors', 'bytes': 383865448}, {'name': 'layers-20.safetensors', 'bytes': 383865472}, {'name': 'layers-21.safetensors', 'bytes': 383865472}, {'name': 'layers-22.safetensors', 'bytes': 383865472}, {'name': 'layers-23.safetensors', 'bytes': 372313760}, {'name': 'layers-24.safetensors', 'bytes': 383865472}, {'name': 'layers-25.safetensors', 'bytes': 383865472}, {'name': 'layers-26.safetensors', 'bytes': 383865472}, {'name': 'layers-27.safetensors', 'bytes': 372313760}, {'name': 'layers-28.safetensors', 'bytes': 383865472}, {'name': 'layers-29.safetensors', 'bytes': 383865472}, {'name': 'layers-3.safetensors', 'bytes': 372313744}, {'name': 'layers-30.safetensors', 'bytes': 383865472}, {'name': 'layers-31.safetensors', 'bytes': 372313760}, {'name': 'layers-32.safetensors', 'bytes': 383865472}, {'name': 'layers-33.safetensors', 'bytes': 383865472}, {'name': 'layers-34.safetensors', 'bytes': 383865472}, {'name': 'layers-35.safetensors', 'bytes': 372313760}, {'name': 'layers-36.safetensors', 'bytes': 383865472}, {'name': 'layers-37.safetensors', 'bytes': 383865472}, {'name': 'layers-38.safetensors', 'bytes': 383865472}, {'name': 'layers-39.safetensors', 'bytes': 372313760}, {'name': 'layers-4.safetensors', 'bytes': 383865448}, {'name': 'layers-40.safetensors', 'bytes': 383865472}, {'name': 'layers-41.safetensors', 'bytes': 383865472}, {'name': 'layers-42.safetensors', 'bytes': 383865472}, {'name': 'layers-43.safetensors', 'bytes': 372313760}, {'name': 'layers-44.safetensors', 'bytes': 383865472}, {'name': 'layers-45.safetensors', 'bytes': 383865472}, {'name': 'layers-46.safetensors', 'bytes': 383865472}, {'name': 'layers-47.safetensors', 'bytes': 372313760}, {'name': 'layers-48.safetensors', 'bytes': 383865472}, {'name': 'layers-49.safetensors', 'bytes': 383865472}, {'name': 'layers-5.safetensors', 'bytes': 383865448}, {'name': 'layers-50.safetensors', 'bytes': 383865472}, {'name': 'layers-51.safetensors', 'bytes': 372313760}, {'name': 'layers-52.safetensors', 'bytes': 383865472}, {'name': 'layers-53.safetensors', 'bytes': 383865472}, {'name': 'layers-54.safetensors', 'bytes': 383865472}, {'name': 'layers-55.safetensors', 'bytes': 372313760}, {'name': 'layers-6.safetensors', 'bytes': 383865448}, {'name': 'layers-7.safetensors', 'bytes': 372313744}, {'name': 'layers-8.safetensors', 'bytes': 383865448}, {'name': 'layers-9.safetensors', 'bytes': 383865448}, {'name': 'manifest.json', 'bytes': 151231}, {'name': 'merges.txt', 'bytes': 3353259}, {'name': 'model-00016-of-00018.safetensors', 'bytes': 3979564040}, {'name': 'model-00017-of-00018.safetensors', 'bytes': 2108759344}, {'name': 'model.safetensors.index.json', 'bytes': 110564}, {'name': 'text-outside.receipt.json', 'bytes': 707}, {'name': 'text-outside.safetensors', 'bytes': 2542797120}, {'name': 'tokenizer.json', 'bytes': 12809320}, {'name': 'tokenizer_config.json', 'bytes': 17928}, {'name': 'vocab.json', 'bytes': 6722759}]}, '/kaggle/input/ntl-olmo2-native24-final-inference-assets/notebook-inputs/notebook-inputs/smollm3-3b-base-offline-weights/smollm3-base': {'status': 'PROVEN', 'files': [{'name': 'README.md', 'bytes': 12340}, {'name': 'config.json', 'bytes': 943}, {'name': 'generation_config.json', 'bytes': 126}, {'name': 'model-00001-of-00002.safetensors', 'bytes': 4966315264}, {'name': 'model-00002-of-00002.safetensors', 'bytes': 1183919744}, {'name': 'model.safetensors.index.json', 'bytes': 26899}, {'name': 'special_tokens_map.json', 'bytes': 151}, {'name': 'tokenizer.json', 'bytes': 17208861}, {'name': 'tokenizer_config.json', 'bytes': 50405}]}}, 'hub_alias_discrepancy': 'Current production editor Copy gives /kaggle/input/ntl-olmo2-native24-final-inference-assets. CPU utility v7 proved the different mixed-utility canonical hub root absent. This draft uses the exact current editor flat root only, no alias fallback; all foreign canonical paths are retained.', 'foreign_version_policy': 'Original v11 attachment metadata is retained. Fresh MCP filename inventories with version=None establish names, not a different pinned version; no latest-version upgrade is performed by this source patch.'}, 'contracts': {'official_train_sha256': '40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707', 'qwen_final_metadata': {'stage': 'official', 'update': 300, 'official_fit_rows': 2699, 'official_holdout_rows': 0, 'native_decoder_layers': 64, 'executed_decoder_layers': 64, 'lora_target_layer_indices': [56, 57, 58, 59, 60, 61, 62, 63], 'source_revision': '26157380225e427827263c34df23018a1e28dff5', 'text_truncation': False, 'inference_max_length': None}, 'qwen_config': {'text_config.num_hidden_layers': 64, 'text_config.hidden_size': 5120}, 'qwen_head_shapes': {'plain.weight': [18, 40960], 'plain.bias': [18], 'centered.weight': [18, 40960], 'centered.bias': [18], 'norms.i.weight': [5120], 'norms.i.bias': [5120], 'norm_count': 8}, 'qwen_lora_rank': 16, 'qwen_lora_parameter_key_rule': 'backbone.layers.{layer}.{submodule}.lora_A|lora_B for every nn.Linear in layers56..63; exact submodules derive from config/adapter metadata, not guessed by this extraction', 'labels': ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5', 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3'], 'literal_path_policy': 'Every required /kaggle/input path preserved verbatim; no aliases/discovery/fallback'}, 'staging_constants': {'NF4_FILES': {'LICENSE': 11544, 'README.md': 27108, 'chat_template.jinja': 8952, 'config.json': 4222, 'generation_config.json': 202, 'merges.txt': 3353259, 'model-00001-of-00007.safetensors': 2542796928, 'model-00002-of-00007.safetensors': 3967729339, 'model-00003-of-00007.safetensors': 3973498949, 'model-00004-of-00007.safetensors': 3992447326, 'model-00005-of-00007.safetensors': 4000148996, 'model-00006-of-00007.safetensors': 98537568, 'model-00007-of-00007.safetensors': 849400424, 'model.safetensors.index.json': 396308, 'tokenizer.json': 12809320, 'tokenizer_config.json': 17928, 'vocab.json': 6722759}, 'ULTRA_FOLD_FILES': {'0': {'config.json': 2739, 'configuration_ultrabert.py': 2209, 'model.safetensors': 755137636, 'modeling_ultrabert.py': 30581, 'tokenizer.json': 3523731, 'tokenizer_config.json': 814}, '1': {'config.json': 2739, 'configuration_ultrabert.py': 2209, 'model.safetensors': 755137636, 'modeling_ultrabert.py': 30581, 'tokenizer.json': 3523731, 'tokenizer_config.json': 814}, '2': {'config.json': 2739, 'configuration_ultrabert.py': 2209, 'model.safetensors': 755137636, 'modeling_ultrabert.py': 30581, 'tokenizer.json': 3523731, 'tokenizer_config.json': 814}, '3': {'config.json': 2739, 'configuration_ultrabert.py': 2209, 'model.safetensors': 755137636, 'modeling_ultrabert.py': 30581, 'tokenizer.json': 3523731, 'tokenizer_config.json': 814}, '4': {'config.json': 2739, 'configuration_ultrabert.py': 2209, 'model.safetensors': 755137636, 'modeling_ultrabert.py': 30581, 'tokenizer.json': 3523731, 'tokenizer_config.json': 814}}}, 'source_manifest_sha256': 'f1496ecaeead9dae775d6a308b24a5a2382e6f8adaad9c677175f3b1e911a945'}
LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5', 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
SORTED_LABELS=sorted(LABELS)
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def safe_relative(name):
    path = Path(str(name))
    if path.is_absolute() or '..' in path.parts or not path.parts or '\x00' in str(name):
        raise ValueError('Unsafe relative asset path')
    return path

def checked_labels(value, ordered=None):
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, list) or len(value) != 18 or set(value) != set(LABELS):
        raise ValueError('Checkpoint label set differs from the 18 official labels')
    if ordered is not None and value != ordered:
        raise ValueError('Checkpoint label order differs from its frozen source head')
    return value

def safetensor_header(path):
    path = Path(path)
    with path.open('rb') as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError('Truncated safetensors length prefix')
        size = struct.unpack('<Q', raw)[0]
        if size < 2 or size > 16 * 1024 * 1024 or size + 8 > path.stat().st_size:
            raise ValueError('Invalid safetensors header length')
        header = json.loads(f.read(size))
    result = {}
    regions=[]
    widths={'BOOL':1,'U8':1,'I8':1,'F8_E4M3':1,'F8_E4M3FN':1,'F8_E5M2':1,'F8_E8M0':1,'I16':2,'U16':2,'F16':2,'BF16':2,'I32':4,'U32':4,'F32':4,'I64':8,'U64':8,'F64':8}
    for key, value in header.items():
        if key == '__metadata__':
            continue
        shape, offsets = value['shape'], value['data_offsets']
        if (not isinstance(shape, list) or not all(isinstance(x, int) and x >= 0 for x in shape)
                or not isinstance(offsets, list) or len(offsets) != 2 or offsets[0] < 0
                or offsets[1] < offsets[0] or offsets[1] > path.stat().st_size - size - 8):
            raise ValueError('Invalid safetensors tensor extent')
        if value['dtype'] not in widths or offsets[1]-offsets[0]!=math.prod(shape)*widths[value['dtype']]:
            raise ValueError('Safetensors dtype/shape byte extent differs')
        if offsets[1]>offsets[0]:regions.append((offsets[0],offsets[1]))
        result[key] = {'shape': shape, 'dtype': value['dtype'], 'data_offsets': offsets}
    regions.sort()
    if any(current[0]<previous[1] for previous,current in zip(regions,regions[1:])):
        raise ValueError('Overlapping safetensors tensor extents')
    if not result:
        raise ValueError('Empty safetensors tensor inventory')
    return result


def read_json(path, maximum=32*1024*1024):
    p=Path(path)
    if p.stat().st_size>maximum: raise ValueError('Bounded JSON reader exceeded: '+p.name)
    return json.loads(p.read_text(encoding='utf-8'))

def expect_shape(header, name, shape):
    if name not in header or header[name]['shape'] != list(shape):
        raise ValueError('Safetensors shape mismatch: '+name)

def qwen_label_alignment(native_labels):
    """Mirror the frozen worker-to-ensemble label-name transport, without reordering weights."""
    if not isinstance(native_labels,list) or not all(isinstance(label,str) for label in native_labels):
        raise ValueError('Qwen native labels must be a string list')
    checked_labels(native_labels)
    permutation=[native_labels.index(label) for label in LABELS]
    if sorted(permutation)!=list(range(18)) or [native_labels[i] for i in permutation]!=LABELS:
        raise ValueError('Qwen native-to-canonical class permutation invalid')
    return {'native_head_label_order':list(native_labels),'canonical_ensemble_label_order':list(LABELS),
            'source_column_permutation_to_canonical':permutation,
            'source_alignment_rule':'value[:, [labels.index(label) for label in LABELS]]',
            'native_head_or_metadata_reordered':False,
            'source_proof':'NF4 reload preserves metadata.labels; own worker saves native labels; orchestrator aligned() reorders by label names before Qwen blending.'}

def required_tail8_lora_keys(index):
    suffix='.weight.quant_state.bitsandbytes__nf4'
    names=[]
    for key in index:
        if not key.startswith('model.language_model.layers.') or not key.endswith(suffix):continue
        module=key[len('model.language_model.'): -len(suffix)]
        match=re.match(r'layers\.(\d+)\.',module)
        if match and int(match.group(1)) in range(56,64):names.append(module)
    layers={int(x.split('.')[1]) for x in names}
    if layers!=set(range(56,64)):raise ValueError('NF4 index does not identify every tail8 layer linear')
    return {f'backbone.{module}.lora_{letter}' for module in names for letter in ['A','B']}

def check_complete_lora_keys(actual,index):
    expected=required_tail8_lora_keys(index)
    if set(actual)!=expected:raise ValueError('Final adapter LoRA keys differ from all NF4 tail8 linears')
    return expected

def linear_state_contract(path,width):
    state=state_inventory(path)
    if state.get('coef',{}).get('shape')!=[18,width] or state.get('intercept',{}).get('shape')!=[18]:
        raise ValueError('Required linear coefficient axes differ: '+str(path))
    if 'classes' in state and state['classes'].get('small_axis_values')!=list(range(18)):
        raise ValueError('Required linear class order differs')
    for key in ['mean','scale']:
        if key in state and state[key]['shape']!=[width]:raise ValueError('Linear scaler axes differ')
    if 'scale' in state:
        with np.load(path,allow_pickle=False) as z:
            if (z['scale']<=0).any():raise ValueError('Linear scale must be positive')
    return state

def gaussian_state_contract(path,width,reduced=384):
    state=state_inventory(path)
    for key,shape in [('mx',[width]),('sx',[width]),('W',[width,reduced]),('m',[18,reduced]),('lam',[reduced])]:
        if state.get(key,{}).get('shape')!=shape:raise ValueError('Required Gaussian state axes differ: '+key)
    with np.load(path,allow_pickle=False) as z:
        if (z['sx']<=0).any() or (z['lam']<0).any():raise ValueError('Gaussian scale/covariance invalid')
    meta=read_json(Path(path).with_suffix('.json'))
    if not isinstance(meta,dict) or not isinstance(meta.get('T1'),(int,float)) or not isinstance(meta.get('Tg'),(int,float)) or meta['T1']<=0 or meta['Tg']<=0:
        raise ValueError('Required Gaussian temperature metadata missing/invalid')
    return state

def style_bank_contract(directory):
    p=Path(directory);width=0
    for i in range(4):
        record=read_json(p/f'view{i}.json');vocab=record.get('vocabulary');params=record.get('params')
        if not isinstance(vocab,dict) or not vocab or set(vocab.values())!=set(range(len(vocab))) or not isinstance(params,dict):
            raise ValueError('Style vectorizer vocabulary/indices/parameters invalid')
        if record.get('view') not in ['char','word','shape','function'] or not isinstance(record.get('weight'),(int,float)) or record['weight']<=0:
            raise ValueError('Style view/weight missing or invalid')
        idf=np.load(p/f'view{i}.idf.npy',allow_pickle=False)
        if idf.shape!=(len(vocab),) or not np.isfinite(idf).all():raise ValueError('Style vocabulary/IDF axes differ')
        width+=len(vocab)
    for name in ['plain.npz','twin.npz']:linear_state_contract(p/name,width)
    return {'views':4,'features':width,'required_linear_states':['plain.npz','twin.npz']}

def state_inventory(path):
    """Read safe numeric arrays only; forbid object arrays and oversized decompression."""
    p=Path(path); result={}
    if p.suffix=='.npz':
        with zipfile.ZipFile(p) as archive:
            if any(info.file_size>512*1024*1024 for info in archive.infolist()):
                raise ValueError('NPZ member exceeds bounded numeric reader')
        context=np.load(p,allow_pickle=False)
        try:
            for key in context.files:
                a=context[key]
                if a.dtype.hasobject:raise ValueError('Object numeric state is not admitted')
                if a.dtype.kind in 'fciu' and not np.isfinite(a).all():raise ValueError('Nonfinite state: '+key)
                result[key]={'shape':list(a.shape),'dtype':str(a.dtype),'finite_numeric':a.dtype.kind in 'fciu'}
                if key in ['classes','labels','pool_layers','layers'] and a.size<=1000:
                    result[key]['small_axis_values']=a.tolist()
        finally:context.close()
    else:
        a=np.load(p,allow_pickle=False,mmap_mode='r')
        if a.dtype.hasobject:raise ValueError('Object NPY is not admitted')
        for start in range(0,a.size,1024*1024):
            if a.dtype.kind in 'fciu' and not np.isfinite(a.reshape(-1)[start:start+1024*1024]).all():
                raise ValueError('Nonfinite NPY state')
        result={'array':{'shape':list(a.shape),'dtype':str(a.dtype),'finite_numeric':a.dtype.kind in 'fciu'}}
    return result

class Audit:
    def __init__(self,root='/kaggle/input',out='/kaggle/working/stage11_preflight'):
        self.root=Path(root);self.out=Path(out);self.checks=[];self.details={};self.warnings=[];self.code=[];self.hashed={};self.header_cache={}
    def local(self,literal):
        p=Path(literal)
        if not p.is_absolute() or not p.is_relative_to('/kaggle/input') or '..' in p.parts:
            raise ValueError('Expected an unmodified literal input path')
        return self.root/p.relative_to('/kaggle/input')
    def rel(self,path):return str(Path(path).relative_to(self.root))
    def hashed_file(self,path):
        p=Path(path)
        if p not in self.hashed:self.hashed[p]=sha(p)
        return self.hashed[p]
    def header(self,path):
        p=Path(path)
        if p not in self.header_cache:self.header_cache[p]=safetensor_header(p)
        return self.header_cache[p]
    def check(self,name,fn):
        try:
            value=fn();self.details[name]=value;self.checks.append({'name':name,'status':'PASS'});print('PREFLIGHT_PASS',name,flush=True);return value
        except Exception as e:
            self.checks.append({'name':name,'status':'FAIL','error_type':type(e).__name__,'error':str(e)[:1600]});print('PREFLIGHT_FAIL',name,type(e).__name__,str(e)[:300],flush=True);return None
    def required(self,record):
        p=self.local(record['path']);kind=record.get('kind','exists')
        if not p.exists():raise ValueError('Mandatory literal mount/path missing: '+record['path'])
        if kind=='file' and (not p.is_file() or p.stat().st_size==0):raise ValueError('Required nonempty file missing')
        if kind=='directory' and (not p.is_dir() or not any(p.iterdir())):raise ValueError('Required nonempty directory missing')
        if 'expected_bytes' in record and p.stat().st_size!=record['expected_bytes']:raise ValueError('Source-pinned byte count differs: '+record['path'])
        return {'path':self.rel(p),'source_record':record['source'],'kind':kind,'bytes':p.stat().st_size if p.is_file() else None,'expected_bytes':record.get('expected_bytes'),'discovered_or_remapped':False}
    def export_text(self,p,role):
        p=Path(p)
        if p.stat().st_size>512*1024:raise ValueError('Private custom code export exceeds bound')
        text=p.read_text(encoding='utf-8');digest=self.hashed_file(p)
        if p.suffix=='.py':ast.parse(text)
        row={'path':self.rel(p),'role':role,'bytes':p.stat().st_size,'sha256':digest}
        previous=next((x for x in self.code if x['sha256']==digest and 'text' in x),None)
        if previous:row['same_text_as']=previous['path']
        else:row['text']=text
        self.code.append(row)
        return {k:v for k,v in row.items() if k!='text'}
    def official(self):
        p=self.local('/kaggle/input/competitions/name-that-llm/train.csv')
        if self.hashed_file(p)!=CONTRACT['contracts']['official_train_sha256']:raise ValueError('Official train SHA256 changed')
        f=pd.read_csv(p,dtype=str,keep_default_na=False)
        if not {'id','prompt_id','text','model'}.issubset(f) or len(f)!=2699 or f.prompt_id.nunique()!=150 or set(f.model)!=set(LABELS) or not f.id.is_unique:
            raise ValueError('Official train schema/count/label contract differs')
        return {'rows':len(f),'prompts':int(f.prompt_id.nunique()),'labels':18,'sha256':self.hashed_file(p),'test_text_or_test_labels_opened':False,'original_rows_exported':False}
    def tokenizer(self,directory):
        p=Path(directory)
        names=['tokenizer.json','tokenizer.model','spm.model','vocab.txt','vocab.json','merges.txt','tokenizer_config.json','special_tokens_map.json','added_tokens.json']
        files=[p/n for n in names if (p/n).is_file()]
        if not any(x.name in names[:5] and x.stat().st_size>0 for x in files):raise ValueError('Tokenizer vocabulary/model missing')
        return [{'name':x.name,'bytes':x.stat().st_size,'sha256':self.hashed_file(x)} for x in files]
    def model(self,literal,require_tokenizer=True):
        p=self.local(literal);cfg=read_json(p/'config.json');indexes=sorted(p.glob('*.index.json'));weights={x.name:x for x in p.glob('*.safetensors')}
        if not weights and not list(p.glob('*.bin')):raise ValueError('Model weight files missing')
        tensor_total=0
        for index in indexes:
            doc=read_json(index);mapping=doc.get('weight_map')
            if not isinstance(mapping,dict) or not mapping:raise ValueError('Empty/invalid model index')
            for key,name in mapping.items():
                shard=p/safe_relative(name)
                if not shard.is_file():raise ValueError('Indexed shard missing: '+name)
                if shard.suffix=='.safetensors' and key not in self.header(shard):raise ValueError('Index tensor missing from safetensors shard: '+key)
            tensor_total+=len(mapping)
        for path in weights.values():self.header(path)
        token=self.tokenizer(p) if require_tokenizer else []
        return {'path':self.rel(p),'model_type':cfg.get('model_type'),'hidden_size':cfg.get('hidden_size',cfg.get('text_config',{}).get('hidden_size')),'layers':cfg.get('num_hidden_layers',cfg.get('text_config',{}).get('num_hidden_layers')),'config_sha256':self.hashed_file(p/'config.json'),'indexes':[{'name':x.name,'sha256':self.hashed_file(x)} for x in indexes],'indexed_tensors':tensor_total,'weight_files':[{'name':x.name,'bytes':x.stat().st_size,'tensor_header_count':len(self.header(x))} for x in weights.values()],'tokenizer':token,'large_weight_values_or_custom_code_loaded':False,'large_weight_payloads_hashed':False}
    def nf4(self):
        literal=CONTRACT['paths']['own_nf4_base'];p=self.local(literal)
        pins=CONTRACT['staging_constants']['NF4_FILES']
        for name,size in pins.items():
            f=p/safe_relative(name)
            if not f.is_file() or f.stat().st_size!=size:raise ValueError('NF4 source-pinned file/bytes changed: '+name)
        cfg=read_json(p/'config.json');text=cfg.get('text_config',{})
        if text.get('num_hidden_layers')!=64 or text.get('hidden_size')!=5120:raise ValueError('NF4 config is not the full64/5120 backbone')
        inv=self.model(literal)
        index=read_json(p/'model.safetensors.index.json')['weight_map'];layers=set()
        for key in index:
            match=re.match(r'model\.language_model\.layers\.(\d+)\.',key)
            if match:layers.add(int(match.group(1)))
        if layers!=set(range(64)):raise ValueError('NF4 index does not cover exactly all64 decoder layers')
        inv.update({'source_pinned_files':len(pins),'source_pinned_total_bytes':sum(pins.values()),'decoder_layers_present':sorted(layers),'native_context_tokens':text.get('max_position_embeddings'),'quantized_weight_values_dequantized':False})
        return inv
    def final_adapter(self):
        p=self.local(CONTRACT['paths']['own_fulltext_adapter']);meta=read_json(p.with_suffix('.json'))
        for key,value in CONTRACT['contracts']['qwen_final_metadata'].items():
            if key not in meta or type(meta[key]) is not type(value) or meta[key]!=value:raise ValueError('Final adapter metadata differs: '+key)
        alignment=qwen_label_alignment(meta.get('labels'));header=self.header(p)
        expected={}
        for name,shape in [('plain.weight',[18,40960]),('plain.bias',[18]),('centered.weight',[18,40960]),('centered.bias',[18])]:expected[name]=shape
        for i in range(8):
            for field in ['weight','bias']:expected[f'norms.{i}.{field}']=[5120]
        lora={k:v for k,v in header.items() if '.lora_A' in k or '.lora_B' in k}
        layers=set()
        for key,item in lora.items():
            match=re.fullmatch(r'backbone\.layers\.(\d+)\.(.+)\.(lora_A|lora_B)',key)
            if not match or int(match.group(1)) not in range(56,64):raise ValueError('Unexpected adapter layer/key: '+key)
            layers.add(int(match.group(1)));shape=item['shape']
            if len(shape)!=2 or shape[0 if match.group(3)=='lora_A' else 1]!=16:raise ValueError('LoRA rank differs: '+key)
            partner=key[:-1]+('B' if key.endswith('A') else 'A')
            if partner not in lora:raise ValueError('LoRA pair missing: '+key)
        if layers!=set(range(56,64)):raise ValueError('Final adapter does not cover every tail8 layer')
        if set(header)!=set(expected)|set(lora):raise ValueError('Unexpected/missing final adapter tensor keys')
        for key,shape in expected.items():expect_shape(header,key,shape)
        # Match each adapter input/output dimension to the original NF4 quant-state metadata.
        base=self.local(CONTRACT['paths']['own_nf4_base']);index=read_json(base/'model.safetensors.index.json')['weight_map'];matched=0
        check_complete_lora_keys(lora,index)
        for key,item in lora.items():
            if not key.endswith('.lora_A'):continue
            weight='model.language_model.'+key[len('backbone.'):].replace('.lora_A','.weight')
            qkey=weight+'.quant_state.bitsandbytes__nf4'
            if qkey not in index:raise ValueError('Adapter linear lacks NF4 quantization-state tensor: '+weight)
            shard=base/safe_relative(index[qkey]);h=self.header(shard)[qkey]
            if h['dtype']!='U8' or math.prod(h['shape'])>65536:raise ValueError('Unbounded/nonbyte NF4 metadata tensor')
            with shard.open('rb') as stream:
                prefix=struct.unpack('<Q',stream.read(8))[0];stream.seek(8+prefix+h['data_offsets'][0]);raw=stream.read(h['data_offsets'][1]-h['data_offsets'][0])
            qstate=json.loads(raw.decode('utf-8'));shape=qstate['shape']
            if item['shape']!=[16,shape[1]] or lora[key[:-1]+'B']['shape']!=[shape[0],16]:raise ValueError('LoRA/NF4 linear dimensions differ: '+weight)
            matched+=1
        return {'path':self.rel(p),'bytes':p.stat().st_size,'sha256':self.hashed_file(p),'metadata_sha256':self.hashed_file(p.with_suffix('.json')),'metadata_contract':CONTRACT['contracts']['qwen_final_metadata'],'labels':meta['labels'],'label_alignment':alignment,'tensors':len(header),'lora_pairs_matched_to_nf4_metadata':matched,'adapter_payload_values_loaded':False,'finite_adapter_values_verified':False,'honest_oof_checkpoint':False}
    def svc(self):
        p=self.local(CONTRACT['paths']['svc_prefit']);meta=read_json(p/'metadata.json');checked_labels(meta['labels'],LABELS)
        if not isinstance(meta.get('views'),list) or len(meta['views'])!=4:raise ValueError('SVC four-view metadata differs')
        width=0;views=[]
        for row in meta['views']:
            name,lo,hi,maximum,weight=row
            if name not in ['char','word','shape','small'] or not 1<=lo<=hi<=6 or weight<=0:raise ValueError('SVC view specification invalid')
            vocabulary=read_json(p/(name+'-vocabulary.json'));idf=np.load(p/(name+'-idf.npy'),allow_pickle=False)
            if not isinstance(vocabulary,list) or len(vocabulary)!=len(set(vocabulary)) or not all(isinstance(x,str) for x in vocabulary) or idf.shape!=(len(vocabulary),) or not np.isfinite(idf).all():raise ValueError('SVC vocabulary/IDF axes invalid')
            width+=len(vocabulary);views.append({'name':name,'features':len(vocabulary),'idf_sha256':self.hashed_file(p/(name+'-idf.npy')),'vocabulary_sha256':self.hashed_file(p/(name+'-vocabulary.json'))})
        state=state_inventory(p/'readout.npz')
        if state.get('classes',{}).get('small_axis_values')!=list(range(18)) or state.get('coef',{}).get('shape')!=[18,width] or state.get('intercept',{}).get('shape')!=[18]:raise ValueError('SVC readout dimension/class axes differ')
        return {'path':self.rel(p),'metadata_sha256':self.hashed_file(p/'metadata.json'),'views':views,'total_features':width,'readout':state,'readout_sha256':self.hashed_file(p/'readout.npz'),'fitting_or_transform_executed':False}
    def prefit(self,key):
        p=self.local(CONTRACT['paths'][key]);files=sorted(x for x in p.rglob('*') if x.is_file());arrays={};json_records={}
        if not files:raise ValueError('Empty prefit bank: '+key)
        active={}
        if key=='complement_style':active=style_bank_contract(p)
        elif key=='legacy_prefit':
            booster=p/'stylometry/booster.txt'
            if not booster.is_file() or booster.stat().st_size==0:raise ValueError('Required stylometry booster missing')
            with booster.open() as f:booster_prefix=f.read(16384)
            if not booster_prefix.startswith('tree') or 'num_class=18' not in booster_prefix:raise ValueError('Stylometry booster multiclass schema differs')
            paths=CONTRACT['explicit_paths']['paths']
            hidden={k:read_json(self.local(paths[name]['path'])/'config.json')['hidden_size'] for k,name in [('debv3l','deberta_base'),('electl','electra_base'),('desk','desklib_base')]}
            widths={'q3_8b':8192,'q25_7b':7168,'q3_4b':5120,'glm9b':8192,**{k:2*v for k,v in hidden.items()}}
            required=[]
            for name in ['q3_8b','q25_7b','q3_4b','glm9b','debv3l']:
                rel=f'frozen_{name}/twin.npz';linear_state_contract(p/rel,widths[name]);required.append(rel)
            for name in ['debv3l','electl','desk']:
                rel=f'frozen_{name}/plain.npz';linear_state_contract(p/rel,widths[name]);required.append(rel)
            for name,head,dim in [('q3_8b','gauss_q3_8b_L22',4096),('glm9b','gauss_glm9b_L8',4096),('q3_4b','gauss_q3_4b_L4',2560),('debv3l','gauss_debv3l_L24',hidden['debv3l'])]:
                rel=f'frozen_{name}/{head}.npz';gaussian_state_contract(p/rel,dim);required.extend([rel,str(Path(rel).with_suffix('.json'))])
            for name in ['q3_4b','electl','desk']:
                rel=f'style_token_masks/{name}.npy';mask=np.load(p/rel,allow_pickle=False)
                if mask.dtype!=np.bool_ or mask.ndim!=1 or len(mask)==0:raise ValueError('Required style tokenizer mask missing/invalid')
                required.append(rel)
            active={'required_files':required+['stylometry/booster.txt'],'all_active_readouts_present_and_axes_verified':True}
        for f in files:
            if f.suffix in ['.npz','.npy']:arrays[str(f.relative_to(p))]={'sha256':self.hashed_file(f),'members':state_inventory(f)}
            elif f.suffix=='.json':
                doc=read_json(f);json_records[str(f.relative_to(p))]={'sha256':self.hashed_file(f),'keys':list(doc)[:40] if isinstance(doc,dict) else None}
        if not arrays:raise ValueError('No safe numeric prefit states')
        return {'path':self.rel(p),'file_count':len(files),'active_schema':active,'safe_numeric_states':arrays,'json_metadata':json_records,'estimators_or_upstream_code_loaded':False}
    def night24(self):
        p=self.local(CONTRACT['paths']['night24_bank']);mp=self.local(CONTRACT['paths']['night24_metadata']);meta=read_json(mp)
        if meta.get('full_fit') is not True or meta.get('fold_adapter') is not False or meta.get('adapter_training') is not False or meta.get('raw_synthetic_data_in_export') is not False:raise ValueError('Night24 fit/export metadata differs')
        rec=meta['banks']['synthetic4404'];checked_labels(rec['labels'],SORTED_LABELS)
        if rec['official_rows']!=2699 or rec['synthetic_rows']!=4404 or rec['layers']!=[2,4,6,8,12,16,20,24] or self.hashed_file(p)!=rec['readout_sha256']:raise ValueError('Night24 bank actualhash/row/layer contract differs')
        if rec['adapter_sha256']!='a6aff409ae2df897717875373d038635fe5012fbf93557c90aea3457fda0e7f8':raise ValueError('Night24 adapter identity differs')
        return {'path':self.rel(p),'sha256':self.hashed_file(p),'metadata_sha256':self.hashed_file(mp),'official_rows':2699,'synthetic_rows':4404,'joblib_deserialized':False,'internal_bank_objects_inspected':False,'attached_readouts':self.export_text(p.parent/'readouts.py','original Night24 source; inlined worker code is reviewed separately')}
    def wheels(self):
        records=[]
        for literal in CONTRACT['paths']['qwen_wheels']:
            p=self.local(literal)
            with zipfile.ZipFile(p) as z:
                names=z.namelist();candidates=[n for n in names if n.endswith('.dist-info/METADATA')]
                wheel=[n for n in names if n.endswith('.dist-info/WHEEL')]
                if len(candidates)!=1 or len(wheel)!=1:raise ValueError('Wheel metadata structure differs')
                if any(z.getinfo(n).file_size>1024*1024 for n in candidates+wheel):raise ValueError('Wheel metadata oversized')
                meta=z.read(candidates[0]).decode();tags=z.read(wheel[0]).decode()
                fields={line.split(': ',1)[0]:line.split(': ',1)[1] for line in meta.splitlines() if ': ' in line and line.split(': ',1)[0] in ['Name','Version','Requires-Python']}
                records.append({'path':self.rel(p),'bytes':p.stat().st_size,'sha256':self.hashed_file(p),'metadata':fields,'wheel_tags':[l[5:] for l in tags.splitlines() if l.startswith('Tag: ')]})
        return {'offline_wheels':records,'installed_or_imported':False,'dependency_runtime_compatibility_verified':False}
    def ultrabert(self):
        records=[];pins=CONTRACT['staging_constants']['ULTRA_FOLD_FILES']
        folders=CONTRACT['explicit_paths']['lists']['ultrabert_folds']['paths']
        if len(folders)!=5:raise ValueError('Expected exactlyfive UltraBERT folds')
        for i,literal in enumerate(folders):
            p=self.local(literal)
            for name,size in pins[str(i)].items():
                f=p/safe_relative(name)
                if not f.is_file() or f.stat().st_size!=size:raise ValueError('UltraBERT pinned file/bytes differ: '+literal+'/'+name)
            cfg=read_json(p/'config.json')
            if cfg.get('model_type')!='ultrabert' or cfg.get('num_labels',len(cfg.get('id2label',{})))!=18:raise ValueError('UltraBERT config/classes differ')
            if cfg.get('id2label'):checked_labels([cfg['id2label'][str(n)] for n in range(18)])
            code=[self.export_text(p/n,'attached UltraBERT custom source; never imported') for n in ['configuration_ultrabert.py','modeling_ultrabert.py']]
            records.append({'model':self.model(literal),'private_code_exports':code,'custom_code_executed':False})
        return {'folds':records}
    def legacy_models(self):
        models={};paths=CONTRACT['explicit_paths']['paths']
        for key in ['qwen3_8b_base','qwen3_14b_base','qwen2_7b_base','qwen3_4b_base','glm_base','deberta_base','electra_base','desklib_base','q35_base']:
            literal=paths[key]['path'];models[key]=self.model(literal)
        for key in ['qwen3_14b','qwen25_14b']:
            literal=CONTRACT['paths'][key];models[key]=self.model(literal)
        return models
    def adapter_inventory(self,literal,labels_order=None):
        p=self.local(literal);files=sorted(x for x in p.rglob('*') if x.is_file());summaries={}
        for f in files:
            if f.suffix=='.safetensors':summaries[str(f.relative_to(p))]={'bytes':f.stat().st_size,'sha256':self.hashed_file(f),'tensor_count':len(self.header(f)),'tensor_shapes':self.header(f) if f.stat().st_size<256*1024*1024 else None}
            elif f.name in ['meta.json','adapter_config.json','COMPLETE.json']:
                doc=read_json(f)
                if 'labels' in doc:checked_labels(doc['labels'],labels_order)
                summaries[str(f.relative_to(p))]={'sha256':self.hashed_file(f),'metadata':doc}
            elif f.name=='readouts.npz':summaries[str(f.relative_to(p))]={'sha256':self.hashed_file(f),'members':state_inventory(f)}
        if not any(x.endswith('.safetensors') for x in summaries):raise ValueError('Adapter/checkpoint safetensors missing')
        return {'path':self.rel(p),'files':summaries,'Torch_or_pickle_deserialized':False}
    def active_adapters(self):
        p=self.local(CONTRACT['paths']['new_two_stage_root']);result={}
        for name in ['lora_qwen3_14b','lora_qwen25_14b','deberta_v3_large']:
            result[name]=self.adapter_inventory('/kaggle/input/'+str((p/name).relative_to(self.root)),LABELS)
        paths=CONTRACT['explicit_paths']['paths']
        for name in ['lora24_public','lora24_seed7','lora24_claude','lora20_q25_instyle']:
            result[name]=self.adapter_inventory(paths[name]['path'],SORTED_LABELS)
        return result
    def atlas(self):
        p=self.local(CONTRACT['explicit_paths']['paths']['atlas_manifest']['path']);cfg=read_json(p)
        if not isinstance(cfg.get('training_config'),dict) or cfg['training_config'].get('max_length',0)<=0:raise ValueError('ATLAS training/token contract missing')
        adapter=p.parent/'adapters.safetensors';header=self.header(adapter)
        exports=[]
        for package in ['ntl_atlas','ntl_rlcd']:
            for f in sorted((p.parent/package).rglob('*.py')):exports.append(self.export_text(f,'attached ATLAS code; active inlined implementation reviewed separately'))
        return {'path':self.rel(p),'metadata_sha256':self.hashed_file(p),'adapter_sha256':self.hashed_file(adapter),'adapter_tensor_count':len(header),'private_code_exports':exports,'custom_code_or_weights_loaded':False}
    def manifest_audits(self):
        """Verify explicit relative-file digest records; never execute serializer formats."""
        seen=set();results=[]
        candidates=[]
        for rec in CONTRACT['required_input_checks']:
            p=self.local(rec['path'])
            if p.is_file() and p.suffix=='.json' and ('manifest' in p.name.lower() or p.name=='COMPLETE.json'):candidates.append(p)
        for literal in [CONTRACT['paths']['own_nf4_base'],CONTRACT['paths']['svc_prefit'],CONTRACT['paths']['legacy_prefit'],CONTRACT['paths']['complement_style']]:
            p=self.local(literal)
            if p.is_dir():candidates.extend(x for x in p.glob('*.json') if 'manifest' in x.name.lower())
        for p in candidates:
            if p in seen:continue
            seen.add(p);doc=read_json(p);checked=[];unresolved=[]
            def examine(record,prefix=None):
                if not isinstance(record,dict):return
                digest=record.get('sha256');name=record.get('relative_path',record.get('file',record.get('name',record.get('path',prefix))))
                if isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest) and isinstance(name,str):
                    try:target=p.parent/safe_relative(name)
                    except ValueError:unresolved.append({'record':name,'reason':'nonrelative manifest file name'});return
                    if target.is_file():
                        if self.hashed_file(target)!=digest:raise ValueError('Manifest actual file digest mismatch: '+self.rel(target))
                        if 'bytes' in record and target.stat().st_size!=record['bytes']:raise ValueError('Manifest actual file size mismatch')
                        checked.append(self.rel(target))
                    else:unresolved.append({'record':name,'reason':'declared relative path not present beside manifest'})
                for k,v in record.items():
                    if k=='files' and isinstance(v,dict):
                        for name,item in v.items():
                            if isinstance(item,str) and re.fullmatch('[0-9a-f]{64}',item):
                                target=p.parent/safe_relative(name)
                                if not target.is_file():unresolved.append({'record':name,'reason':'digest-map target unavailable'})
                                elif self.hashed_file(target)!=item:raise ValueError('Manifest digest-map mismatch: '+self.rel(target))
                                else:checked.append(self.rel(target))
                            elif isinstance(item,dict):examine(item,name)
                    elif isinstance(v,dict):examine(v)
                    elif isinstance(v,list):
                        for item in v:
                            if isinstance(item,dict):examine(item)
            examine(doc)
            results.append({'path':self.rel(p),'sha256':self.hashed_file(p),'declared_relative_file_hashes_verified':sorted(set(checked)),'unresolved_hash_records':unresolved})
        if any(r['unresolved_hash_records'] for r in results):self.warnings.append('Some historical manifest hash records use unresolved/nonrelative paths. They are explicitly listed, not treated as verified. Exact runtime byte pins and active schema checks remain separate.')
        return results
    def resources(self):
        packages={}
        for name in ['numpy','pandas','pyarrow','torch','transformers','accelerate','bitsandbytes','safetensors','tokenizers','scikit-learn','scipy','joblib','lightgbm']:
            try:packages[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:packages[name]=None
        mem={}
        p=Path('/proc/meminfo')
        if p.is_file():
            for line in p.read_text().splitlines():
                if line.split(':')[0] in ['MemTotal','MemAvailable','SwapTotal','SwapFree']:mem[line.split(':')[0]]=line.split(':')[1].strip()
        disks={}
        for literal in ['/tmp','/kaggle/temp','/kaggle/working']:
            p=Path(literal)
            if p.exists():
                d=shutil.disk_usage(p);disks[literal]={'total_bytes':d.total,'free_bytes':d.free}
        return {'python':sys.version,'platform':platform.platform(),'cpu_count':os.cpu_count(),'packages_by_distribution_metadata_only':packages,'memory':mem,'disk':disks,'nf4_expected_staging_bytes':sum(CONTRACT['staging_constants']['NF4_FILES'].values()),'gpu_probe_run':False,'model_imports':False,'internal_runtime_budget_enforces_deadlines':False,'runtime_feasibility_verified':False}
    def alternative_hub_inventory(self):
        expected=self.local('/kaggle/input/ntl-olmo2-native24-final-inference-assets');alternatives=[]
        if not expected.exists():
            for p in self.root.glob('**/ntl627-own-fulltext'):
                if p.is_dir():alternatives.append(self.rel(p))
        return {'expected_literal_root':str(expected),'literal_root_present':expected.is_dir(),'metadata_only_alternative_paths':alternatives[:30],'aliases_created':False,'paths_remapped':False}
    def execute(self):
        started=time.monotonic();self.check('official_train_contract',self.official)
        for i,record in enumerate(CONTRACT['required_input_checks']):self.check(f'mandatory_{i+1:03d}',lambda record=record:self.required(record))
        self.check('literal_hub_resolution',lambda:self.alternative_hub_inventory())
        self.check('fulltext_nf4_base',self.nf4)
        self.check('fulltext_final_adapter',self.final_adapter)
        self.check('reference_svc_prefit',self.svc)
        self.check('legacy_prefit_numeric_states',lambda:self.prefit('legacy_prefit'))
        self.check('complement_style_numeric_states',lambda:self.prefit('complement_style'))
        self.check('night24_bank_hash_and_schema',self.night24)
        self.check('offline_qwen_wheels',self.wheels)
        self.check('ultrabert_five_folds',self.ultrabert)
        self.check('legacy_and_reference_models',self.legacy_models)
        self.check('active_adapters',self.active_adapters)
        self.check('atlas_bundle',self.atlas)
        self.check('published_manifest_actual_hashes',self.manifest_audits)
        self.check('cpu_environment_and_resources',self.resources)
        failed=[c for c in self.checks if c['status']!='PASS']
        self.out.mkdir(parents=True,exist_ok=True)
        private={'scope':'PRIVATE static source/config text for independent review; no imports or execution','files':self.code}
        (self.out/'PRIVATE_CODE_EXPORT.json').write_text(json.dumps(private,indent=2)+'\n')
        mandatory=[x for x in self.checks if x['name'].startswith('mandatory_')]
        report={'stage':11,'source_version':SOURCE_VERSION,'source_capture_json_sha256':CONTRACT['capture_json_sha256'],'source_manifest_sha256':CONTRACT['source_manifest_sha256'],'audit_script_sha256':sha(Path(__file__)),'status':'PASS_CPU_INPUT_CONTRACT_ONLY' if not failed else 'BLOCKED_INPUT_CONTRACT','passed':len(self.checks)-len(failed),'total':len(self.checks),'mandatory_source_checks_passed':sum(c['status']=='PASS' for c in mandatory),'mandatory_source_checks_total':len(mandatory),'checks':self.checks,'details':self.details,'warnings':self.warnings,'elapsed_seconds':time.monotonic()-started,'hashed_file_count':len(self.hashed),'private_code_export_files':len(self.code),'private_code_export_sha256':sha(self.out/'PRIVATE_CODE_EXPORT.json'),'upstream_or_custom_code_executed':False,'torch_transformers_imported':False,'joblib_pickle_torch_deserialized':False,'training_or_inference_started':False,'test_rows_labels_or_predictions_exported':False,'runtime_feasibility_verified':False,'limitations':['CPU inventory/schema/hash checks are not full model inference or a score.','Safetensors headers verify extents/layout only; tensor payload finiteness and exact runtime model construction remain unverified.','Joblib/Torch payload internals are not deserialized or certified by this audit.','Reported distribution versions do not establish runtime API compatibility.','Source preview is exact-public-format only; no hidden-runtime evidence.','Final adapted checkpoints saw all2699 official labels; training-row scores are not honest OOF.','Some author historical manifest records may require separate mapping review; unresolved records are never counted as verified.','RuntimeBudget in frozen source is untimed; a CPU pass does not authorize a hidden-test runtime claim.']}
        (self.out/'INPUT_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('STAGE11_CPU_AUDIT_SUMMARY',json.dumps({k:report[k] for k in ['status','passed','total','mandatory_source_checks_passed','mandatory_source_checks_total','training_or_inference_started']}),flush=True)
        return report

def self_test():
    checks={}
    def accepts(name,fn):
        try:fn();checks[name]=True
        except Exception:checks[name]=False
    def rejects(name,fn):
        try:fn();checks[name]=False
        except Exception:checks[name]=True
    for value in ['/absolute','../outside','folder/../outside']:
        rejects('unsafe_path_'+str(len(checks)),lambda value=value:safe_relative(value))
    rejects('wrong_label_set',lambda:checked_labels(['wrong']*18))
    rejects('wrong_label_order',lambda:checked_labels(SORTED_LABELS,LABELS))
    accepts('qwen_canonical_native_order',lambda:qwen_label_alignment(list(LABELS)))
    accepts('qwen_sorted_native_order',lambda:qwen_label_alignment(list(SORTED_LABELS)))
    accepts('qwen_reversed_native_order',lambda:qwen_label_alignment(list(reversed(LABELS))))
    rejects('qwen_duplicate_native_label',lambda:qwen_label_alignment(LABELS[:-1]+[LABELS[0]]))
    rejects('qwen_missing_native_label',lambda:qwen_label_alignment(LABELS[:-1]))
    rejects('qwen_unknown_native_label',lambda:qwen_label_alignment(LABELS[:-1]+['unknown']))
    rejects('qwen_nonstring_native_label',lambda:qwen_label_alignment(LABELS[:-1]+[None]))
    def permutation_roundtrip():
        native=list(reversed(SORTED_LABELS));alignment=qwen_label_alignment(native)
        native_columns=np.asarray([[LABELS.index(label) for label in native]]*2)
        if not np.array_equal(native_columns[:,alignment['source_column_permutation_to_canonical']],np.asarray([list(range(18))]*2)):
            raise ValueError('Label-name column alignment roundtrip differs')
    accepts('qwen_label_name_numeric_column_roundtrip',permutation_roundtrip)
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);a=Audit(root,root/'out')
        rejects('literal_missing_path',lambda:a.required({'path':'/kaggle/input/missing','source':'fixture','kind':'file'}))
        rejects('literal_outside_mount',lambda:a.local('/somewhere/else'))
        rejects('literal_path_traversal',lambda:a.local('/kaggle/input/../escape'))
        f=root/'fixture';f.write_bytes(b'123')
        accepts('literal_exact_path_and_bytes',lambda:a.required({'path':'/kaggle/input/fixture','source':'fixture','kind':'file','expected_bytes':3}))
        rejects('wrong_pinned_bytes',lambda:a.required({'path':'/kaggle/input/fixture','source':'fixture','kind':'file','expected_bytes':4}))
        (root/'empty').write_bytes(b'')
        rejects('zero_byte_file',lambda:a.required({'path':'/kaggle/input/empty','source':'fixture','kind':'file'}))
        rejects('wrong_file_kind',lambda:a.required({'path':'/kaggle/input/fixture','source':'fixture','kind':'directory'}))
        np.savez(root/'object.npz',bad=np.asarray([{'no':'pickle'}],dtype=object))
        rejects('npz_object_payload',lambda:state_inventory(root/'object.npz'))
        np.savez(root/'nan.npz',coef=np.asarray([[np.nan]]))
        rejects('npz_nonfinite_payload',lambda:state_inventory(root/'nan.npz'))
        np.save(root/'nan.npy',np.asarray([np.inf]),allow_pickle=False)
        rejects('npy_nonfinite_payload',lambda:state_inventory(root/'nan.npy'))
        np.savez(root/'safe.npz',classes=np.arange(18),coef=np.ones((18,2)),intercept=np.zeros(18))
        accepts('safe_numeric_npz',lambda:state_inventory(root/'safe.npz'))
        bad=root/'bad.safetensors';bad.write_bytes(struct.pack('<Q',100)+b'{}')
        rejects('truncated_safetensors_header',lambda:safetensor_header(bad))
        def fixture_header(name,header,payload):
            raw=json.dumps(header).encode();p=root/name;p.write_bytes(struct.pack('<Q',len(raw))+raw+payload);return p
        wrong=fixture_header('span.safetensors',{'w':{'dtype':'F32','shape':[2],'data_offsets':[0,4]}},b'1234')
        rejects('safetensor_shape_span_mismatch',lambda:safetensor_header(wrong))
        overlap=fixture_header('overlap.safetensors',{'a':{'dtype':'U8','shape':[2],'data_offsets':[0,2]},'b':{'dtype':'U8','shape':[2],'data_offsets':[1,3]}},b'123')
        rejects('safetensor_overlapping_payloads',lambda:safetensor_header(overlap))
        unknown=fixture_header('dtype.safetensors',{'w':{'dtype':'INVALID','shape':[1],'data_offsets':[0,1]}},b'1')
        rejects('safetensor_unknown_dtype',lambda:safetensor_header(unknown))
        rejects('adapter_shape_mismatch',lambda:expect_shape({'w':{'shape':[18,1]}},'w',[18,2]))
        rejects('adapter_missing_tensor',lambda:expect_shape({},'w',[18,2]))
        accepts('adapter_shape_match',lambda:expect_shape({'w':{'shape':[18,2]}},'w',[18,2]))
        (root/'fake_style').mkdir();np.savez(root/'fake_style/plain.npz',coef=np.ones((18,2)),intercept=np.zeros(18))
        rejects('arbitrary_numeric_file_cannot_satisfy_style_bank',lambda:style_bank_contract(root/'fake_style'))
        rejects('wrong_linear_state_width',lambda:linear_state_contract(root/'safe.npz',3))
        index={f'model.language_model.layers.{i}.{part}.weight.quant_state.bitsandbytes__nf4':'fixture' for i in range(56,64) for part in ['mlp.gate_proj','mlp.down_proj']}
        complete=required_tail8_lora_keys(index)
        accepts('complete_tail8_lora_keyset',lambda:check_complete_lora_keys(complete,index))
        reduced=complete-{f'backbone.layers.56.mlp.gate_proj.lora_{x}' for x in ['A','B']}
        rejects('missing_entire_lora_pair_despite_layer_coverage',lambda:check_complete_lora_keys(reduced,index))
        hub=root/'unrelated'/'ntl627-own-fulltext';hub.mkdir(parents=True)
        accepts('alternative_paths_inventory_only',lambda: a.alternative_hub_inventory()['metadata_only_alternative_paths']==['unrelated/ntl627-own-fulltext'] or (_ for _ in ()).throw(ValueError()))
        rejects('alternative_does_not_satisfy_literal_mount',lambda:a.required({'path':'/kaggle/input/ntl-olmo2-native24-final-inference-assets/ntl627-own-fulltext','kind':'directory','source':'fixture'}))
    result={'status':'PASS' if all(checks.values()) else 'FAIL','passed':sum(checks.values()),'total':len(checks),'checks':checks,'source_or_model_executed':False}
    if not all(checks.values()):raise AssertionError(json.dumps(result))
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='/kaggle/input');parser.add_argument('--out',default='/kaggle/working/stage11_preflight');parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2))
    else:
        result=Audit(args.root,args.out).execute()
        print('STAGE11_INPUT_AUDIT_JSON_BEGIN');print(json.dumps(result,indent=2,allow_nan=False));print('STAGE11_INPUT_AUDIT_JSON_END')
