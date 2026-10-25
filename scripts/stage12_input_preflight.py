"""CPU-only source-selector/input audit for Dhruv nlm-v27 V1 362013743.
Uses stdlib, NumPy and pandas. Never imports Torch, Transformers, custom model
code or a pickle/joblib loader. Torch ZIP metadata may be inspected statically
with pickletools; reducers are never imported or called.
"""
from __future__ import annotations
import argparse, ast, collections, csv, glob, hashlib, importlib.metadata, io, json, math, os, pickletools, platform, re, shutil, struct, sys, tempfile, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
SOURCE_VERSION=362013743
COPIED_INPUT_IDS=[{'sourceType': 'competition', 'sourceId': 170484}, {'sourceType': 'datasetVersion', 'sourceId': 20736096}, {'sourceType': 'datasetVersion', 'sourceId': 20819121}, {'sourceType': 'datasetVersion', 'sourceId': 20831302}, {'sourceType': 'datasetVersion', 'sourceId': 20847146}, {'sourceType': 'datasetVersion', 'sourceId': 20858502}, {'sourceType': 'datasetVersion', 'sourceId': 20934027}, {'sourceType': 'datasetVersion', 'sourceId': 21008519}, {'sourceType': 'datasetVersion', 'sourceId': 21011978}, {'sourceType': 'datasetVersion', 'sourceId': 21062729}, {'sourceType': 'datasetVersion', 'sourceId': 21097171}, {'sourceType': 'datasetVersion', 'sourceId': 21116786}, {'sourceType': 'datasetVersion', 'sourceId': 21119693}, {'sourceType': 'datasetVersion', 'sourceId': 21145399}, {'sourceType': 'datasetVersion', 'sourceId': 21203279}, {'sourceType': 'kernelVersion', 'sourceId': 358683059}, {'sourceType': 'kernelVersion', 'sourceId': 358745836}, {'sourceType': 'kernelVersion', 'sourceId': 358758062}, {'sourceType': 'kernelVersion', 'sourceId': 358770408}, {'sourceType': 'kernelVersion', 'sourceId': 358911769}, {'sourceType': 'kernelVersion', 'sourceId': 358928013}, {'sourceType': 'kernelVersion', 'sourceId': 358948428}, {'sourceType': 'kernelVersion', 'sourceId': 359652568}, {'sourceType': 'modelInstanceVersion', 'sourceId': 104433}, {'sourceType': 'modelInstanceVersion', 'sourceId': 166254}, {'sourceType': 'modelInstanceVersion', 'sourceId': 166265}, {'sourceType': 'modelInstanceVersion', 'sourceId': 363135}, {'sourceType': 'modelInstanceVersion', 'sourceId': 363139}, {'sourceType': 'modelInstanceVersion', 'sourceId': 363140}, {'sourceType': 'modelInstanceVersion', 'sourceId': 363149}, {'sourceType': 'modelInstanceVersion', 'sourceId': 363153}]
COPIED_NOTEBOOK_VERSION=362254770
CONTRACT={'source_version': 362013743, 'capture_json_sha256': '0a65325bd0d001195522cf4a55d1a907465d285403fa180f46760ffd995cda6a', 'main_selectors': [{'name': 'B8', 'glob': '/kaggle/input/**/config.json', 'substring_filters_lower': ['/qwen-3/', '/8b-base/'], 'ordering': 'sorted paths; first matching all substrings', 'kind': 'model_directory', 'required': True}, {'name': 'B14', 'glob': '/kaggle/input/**/config.json', 'substring_filters_lower': ['/qwen-3/', '/14b-base/'], 'ordering': 'sorted paths; first matching all substrings', 'kind': 'model_directory', 'required': True}, {'name': 'B25', 'glob': '/kaggle/input/**/config.json', 'substring_filters_lower': ['/qwen2.5/', '/7b/'], 'ordering': 'sorted paths; first matching all substrings', 'kind': 'model_directory', 'required': True}, {'name': 'I8', 'glob': '/kaggle/input/**/config.json', 'substring_filters_lower': ['/qwen-3/', '/8b/'], 'ordering': 'sorted paths; first matching all substrings', 'kind': 'model_directory', 'required': True}, {'name': 'I14', 'glob': '/kaggle/input/**/config.json', 'substring_filters_lower': ['/qwen-3/', '/14b/'], 'ordering': 'sorted paths; first matching all substrings', 'kind': 'model_directory', 'required': True}, {'name': 'BL', 'glob': '/kaggle/input/**/config.json', 'substring_filters_lower': ['/llama-3.1/', '/8b/'], 'ordering': 'sorted paths; first matching all substrings', 'kind': 'model_directory', 'required': True}, {'name': 'B25_14', 'glob': '/kaggle/input/**/config.json', 'substring_filters_lower': ['/qwen2.5/', '/14b/'], 'ordering': 'sorted paths; first matching all substrings', 'kind': 'model_directory', 'required': True}, {'name': 'ts_q3b8_f0', 'family': 'A8', 'glob': '/kaggle/input/**/ts_q3b8_f0/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': False}, {'name': 'ts_q3b8_f1', 'family': 'A8', 'glob': '/kaggle/input/**/ts_q3b8_f1/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': False}, {'name': 'ts_q3b8_f2', 'family': 'A8', 'glob': '/kaggle/input/**/ts_q3b8_f2/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': False}, {'name': 'ts_q3b8_f3', 'family': 'A8', 'glob': '/kaggle/input/**/ts_q3b8_f3/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': False}, {'name': 'ts_q3b8_f4', 'family': 'A8', 'glob': '/kaggle/input/**/ts_q3b8_f4/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': False}, {'name': 'ts_q3b8', 'family': 'A8', 'glob': '/kaggle/input/**/ts_q3b8/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': False}, {'name': 'ts_q3b14_f0', 'family': 'A14', 'glob': '/kaggle/input/**/ts_q3b14_f0/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': False}, {'name': 'ts_q3b14_f2', 'family': 'A14', 'glob': '/kaggle/input/**/ts_q3b14_f2/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': False}, {'name': 'ts_q3b14_f4', 'family': 'A14', 'glob': '/kaggle/input/**/ts_q3b14_f4/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': False}, {'name': 'ts_q3b14', 'family': 'A14', 'glob': '/kaggle/input/**/ts_q3b14/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': False}, {'name': 'ts_q25_7b', 'family': 'A25', 'glob': '/kaggle/input/**/ts_q25_7b/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': False}, {'name': 'ts_llama8_f0', 'family': 'AL', 'glob': '/kaggle/input/**/ts_llama8_f0/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': False}, {'name': 'ts_llama8_f2', 'family': 'AL', 'glob': '/kaggle/input/**/ts_llama8_f2/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': False}, {'name': 'ts_llama8_f4', 'family': 'AL', 'glob': '/kaggle/input/**/ts_llama8_f4/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': False}, {'name': 'ts_llama8', 'family': 'AL', 'glob': '/kaggle/input/**/ts_llama8/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': False}, {'name': 'ts8_s7_f0', 'family': 'A8S7', 'glob': '/kaggle/input/**/ts8_s7_f0/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': False}, {'name': 'ts8_s7_f2', 'family': 'A8S7', 'glob': '/kaggle/input/**/ts8_s7_f2/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': False}, {'name': 'ts8_s7_f4', 'family': 'A8S7', 'glob': '/kaggle/input/**/ts8_s7_f4/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': False}, {'name': 'ts8_s7', 'family': 'A8S7', 'glob': '/kaggle/input/**/ts8_s7/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': False}, {'name': 'q8v5_f0', 'family': 'AQ8V5', 'glob': '/kaggle/input/**/q8v5_f0/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': True}, {'name': 'q8v5_f1', 'family': 'AQ8V5', 'glob': '/kaggle/input/**/q8v5_f1/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': True}, {'name': 'q8v5_f2', 'family': 'AQ8V5', 'glob': '/kaggle/input/**/q8v5_f2/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': True}, {'name': 'q8v5_f3', 'family': 'AQ8V5', 'glob': '/kaggle/input/**/q8v5_f3/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': True}, {'name': 'q8v5_f4', 'family': 'AQ8V5', 'glob': '/kaggle/input/**/q8v5_f4/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': True}, {'name': 'q8v5', 'family': 'AQ8V5', 'glob': '/kaggle/input/**/q8v5/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': True}, {'name': 'q14v5_f0', 'family': 'AQ14V5', 'glob': '/kaggle/input/**/q14v5_f0/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': True}, {'name': 'q14v5_f1', 'family': 'AQ14V5', 'glob': '/kaggle/input/**/q14v5_f1/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': True}, {'name': 'q14v5_f2', 'family': 'AQ14V5', 'glob': '/kaggle/input/**/q14v5_f2/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': True}, {'name': 'q14v5_f3', 'family': 'AQ14V5', 'glob': '/kaggle/input/**/q14v5_f3/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': True}, {'name': 'q14v5_f4', 'family': 'AQ14V5', 'glob': '/kaggle/input/**/q14v5_f4/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': True}, {'name': 'q14v5', 'family': 'AQ14V5', 'glob': '/kaggle/input/**/q14v5/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': True}, {'name': 'll8v5_f0', 'family': 'ALL8V5', 'glob': '/kaggle/input/**/ll8v5_f0/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': True}, {'name': 'll8v5_f1', 'family': 'ALL8V5', 'glob': '/kaggle/input/**/ll8v5_f1/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': True}, {'name': 'll8v5_f2', 'family': 'ALL8V5', 'glob': '/kaggle/input/**/ll8v5_f2/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': True}, {'name': 'll8v5_f3', 'family': 'ALL8V5', 'glob': '/kaggle/input/**/ll8v5_f3/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': True}, {'name': 'll8v5_f4', 'family': 'ALL8V5', 'glob': '/kaggle/input/**/ll8v5_f4/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': True}, {'name': 'll8v5', 'family': 'ALL8V5', 'glob': '/kaggle/input/**/ll8v5/heads.pt', 'ordering': 'glob-native first, UNSORTED', 'kind': 'main_adapter_directory_from_head_parent', 'required': True, 'active_main_family': True}, {'name': 'lora8b_v1', 'glob': '/kaggle/input/**/lora8b_v1/meta.json', 'ordering': 'glob-native first, UNSORTED', 'kind': 'metadata_parent', 'required': True, 'metadata_cfg_required': True}, {'name': 'lora14b_v1', 'glob': '/kaggle/input/**/lora14b_v1/meta.json', 'ordering': 'glob-native first, UNSORTED', 'kind': 'metadata_parent', 'required': True, 'metadata_cfg_required': True}, {'name': 'lora_qwen3_14b', 'glob': '/kaggle/input/**/lora_qwen3_14b/meta.json', 'ordering': 'glob-native first, UNSORTED', 'kind': 'metadata_parent', 'required': True, 'metadata_cfg_required': True}, {'name': 'lora_qwen25_14b', 'glob': '/kaggle/input/**/lora_qwen25_14b/meta.json', 'ordering': 'glob-native first, UNSORTED', 'kind': 'metadata_parent', 'required': True, 'metadata_cfg_required': True}, {'name': 'HEADS', 'glob': '/kaggle/input/**/heads/q3_8bx.npz', 'ordering': 'glob-native first, UNSORTED', 'kind': 'file_or_parent', 'required': True}, {'name': 'EXTRA', 'glob': '/kaggle/input/**/nlm_matched_clean.parquet', 'ordering': 'glob-native first, UNSORTED', 'kind': 'file_or_parent', 'required': True}, {'name': 'POOL', 'glob': '/kaggle/input/**/nlm_pool_v5.parquet', 'ordering': 'glob-native first, UNSORTED', 'kind': 'file_or_parent', 'required': True}, {'name': 'ATLAS', 'glob': '/kaggle/input/**/atlas_bundle.json', 'ordering': 'glob-native first, UNSORTED', 'kind': 'file_or_parent', 'required': True}, {'name': 'DEB', 'glob': '/kaggle/input/**/deberta_v3_large/meta.json', 'ordering': 'glob-native first, UNSORTED', 'kind': 'file_or_parent', 'required': True}], 'main_adapter_families': {'A8': ['ts_q3b8_f0', 'ts_q3b8_f1', 'ts_q3b8_f2', 'ts_q3b8_f3', 'ts_q3b8_f4', 'ts_q3b8'], 'A14': ['ts_q3b14_f0', 'ts_q3b14_f2', 'ts_q3b14_f4', 'ts_q3b14'], 'A25': ['ts_q25_7b'], 'AL': ['ts_llama8_f0', 'ts_llama8_f2', 'ts_llama8_f4', 'ts_llama8'], 'A8S7': ['ts8_s7_f0', 'ts8_s7_f2', 'ts8_s7_f4', 'ts8_s7'], 'AQ8V5': ['q8v5_f0', 'q8v5_f1', 'q8v5_f2', 'q8v5_f3', 'q8v5_f4', 'q8v5'], 'AQ14V5': ['q14v5_f0', 'q14v5_f1', 'q14v5_f2', 'q14v5_f3', 'q14v5_f4', 'q14v5'], 'ALL8V5': ['ll8v5_f0', 'll8v5_f1', 'll8v5_f2', 'll8v5_f3', 'll8v5_f4', 'll8v5']}, 'complement_selectors': [{'name': 'glm9b', 'glob': '/kaggle/input/**/config.json', 'model_type': 'glm4', 'hidden_size': 4096, 'ordering': 'sorted paths, first matching config json; no source weight-existence filter', 'required': True}, {'name': 'q3_8b', 'glob': '/kaggle/input/**/config.json', 'model_type': 'qwen3', 'hidden_size': 4096, 'ordering': 'sorted paths, first matching config json; no source weight-existence filter', 'required': True}, {'name': 'q25_7b', 'glob': '/kaggle/input/**/config.json', 'model_type': 'qwen2', 'hidden_size': 3584, 'ordering': 'sorted paths, first matching config json; no source weight-existence filter', 'required': True}, {'name': 'q3_4b', 'glob': '/kaggle/input/**/config.json', 'model_type': 'qwen3', 'hidden_size': 2560, 'ordering': 'sorted paths, first matching config json; no source weight-existence filter', 'required': True}, {'name': 'debv3l', 'glob': '/kaggle/input/**/deberta-v3-large/config.json', 'directory_name': 'deberta-v3-large', 'ordering': 'sorted paths, first parseable config; source does not check model_type when directory_name supplied', 'required': True}, {'name': 'electl', 'glob': '/kaggle/input/**/electra-large-discriminator/config.json', 'directory_name': 'electra-large-discriminator', 'ordering': 'sorted paths, first parseable config; source does not check model_type when directory_name supplied', 'required': True}, {'name': 'desk', 'glob': '/kaggle/input/**/desklib-detector/config.json', 'directory_name': 'desklib-detector', 'ordering': 'sorted paths, first parseable config; source does not check model_type when directory_name supplied', 'required': True}, {'name': 'lora24_public', 'glob': '/kaggle/input/**/lora24/final/COMPLETE.json', 'ordering': 'sorted paths, first', 'source_allows_missing_member_skip': True, 'trunk_spec': ['qwen3', 4096, 24], 'pool_layers': [2, 4, 6, 8, 12, 16, 20, 24], 'norms_file': 'heads.safetensors', 'anchor_file': 'anchor_features.npz', 'relative_assets': ['adapter/adapter_config.json', 'adapter/adapter_model.safetensors', 'heads.safetensors', 'readouts.npz', 'tokenizer']}, {'name': 'lora24_seed7', 'glob': '/kaggle/input/**/name-that-llm-train-lora24-s7/lora_out/job_final/COMPLETE.json', 'ordering': 'sorted paths, first', 'source_allows_missing_member_skip': True, 'trunk_spec': ['qwen3', 4096, 24], 'pool_layers': [2, 4, 6, 8, 12, 16, 20, 24], 'norms_file': 'norms.safetensors', 'anchor_file': 'valid_features.npz', 'relative_assets': ['adapter/adapter_config.json', 'adapter/adapter_model.safetensors', 'norms.safetensors', 'readouts.npz', 'tokenizer']}, {'name': 'lora24_claude', 'glob': '/kaggle/input/**/ntl-qwen3-8b-lora-claude-head/**/final/COMPLETE.json', 'ordering': 'sorted paths, first', 'source_allows_missing_member_skip': True, 'trunk_spec': ['qwen3', 4096, 24], 'pool_layers': [2, 4, 6, 8, 12, 16, 20, 24], 'norms_file': 'heads.safetensors', 'anchor_file': 'anchor_features.npz', 'relative_assets': ['adapter/adapter_config.json', 'adapter/adapter_model.safetensors', 'heads.safetensors', 'readouts.npz', 'tokenizer']}, {'name': 'lora20_q25_instyle', 'glob': '/kaggle/input/**/ntl-train-lora-qwen25-instyle/lora_out/job_final/COMPLETE.json', 'ordering': 'sorted paths, first', 'source_allows_missing_member_skip': True, 'trunk_spec': ['qwen2', 3584, 20], 'pool_layers': [2, 4, 6, 8, 10, 12, 16, 20], 'norms_file': 'norms.safetensors', 'anchor_file': 'valid_features.npz', 'relative_assets': ['adapter/adapter_config.json', 'adapter/adapter_model.safetensors', 'norms.safetensors', 'readouts.npz', 'tokenizer']}, {'name': 'tokattn_deb', 'glob': '/kaggle/input/**/tokattn_debl23_final.pt', 'ordering': 'sorted paths, first', 'required': True, 'serialized_torch_state': True}, {'name': 'tokattn_q35', 'glob': '/kaggle/input/**/tokattn_final.pt', 'required': False, 'reason': 'source cell5 sets NTL_DROP_Q35=1; not active in this frozen run'}, {'name': 'ultrabert', 'glob': '/kaggle/input/**/ultrabert-380m-ntl', 'ordering': 'glob-native first root with ==5 sorted fold* dirs whose config model_type=ultrabert', 'required': True, 'trust_remote_code': True}], 'active_main_members': {'q14v5': {'base': 'B14', 'adapters': 'AQ14V5', 'keep': 27, 'layers': [2, 4, 6, 8, 12, 16, 20, 24], 'hidden_size': 5120, 'head_feature_dimension': 40960, 'weight': 3.0, 'bs': 8, 'max_len': 384, 'twin': 0.35, 'gauss': 0.7}, 'q8v5': {'base': 'B8', 'adapters': 'AQ8V5', 'keep': 24, 'layers': [2, 4, 6, 8, 12, 16, 20, 24], 'hidden_size': 4096, 'head_feature_dimension': 32768, 'weight': 2.0, 'bs': 16, 'max_len': 384, 'twin': 0.35, 'gauss': 0.44}, 'll8v5': {'base': 'BL', 'adapters': 'ALL8V5', 'keep': 24, 'layers': [2, 4, 6, 8, 12, 16, 20, 24], 'hidden_size': 4096, 'head_feature_dimension': 32768, 'weight': 1.0, 'bs': 16, 'max_len': 384, 'twin': 0.35}, 'n14b': {'base': 'B14', 'meta_selector': 'lora_qwen3_14b', 'weight': 2.5, 'twin': 0.35, 'no_final_norm': True, 'keep_layers_max_len_eos': 'from cfg in actual metadata'}, 'q25_14': {'base': 'B25_14', 'meta_selector': 'lora_qwen25_14b', 'weight': 1.0, 'twin': 0.35, 'no_final_norm': True, 'keep_layers_max_len_eos': 'from cfg in actual metadata'}, 'deb': {'meta_selector': 'DEB', 'weight': 1.0, 'encoder_relative_dir': 'encoder', 'head_relative_file': 'head.safetensors'}}, 'contracts': {'labels_main_infer': ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'kimi-k3', 'qwen3.8-max'], 'labels_complement': ['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'kimi-k3', 'qwen3.8-max'], 'all_main_head_rows': 18, 'l24_heads_expected_keys': ['plain.weight', 'plain.bias', 'centered.weight', 'centered.bias', 'norms.0.weight', 'norms.0.bias', 'norms.1.weight', 'norms.1.bias', 'norms.2.weight', 'norms.2.bias', 'norms.3.weight', 'norms.3.bias', 'norms.4.weight', 'norms.4.bias', 'norms.5.weight', 'norms.5.bias', 'norms.6.weight', 'norms.6.bias', 'norms.7.weight', 'norms.7.bias'], 'l24_heads_loader': 'heads.pt preferred; heads.safetensors fallback only when heads.pt absent. adir selector itself mandates heads.pt. Optional meta.json labels remap to sorted18; absent metadata leaves source-assumed axis unverifiable on CPU without static serialized-state inspection.', 'l24_adapter_relative': 'adapter/adapter_config.json and adapter/adapter_model.safetensors if adapter subdir config exists; otherwise root files', 'l24_adapter_rank_scale': 'r >0; scale=lora_alpha/r; paired lora_A/lora_B matrices; shapes must match target hidden/model submodule and kept layers', 'complement_readouts_required': ['classes', 'pool_layers', 'plain_coef', 'plain_intercept', 'center_coef', 'center_intercept'], 'complement_readout_axis': 'classes exact sorted18; pool_layers exact source LORA_RUNS layers; coef [18,hidden*layers],intercept[18]; optional claude_coef[5,D]/claude_intercept[5]', 'deberta_head_shapes': 'weight[18,encoder.hidden_size],bias[18]; metadata labels remapped by name; max_len frommeta', 'ATLAS': 'atlas_bundle.json, adapters.safetensors, ntl_atlas/model.py and ntl_rlcd/data.py imported from bundle root; label permutation to sorted18', 'parquet_schema': {'nlm_matched_clean.parquet': ['text', 'model'], 'nlm_pool_v5.parquet': ['text', 'model']}, 'extra_csv_sources': [['**/ntl_extra_passages_claude.csv', 'model', None], ['**/gen_passages.csv', 'model', None], ['**/ntl_generations.csv', 'label', 'finish_reason'], ['**/ntl_proxy_generations.csv', 'label', None], ['**/ntl_generated*.csv', 'model', 'finish'], ['**/passages_clean.csv', 'model', None]], 'runtime_flags': {'NTL_DROP_Q35': '1', 'NTL_SMOKE': 'mustnot1', 'NTL_FAKE_TEST': 'mustabsentforofficial'}, 'ATLAS_budget_formula': 'int(8.3*3600-COMP_SEC-600); actual complement wall time subtracted', 'source_hardcoded_file_sha256s_present': False}, 'source_manifest_sha256': 'ea6ede6cf6641f23217b7f3bdc5d3b3366a7261a22d99418778fae189b7a73ee'}
PRIOR_VENDOR_PINS={'models/qwen-lm/qwen-3/transformers/8b-base/1': {'path': 'models/qwen-lm/qwen-3/transformers/8b-base/1', 'model_type': 'qwen3', 'hidden_size': 4096, 'layers': 36, 'config_sha256': '3bd01d7ad7a2e203ecbbe84e24087a51c6d2a108ee4bcc42d0016bf49564983a', 'indexes': [{'name': 'model.safetensors.index.json', 'sha256': 'cd39d5e44c5aadfd1954b598883aea1c4a98822a2f7723d214ef2278b7fec2e4'}], 'indexed_tensors': 399, 'weight_files': [{'name': 'model-00005-of-00005.safetensors', 'bytes': 1244659840, 'tensor_header_count': 1}, {'name': 'model-00001-of-00005.safetensors', 'bytes': 3996250744, 'tensor_header_count': 81}, {'name': 'model-00002-of-00005.safetensors', 'bytes': 3993160032, 'tensor_header_count': 114}, {'name': 'model-00004-of-00005.safetensors', 'bytes': 3187841392, 'tensor_header_count': 92}, {'name': 'model-00003-of-00005.safetensors', 'bytes': 3959604768, 'tensor_header_count': 111}], 'tokenizer': [{'name': 'tokenizer.json', 'bytes': 7031645, 'sha256': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'}, {'name': 'vocab.json', 'bytes': 2776833, 'sha256': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}, {'name': 'merges.txt', 'bytes': 1671853, 'sha256': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5'}, {'name': 'tokenizer_config.json', 'bytes': 9675, 'sha256': '5a7303fcb1a27ede63134a2cbd61d5282c247ca6d769ce4746d4ffa124aedd63'}], 'large_weight_values_or_custom_code_loaded': False, 'large_weight_payloads_hashed': False}, 'models/qwen-lm/qwen-3/transformers/14b-base/1': {'path': 'models/qwen-lm/qwen-3/transformers/14b-base/1', 'model_type': 'qwen3', 'hidden_size': 5120, 'layers': 40, 'config_sha256': 'eb65771c8f108f88e4d87f771e1c9d40b7e35bc8070be8c97246c9446ccdf179', 'indexes': [{'name': 'model.safetensors.index.json', 'sha256': '8bb45057e5de5b87096da94fab4492dfa0d7b9fc1b7fe9bc0d91cb9cd6c19bcb'}], 'indexed_tensors': 443, 'weight_files': [{'name': 'model-00001-of-00008.safetensors', 'bytes': 3841788544, 'tensor_header_count': 41}, {'name': 'model-00003-of-00008.safetensors', 'bytes': 3963750880, 'tensor_header_count': 66}, {'name': 'model-00002-of-00008.safetensors', 'bytes': 3963750816, 'tensor_header_count': 66}, {'name': 'model-00007-of-00008.safetensors', 'bytes': 3963750880, 'tensor_header_count': 66}, {'name': 'model-00008-of-00008.safetensors', 'bytes': 1912371880, 'tensor_header_count': 6}, {'name': 'model-00005-of-00008.safetensors', 'bytes': 3963750880, 'tensor_header_count': 66}, {'name': 'model-00006-of-00008.safetensors', 'bytes': 3963750880, 'tensor_header_count': 66}, {'name': 'model-00004-of-00008.safetensors', 'bytes': 3963750880, 'tensor_header_count': 66}], 'tokenizer': [{'name': 'tokenizer.json', 'bytes': 7031645, 'sha256': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'}, {'name': 'vocab.json', 'bytes': 2776833, 'sha256': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}, {'name': 'merges.txt', 'bytes': 1671853, 'sha256': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5'}, {'name': 'tokenizer_config.json', 'bytes': 9675, 'sha256': '5a7303fcb1a27ede63134a2cbd61d5282c247ca6d769ce4746d4ffa124aedd63'}], 'large_weight_values_or_custom_code_loaded': False, 'large_weight_payloads_hashed': False}, 'models/qwen-lm/qwen2.5/transformers/7b/1': {'path': 'models/qwen-lm/qwen2.5/transformers/7b/1', 'model_type': 'qwen2', 'hidden_size': 3584, 'layers': 28, 'config_sha256': '267ce68584c5f24c3b267d934db2de68dd21d1ca677fb78ed809eb60067f7642', 'indexes': [{'name': 'model.safetensors.index.json', 'sha256': '624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028'}], 'indexed_tensors': 339, 'weight_files': [{'name': 'model-00003-of-00004.safetensors', 'bytes': 3864726424, 'tensor_header_count': 97}, {'name': 'model-00001-of-00004.safetensors', 'bytes': 3945441440, 'tensor_header_count': 82}, {'name': 'model-00004-of-00004.safetensors', 'bytes': 3556377672, 'tensor_header_count': 63}, {'name': 'model-00002-of-00004.safetensors', 'bytes': 3864726352, 'tensor_header_count': 97}], 'tokenizer': [{'name': 'tokenizer.json', 'bytes': 7031645, 'sha256': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'}, {'name': 'vocab.json', 'bytes': 2776833, 'sha256': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}, {'name': 'merges.txt', 'bytes': 1671839, 'sha256': '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3'}, {'name': 'tokenizer_config.json', 'bytes': 7228, 'sha256': 'c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9'}], 'large_weight_values_or_custom_code_loaded': False, 'large_weight_payloads_hashed': False}, 'models/qwen-lm/qwen-3/transformers/4b-base/1': {'path': 'models/qwen-lm/qwen-3/transformers/4b-base/1', 'model_type': 'qwen3', 'hidden_size': 2560, 'layers': 36, 'config_sha256': '304b2545a258d35620f1d4bf46940c0471d9baa00715ff8e77f84c2fca5057c1', 'indexes': [{'name': 'model.safetensors.index.json', 'sha256': 'd6c42883a895dfef5b0080ed2116a1bcd764f558406b98923d675978a1abf29c'}], 'indexed_tensors': 398, 'weight_files': [{'name': 'model-00003-of-00003.safetensors', 'bytes': 99630640, 'tensor_header_count': 5}, {'name': 'model-00001-of-00003.safetensors', 'bytes': 3957900840, 'tensor_header_count': 174}, {'name': 'model-00002-of-00003.safetensors', 'bytes': 3987450520, 'tensor_header_count': 219}], 'tokenizer': [{'name': 'tokenizer.json', 'bytes': 7031645, 'sha256': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'}, {'name': 'vocab.json', 'bytes': 2776833, 'sha256': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}, {'name': 'merges.txt', 'bytes': 1671853, 'sha256': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5'}, {'name': 'tokenizer_config.json', 'bytes': 9675, 'sha256': '5a7303fcb1a27ede63134a2cbd61d5282c247ca6d769ce4746d4ffa124aedd63'}], 'large_weight_values_or_custom_code_loaded': False, 'large_weight_payloads_hashed': False}, 'notebooks/lucasboesen/name-that-llm-glm4-9b-trunc25/glm4_9b_trunc25': {'path': 'notebooks/lucasboesen/name-that-llm-glm4-9b-trunc25/glm4_9b_trunc25', 'model_type': 'glm4', 'hidden_size': 4096, 'layers': 25, 'config_sha256': '22b5515c8520e7aa3463b0fc07e601e99a11119c736efb754035ea59637fa3ed', 'indexes': [], 'indexed_tensors': 0, 'weight_files': [{'name': 'model.safetensors', 'bytes': 11440008912, 'tensor_header_count': 327}], 'tokenizer': [{'name': 'tokenizer.json', 'bytes': 19966496, 'sha256': '76ebeac0d8bd7879ead7b43c16b44981f277e47225de2bd7de9ae1a6cc664a8c'}, {'name': 'tokenizer_config.json', 'bytes': 396, 'sha256': 'de548df866322721bf61112987f91e3de68bd9525df6eb1c558985ea79c43269'}], 'large_weight_values_or_custom_code_loaded': False, 'large_weight_payloads_hashed': False}, 'notebooks/lucasboesen/name-that-llm-extract-encoders2/deberta-v3-large': {'path': 'notebooks/lucasboesen/name-that-llm-extract-encoders2/deberta-v3-large', 'model_type': 'deberta-v2', 'hidden_size': 1024, 'layers': 24, 'config_sha256': '9125207e9fc1f5913ca71c99936e671ab44280317133d02f909726a8af0b231f', 'indexes': [], 'indexed_tensors': 0, 'weight_files': [{'name': 'model.safetensors', 'bytes': 1736094384, 'tensor_header_count': 390}], 'tokenizer': [{'name': 'tokenizer.json', 'bytes': 8339876, 'sha256': 'e9ddd853a4a31023721c993458663f2019e3154a05dacf0d2b3c217e53d9bdc8'}, {'name': 'tokenizer_config.json', 'bytes': 538, 'sha256': '0dbe41323167ba5f2e1e876ccd05068318dd2378b922bdfd924a389dba3bcc7a'}], 'large_weight_values_or_custom_code_loaded': False, 'large_weight_payloads_hashed': False}, 'notebooks/lucasboesen/name-that-llm-electra-large-weights/electra-large-discriminator': {'path': 'notebooks/lucasboesen/name-that-llm-electra-large-weights/electra-large-discriminator', 'model_type': 'electra', 'hidden_size': 1024, 'layers': 24, 'config_sha256': '01244920f29102fda67359ad1ec89d09c8076378d46acd4517f656debd1e4303', 'indexes': [], 'indexed_tensors': 0, 'weight_files': [{'name': 'model.safetensors', 'bytes': 1336413848, 'tensor_header_count': 389}], 'tokenizer': [{'name': 'tokenizer.json', 'bytes': 711396, 'sha256': 'd241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66'}, {'name': 'tokenizer_config.json', 'bytes': 351, 'sha256': 'e1c2a61a99bda00f6c55303a210b30e2f92dcf8b555e215812e2eb583e177ffd'}], 'large_weight_values_or_custom_code_loaded': False, 'large_weight_payloads_hashed': False}, 'notebooks/lucasboesen/name-that-llm-desklib-weights/desklib-detector': {'path': 'notebooks/lucasboesen/name-that-llm-desklib-weights/desklib-detector', 'model_type': 'deberta-v2', 'hidden_size': 1024, 'layers': 24, 'config_sha256': 'fd728d4c20ca32f0cd98851c8dc1a1c1ba68740dc33a7a6f93b99f053e378833', 'indexes': [], 'indexed_tensors': 0, 'weight_files': [{'name': 'model.safetensors', 'bytes': 1736094384, 'tensor_header_count': 390}], 'tokenizer': [{'name': 'tokenizer.json', 'bytes': 8339875, 'sha256': '2c070ff353c36f85e20919da787a30e1e8e52b626d0e026e044f373378f6d0bf'}, {'name': 'tokenizer_config.json', 'bytes': 532, 'sha256': 'd8089020d294f20cb21eca751b5f7f68a0b62ea2716a78576cbd587605382432'}], 'large_weight_values_or_custom_code_loaded': False, 'large_weight_payloads_hashed': False}, 'datasets/bbucxi/qwen35-9b-safetensors': {'path': 'datasets/bbucxi/qwen35-9b-safetensors', 'model_type': 'qwen3_5', 'hidden_size': 4096, 'layers': 32, 'config_sha256': 'd0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05', 'indexes': [{'name': 'model.safetensors.index.json', 'sha256': '26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5'}], 'indexed_tensors': 775, 'weight_files': [{'name': 'model.safetensors-00001-of-00004.safetensors', 'bytes': 5276436216, 'tensor_header_count': 14}, {'name': 'model.safetensors-00002-of-00004.safetensors', 'bytes': 5335161512, 'tensor_header_count': 53}, {'name': 'model.safetensors-00003-of-00004.safetensors', 'bytes': 5368717440, 'tensor_header_count': 63}, {'name': 'model.safetensors-00004-of-00004.safetensors', 'bytes': 3325995712, 'tensor_header_count': 645}], 'tokenizer': [{'name': 'tokenizer.json', 'bytes': 12807982, 'sha256': '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'}, {'name': 'vocab.json', 'bytes': 6722759, 'sha256': 'ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003'}, {'name': 'merges.txt', 'bytes': 3353259, 'sha256': 'a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d'}, {'name': 'tokenizer_config.json', 'bytes': 16710, 'sha256': '316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8'}], 'large_weight_values_or_custom_code_loaded': False, 'large_weight_payloads_hashed': False}, 'models/qwen-lm/qwen2.5/transformers/14b/1': {'path': 'models/qwen-lm/qwen2.5/transformers/14b/1', 'model_type': 'qwen2', 'hidden_size': 5120, 'layers': 48, 'config_sha256': 'fdb89460be9a6383451437cdf632a3e087522efa682265461f33507e7069f026', 'indexes': [{'name': 'model.safetensors.index.json', 'sha256': '7b4b616a213f09ae889be05d1b135198589496e817d793e2605d5c5b1fe3ae5d'}], 'indexed_tensors': 579, 'weight_files': [{'name': 'model-00001-of-00008.safetensors', 'bytes': 3885134104, 'tensor_header_count': 56}, {'name': 'model-00003-of-00008.safetensors', 'bytes': 3995328064, 'tensor_header_count': 85}, {'name': 'model-00002-of-00008.safetensors', 'bytes': 3995327992, 'tensor_header_count': 85}, {'name': 'model-00007-of-00008.safetensors', 'bytes': 3995328064, 'tensor_header_count': 85}, {'name': 'model-00008-of-00008.safetensors', 'bytes': 1698724408, 'tensor_header_count': 5}, {'name': 'model-00005-of-00008.safetensors', 'bytes': 3979614448, 'tensor_header_count': 91}, {'name': 'model-00006-of-00008.safetensors', 'bytes': 3995328064, 'tensor_header_count': 85}, {'name': 'model-00004-of-00008.safetensors', 'bytes': 3995348776, 'tensor_header_count': 87}], 'tokenizer': [{'name': 'tokenizer.json', 'bytes': 7031645, 'sha256': 'c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'}, {'name': 'vocab.json', 'bytes': 2776833, 'sha256': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910'}, {'name': 'merges.txt', 'bytes': 1671839, 'sha256': '599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3'}, {'name': 'tokenizer_config.json', 'bytes': 7228, 'sha256': 'c91efca15ceff6e9ee9424db58a6f59cd41294e550a86cbd07e3c1fb500b34f9'}], 'large_weight_values_or_custom_code_loaded': False, 'large_weight_payloads_hashed': False}}
LABELS=['claude-fable-5', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'glm-5.3', 'glm-5.3-flash', 'gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'kimi-k3', 'qwen3.8-max']
TRAIN_SHA="40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707"
def safe_relative(name):
    path = Path(str(name))
    if path.is_absolute() or '..' in path.parts or not path.parts or '\x00' in str(name):
        raise ValueError('Unsafe relative asset path')
    return path

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

def npz_header_inventory(path):
    """Inspect bounded NPY headers inside ZIP; never interpret object payloads."""
    result={}
    with zipfile.ZipFile(path) as archive:
        names=[info.filename for info in archive.infolist()]
        if len(names)!=len(set(names)):raise ValueError('Duplicate NPZ archive member names')
        for info in archive.infolist():
            safe_relative(info.filename)
            if not info.filename.endswith('.npy'):raise ValueError('Unexpected non-NPY NPZ member')
            if info.file_size>512*1024*1024:raise ValueError('NPZ member exceeds bounded numeric reader')
            with archive.open(info) as stream:
                if stream.read(6)!=b'\x93NUMPY':raise ValueError('Malformed NPY member magic')
                version=tuple(stream.read(2))
                if version not in [(1,0),(2,0),(3,0)]:raise ValueError('Unsupported NPY member version')
                prefix_bytes=2 if version==(1,0) else 4;raw_length=stream.read(prefix_bytes)
                if len(raw_length)!=prefix_bytes:raise ValueError('Truncated NPY header length')
                size=struct.unpack('<H' if prefix_bytes==2 else '<I',raw_length)[0]
                if size<2 or size>65536 or size+8+prefix_bytes>info.file_size:raise ValueError('NPY header length exceeds bound')
                raw=stream.read(size)
                if len(raw)!=size:raise ValueError('Truncated NPY header')
                header=ast.literal_eval(raw.decode('utf-8' if version==(3,0) else 'latin1'))
                if not isinstance(header,dict) or set(header)!={'descr','fortran_order','shape'}:raise ValueError('NPY header schema differs')
                shape=header['shape'];dtype=np.dtype(header['descr'])
                if not isinstance(shape,tuple) or not all(isinstance(x,int) and not isinstance(x,bool) and x>=0 for x in shape) or not isinstance(header['fortran_order'],bool):raise ValueError('NPY shape/order metadata invalid')
                if not dtype.hasobject and math.prod(shape)*dtype.itemsize!=info.file_size-(size+8+prefix_bytes):raise ValueError('NPY numeric shape/payload extent differs')
                result[info.filename[:-4]]={'shape':list(shape),'dtype':str(dtype),'hasobject':bool(dtype.hasobject),'fortran_order':header['fortran_order'],'uncompressed_bytes':info.file_size}
    return result

def state_inventory(path,required_members=(),allow_unused_object_metadata=False):
    """Read safe numeric members; optionally inventory unconsumed object metadata headers only."""
    p=Path(path); result={}
    if p.suffix=='.npz':
        headers=npz_header_inventory(p);required_members=set(required_members)
        if not required_members.issubset(headers):raise ValueError('Source-consumed NPZ member missing: '+str(sorted(required_members-set(headers))))
        context=np.load(p,allow_pickle=False)
        try:
            for key in context.files:
                if headers[key]['hasobject']:
                    if key in required_members or not allow_unused_object_metadata:raise ValueError('Object NPZ member is required or not admitted: '+key)
                    result[key]={**headers[key],'finite_numeric':False,'payload_loaded':False,'source_consumed':False,'inspection':'HEADER_ONLY_UNUSED_OBJECT_METADATA'}
                    continue
                a=context[key]
                if a.dtype.hasobject:raise ValueError('Object numeric state is not admitted')
                if a.dtype.kind in 'fciu' and not np.isfinite(a).all():raise ValueError('Nonfinite state: '+key)
                result[key]={'shape':list(a.shape),'dtype':str(a.dtype),'finite_numeric':a.dtype.kind in 'fciu','payload_loaded':True,'source_consumed':key in required_members}
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


class UnsupportedPickleMetadata(Exception):pass

# A metadata-only pickle opcode interpreter. No unpickler, imports, globals or
# Python reducer calls are used. Unsupported syntax leaves archive-only evidence.
def static_torch_metadata(data,storage_sizes):
    stack=[];memo={};mark=object();globals_seen=[]
    dtype_width={'FloatStorage':4,'DoubleStorage':8,'HalfStorage':2,'BFloat16Storage':2,'LongStorage':8,'IntStorage':4,'ShortStorage':2,'CharStorage':1,'ByteStorage':1,'BoolStorage':1}
    def take_mark():
        for i in range(len(stack)-1,-1,-1):
            if stack[i] is mark:
                values=stack[i+1:];del stack[i:];return values
        raise ValueError('Malformed metadata pickle mark')
    for op,arg,pos in pickletools.genops(data):
        name=op.name
        if name in ['PROTO','FRAME']:continue
        if name=='MARK':stack.append(mark)
        elif name in ['EMPTY_DICT','EMPTY_LIST','EMPTY_TUPLE']:stack.append({} if name=='EMPTY_DICT' else [] if name=='EMPTY_LIST' else ())
        elif name in ['NONE','NEWTRUE','NEWFALSE']:stack.append(None if name=='NONE' else name=='NEWTRUE')
        elif name in ['BININT','BININT1','BININT2','INT','LONG','LONG1','LONG4','FLOAT','BINFLOAT','UNICODE','BINUNICODE','SHORT_BINUNICODE','BINUNICODE8','BINBYTES','SHORT_BINBYTES','BINBYTES8','STRING','BINSTRING','SHORT_BINSTRING']:stack.append(arg)
        elif name=='TUPLE':stack.append(tuple(take_mark()))
        elif name in ['TUPLE1','TUPLE2','TUPLE3']:
            count=int(name[-1]);values=tuple(stack[-count:]);del stack[-count:];stack.append(values)
        elif name=='LIST':stack.append(list(take_mark()))
        elif name=='DICT':
            values=take_mark()
            if len(values)%2:raise ValueError('Malformed metadata dictionary')
            stack.append(dict(zip(values[::2],values[1::2])))
        elif name=='APPEND':value=stack.pop();stack[-1].append(value)
        elif name=='APPENDS':values=take_mark();stack[-1].extend(values)
        elif name=='SETITEM':value=stack.pop();key=stack.pop();stack[-1][key]=value
        elif name=='SETITEMS':
            values=take_mark()
            if len(values)%2:raise ValueError('Malformed metadata setitems')
            stack[-1].update(zip(values[::2],values[1::2]))
        elif name in ['BINPUT','LONG_BINPUT','PUT']:memo[int(arg)]=stack[-1]
        elif name=='MEMOIZE':memo[len(memo)]=stack[-1]
        elif name in ['BINGET','LONG_BINGET','GET']:stack.append(memo[int(arg)])
        elif name=='GLOBAL':
            module,symbol=arg.split(' ',1);global_name=(module,symbol);globals_seen.append('.'.join(global_name));stack.append(('GLOBAL',module,symbol))
        elif name=='STACK_GLOBAL':
            symbol=stack.pop();module=stack.pop();globals_seen.append(str(module)+'.'+str(symbol));stack.append(('GLOBAL',module,symbol))
        elif name=='BINPERSID':
            item=stack.pop()
            if not isinstance(item,tuple) or len(item)<5 or item[0]!='storage':raise UnsupportedPickleMetadata('Unsupported persistent identifier')
            _,dtype,key,device,count,*_=item
            if not isinstance(dtype,tuple) or dtype[:2]!=('GLOBAL','torch') or dtype[2] not in dtype_width:raise UnsupportedPickleMetadata('Unsupported storage type')
            if not isinstance(key,str) or not key.isdigit() or not isinstance(count,int) or count<0:raise ValueError('Malformed Torch storage descriptor')
            if storage_sizes.get(key)!=count*dtype_width[dtype[2]]:raise ValueError('Torch ZIP storage byte extent differs from metadata')
            stack.append({'_storage':True,'key':key,'count':count,'dtype':dtype[2],'width':dtype_width[dtype[2]]})
        elif name=='REDUCE':
            args=stack.pop();function=stack.pop()
            if not isinstance(function,tuple) or function[0]!='GLOBAL' or not isinstance(args,tuple):raise UnsupportedPickleMetadata('Unsupported reducer descriptor')
            _,module,symbol=function
            if (module,symbol)==('collections','OrderedDict'):
                if args:raise UnsupportedPickleMetadata('Nonempty OrderedDict constructor')
                stack.append({})
            elif module=='torch._utils' and symbol in ['_rebuild_tensor','_rebuild_tensor_v2']:
                if len(args)<4:raise ValueError('Malformed Torch tensor metadata')
                storage,offset,shape,stride=args[:4]
                if not isinstance(storage,dict) or storage.get('_storage') is not True:raise UnsupportedPickleMetadata('Unsupported tensor storage')
                if not isinstance(offset,int) or offset<0 or not isinstance(shape,tuple) or not isinstance(stride,tuple) or len(shape)!=len(stride) or not all(isinstance(x,int) and x>=0 for x in shape+stride):raise ValueError('Malformed tensor shape/stride')
                extent=offset+(sum((dim-1)*step for dim,step in zip(shape,stride)) if all(shape) else 0)
                if (math.prod(shape)>0 and extent>=storage['count']) or offset>storage['count']:raise ValueError('Tensor exceeds storage bounds')
                stack.append({'_tensor':True,'shape':list(shape),'stride':list(stride),'storage_offset':offset,'storage_key':storage['key'],'dtype':storage['dtype']})
            elif module=='torch._utils' and symbol in ['_rebuild_parameter','_rebuild_parameter_with_state'] and args and isinstance(args[0],dict) and args[0].get('_tensor'):stack.append(args[0])
            else:raise UnsupportedPickleMetadata('Unsupported reducer '+module+'.'+symbol)
        elif name=='BUILD':
            state=stack.pop()
            if not isinstance(stack[-1],dict) or not isinstance(state,dict) or set(state)-{'_metadata'}:raise UnsupportedPickleMetadata('Unsupported BUILD metadata')
        elif name=='STOP':
            if len(stack)!=1:raise ValueError('Malformed metadata pickle stack')
            value=stack[0]
            if not isinstance(value,dict):raise UnsupportedPickleMetadata('Torch state is not a mapping')
            tensors={k:v for k,v in value.items() if isinstance(k,str) and isinstance(v,dict) and v.get('_tensor')}
            return {'tensors':tensors,'globals':sorted(set(globals_seen)),'non_tensor_keys':[str(k) for k in value if k not in tensors]}
        else:raise UnsupportedPickleMetadata('Unsupported opcode '+name)
    raise ValueError('No pickle STOP in Torch archive metadata')

def torch_zip_inventory(path):
    p=Path(path)
    if not zipfile.is_zipfile(p):return {'format':'non-ZIP Torch serialization','tensor_schema_inspected':False,'limitation':'Legacy Torch serialization not parsed or loaded','bytes':p.stat().st_size}
    with zipfile.ZipFile(p) as z:
        infos=z.infolist()
        if len(infos)>20000 or any(i.file_size>1024**3 for i in infos) or sum(i.file_size for i in infos)>2*1024**3:raise ValueError('Torch archive exceeds bounded CPU inspection')
        names=[i.filename for i in infos]
        for name in names:safe_relative(name)
        if z.testzip() is not None:raise ValueError('Torch ZIP CRC failure')
        metadata=[n for n in names if n.endswith('/data.pkl') or n=='data.pkl']
        if len(metadata)!=1:raise ValueError('Torch ZIP missing/ambiguous data.pkl')
        if z.getinfo(metadata[0]).file_size>16*1024*1024:raise ValueError('Torch metadata pickle exceeds bound')
        prefix=metadata[0][:-len('data.pkl')];storage_sizes={n[len(prefix+'data/'):]:z.getinfo(n).file_size for n in names if n.startswith(prefix+'data/')}
        raw=z.read(metadata[0]);literal_globals=sorted({str(arg).replace(' ','.',1) for op,arg,_ in pickletools.genops(raw) if op.name=='GLOBAL'})
        result={'format':'Torch ZIP','bytes':p.stat().st_size,'member_count':len(infos),'storage_count':len(storage_sizes),'metadata_sha256':hashlib.sha256(raw).hexdigest(),'archive_crc_verified':True,'pickle_loaded':False,'reducers_executed':False,'literal_pickle_GLOBAL_references':literal_globals}
        try:state=static_torch_metadata(raw,storage_sizes)
        except UnsupportedPickleMetadata as error:
            result.update(tensor_schema_inspected=False,limitation=str(error))
        else:result.update(tensor_schema_inspected=True,**state)
        return result

def labels_and_permutation(native):
    if not isinstance(native,list) or len(native)!=18 or not all(isinstance(x,str) for x in native) or set(native)!=set(LABELS):raise ValueError('Label metadata is not exactly the18canonical classes')
    perm=[native.index(x) for x in LABELS]
    if sorted(perm)!=list(range(18)):raise ValueError('Invalid label permutation')
    return {'native_order':native,'canonical_order':LABELS,'source_column_permutation':perm,'head_or_metadata_reordered':False}

def ultrabert_roots_in_source_order(roots):
    candidates=[]
    for root in roots:
        folds=[p for p in sorted(root.glob('fold*')) if (p/'config.json').is_file() and read_json(p/'config.json').get('model_type')=='ultrabert']
        if len(folds)==5:candidates.append((root,folds))
    return candidates

class Audit:
    def __init__(self,root='/kaggle/input',out='/kaggle/working/stage12_preflight',seconds=900):
        self.root=Path(root);self.out=Path(out);self.started=time.monotonic();self.deadline=self.started+seconds;self.seconds=seconds;self.checks=[];self.details={};self.warnings=[];self.code=[];self.hashes={};self.heads={};self.selected={};self.selection_evidence={};self.configs=None
    def check(self,name,fn):
        try:
            value=fn();self.details[name]=value;self.checks.append({'name':name,'status':'PASS'});print('PREFLIGHT_PASS',name,flush=True);return value
        except Exception as e:
            self.checks.append({'name':name,'status':'FAIL','error_type':type(e).__name__,'error':str(e)[:1500]});print('PREFLIGHT_FAIL',name,type(e).__name__,str(e)[:300],flush=True);return None
    def rel(self,p):return str(Path(p).relative_to(self.root))
    def hash(self,p):
        p=Path(p)
        if p not in self.hashes:
            digest=hashlib.sha256()
            with p.open('rb') as f:
                while chunk:=f.read(8*1024*1024):
                    if time.monotonic()>self.deadline:raise TimeoutError('Bounded900-second critical-file hash audit budget exhausted')
                    digest.update(chunk)
            self.hashes[p]=digest.hexdigest()
        return self.hashes[p]
    def header(self,p):
        p=Path(p)
        if p not in self.heads:self.heads[p]=safetensor_header(p)
        return self.heads[p]
    def matches(self,pattern,ordered=False):
        if not pattern.startswith('/kaggle/input/'):raise ValueError('Source pattern is outside input root')
        hits=glob.glob(str(self.root/pattern[len('/kaggle/input/'):]),recursive=True)
        # The frozen source sorts glob STRINGS. Path component sorting changes
        # prefix cases such as 8b-base/ versus 8b/ and selects a different model.
        if ordered:hits=sorted(hits)
        return [Path(p) for p in hits]
    def resolve(self,spec,complement=False):
        ordered='sorted' in spec.get('ordering','') and 'UNSORTED' not in spec.get('ordering','')
        hits=self.matches(spec['glob'],ordered)
        if 'substring_filters_lower' in spec:hits=[p for p in hits if all(token in str(p).lower() for token in spec['substring_filters_lower'])]
        if 'model_type' in spec:
            filtered=[]
            for p in hits:
                try:c=read_json(p)
                except Exception:continue
                if c.get('model_type')==spec['model_type'] and ('hidden_size' not in spec or c.get('hidden_size')==spec['hidden_size']):filtered.append(p)
            hits=filtered
        if spec.get('directory_name'):
            filtered=[]
            for p in hits:
                try:read_json(p)
                except Exception:continue
                filtered.append(p)
            hits=filtered
        evidence={'source_pattern':spec['glob'],'source_ordering':spec.get('ordering'),'candidate_paths_in_source_order':[self.rel(p) for p in hits],'source_would_choose':self.rel(hits[0]) if hits else None,'source_allowed_missing_skip':spec.get('source_allows_missing_member_skip',False)}
        self.selection_evidence[('comp_' if complement else '')+spec['name']]=evidence
        if not hits:
            if spec.get('source_allows_missing_member_skip') or spec.get('required') is False:
                self.warnings.append('Source permits/declares inactive missing member: '+spec['name']);return None
            raise ValueError('Required source selector found no match: '+spec['name'])
        selected=hits[0]
        if len(hits)>1:
            relative=self.rel(selected.parent)
            pin=PRIOR_VENDOR_PINS.get(relative)
            if selected.name=='config.json' and pin and self.hash(selected)==pin['config_sha256']:
                evidence['multiple_matches_review']='Original sorted first choice exactly matches the previously audited vendor configuration; complete candidate list retained.'
            else:raise ValueError('Ambiguous source first-match selector requires explicit review: '+spec['name'])
        self.selected[('comp_' if complement else '')+spec['name']]=selected
        return selected
    def tokenizer(self,p):
        names=['tokenizer.json','tokenizer.model','spm.model','vocab.txt','vocab.json','merges.txt','tokenizer_config.json','special_tokens_map.json','added_tokens.json'];files=[p/name for name in names if (p/name).is_file()]
        if not any(f.name in names[:5] and f.stat().st_size>0 for f in files):raise ValueError('Tokenizer vocabulary/model missing: '+self.rel(p))
        return [{'name':f.name,'bytes':f.stat().st_size,'sha256':self.hash(f)} for f in files]
    def model(self,p):
        if p.is_file():p=p.parent
        cfg=read_json(p/'config.json');files=sorted(p.glob('*.safetensors'));bins=sorted(p.glob('*.bin'));indexes=sorted(p.glob('*.index.json'))
        if not files and not bins:raise ValueError('Selected config has no actual model weights: '+self.rel(p))
        for index in indexes:
            doc=read_json(index);mapping=doc.get('weight_map',{})
            if not mapping:raise ValueError('Empty model weight index')
            for key,relative in mapping.items():
                shard=p/safe_relative(relative)
                if not shard.is_file():raise ValueError('Indexed model shard missing: '+relative)
                if shard.suffix=='.safetensors' and key not in self.header(shard):raise ValueError('Index tensor absent from actual shard: '+key)
        for f in files:self.header(f)
        relative=self.rel(p);pin=PRIOR_VENDOR_PINS.get(relative);reused=None
        if pin:
            if self.hash(p/'config.json')!=pin['config_sha256']:raise ValueError('Previously audited vendor config changed')
            expected={r['name']:r['bytes'] for r in pin['weight_files']}
            actual={f.name:f.stat().st_size for f in files}
            if actual!=expected:raise ValueError('Previously audited vendor safetensors inventory/bytes changed')
            reused={'prior_stage':11,'prior_actual_input_report':'REAL_INPUT_AUDIT_V2.json','config_and_weight_inventory_matches':True,'large_payload_hashes_were_not_proven_by_prior_audit':True}
        return {'path':relative,'model_type':cfg.get('model_type'),'hidden_size':cfg.get('hidden_size',cfg.get('text_config',{}).get('hidden_size')),'num_hidden_layers':cfg.get('num_hidden_layers',cfg.get('text_config',{}).get('num_hidden_layers')),'config_sha256':self.hash(p/'config.json'),'tokenizer':self.tokenizer(p),'weight_files':[{'name':f.name,'bytes':f.stat().st_size,'header_tensor_count':len(self.header(f))} for f in files],'index_sha256':{f.name:self.hash(f) for f in indexes},'prior_vendor_reuse':reused,'large_weight_payloads_hashed':False,'model_or_custom_code_loaded':False}
    def export_text(self,p,role):
        if p.stat().st_size>512*1024:raise ValueError('Private code export exceeds bound')
        text=p.read_text();digest=self.hash(p)
        if p.suffix=='.py':ast.parse(text)
        record={'path':self.rel(p),'bytes':p.stat().st_size,'sha256':digest,'role':role};old=next((x for x in self.code if x['sha256']==digest and 'text' in x),None)
        if old:record['same_text_as']=old['path']
        else:record['text']=text
        self.code.append(record);return {k:v for k,v in record.items() if k!='text'}
    def official(self):
        options=[self.root/'competitions/name-that-llm',self.root/'name-that-llm'];p=next((p for p in options if (p/'train.csv').is_file()),None)
        if p is None:raise ValueError('Official competition files unavailable')
        f=pd.read_csv(p/'train.csv',dtype=str,keep_default_na=False)
        if self.hash(p/'train.csv')!=TRAIN_SHA or len(f)!=2699 or f.prompt_id.nunique()!=150 or set(f.model)!=set(LABELS) or not f.id.is_unique:raise ValueError('Official train identity/schema differs')
        if pd.read_csv(p/'test.csv',nrows=0).columns.tolist()!=['id','text']:raise ValueError('Official test header differs')
        return {'path':self.rel(p),'rows':2699,'prompts':150,'classes':18,'train_sha256':self.hash(p/'train.csv'),'test_passages_or_labels_read':False,'original_rows_exported':False}
    def main_selector(self,spec):
        p=self.resolve(spec)
        if p is None:raise ValueError('Unexpected missing mandatory main selector')
        if spec['kind']=='model_directory':return self.model(p)
        if spec['kind']=='serialized_head_parent':return self.checkpoint(p.parent,spec['name'])
        if p.name=='heads.pt':return self.checkpoint(p.parent,spec['name'])
        if spec.get('metadata_cfg_required'):
            meta=read_json(p)
            if not isinstance(meta.get('cfg'),dict):raise ValueError('Main metadata cfg absent')
            return self.nishant(p.parent,spec['name'],active=spec['name'] in ['lora_qwen3_14b','lora_qwen25_14b'])
        if spec['name'] in ['POOL','EXTRA']:return self.parquet(p)
        if spec['name']=='HEADS':return self.head_npz(p)
        if spec['name']=='ATLAS':return self.atlas(p.parent)
        if spec['name']=='DEB':return self.deberta(p.parent)
        return {'path':self.rel(p),'bytes':p.stat().st_size,'sha256':self.hash(p)}
    def head_npz(self,p):
        state=state_inventory(p)
        required=['W','b','mu','sd','layers']
        if not all(key in state for key in required):raise ValueError('Legacy NPZ head keys missing')
        width=state['W']['shape'][0]
        if state['W']['shape']!=[width,18] or state['b']['shape']!=[18] or state['mu']['shape']!=[width] or state['sd']['shape']!=[width]:raise ValueError('Legacy NPZ classifier axes differ')
        return {'path':self.rel(p),'sha256':self.hash(p),'state':state,'active_main_member':False}
    def parquet(self,p):
        f=pd.read_parquet(p)
        if not {'text','model'}.issubset(f) or len(f)==0 or not f.model.isin(LABELS).all() or f.text.isna().any() or not f.text.map(lambda x:isinstance(x,str) and bool(x.strip())).all():raise ValueError('Required source pool schema/text/label contract differs')
        return {'path':self.rel(p),'sha256':self.hash(p),'rows':len(f),'columns':list(f.columns),'label_counts':f.model.value_counts().sort_index().to_dict(),'all18_classes_present':set(f.model)==set(LABELS),'original_rows_exported':False}
    def adapter_pairs(self,p,hidden=None,keep=None,base=None):
        cfg=read_json(p/'adapter_config.json');header=self.header(p/'adapter_model.safetensors');rank=cfg.get('r');alpha=cfg.get('lora_alpha')
        if not isinstance(rank,int) or rank<=0 or not isinstance(alpha,(int,float)) or not math.isfinite(alpha) or alpha<=0:raise ValueError('Adapter rank/scale invalid')
        count=0;layers=set();dimension_checks=0;unresolved_targets=[]
        for key,item in header.items():
            if '.lora_A.' not in key:continue
            partner=key.replace('.lora_A.','.lora_B.')
            if partner not in header or len(item['shape'])!=2 or len(header[partner]['shape'])!=2 or item['shape'][0]!=rank or header[partner]['shape'][1]!=rank:raise ValueError('LoRA pair shape/rank mismatch')
            match=re.search(r'layers\.(\d+)\.',key)
            if not match:raise ValueError('Adapter target lacks decoder layer number')
            layer=int(match.group(1));layers.add(layer)
            if keep is not None and layer>=keep:raise ValueError('Adapter contains a layer outside source truncated trunk')
            if base:
                cfg_base=read_json(base/'config.json');cfg_base=cfg_base.get('text_config',cfg_base);H=cfg_base['hidden_size'];I=cfg_base['intermediate_size'];hd=cfg_base.get('head_dim',H//cfg_base['num_attention_heads']);Q=cfg_base['num_attention_heads']*hd;KV=cfg_base.get('num_key_value_heads',cfg_base['num_attention_heads'])*hd
                module=re.search(r'layers\.\d+\.(.+)\.lora_A\.',key).group(1)
                widths={'self_attn.q_proj':(H,Q),'self_attn.k_proj':(H,KV),'self_attn.v_proj':(H,KV),'self_attn.o_proj':(Q,H),'mlp.gate_proj':(H,I),'mlp.up_proj':(H,I),'mlp.down_proj':(I,H)}
                if module in widths:
                    inp,out=widths[module]
                    if item['shape']!=[rank,inp] or header[partner]['shape']!=[out,rank]:raise ValueError('LoRA matrices differ from selected base projection: '+module)
                    dimension_checks+=1
                else:unresolved_targets.append(module)
            count+=1
        if count==0:raise ValueError('No valid LoRA A/B pairs in adapter')
        return {'config_sha256':self.hash(p/'adapter_config.json'),'adapter_sha256':self.hash(p/'adapter_model.safetensors'),'adapter_bytes':(p/'adapter_model.safetensors').stat().st_size,'rank':rank,'alpha':alpha,'pairs':count,'layer_indices':sorted(layers),'tensor_header_count':len(header),'selected_base_projection_dimension_checks':dimension_checks,'unresolved_base_projection_targets':sorted(set(unresolved_targets)),'tensor_payload_finiteness_verified':False}
    def checkpoint(self,p,name):
        family=next((f for f,names in CONTRACT['main_adapter_families'].items() if name in names),None)
        hidden=5120 if family in ['A14','AQ14V5'] else 3584 if family=='A25' else 4096
        active=family in ['AQ8V5','AQ14V5','ALL8V5'];keep=27 if family=='AQ14V5' else 24 if active else None
        adp=p/'adapter' if (p/'adapter/adapter_config.json').is_file() else p
        base_name='B14' if family in ['A14','AQ14V5'] else 'B25' if family=='A25' else 'BL' if family in ['AL','ALL8V5'] else 'B8'
        base_selected=self.selected.get(base_name);base=base_selected.parent if base_selected is not None else None
        adapter=self.adapter_pairs(adp,hidden,keep,base)
        head=p/'heads.pt';archive=torch_zip_inventory(head);archive['sha256']=self.hash(head)
        schema_verified=archive.get('tensor_schema_inspected',False)
        if schema_verified:
            tensors=archive['tensors'];norm_indices=sorted(int(re.fullmatch(r'norms\.(\d+)\.weight',key).group(1)) for key in tensors if re.fullmatch(r'norms\.(\d+)\.weight',key));norm_count=len(norm_indices)
            if not norm_count or norm_indices!=list(range(norm_count)) or (active and norm_count!=8):raise ValueError('Serialized head norm-count contract differs')
            expected={f'{prefix}.{field}':([18,hidden*norm_count] if field=='weight' else [18]) for prefix in ['plain','centered'] for field in ['weight','bias']}
            for i in range(norm_count):
                for field in ['weight','bias']:expected[f'norms.{i}.{field}']=[hidden]
            for key,shape in expected.items():
                if key not in tensors or tensors[key]['shape']!=shape:raise ValueError('Serialized head static tensor schema mismatch: '+name+'/'+key)
            archive['required_head_schema_verified_statically']=True
        else:self.warnings.append('Torch archive integrity/hash verified but tensor metadata not parsed: '+self.rel(head)+'; '+archive.get('limitation',''))
        alignment=None
        if (p/'meta.json').is_file():
            meta=read_json(p/'meta.json')
            if 'labels' in meta:alignment=labels_and_permutation(meta['labels'])
        if alignment is None:self.warnings.append('Source assumes canonical label rows without named meta.json: '+self.rel(p))
        return {'path':self.rel(p),'family':family,'active_v5_member':active,'hidden_size_expected':hidden,'pool_layers_expected':[2,4,6,8,12,16,20,24] if active else 'Legacy initialization selector; no invented inactive pooling recipe','head':archive,'adapter':adapter,'label_alignment':alignment,'serialized_head_loaded':False}
    def nishant(self,p,name,active=False):
        meta=read_json(p/'meta.json');cfg=meta['cfg'];alignment=labels_and_permutation(meta['labels']);keep=cfg.get('keep');layers=cfg.get('layers');hidden=5120 if name in ['lora_qwen3_14b','lora_qwen25_14b','lora14b_v1'] else 4096
        if not isinstance(keep,int) or keep<=0 or not isinstance(layers,list) or not layers or not all(isinstance(n,int) and 0<n<=keep for n in layers) or not isinstance(cfg.get('max_len'),int) or cfg['max_len']<=1:raise ValueError('Nishant pooling/depth/token metadata invalid')
        if active and cfg.get('eos_id')!=151643:raise ValueError('Active part-two training EOS must be151643')
        head=self.header(p/'heads.safetensors')
        for prefix in ['plain','centered']:
            for field,shape in [('weight',[18,hidden*len(layers)]),('bias',[18])]:
                if head.get(prefix+'.'+field,{}).get('shape')!=shape:raise ValueError('Nishant head class/feature dimensions differ')
        for i in range(len(layers)):
            for field in ['weight','bias']:
                if head.get(f'norms.{i}.{field}',{}).get('shape')!=[hidden]:raise ValueError('Nishant normalization axis differs')
        return {'path':self.rel(p),'metadata_sha256':self.hash(p/'meta.json'),'cfg':cfg,'label_alignment':alignment,'head_sha256':self.hash(p/'heads.safetensors'),'head_tensor_count':len(head),'adapter':self.adapter_pairs(p/'adapter',hidden,keep),'active_main_member':active}
    def deberta(self,p):
        meta=read_json(p/'meta.json');alignment=labels_and_permutation(meta['labels']);model=self.model(p/'encoder');hidden=model['hidden_size'];head=self.header(p/'head.safetensors')
        if head.get('weight',{}).get('shape')!=[18,hidden] or head.get('bias',{}).get('shape')!=[18] or not isinstance(meta.get('max_len'),int) or meta['max_len']<=0:raise ValueError('Adapted DeBERTa head/token schema differs')
        return {'path':self.rel(p),'model':model,'label_alignment':alignment,'max_len':meta['max_len'],'metadata_sha256':self.hash(p/'meta.json'),'head_sha256':self.hash(p/'head.safetensors')}
    def atlas(self,p):
        cfg=read_json(p/'atlas_bundle.json')
        if not isinstance(cfg.get('training_config'),dict) or cfg['training_config'].get('max_length',0)<=0:raise ValueError('ATLAS config missing')
        files=[]
        for relative in ['ntl_atlas/model.py','ntl_rlcd/data.py']:
            if not (p/relative).is_file():raise ValueError('ATLAS imported package source missing')
        for package in ['ntl_atlas','ntl_rlcd']:
            for f in sorted((p/package).rglob('*.py')):files.append(self.export_text(f,'ATLAS local custom code; no imports'))
        adapter=p/'adapters.safetensors';return {'path':self.rel(p),'config_sha256':self.hash(p/'atlas_bundle.json'),'adapter_sha256':self.hash(adapter),'adapter_tensor_count':len(self.header(adapter)),'private_code_exports':files}
    def comp_selector(self,spec):
        if spec.get('required') is False:return {'active':False,'source_reason':spec.get('reason'),'weights_or_head_required':False}
        if spec['name']=='ultrabert':return self.ultrabert(spec)
        p=self.resolve(spec,True)
        if p is None:return {'status':'SOURCE_ALLOWED_MISSING_MEMBER_SKIP','member':spec['name'],'emitted_in_our_source_stack':False}
        if p.name=='config.json':return self.model(p)
        if p.name=='COMPLETE.json':
            ck=p.parent;hidden=spec['trunk_spec'][1];layers=spec['pool_layers']
            header_inventory=npz_header_inventory(ck/'readouts.npz')
            required={'classes','pool_layers','plain_coef','plain_intercept'}
            if spec['name'] in ['lora24_public','lora24_claude']:required.update(['center_coef','center_intercept'])
            if 'claude_coef' in header_inventory:required.update(['claude_coef','claude_intercept'])
            state=state_inventory(ck/'readouts.npz',required,allow_unused_object_metadata=True)
            with np.load(ck/'readouts.npz',allow_pickle=False) as z:
                if z['classes'].tolist()!=LABELS or z['pool_layers'].tolist()!=layers:raise ValueError('Complement class/layer axes differ')
            width=hidden*len(layers)
            shapes=[('plain_coef',[18,width]),('plain_intercept',[18])]
            if spec['name'] in ['lora24_public','lora24_claude']:shapes.extend([('center_coef',[18,width]),('center_intercept',[18])])
            if 'claude_coef' in header_inventory:shapes.extend([('claude_coef',[5,width]),('claude_intercept',[5])])
            for key,shape in shapes:
                if state.get(key,{}).get('shape')!=shape:raise ValueError('Complement readout dimensions differ')
                if np.dtype(state[key]['dtype']).kind not in 'fiu':raise ValueError('Source-consumed readout must be real numeric: '+key)
            norms=self.header(ck/spec['norms_file']);self.tokenizer(ck/'tokenizer')
            for i in range(len(layers)):
                for field in ['weight','bias']:
                    if norms.get(f'norms.{i}.{field}',{}).get('shape')!=[hidden] and norms.get(f'{i}.{field}',{}).get('shape')!=[hidden]:raise ValueError('Complement norm shape missing/different')
            anchor=None
            if (ck/spec['anchor_file']).is_file():
                anchor=state_inventory(ck/spec['anchor_file'],['features'],allow_unused_object_metadata=True)
                if anchor['features']['shape']!=[3,width] or np.dtype(anchor['features']['dtype']).kind not in 'fiu':raise ValueError('Source-consumed parity anchor features dimensions/dtype differ')
            return {'path':self.rel(ck),'complete_sha256':self.hash(p),'readouts_sha256':self.hash(ck/'readouts.npz'),'norms_sha256':self.hash(ck/spec['norms_file']),'adapter':self.adapter_pairs(ck/'adapter',hidden,spec['trunk_spec'][2]),'state':state,'source_consumed_readout_members':sorted(required),'unused_object_metadata_members':[key for key,value in state.items() if value.get('inspection')=='HEADER_ONLY_UNUSED_OBJECT_METADATA'],'anchor_state':anchor,'anchor_present':anchor is not None,'object_payloads_loaded':False}
        if p.suffix=='.pt':return {'path':self.rel(p),'sha256':self.hash(p),'archive':torch_zip_inventory(p),'serialized_checkpoint_loaded':False}
        raise ValueError('Unrecognised complement asset selector')
    def ultrabert(self,spec):
        candidates=ultrabert_roots_in_source_order(self.matches(spec['glob']))
        self.selection_evidence['comp_ultrabert']={'candidate_paths_in_source_order':[self.rel(p) for p,_ in candidates],'source_would_choose':self.rel(candidates[0][0]) if candidates else None}
        if len(candidates)!=1:raise ValueError('Missing/ambiguous UltraBERT source root')
        root,folds=candidates[0];records=[]
        for fold in folds:
            cfg=read_json(fold/'config.json')
            if cfg.get('id2label'):labels_and_permutation([cfg['id2label'][str(i)] for i in range(18)])
            records.append({'model':self.model(fold),'private_code_exports':[self.export_text(f,'UltraBERT trust_remote_code source; no imports') for f in sorted(fold.glob('*.py'))]})
        return {'path':self.rel(root),'folds':records,'custom_code_executed':False}
    def extra_csvs(self):
        records=[]
        for pattern,label,finish in CONTRACT['contracts']['extra_csv_sources']:
            for p in self.matches('/kaggle/input/'+pattern,True):
                f=pd.read_csv(p)
                if not {'text','prompt_id',label}.issubset(f):raise ValueError('Complement extra CSV schema differs: '+self.rel(p))
                records.append({'path':self.rel(p),'sha256':self.hash(p),'rows':len(f),'label_column':label,'finish_filter_column':finish,'nonempty_allowed_label_rows':int(f[label].isin(LABELS).sum())})
        return {'source_csvs':records,'original_rows_exported':False,'provider_or_prompt_independence_certified':False}
    def wheels(self):
        records=[]
        for p in self.matches('/kaggle/input/**/bitsandbytes-*.whl'):
            if not zipfile.is_zipfile(p):raise ValueError('Malformed attached offline wheel')
            with zipfile.ZipFile(p) as z:
                metadata=[n for n in z.namelist() if n.endswith('.dist-info/METADATA')]
                if len(metadata)!=1 or z.getinfo(metadata[0]).file_size>1024*1024:raise ValueError('Wheel metadata invalid')
                fields={l.split(': ',1)[0]:l.split(': ',1)[1] for l in z.read(metadata[0]).decode().splitlines() if ': ' in l and l.split(': ',1)[0] in ['Name','Version','Requires-Python']}
            records.append({'path':self.rel(p),'sha256':self.hash(p),'bytes':p.stat().st_size,'metadata':fields})
        return {'source_installs_all_glob_matches':True,'matches':records,'wheels_installed_or_imported':False,'source_allows_empty_match':True}
    def resources(self):
        packages={}
        for name in ['numpy','pandas','pyarrow','torch','transformers','bitsandbytes','accelerate','safetensors','scikit-learn','scipy','joblib','lightgbm']:
            try:packages[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:packages[name]=None
        memory={}
        if Path('/proc/meminfo').is_file():memory={l.split(':')[0]:l.split(':')[1].strip() for l in Path('/proc/meminfo').read_text().splitlines() if l.split(':')[0] in ['MemTotal','MemAvailable']}
        cgroup={}
        for name in ['/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current']:
            if Path(name).is_file():cgroup[name]=Path(name).read_text().strip()
        disk={}
        for name in ['/tmp','/kaggle/temp','/kaggle/working']:
            if Path(name).exists():
                usage=shutil.disk_usage(name);disk[name]={'total_bytes':usage.total,'free_bytes':usage.free}
        return {'python':sys.version,'platform':platform.platform(),'packages_by_distribution_metadata_only':packages,'memory':memory,'cgroup_memory':cgroup,'disk':disk,'cpu_count':os.cpu_count(),'gpu_probe_run':False,'source_preview_seconds_approx':1.46*3600,'source_atlas_budget_formula':'int(8.3*3600 - complement_seconds -600)','full_hidden_runtime_verified':False}
    def execute(self):
        self.check('official_train_identity',self.official)
        for spec in CONTRACT['main_selectors']:self.check('main_'+spec['name'],lambda spec=spec:self.main_selector(spec))
        for spec in CONTRACT['complement_selectors']:self.check('comp_'+spec['name'],lambda spec=spec:self.comp_selector(spec))
        self.check('complement_extra_csv_inventory',self.extra_csvs);self.check('offline_bitsandbytes_wheels',self.wheels);self.check('cpu_environment_resources',self.resources)
        failed=[c for c in self.checks if c['status']!='PASS'];self.out.mkdir(parents=True,exist_ok=True)
        export={'scope':'PRIVATE actual attached custom code/config text; no imports or execution','files':self.code};private=self.out/'PRIVATE_CODE_EXPORT.json';private.write_text(json.dumps(export,indent=2)+'\n')
        schemas=[v.get('head',{}) for k,v in self.details.items() if k.startswith('main_') and isinstance(v,dict) and 'head' in v]
        report={'stage':12,'source_version':SOURCE_VERSION,'source_capture_json_sha256':CONTRACT['capture_json_sha256'],'source_manifest_sha256':CONTRACT['source_manifest_sha256'],'audit_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'status':'PASS_CPU_INPUT_CONTRACT_ONLY' if not failed else 'BLOCKED_INPUT_CONTRACT','passed':len(self.checks)-len(failed),'total':len(self.checks),'copied_notebook_preparation_version':COPIED_NOTEBOOK_VERSION,'copied_input_ids_from_UI_freeze':COPIED_INPUT_IDS,'input_version_reconciliation_scope':'Own copied UI IDs frozen; source author original input IDs not independently exported. Runtime identities are recorded separately by paths and hashes.','main_selectors_total':53,'main_serialized_head_selectors_total':37,'checks':self.checks,'details':self.details,'selection_evidence':self.selection_evidence,'warnings':self.warnings,'critical_file_hashes':{self.rel(p):digest for p,digest in self.hashes.items() if p.is_relative_to(self.root)},'hash_elapsed_seconds':time.monotonic()-self.started,'hash_budget_seconds':self.seconds,'private_code_export_files':len(self.code),'private_code_export_sha256':hashlib.sha256(private.read_bytes()).hexdigest(),'serialized_head_archives_inspected':len(schemas),'serialized_head_static_tensor_schemas_inspected':sum(x.get('tensor_schema_inspected',False) for x in schemas),'Torch_custom_code_or_upstream_imported':False,'pickle_joblib_Torch_load_called':False,'training_or_inference_started':False,'test_rows_or_predictions_exported':False,'hidden_runtime_verified':False,'limitations':['CPU file/metadata checks are not GPU model inference or a new official score.','Torch ZIP checks never call an unpickler or reducer. Unsupported serialized syntax is archive-only evidence and explicitly reported, not certified tensor behavior.','Safetensors headers do not verify finite payload values. Vendor model payloads are not fully hashed; prior configuration/index/inventory matches are recorded as such.','The source has no complete expectedSHA256 input pins. Actual critical-file hashes are receipts, not proof of equality to an unpublished author hash.','A missing legacy complement member may be an allowed sourceSKIP; actual membership is explicitly reported and must be compared with source execution evidence.','Source fullpreview costs about1.46hours; a CPU pass cannot establish the full hidden test fits the9hour execution limit.','Final/fold weights may include official labels; assigningfoldsafterwarddoesnotcreatehonestOOF.']}
        (self.out/'INPUT_AUDIT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('STAGE12_CPU_AUDIT_SUMMARY',json.dumps({k:report[k] for k in ['status','passed','total','serialized_head_archives_inspected','serialized_head_static_tensor_schemas_inspected','training_or_inference_started']}),flush=True);return report

def self_test():
    tests={}
    def accepts(name,fn):
        try:fn();tests[name]=True
        except Exception:tests[name]=False
    def rejects(name,fn):
        try:fn();tests[name]=False
        except Exception:tests[name]=True
    for value in ['/absolute','../outside','x/../outside']:rejects('unsafe_path_'+str(len(tests)),lambda value=value:safe_relative(value))
    accepts('canonical_label_order',lambda:labels_and_permutation(list(LABELS)))
    accepts('reversed_native_label_order',lambda:labels_and_permutation(list(reversed(LABELS))))
    rejects('duplicate_label',lambda:labels_and_permutation(LABELS[:-1]+[LABELS[0]]))
    rejects('unknown_label',lambda:labels_and_permutation(LABELS[:-1]+['unknown']))
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);a=Audit(root,root/'out',900)
        rejects('missing_required_selector',lambda:a.resolve({'name':'x','glob':'/kaggle/input/**/heads.pt','required':True,'ordering':'UNSORTED'}))
        accepts('allowed_missing_complement_skip',lambda:a.resolve({'name':'x','glob':'/kaggle/input/**/absent/COMPLETE.json','source_allows_missing_member_skip':True,'ordering':'sorted'}) is None or (_ for _ in ()).throw(ValueError()))
        for name in ['a','b']:(root/name).mkdir();(root/name/'heads.pt').write_bytes(b'fixture')
        rejects('ambiguous_unreviewed_selector',lambda:a.resolve({'name':'x','glob':'/kaggle/input/**/heads.pt','required':True,'ordering':'UNSORTED'}))
        bad=root/'bad.safetensors';bad.write_bytes(struct.pack('<Q',1000)+b'{}');rejects('truncated_safetensors',lambda:safetensor_header(bad))
        np.savez(root/'object.npz',x=np.asarray([{'bad':True}],dtype=object));rejects('npz_pickle_object',lambda:state_inventory(root/'object.npz'))
        np.savez(root/'nan.npz',x=np.asarray([np.nan]));rejects('npz_nonfinite',lambda:state_inventory(root/'nan.npz'))
        np.savez(root/'unused_metadata.npz',features=np.ones((3,8)),ids=np.asarray(['one','two','three'],dtype=object))
        def unused_object_header_only():
            record=state_inventory(root/'unused_metadata.npz',['features'],allow_unused_object_metadata=True)
            if record['ids']['inspection']!='HEADER_ONLY_UNUSED_OBJECT_METADATA' or record['ids']['payload_loaded'] is not False or record['features']['shape']!=[3,8]:raise ValueError('Unused object metadata was not safely excluded')
        accepts('unused_object_metadata_header_inventory_without_payload',unused_object_header_only)
        rejects('source_consumed_object_member_still_blocks',lambda:state_inventory(root/'unused_metadata.npz',['ids'],allow_unused_object_metadata=True))
        rejects('missing_source_consumed_npz_member_still_blocks',lambda:state_inventory(root/'unused_metadata.npz',['absent'],allow_unused_object_metadata=True))
        np.savez(root/'bad_features.npz',features=np.full((3,8),np.nan),ids=np.asarray(['one'],dtype=object))
        rejects('nonfinite_consumed_numeric_member_still_blocks',lambda:state_inventory(root/'bad_features.npz',['features'],allow_unused_object_metadata=True))
        for model_name in ['8b','8b-base']:
            folder=root/'models/qwen-3'/model_name/'1';folder.mkdir(parents=True);(folder/'config.json').write_text('{}')
        def source_string_sort_precedes_Path_conversion():
            hits=a.matches('/kaggle/input/models/qwen-3/**/config.json',ordered=True)
            expected=sorted([str(root/'models/qwen-3'/name/'1/config.json') for name in ['8b','8b-base']])
            if [str(p) for p in hits]!=expected or '/8b-base/' not in str(hits[0]):raise ValueError('Source string sorting differs from audited selection')
        accepts('source_string_sort_selects_8b_base_before_8b',source_string_sort_precedes_Path_conversion)
        # Primitive mapping opcode fixture, parsed without a pickle loader.
        accepts('static_primitive_pickle_metadata',lambda:static_torch_metadata(b'\x80\x02}q\x00X\x01\x00\x00\x00xq\x01K\x03s.',{}))
        rejects('unsupported_reducer_not_executed',lambda:static_torch_metadata(b'\x80\x02cos\nsystem\nX\x04\x00\x00\x00oops\x85R.',{}))
        archive=root/'head.pt'
        with zipfile.ZipFile(archive,'w') as z:z.writestr('archive/data.pkl',b'\x80\x02}q\x00X\x01\x00\x00\x00xq\x01K\x03s.')
        accepts('torch_zip_integrity_metadata_only',lambda:torch_zip_inventory(archive))
        def text(value):
            raw=value.encode();return b'X'+struct.pack('<I',len(raw))+raw
        storage=b'('+text('storage')+b'ctorch\nFloatStorage\n'+text('0')+text('cpu')+b'K\x24tQ'
        tensor_pickle=b'\x80\x02}'+text('plain.weight')+b'ctorch._utils\n_rebuild_tensor_v2\n('+storage+b'K\x00K\x12K\x02\x86K\x02K\x01\x86\x89ccollections\nOrderedDict\n)RtRs.'
        tensor_archive=root/'tensorhead.pt'
        with zipfile.ZipFile(tensor_archive,'w') as z:z.writestr('archive/data.pkl',tensor_pickle);z.writestr('archive/data/0',bytes(36*4))
        def valid_static_tensor():
            record=torch_zip_inventory(tensor_archive)
            if not record['tensor_schema_inspected'] or record['tensors']['plain.weight']['shape']!=[18,2]:raise ValueError('Static Torch metadata tensor fixture differs')
        accepts('torch_tensor_shape_without_unpickle',valid_static_tensor)
        rejects('torch_storage_extent_mismatch',lambda:static_torch_metadata(tensor_pickle,{'0':1}))
        badzip=root/'badhead.pt'
        with zipfile.ZipFile(badzip,'w') as z:z.writestr('archive/../escape',b'x');z.writestr('archive/data.pkl',b'\x80\x02}.')
        rejects('torch_zip_path_escape',lambda:torch_zip_inventory(badzip))
        six=root/'six/ultrabert-380m-ntl';five=root/'five/ultrabert-380m-ntl'
        for folder,count in [(six,6),(five,5)]:
            for i in range(count):
                fold=folder/f'fold{i}';fold.mkdir(parents=True);(fold/'config.json').write_text(json.dumps({'model_type':'ultrabert'}))
        def exactly_five_candidate():
            candidates=ultrabert_roots_in_source_order([six,five])
            if len(candidates)!=1 or candidates[0][0]!=five or len(candidates[0][1])!=5:raise ValueError('Source five-fold selector differs')
        accepts('ultrabert_six_fold_root_skipped_for_valid_five_fold_root',exactly_five_candidate)
        def six_is_not_eligible():
            if ultrabert_roots_in_source_order([six]):raise ValueError('Six-fold root was incorrectly eligible')
        accepts('ultrabert_requires_exactly_five_not_at_least_five',six_is_not_eligible)
        a.deadline=time.monotonic()-1
        rejects('critical_hash_budget_is_bounded',lambda:a.hash(root/'a/heads.pt'))
    result={'status':'PASS' if all(tests.values()) else 'FAIL','passed':sum(tests.values()),'total':len(tests),'checks':tests,'upstream_or_model_executed':False,'pickle_or_Torch_loader_called':False}
    if not all(tests.values()):raise AssertionError(json.dumps(result))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='/kaggle/input');p.add_argument('--out',default='/kaggle/working/stage12_preflight');p.add_argument('--seconds',type=int,default=900);p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:print(json.dumps(self_test(),indent=2))
    else:
        report=Audit(a.root,a.out,a.seconds).execute();print('STAGE12_INPUT_AUDIT_JSON_BEGIN');print(json.dumps(report,indent=2,allow_nan=False));print('STAGE12_INPUT_AUDIT_JSON_END')
