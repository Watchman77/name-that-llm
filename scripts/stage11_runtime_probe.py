"""Bounded diagnostic of the frozen V28 full-text Qwen branch only.

Default admission is locked. This is not a competition submission, an accuracy
experiment or a complete ensemble runtime check. Frozen worker text is embedded
for reproducibility; local build/tests never import or execute those workers.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import numpy as np
import pandas as pd


SOURCE_VERSION = 362014737
SOURCE_CAPTURE_SHA256 = '143351168ff87d0cd0f2712c29778dfad8acd34c1527088a19d4b42141869413'
FROZEN_MODULES = {'stage_dataset_weights.py': '"""Sequential, explicit-file Dataset staging for local random reads."""\nimport json\nimport shutil\nimport tempfile\nimport time\nfrom contextlib import contextmanager\nfrom pathlib import Path\n\nNF4_FILES = {\'LICENSE\': 11544, \'README.md\': 27108, \'chat_template.jinja\': 8952, \'config.json\': 4222, \'generation_config.json\': 202, \'merges.txt\': 3353259, \'model-00001-of-00007.safetensors\': 2542796928, \'model-00002-of-00007.safetensors\': 3967729339, \'model-00003-of-00007.safetensors\': 3973498949, \'model-00004-of-00007.safetensors\': 3992447326, \'model-00005-of-00007.safetensors\': 4000148996, \'model-00006-of-00007.safetensors\': 98537568, \'model-00007-of-00007.safetensors\': 849400424, \'model.safetensors.index.json\': 396308, \'tokenizer.json\': 12809320, \'tokenizer_config.json\': 17928, \'vocab.json\': 6722759}\nULTRA_FOLD_FILES = {0: {\'config.json\': 2739, \'configuration_ultrabert.py\': 2209, \'model.safetensors\': 755137636, \'modeling_ultrabert.py\': 30581, \'tokenizer.json\': 3523731, \'tokenizer_config.json\': 814}, 1: {\'config.json\': 2739, \'configuration_ultrabert.py\': 2209, \'model.safetensors\': 755137636, \'modeling_ultrabert.py\': 30581, \'tokenizer.json\': 3523731, \'tokenizer_config.json\': 814}, 2: {\'config.json\': 2739, \'configuration_ultrabert.py\': 2209, \'model.safetensors\': 755137636, \'modeling_ultrabert.py\': 30581, \'tokenizer.json\': 3523731, \'tokenizer_config.json\': 814}, 3: {\'config.json\': 2739, \'configuration_ultrabert.py\': 2209, \'model.safetensors\': 755137636, \'modeling_ultrabert.py\': 30581, \'tokenizer.json\': 3523731, \'tokenizer_config.json\': 814}, 4: {\'config.json\': 2739, \'configuration_ultrabert.py\': 2209, \'model.safetensors\': 755137636, \'modeling_ultrabert.py\': 30581, \'tokenizer.json\': 3523731, \'tokenizer_config.json\': 814}}\n\n@contextmanager\ndef staged_dataset_tree(source, files, *, label, input_root=\'/kaggle/input\', temp_root=\'/tmp\'):\n    source = Path(source)\n    input_root = Path(input_root)\n    temp_root = Path(temp_root)\n    assert source.is_dir() and source.resolve().is_relative_to(input_root.resolve())\n    assert label and label.replace(\'-\', \'\').isalnum()\n    assert files and all(\'/\' not in name and \'..\' not in name and size > 0\n                         for name, size in files.items())\n    total = sum(files.values())\n    for name, size in files.items():\n        item = source / name\n        assert item.is_file() and item.stat().st_size == size, (label, name, size)\n    free = shutil.disk_usage(temp_root).free\n    assert free > total + 2 * 2**30, (\'Insufficient /tmp disk for explicit staging\', label, total, free)\n    destination = Path(tempfile.mkdtemp(prefix=\'ntl-stage-\' + label + \'-\', dir=temp_root))\n    assert destination.resolve().is_relative_to(temp_root.resolve())\n    start = time.monotonic()\n    try:\n        for name, size in files.items():\n            target = destination / name\n            with (source / name).open(\'rb\') as reader, target.open(\'wb\') as writer:\n                while block := reader.read(16 * 1024 * 1024):\n                    writer.write(block)\n            assert target.stat().st_size == size\n            print(json.dumps({\'stage\': label, \'file\': name, \'bytes\': size,\n                              \'elapsed_seconds\': round(time.monotonic() - start, 2)}), flush=True)\n        yield destination\n    finally:\n        assert destination.resolve().is_relative_to(temp_root.resolve())\n        shutil.rmtree(destination)\n        print(json.dumps({\'stage\': label, \'event\': \'STAGING_CLEANED\'}), flush=True)\n', 'qwen_nf4_loader.py': '"""Read-only inference for the trained full64 NF4 Qwen classifier.\n\nThe quantized backbone is frozen. No training data, optimizer or fitting is used.\n"""\nimport os\nos.environ.setdefault(\'TOKENIZERS_PARALLELISM\', \'false\')\nimport argparse, gc, hashlib, json, math, struct, time\nfrom pathlib import Path\nimport numpy as np\nimport torch\nfrom torch import nn\nfrom torch.nn import functional as F\nfrom accelerate.utils import set_module_tensor_to_device\nfrom bitsandbytes.functional import QuantState, dequantize_4bit\nfrom safetensors.torch import load_file\nfrom transformers import AutoTokenizer\nfrom transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig\nfrom transformers.models.qwen3_5 import modeling_qwen3_5 as qmodel\n\nclass HostEmbedding(nn.Module):\n    def __init__(self, weight):\n        super().__init__(); self.weight_cpu = weight\n    def forward(self, ids):\n        return F.embedding(ids.cpu(), self.weight_cpu).to(ids.device)\n\nclass NF4InferenceLinear(nn.Module):\n    def __init__(self, packed, state, shape, rank, device, resident):\n        super().__init__()\n        self.packed = packed.to(device) if resident else packed\n        self.state, self.compute_device = state, device\n        self.in_features, self.out_features = shape\n        if rank:\n            self.lora_A = nn.Parameter(torch.empty(rank, shape[0], device=device, dtype=torch.float32), requires_grad=False)\n            self.lora_B = nn.Parameter(torch.empty(shape[1], rank, device=device, dtype=torch.float32), requires_grad=False)\n        else:\n            self.register_parameter(\'lora_A\', None); self.register_parameter(\'lora_B\', None)\n    def forward(self, value):\n        original_device = value.device\n        value = value.to(self.compute_device)\n        weight = dequantize_4bit(self.packed.to(self.compute_device), self.state).to(value.dtype)\n        base = F.linear(value, weight)\n        del weight\n        if self.lora_A is not None:\n            with torch.autocast(\'cuda\', dtype=value.dtype):\n                extra = F.linear(F.linear(value, self.lora_A), self.lora_B)\n            base = base + 2. * extra.to(base.dtype)\n        return base.to(original_device)\n\nclass NF4BackboneClassifier:\n    def __init__(self, base, adapter, *, portable=False, resident=False):\n        started = time.perf_counter()\n        self.base = Path(base); self.device = torch.device(\'cuda:0\')\n        self.dtype = torch.float16 if portable else torch.bfloat16\n        self.portable = portable\n        torch.set_num_threads(8)\n        conf = json.loads((self.base/\'config.json\').read_text())\n        cfg = Qwen3_5TextConfig(**conf[\'text_config\'])\n        assert cfg.num_hidden_layers == 64 and cfg.hidden_size == 5120\n        cfg._attn_implementation = \'sdpa\'; cfg.use_cache = False\n        with torch.device(\'meta\'): self.model = qmodel.Qwen3_5TextModel(cfg)\n        self.model.norm = nn.Identity()\n        self.index = json.loads((self.base/\'model.safetensors.index.json\').read_text())[\'weight_map\']\n        self.headers, self.maps = {}, {}\n        prefix = \'model.language_model.\'\n        ngpu = torch.cuda.device_count() if resident else 1\n        count = 0\n        for name, module in list(self.model.named_modules()):\n            if not isinstance(module, nn.Linear): continue\n            layer = int(name.split(\'.\')[1]); device = torch.device(f\'cuda:{layer*ngpu//64}\')\n            key = prefix+name+\'.weight\'\n            state = QuantState.from_dict({k:self.tensor(k) for k in self.index if k.startswith(key+\'.\')}, device=device)\n            assert tuple(state.shape) == (module.out_features, module.in_features)\n            assert module.bias is None\n            replacement = NF4InferenceLinear(self.tensor(key), state, (module.in_features, module.out_features), 16 if layer>=56 else 0, device, resident)\n            parent, _, attribute = name.rpartition(\'.\')\n            setattr(self.model.get_submodule(parent), attribute, replacement)\n            count += 1\n        self.model.embed_tokens = HostEmbedding(self.tensor(prefix+\'embed_tokens.weight\'))\n        self.model.rotary_emb = qmodel.Qwen3_5TextRotaryEmbedding(cfg, device=self.device)\n        for name, parameter in self.model.named_parameters():\n            if \'.lora_A\' in name or \'.lora_B\' in name: continue\n            value = self.tensor(prefix+name)\n            set_module_tensor_to_device(self.model, name, self.device, value=value, dtype=value.dtype)\n        assert not any(p.is_meta for p in self.model.parameters())\n        assert not any(b.is_meta for b in self.model.buffers())\n        if portable:\n            # The reference PyTorch implementation also runs on Turing/T4.\n            # No Ampere-only FLA Triton kernels are required by this route.\n            for layer in self.model.layers:\n                if hasattr(layer, \'linear_attn\'):\n                    layer.linear_attn.chunk_gated_delta_rule = qmodel.torch_chunk_gated_delta_rule\n                    layer.linear_attn.recurrent_gated_delta_rule = qmodel.torch_recurrent_gated_delta_rule\n                    layer.linear_attn.causal_conv1d_fn = None\n                    layer.linear_attn.causal_conv1d_update = None\n        self.depths = [8,16,24,32,40,48,56,64]\n        self.norms = nn.ModuleList([nn.LayerNorm(5120) for _ in self.depths]).to(self.device)\n        self.plain = nn.Linear(40960,18).to(self.device)\n        self.centered = nn.Linear(40960,18).to(self.device)\n        self.tokenizer = AutoTokenizer.from_pretrained(self.base, local_files_only=True)\n        self.tokenizer.pad_token_id=248044; self.tokenizer.padding_side=\'right\'\n        self.reload(adapter)\n        self.model.eval(); self.norms.eval(); self.plain.eval(); self.centered.eval()\n        print(json.dumps(dict(event=\'QWEN38_INFERENCE_LOADED\', seconds=time.perf_counter()-started, linears=count, layers=64, portable=portable, resident=resident, gpu_count=ngpu)),flush=True)\n\n    def tensor(self, key):\n        path = self.base/self.index[key]\n        if path not in self.headers:\n            with path.open(\'rb\') as f:\n                size=struct.unpack(\'<Q\',f.read(8))[0]\n                assert size<16*2**20\n                self.headers[path]=(8+size,json.loads(f.read(size)))\n            self.maps[path]=np.memmap(path,dtype=np.uint8,mode=\'r\')\n        offset, header=self.headers[path]; item=header[key]; lo,hi=item[\'data_offsets\']\n        dtype={\'U8\':torch.uint8,\'F16\':torch.float16,\'BF16\':torch.bfloat16,\'F32\':torch.float32,\'I64\':torch.int64}[item[\'dtype\']]\n        assert hi-lo==math.prod(item[\'shape\'])*torch.empty((),dtype=dtype).element_size()\n        return torch.frombuffer(self.maps[path][offset+lo:offset+hi],dtype=dtype).reshape(item[\'shape\'])\n\n    def reload(self, adapter):\n        adapter=Path(adapter)\n        metadata=json.loads(adapter.with_suffix(\'.json\').read_text())\n        assert metadata[\'native_decoder_layers\']==metadata[\'executed_decoder_layers\']==64\n        assert metadata[\'lora_target_layer_indices\']==list(range(56,64))\n        assert metadata[\'source_revision\']==\'26157380225e427827263c34df23018a1e28dff5\'\n        self.labels=metadata[\'labels\']; assert len(set(self.labels))==18\n        saved=load_file(str(adapter),device=\'cpu\')\n        wanted={f\'backbone.{n}\':p for n,p in self.model.named_parameters() if \'.lora_A\' in n or \'.lora_B\' in n}\n        for prefix,module in [(\'norms\',self.norms),(\'plain\',self.plain),(\'centered\',self.centered)]:\n            wanted.update({f\'{prefix}.{n}\':p for n,p in module.named_parameters()})\n        assert set(saved)==set(wanted),(set(saved)-set(wanted),set(wanted)-set(saved))\n        with torch.no_grad():\n            for key,param in wanted.items():\n                assert saved[key].shape==param.shape and torch.isfinite(saved[key]).all(),key\n                param.copy_(saved[key])\n        del saved\n', 'fulltext_pooling.py': '"""Keep eight pooled readouts, without retaining every layer\'s token states."""\nfrom contextlib import nullcontext\nimport torch\nfrom torch import nn\n\nclass MeanPoolHooks:\n    def __init__(self,model,depths,norms,attention_mask):\n        assert isinstance(model.norm,nn.Identity)\n        assert len(depths)==len(norms)\n        self.model,self.depths,self.norms=model,list(depths),norms\n        self.mask=attention_mask.unsqueeze(-1).float();self.den=self.mask.sum(1)\n        assert (self.den>0).all()\n        self.values,self.handles={},[]\n    def hook_for(self,depth,norm):\n        def hook(module,inputs,output):\n            assert isinstance(output,torch.Tensor) and output.ndim==3\n            assert output.shape[:2]==self.mask.shape[:2] and depth not in self.values\n            context=torch.autocast(\'cuda\',enabled=False) if output.is_cuda else nullcontext()\n            with context:self.values[depth]=norm((output.float()*self.mask).sum(1)/self.den)\n        return hook\n    def __enter__(self):\n        for depth,norm in zip(self.depths,self.norms):\n            assert 1<=depth<=len(self.model.layers)\n            self.handles.append(self.model.layers[depth-1].register_forward_hook(self.hook_for(depth,norm)))\n        return self\n    def __exit__(self,*args):\n        for handle in self.handles:handle.remove()\n        self.handles.clear()\n    def features(self):\n        assert set(self.values)==set(self.depths)\n        return torch.cat([self.values[depth] for depth in self.depths],dim=-1)\n', 'qwen_fulltext_inference.py': '"""Full-passage inference for the new NF4-trained own classifier.\n\nUse the SAME verified NF4 quantization as training, not the previous mixed-FP8\ncheckpoint. All 64 layers execute. Only the existing tail8 adapters are loaded.\nNo fitting, model discovery, cropping, network requests or fixed token budget.\n"""\nimport gc\nimport json\nimport time\nfrom pathlib import Path\nimport numpy as np\nimport torch\nfrom qwen_nf4_loader import NF4BackboneClassifier as NF4Classifier\nfrom fulltext_pooling import MeanPoolHooks\n\n\nclass FullTextClassifier(NF4Classifier):\n    def __init__(self,*args,**kwargs):\n        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False\n        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False\n        super().__init__(*args,**kwargs)\n\n    def reload(self,adapter):\n        metadata=json.loads(Path(adapter).with_suffix(\'.json\').read_text())\n        if metadata.get(\'text_truncation\') is not False or metadata.get(\'inference_max_length\',\'absent\') is not None:\n            raise ValueError(\'Expected the new full-text trained adapter, not the historical cropped checkpoint\')\n        super().reload(adapter)\n\n    @torch.inference_mode()\n    def predict(self,texts,batch_size=3,return_features=False):\n        if not texts:\n            raise ValueError(\'Empty prediction request\')\n        raw=self.tokenizer(list(texts),add_special_tokens=False,truncation=False)[\'input_ids\']\n        tokens=[ids+[248044] for ids in raw]\n        if any(len(ids)>self.model.config.max_position_embeddings for ids in tokens):\n            raise ValueError(\'Passage exceeds native context; do not silently crop it\')\n        lengths=np.asarray([len(ids) for ids in tokens])\n        order=np.argsort(lengths,kind=\'stable\')\n        output=np.empty((len(tokens),18),dtype=np.float32)\n        features=np.empty((len(tokens),40960),dtype=np.float32) if return_features else None\n\n        def batch_predict(rows):\n            selected=[tokens[i] for i in rows]\n            batch=self.tokenizer.pad({\'input_ids\':selected},padding=True,pad_to_multiple_of=64,return_tensors=\'pt\').to(self.device)\n            assert batch.attention_mask.sum(1).tolist()==[len(ids) for ids in selected]\n            with MeanPoolHooks(self.model,self.depths,self.norms,batch.attention_mask) as readouts:\n                with torch.autocast(\'cuda\',dtype=self.dtype):\n                    self.model(inputs_embeds=self.model.embed_tokens(batch.input_ids),attention_mask=batch.attention_mask,output_hidden_states=False,use_cache=False)\n                z=readouts.features()\n                value=self.plain(z)\n                if not torch.isfinite(value).all() or not torch.isfinite(z).all():\n                    raise FloatingPointError(\'Nonfinite full-text Qwen output\')\n                output[rows]=value.float().cpu().numpy()\n                if return_features:\n                    features[rows]=z.float().cpu().numpy()\n        for start in range(0,len(order),batch_size):\n            rows=order[start:start+batch_size]\n            began=time.perf_counter()\n            oom=False\n            try:\n                batch_predict(rows)\n            except torch.cuda.OutOfMemoryError:\n                oom=True\n            # Leave the exception scope first, so its traceback no longer owns\n            # the failed batch tensors when CUDA memory is reclaimed.\n            if oom:\n                gc.collect();torch.cuda.empty_cache()\n                for i in rows:\n                    batch_predict([i])\n            torch.cuda.synchronize()\n            print(json.dumps(dict(event=\'OWN_QWEN_FULLTEXT_BATCH\',done=min(start+batch_size,len(order)),total=len(order),seconds=time.perf_counter()-began,max_text_tokens=int(lengths[rows].max()-1),truncated=False)),flush=True)\n        return (output,features) if return_features else output\n', 'own_qwen_worker.py': '"""Run only the new own full-text expert, using explicit artifact bindings."""\nimport os\nos.environ.update(HF_HUB_OFFLINE=\'1\',TRANSFORMERS_OFFLINE=\'1\',TOKENIZERS_PARALLELISM=\'false\')\nimport argparse,json,time\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport torch\nfrom qwen_fulltext_inference import FullTextClassifier\n\ndef main():\n    p=argparse.ArgumentParser();p.add_argument(\'--query\',required=True);p.add_argument(\'--paths\',required=True);p.add_argument(\'--out\',required=True)\n    a=p.parse_args();paths=json.loads(Path(a.paths).read_text())\n    assert torch.cuda.device_count()==2,\'Resident full-text Qwen uses both T4 GPUs\'\n    frame=pd.read_csv(a.query,dtype={\'id\':str});texts=frame.text.fillna(\'\').tolist()\n    started=time.monotonic()\n    model=FullTextClassifier(paths[\'own_nf4_base\'],paths[\'own_fulltext_adapter\'],portable=True,resident=True)\n    logits=model.predict(texts,batch_size=3)\n    assert logits.shape==(len(frame),18) and np.isfinite(logits).all()\n    np.savez_compressed(a.out,ids=frame.id.to_numpy(dtype=str),labels=np.asarray(model.labels),logits=logits)\n    Path(a.out+\'.json\').write_text(json.dumps(dict(rows=len(frame),seconds=time.monotonic()-started,truncation=False,decoder_layers=64,adapter_layers=list(range(56,64)),peak_reserved_bytes=[torch.cuda.max_memory_reserved(i) for i in range(2)]),indent=2))\nif __name__==\'__main__\':main()\n'}
FROZEN_MODULE_SHA256 = {'stage_dataset_weights.py': '6694833b30ced8c2d18966164fa4152e03224634c027121edbb7736f80a6f2bd', 'qwen_nf4_loader.py': '06313ca9a045bd63eed437e93e597b956aa1f6c9d21473f2815669dfef4159be', 'fulltext_pooling.py': '78a8b4dcc231d707dc9b614a9c8c9f86ac3ab774aa8fb4ef69b537d779dc18f7', 'qwen_fulltext_inference.py': '3f7624cd33a72f9816fe463ef17ddf5fd42639f03d5e9dfba7ede6835c72557d', 'own_qwen_worker.py': 'a3e89bd76581983097b1be6630ddca06ac60ec99e3d9decdae9814e28b00ec89'}
SOURCE_PATHS = {'test_csv': '/kaggle/input/competitions/name-that-llm/test.csv', 'new_two_stage_root': '/kaggle/input/datasets/nishantdahal/ntl-handwriting-blend', 'qwen3_14b': '/kaggle/input/models/qwen-lm/qwen-3/transformers/14b-base/1', 'qwen25_14b': '/kaggle/input/models/qwen-lm/qwen2.5/transformers/14b/1', 'svc_prefit': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/ntl627-svc-prefit', 'qwen_wheels': ['/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/accelerate-1.15.0-py3-none-any.whl', '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/bitsandbytes-0.50.2-py3-none-manylinux_2_24_x86_64.whl', '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/huggingface_hub-1.32.0-py3-none-any.whl', '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/safetensors-0.8.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl', '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl', '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-offline-linux/transformers-5.14.1-py3-none-any.whl'], 'legacy_runtime': '/kaggle/temp/ntl636-fulltext-qwen-external3200-runtime/legacy', 'legacy_prefit': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/ntl611-prefit', 'complement_style': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/ntl636-updated-style', 'own_nf4_base': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/name-that-llm-artifacts/qwen38-27b-nf4', 'own_fulltext_adapter': '/kaggle/input/ntl-olmo2-native24-final-inference-assets/ntl627-own-fulltext/ntl627-own-fulltext/final.safetensors', 'night24_bank': '/kaggle/input/datasets/kirilltym/ntl-spectral-style-heads-paired-4404/night24_readout_banks/synthetic4404.joblib', 'night24_metadata': '/kaggle/input/datasets/kirilltym/ntl-spectral-style-heads-paired-4404/night24_readout_banks/COMPLETE.json'}
NF4_FILES = {'LICENSE': 11544, 'README.md': 27108, 'chat_template.jinja': 8952, 'config.json': 4222, 'generation_config.json': 202, 'merges.txt': 3353259, 'model-00001-of-00007.safetensors': 2542796928, 'model-00002-of-00007.safetensors': 3967729339, 'model-00003-of-00007.safetensors': 3973498949, 'model-00004-of-00007.safetensors': 3992447326, 'model-00005-of-00007.safetensors': 4000148996, 'model-00006-of-00007.safetensors': 98537568, 'model-00007-of-00007.safetensors': 849400424, 'model.safetensors.index.json': 396308, 'tokenizer.json': 12809320, 'tokenizer_config.json': 17928, 'vocab.json': 6722759}
LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra', 'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash', 'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5', 'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']


TIMEOUT_SECONDS = 900
SAMPLE_ROWS = 12
TOKEN_SELECTOR = r'''
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
base, train_csv, query_csv, metadata = sys.argv[1:]
f=pd.read_csv(train_csv,usecols=['id','text'],dtype={'id':str})
if len(f)!=2699 or not f.id.is_unique or f.id.isna().any():raise ValueError('Official training ID contract differs')
f['text']=f.text.fillna('').astype(str)
t=AutoTokenizer.from_pretrained(base,local_files_only=True,trust_remote_code=False)
raw=t(f.text.tolist(),add_special_tokens=False,truncation=False)['input_ids']
lengths=np.asarray([len(x)+1 for x in raw],dtype=np.int64)
if len(lengths)!=2699 or (lengths<=0).any():raise ValueError('Invalid exact untruncated token lengths')
order=np.argsort(lengths,kind='stable')
ranks=np.rint(np.linspace(0,len(order)-1,12)).astype(int)
chosen=np.sort(order[ranks])
if len(np.unique(chosen))!=12 or lengths[chosen].min()!=lengths.min() or lengths[chosen].max()!=lengths.max():raise ValueError('Quantile selection lost extremes')
f.iloc[chosen][['id','text']].to_csv(query_csv,index=False)
Path(metadata).write_text(json.dumps({'scope':'PRIVATE selected row positions, IDs and token lengths; no accuracy labels loaded','rows':12,'population_rows':2699,'selected_positions':chosen.tolist(),'ids':f.iloc[chosen].id.tolist(),'selected_token_lengths_including_end':lengths[chosen].tolist(),'population_token_quantiles':np.quantile(lengths,[0,.25,.5,.75,.9,.95,.99,1]).tolist(),'sum_population_tokens_including_end':int(lengths.sum()),'selection':'12 evenly spaced stable token-length ranks including minimum and maximum','labels_loaded':False,'truncation':False,'end_token_id':248044},indent=2))
print(json.dumps({'event':'PROBE_QUERY_SELECTED','rows':12,'population_rows':2699,'population_max_tokens':int(lengths.max()),'sample_max_tokens':int(lengths[chosen].max()),'labels_loaded':False}),flush=True)
'''


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def validate_admission(admission):
    if not isinstance(admission,dict):raise ValueError('Probe locked: an actual passing CPU audit receipt is required')
    if admission.get('source_version_id')!=SOURCE_VERSION:raise ValueError('Admission source version differs')
    if type(admission.get('cpu_audit_version_id')) is not int or admission['cpu_audit_version_id']<=0:raise ValueError('Actual CPU audit version ID is missing')
    if not re.fullmatch('[0-9a-f]{64}',str(admission.get('cpu_audit_report_sha256',''))):raise ValueError('CPU audit report SHA256 is missing')
    if admission.get('cpu_audit_status')!='PASS_CPU_INPUT_CONTRACT_ONLY':raise ValueError('CPU audit did not pass')
    if admission.get('mandatory_source_checks_passed')!=160 or admission.get('mandatory_source_checks_total')!=160:raise ValueError('All 160 source path checks must pass')
    return dict(admission)


def terminate_group(proc,grace=5.0):
    if proc.poll() is None:
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait(timeout=grace)
    # A surviving descendant could keep a pipe open after its parent exits.
    try:os.killpg(proc.pid,signal.SIGKILL)
    except ProcessLookupError:pass


def run_deadline(command,log_path,*,seconds,env=None,cwd=None,emit=True):
    if not seconds>0:raise ValueError('Positive deadline required')
    began=time.monotonic()
    with Path(log_path).open('w',encoding='utf-8',buffering=1) as log:
        proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,
                              errors='replace',bufsize=1,env=env,cwd=cwd,start_new_session=True)
        def tee():
            try:
                for line in proc.stdout:
                    log.write(line)
                    if emit:print(line,end='',flush=True)
            finally:proc.stdout.close()
        thread=threading.Thread(target=tee,daemon=True);thread.start()
        timed_out=False
        try:code=proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out=True;terminate_group(proc);code=proc.returncode
        finally:
            if proc.poll() is None:terminate_group(proc)
            thread.join(timeout=5)
            if thread.is_alive():
                terminate_group(proc);thread.join(timeout=5)
        return {'returncode':code,'timed_out':timed_out,'elapsed_seconds':time.monotonic()-began}


def decode_events(log_path):
    events=[]
    for line in Path(log_path).read_text(encoding='utf-8',errors='replace').splitlines():
        try:event=json.loads(line)
        except ValueError:continue
        if isinstance(event,dict):events.append(event)
    return events


def validate_outputs(npz_path,metadata_path,query_path):
    query=pd.read_csv(query_path,usecols=['id','text'],dtype={'id':str})
    with np.load(npz_path,allow_pickle=False) as z:
        if set(z.files)!={'ids','labels','logits'}:raise ValueError('Unexpected worker NPZ fields')
        ids=z['ids'];labels=z['labels'];logits=z['logits']
        if ids.dtype.hasobject or labels.dtype.hasobject or logits.dtype.hasobject:raise ValueError('Pickle/object arrays are not admitted')
        if ids.ndim!=1 or labels.ndim!=1 or ids.astype(str).tolist()!=query.id.tolist():raise ValueError('Worker ID order/coverage differs')
        if len(set(ids.astype(str)))!=len(query) or query.id.isna().any():raise ValueError('Worker IDs duplicate or missing')
        raw_labels=labels.astype(str).tolist()
        if len(raw_labels)!=18 or len(set(raw_labels))!=18 or set(raw_labels)!=set(LABELS):raise ValueError('Worker label set differs')
        permutation=[raw_labels.index(label) for label in LABELS]
        if logits.shape!=(len(query),18) or logits.dtype.kind not in 'fiu' or not np.isfinite(logits).all():raise ValueError('Nonfinite or invalid N-by18 logits')
        aligned_logits=logits[:,permutation].astype(np.float64)
        shifted=aligned_logits-aligned_logits.max(1,keepdims=True)
        p=np.exp(shifted);p/=p.sum(1,keepdims=True)
        if not np.isfinite(p).all() or (p<0).any() or not np.allclose(p.sum(1),1,atol=1e-10):raise ValueError('Invalid diagnostic softmax')
    meta=json.loads(Path(metadata_path).read_text())
    if meta.get('rows')!=len(query) or meta.get('truncation') is not False or meta.get('decoder_layers')!=64 or meta.get('adapter_layers')!=list(range(56,64)):raise ValueError('Worker full-depth/no-truncation metadata differs')
    peaks=meta.get('peak_reserved_bytes')
    if not isinstance(peaks,list) or len(peaks)!=2 or not all(type(x)is int and x>0 for x in peaks):raise ValueError('Expected memory peaks for two GPUs')
    if not isinstance(meta.get('seconds'),(int,float)) or not math.isfinite(meta['seconds']) or meta['seconds']<=0:raise ValueError('Invalid worker duration')
    return {'rows':len(query),'classes':18,'exact_id_order':True,'known_unique_label_set':True,
            'saved_worker_label_order':raw_labels,'canonical_label_order':LABELS,
            'canonical_permutation_from_saved':permutation,'saved_worker_array_rewritten':False,
            'finite_logits':True,'normalized_diagnostic_softmax':True,'worker_seconds':meta['seconds'],
            'decoder_layers':64,'adapter_layers':list(range(56,64)),'truncation':False,
            'peak_reserved_bytes':peaks,'npz_sha256':digest(npz_path),
            'metadata_sha256':digest(metadata_path),'original_ids_text_or_row_outputs_in_public_report':False}


def _phase(out,name):
    save_json(Path(out)/'CURRENT_PHASE.json',{'phase':name,'monotonic':time.monotonic()})


def _run_inner(runtime,out):
    """Runs only inside the Kaggle watchdog child after admission validation."""
    runtime=Path(runtime);out=Path(out);began=time.monotonic();metrics={}
    paths=dict(SOURCE_PATHS)
    for name,source in FROZEN_MODULES.items():
        if hashlib.sha256(source.encode()).hexdigest()!=FROZEN_MODULE_SHA256[name]:raise ValueError('Frozen source module checksum differs')
        (runtime/name).write_text(source,encoding='utf-8')
    env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
             HF_HUB_DISABLE_TELEMETRY='1',CUDA_VISIBLE_DEVICES='0,1')
    _phase(out,'offline_wheel_install')
    target=runtime/'qwen-python'
    installed=time.monotonic()
    subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps','--disable-pip-version-check',
                    '--target',str(target),*paths['qwen_wheels']],env=env,check=True)
    metrics['offline_wheel_install_seconds']=time.monotonic()-installed
    child_env=dict(env,PYTHONPATH=str(target))
    _phase(out,'CPU_token_length_selection')
    selector=runtime/'select_unlabeled_query.py';selector.write_text(TOKEN_SELECTOR,encoding='utf-8')
    query=out/'query_PRIVATE.csv';lengths=out/'token_lengths_PRIVATE.json'
    selected=time.monotonic()
    tokenizer_env=dict(child_env,CUDA_VISIBLE_DEVICES='',USE_TORCH='0',USE_TF='0',USE_FLAX='0')
    subprocess.run([sys.executable,'-u',str(selector),paths['own_nf4_base'],
                    '/kaggle/input/competitions/name-that-llm/train.csv',str(query),str(lengths)],
                   env=tokenizer_env,check=True,cwd=runtime)
    metrics['CPU_token_length_selection_seconds']=time.monotonic()-selected
    length_meta=json.loads(lengths.read_text())
    values=np.asarray(length_meta['selected_token_lengths_including_end'],dtype=np.int64)
    if len(values)!=SAMPLE_ROWS or (values<=0).any() or length_meta['labels_loaded']is not False:raise ValueError('Invalid unlabeled sample metadata')
    metrics['sample_token_length_quantiles']=np.quantile(values,[0,.25,.5,.75,1]).tolist()
    metrics['sample_total_tokens_including_end']=int(values.sum())
    metrics['population_token_length_quantiles']=length_meta['population_token_quantiles']
    metrics['population_rows']=length_meta['population_rows']
    metrics['selection']='12 evenly spaced stable token-length ranks including minimum and maximum; labels excluded'
    metrics['labels_loaded']=False
    _phase(out,'NF4_staging')
    import importlib.util
    spec=importlib.util.spec_from_file_location('stage11_frozen_staging',runtime/'stage_dataset_weights.py')
    staging=importlib.util.module_from_spec(spec);spec.loader.exec_module(staging)
    if staging.NF4_FILES!=NF4_FILES:raise ValueError('NF4 staging inventory differs')
    stage_root=runtime/'staging';stage_root.mkdir()
    copy_started=time.monotonic()
    with staging.staged_dataset_tree(paths['own_nf4_base'],NF4_FILES,label='own-nf4',temp_root=str(stage_root)) as local_base:
        metrics['nf4_staging_seconds']=time.monotonic()-copy_started
        paths['own_nf4_base']=str(local_base)
        paths_file=runtime/'probe_paths.json';save_json(paths_file,paths)
        _phase(out,'Qwen_model_load_and_inference')
        worker_log=out/'own_qwen_worker_PRIVATE.log';worker_started=time.monotonic()
        with worker_log.open('w',encoding='utf-8',buffering=1) as log:
            process=subprocess.Popen([sys.executable,'-u',str(runtime/'own_qwen_worker.py'),
                                      '--query',str(query),'--paths',str(paths_file),'--out',str(out/'qwen_PRIVATE.npz')],
                                     stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',
                                     bufsize=1,env=child_env,cwd=runtime)
            for line in process.stdout:
                log.write(line);print(line,end='',flush=True)
            code=process.wait()
        if code:raise RuntimeError('Frozen Qwen worker exited '+str(code)+'; inspect private worker log')
        metrics['Qwen_worker_wall_seconds']=time.monotonic()-worker_started
    metrics['nf4_copy_inference_cleanup_wall_seconds']=time.monotonic()-copy_started
    _phase(out,'saved_output_validation')
    metrics['output_checks']=validate_outputs(out/'qwen_PRIVATE.npz',out/'qwen_PRIVATE.npz.json',query)
    events=decode_events(out/'own_qwen_worker_PRIVATE.log')
    loaded=[e for e in events if e.get('event')=='QWEN38_INFERENCE_LOADED']
    batches=[e for e in events if e.get('event')=='OWN_QWEN_FULLTEXT_BATCH']
    if len(loaded)!=1 or not isinstance(loaded[0].get('seconds'),(int,float)) or not math.isfinite(loaded[0]['seconds']) or loaded[0]['seconds']<=0 or loaded[0].get('gpu_count')!=2 or loaded[0].get('layers')!=64 or loaded[0].get('portable')is not True or loaded[0].get('resident')is not True:raise ValueError('Expected full64 portable resident loader evidence')
    if len(batches)!=math.ceil(SAMPLE_ROWS/3) or batches[-1].get('done')!=SAMPLE_ROWS or any(e.get('total')!=SAMPLE_ROWS or e.get('truncated')is not False or not isinstance(e.get('seconds'),(int,float)) or not math.isfinite(e['seconds']) or e['seconds']<=0 or type(e.get('max_text_tokens'))is not int or e['max_text_tokens']<0 for e in batches):raise ValueError('Missing/invalid complete batch evidence')
    if max(e['max_text_tokens'] for e in batches)!=int(values.max()-1):raise ValueError('Worker token-length evidence differs from exact tokenizer selection')
    metrics['model_load_seconds']=loaded[0]['seconds']
    metrics['model_linear_count']=loaded[0].get('linears')
    metrics['batches']=[{k:e[k] for k in ['done','total','seconds','max_text_tokens','truncated']} for e in batches]
    inference_seconds=sum(e['seconds'] for e in batches)
    metrics['batch_inference_total_seconds']=inference_seconds
    metrics['sample_rows_per_inference_second']=SAMPLE_ROWS/inference_seconds
    metrics['sample_actual_tokens_per_inference_second']=int(values.sum())/inference_seconds
    metrics['oom_retry_count']=None
    metrics['oom_retry_observability']='Unknown: the unchanged frozen worker retries OOM batches but does not emit a retry counter'
    metrics['elapsed_inner_seconds']=time.monotonic()-began
    save_json(out/'INNER_RESULT.json',metrics)
    _phase(out,'COMPLETE_QWEN_BRANCH_ONLY')
    return metrics


def run_probe(admission=None,*,output_root='/kaggle/working/stage11_qwen_probe',timeout_seconds=TIMEOUT_SECONDS):
    admitted=validate_admission(admission)
    if type(timeout_seconds)is not int or not 1<=timeout_seconds<=TIMEOUT_SECONDS:raise ValueError('Deadline must be at most 900 seconds')
    if not Path('/kaggle/input').is_dir():raise ValueError('Actual Kaggle inputs are required; local model execution is forbidden')
    out=Path(output_root);out.mkdir(parents=True,exist_ok=False)
    Path('/kaggle/temp').mkdir(parents=True,exist_ok=True)
    runtime=Path(tempfile.mkdtemp(prefix='stage11-qwen-probe-',dir='/kaggle/temp'))
    report={'stage':11,'mode':'QWEN_BRANCH_RUNTIME_DIAGNOSTIC_ONLY','source_version':SOURCE_VERSION,
            'source_capture_sha256':SOURCE_CAPTURE_SHA256,'admission':admitted,
            'accuracy_evaluated':False,'official_public_score':None,'whole_ensemble_runtime_verified':False,
            'hidden_runtime_verified':False,'production_source_cells_modified':False,
            'source_module_sha256':FROZEN_MODULE_SHA256,'timeout_seconds':timeout_seconds,
            'nf4_expected_copy_bytes':sum(NF4_FILES.values()),'sample_rows':SAMPLE_ROWS,
            'labels_loaded_for_selection':False,'original_ids_text_or_row_outputs_in_public_report':False,
            'limitations':['Qwen branch only; complete reference/complement/ATLAS runtime is unmeasured.',
                           'Official-text timing does not establish hidden-text throughput or context limits.',
                           'The length-stratified sample intentionally stresses the tail and is not an unbiased hidden average.',
                           'Final adapter fitted all official labels; in-sample accuracy is not validation.',
                           'No OOM-retry counter is observable in the unchanged worker.']}
    packages={}
    for name in ['torch','numpy','pandas','transformers','accelerate','bitsandbytes','safetensors','tokenizers']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    report['parent_package_distribution_versions']=packages
    try:
        gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv,noheader,nounits'],
                           capture_output=True,text=True,timeout=10,check=True)
        report['nvidia_smi_devices']=[x.strip() for x in gpu.stdout.splitlines() if x.strip()]
    except Exception as e:report['nvidia_smi_observation_error']=type(e).__name__
    began=time.monotonic()
    try:
        result=run_deadline([sys.executable,'-u',str(Path(__file__)),
                             '--inner','--runtime',str(runtime),'--output',str(out)],
                            out/'watchdog_PRIVATE.log',seconds=timeout_seconds)
        report['watchdog']=result
        if result['timed_out']:report['status']='BLOCKED_QWEN_PROBE_TIMEOUT'
        elif result['returncode']!=0:report['status']='BLOCKED_QWEN_PROBE_FAILURE'
        else:
            report['metrics']=json.loads((out/'INNER_RESULT.json').read_text())
            report['status']='PASS_QWEN_BRANCH_PROBE_ONLY'
    except Exception as e:
        report['status']='BLOCKED_PROBE_ORCHESTRATION';report['error_type']=type(e).__name__;report['error']=str(e)[:1000]
    finally:
        phase=out/'CURRENT_PHASE.json'
        if phase.exists():report['last_phase']=json.loads(phase.read_text())
        shutil.rmtree(runtime,ignore_errors=True)
        report['scoped_runtime_and_nf4_stage_removed']=not runtime.exists()
        report['elapsed_watchdog_wall_seconds']=time.monotonic()-began
        save_json(out/'RUNTIME_PROBE.json',report)
        print('STAGE11_RUNTIME_PROBE_JSON_BEGIN');print(json.dumps(report,indent=2,allow_nan=False));print('STAGE11_RUNTIME_PROBE_JSON_END',flush=True)
    if report['status']!='PASS_QWEN_BRANCH_PROBE_ONLY':raise RuntimeError(report['status']+'; inspect private logs and aggregate RUNTIME_PROBE.json')
    return report


def self_test():
    checks={}
    def expect(name,fn,raises=False):
        try:fn()
        except Exception:checks[name]=bool(raises)
        else:checks[name]=not raises
    good={'source_version_id':SOURCE_VERSION,'cpu_audit_version_id':123,'cpu_audit_report_sha256':'a'*64,
          'cpu_audit_status':'PASS_CPU_INPUT_CONTRACT_ONLY','mandatory_source_checks_passed':160,'mandatory_source_checks_total':160}
    expect('locked_without_admission',lambda:validate_admission(None),True)
    expect('valid_actual_receipt_shape',lambda:validate_admission(good))
    for field,value in [('source_version_id',1),('cpu_audit_version_id',None),('cpu_audit_report_sha256','bad'),('cpu_audit_status','BLOCKED_INPUT_CONTRACT'),('mandatory_source_checks_passed',159)]:
        expect('reject_'+field,lambda field=field,value=value:validate_admission(dict(good,**{field:value})),True)
    for name,source in FROZEN_MODULES.items():
        expect('frozen_hash_'+name,lambda name=name,source=source: (_ for _ in ()).throw(ValueError()) if hashlib.sha256(source.encode()).hexdigest()!=FROZEN_MODULE_SHA256[name] else None)
        compile(source,name,'exec')
    with tempfile.TemporaryDirectory() as td:
        td=Path(td);query=td/'query.csv';npz=td/'worker.npz';meta=td/'worker.npz.json'
        pd.DataFrame({'id':['a','b'],'text':['one','two']}).to_csv(query,index=False)
        md={'rows':2,'seconds':1.0,'truncation':False,'decoder_layers':64,'adapter_layers':list(range(56,64)),'peak_reserved_bytes':[1,2]}
        save_json(meta,md)
        def write(**override):
            value={'ids':np.array(['a','b']),'labels':np.array(LABELS),'logits':np.zeros((2,18),dtype=np.float32)};value.update(override);np.savez(npz,**value)
        write();expect('finite_ordered_saved_output',lambda:validate_outputs(npz,meta,query))
        write(ids=np.array(['b','a']));expect('reject_reordered_ids',lambda:validate_outputs(npz,meta,query),True)
        write(labels=np.array(LABELS[::-1]));expect('accept_source_label_permutation',lambda:validate_outputs(npz,meta,query))
        permuted=validate_outputs(npz,meta,query);checks['named_canonical_permutation_is_exact']=permuted['canonical_permutation_from_saved']==list(range(17,-1,-1)) and permuted['saved_worker_array_rewritten']is False
        write(labels=np.array([LABELS[0]]*18));expect('reject_duplicate_label_names',lambda:validate_outputs(npz,meta,query),True)
        write(labels=np.array(LABELS[:-1]+['foreign']));expect('reject_foreign_label_name',lambda:validate_outputs(npz,meta,query),True)
        write(labels=np.array(LABELS[:-1]));expect('reject_missing_label_name',lambda:validate_outputs(npz,meta,query),True)
        write(logits=np.full((2,18),np.nan));expect('reject_nonfinite_logits',lambda:validate_outputs(npz,meta,query),True)
        write(logits=np.zeros((2,17)));expect('reject_wrong_class_shape',lambda:validate_outputs(npz,meta,query),True)
        write(ids=np.array(['a','a']));expect('reject_duplicate_ids',lambda:validate_outputs(npz,meta,query),True)
        write();save_json(meta,dict(md,truncation=True));expect('reject_truncation_metadata',lambda:validate_outputs(npz,meta,query),True)
        save_json(meta,dict(md,peak_reserved_bytes=[1]));expect('reject_single_gpu_metadata',lambda:validate_outputs(npz,meta,query),True)
        # A real bounded dummy process tree checks the watchdog without any models.
        code='import subprocess,sys,time; subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); print("dummy ready",flush=True); time.sleep(30)'
        result=run_deadline([sys.executable,'-u','-c',code],td/'deadline.log',seconds=.2,emit=False)
        checks['hard_deadline_terminates_process_tree']=result['timed_out'] and result['elapsed_seconds']<8 and result['returncode']!=0
    report={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'passed':sum(checks.values()),'total':len(checks),'source_modules_executed':False,'model_or_gpu_executed':False,'dummy_subprocess_only':True}
    if not all(checks.values()):raise AssertionError(json.dumps(report))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inner',action='store_true');parser.add_argument('--runtime');parser.add_argument('--output')
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--admission-json')
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2))
    elif args.inner:
        try:_run_inner(args.runtime,args.output)
        except BaseException:
            Path(args.output,'INNER_ERROR_PRIVATE.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
    elif args.admission_json:run_probe(json.loads(Path(args.admission_json).read_text()))
    else:raise SystemExit('Locked by default: provide an actual passing CPU audit admission receipt.')
