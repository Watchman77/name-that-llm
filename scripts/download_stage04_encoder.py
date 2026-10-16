"""Online preparation only; modelling runs separately with local files/offline flags."""
from pathlib import Path
import argparse
from huggingface_hub import snapshot_download

MODEL_ID = 'sentence-transformers/all-mpnet-base-v2'
REVISION = 'e8c3b32edf5434bc2275fc9bab85f82640a19130'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=Path, required=True)
    args = parser.parse_args()
    path = snapshot_download(MODEL_ID, revision=REVISION, local_dir=args.model_dir,
        allow_patterns=['config.json','tokenizer.json','tokenizer_config.json',
            'special_tokens_map.json','vocab.txt','model.safetensors','sentence_bert_config.json',
            'README.md','1_Pooling/config.json','modules.json','config_sentence_transformers.json'])
    print('Pinned public checkpoint ready:', path)
