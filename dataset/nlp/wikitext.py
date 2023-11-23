import json
import os.path
import numpy as np
from tqdm import trange
from paddle.io import Dataset
from transformers import AutoTokenizer
from typing import Any
from dataclasses import dataclass
from config.base import BaseConfig
from dataset.registry import NLP_DATASET
from dataset import generate_mask, pad_sequence


@dataclass
class Wikitext103Config(BaseConfig):
    tokenizer: Any = None
    split: str = None
    cache_dir: str = None
    max_len: int = None


@NLP_DATASET.register_module
class Wikitext103Dataset(Dataset):
    config_class = Wikitext103Config

    def __init__(self, config: Wikitext103Config):
        super().__init__()
        self.tokenizer = config.tokenizer

        with open(os.path.join(config.cache_dir, f'{config.split}.json'), 'r+') as f:
            dataset = json.load(f)

        self.data = []
        for idx in trange(len(dataset)):
            self.data.append(self.tokenizer(dataset[idx]['text'])['input_ids'][:config.max_len])
        print(f"[!] collect {len(self.data)} for {config.split} set")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return {
            'input_ids': np.array(self.data[i][:-1]),
            'labels': np.array(self.data[i][1:]),
            'attention_mask': generate_mask(np.array(self.data[i][:-1]), pad_token_idx=self.tokenizer.eos_token_id)
        }

    def collate_fn(self, batch):
        input_ids = pad_sequence(
            [sample['input_ids'] for sample in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = pad_sequence(
            [sample['labels'] for sample in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        mask = generate_mask(input_ids, pad_token_idx=self.tokenizer.eos_token_id)
        encoded_result = {'input_ids': input_ids, 'labels': labels, 'attention_mask': mask}
        return encoded_result
