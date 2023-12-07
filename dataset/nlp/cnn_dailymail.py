import json
import os.path
import numpy as np
from tqdm import trange
from typing import Any
from paddle.io import Dataset
from dataclasses import dataclass
from config.base import BaseConfig
from dataset.registry import NLP_DATASET
from dataset import generate_mask, pad_sequence

import logging.config

logger = logging.getLogger(__name__)


@dataclass
class CNNDailyMailConfig(BaseConfig):
    tokenizer: Any = None
    split: str = None
    cache_dir: str = None
    max_sample: int = None
    context_len: int = None
    response_len: int = None


@NLP_DATASET.register_module
class CNNDailyMailDataset(Dataset):
    config_class = CNNDailyMailConfig

    def __init__(self, config: CNNDailyMailConfig):
        super().__init__()
        self.tokenizer = config.tokenizer
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids('[SEP]')

        with open(os.path.join(config.cache_dir, f'{config.split}.json'), 'r+') as f:
            dataset = json.load(f)

        num_sample = min(len(dataset['article']), config.max_sample)
        self.data = []
        for idx in trange(num_sample, desc=f"Tokenizing cnn_dailymail dataset"):
            article, highlight = dataset['article'][idx].strip(), dataset['highlights'][idx].strip()
            article_ids = self.tokenizer.encode(article)[:config.context_len]
            highlight_ids = self.tokenizer.encode(highlight)[:config.response_len]
            if config.split == 'test':
                tokens = [self.tokenizer.eos_token_id] + article_ids + [self.sep_token_id]
            else:
                tokens = [self.tokenizer.eos_token_id] + article_ids + [self.sep_token_id] + highlight_ids + [
                    self.sep_token_id]
            self.data.append(tokens)
        logger.info(f"[!] collect {len(self.data)} for {config.split} set")

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
