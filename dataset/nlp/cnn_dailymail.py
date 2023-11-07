import json

import numpy as np
import paddle
from tqdm import trange
from datasets import load_dataset
from paddle.io import Dataset
from transformers import AutoTokenizer
# from paddle.static.nn import sequence_pad
from paddle.static.nn import sequence_pad
from ..utils import generate_mask
from dataset.registry import NLP_DATASET


@NLP_DATASET.register_module
class CNNDailyMailDataset(Dataset):
    def __init__(self, tokenizer, split, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.add_tokens(['[SEP]'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids('[SEP]')

        with open(f"{args['cache_dir']}/{split}.json", 'r+') as f:
            dataset = json.load(f)

        self.data = []
        for idx in trange(1000):
            article, highlight = dataset['article'][idx].strip(), dataset['highlights'][idx].strip()
            article_ids = self.tokenizer.encode(article)[:args['context_len']]
            highlight_ids = self.tokenizer.encode(highlight)[:args['response_len']]
            if split == 'test':
                tokens = [self.tokenizer.eos_token_id] + article_ids + [self.sep_token_id]
            else:
                tokens = [self.tokenizer.eos_token_id] + article_ids + [self.sep_token_id] + highlight_ids + [
                    self.sep_token_id]
            self.data.append(tokens)
        print(f"[!] collect {len(self.data)} for {split} set")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return np.array(self.data[i][:-1]), np.array(self.data[i][1:])

    def collate(self, batch):
        input_ids = sequence_pad(batch, pad_value=self.tokenizer.pad_token_id)
        mask = generate_mask(input_ids, pad_token_idx=self.tokenizer.eos_token_id)
        encoded_result = {'input_ids': input_ids, 'attention_mask': mask}
        return encoded_result
