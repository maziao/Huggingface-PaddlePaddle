import json
import os.path
import numpy as np
from tqdm import trange
from paddle.io import Dataset
from transformers import AutoTokenizer
from dataset.registry import NLP_DATASET
from dataset import generate_mask, pad_sequence


@NLP_DATASET.register_module
class CNNDailyMailDataset(Dataset):
    def __init__(self, tokenizer, split, args):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids('[SEP]')

        with open(os.path.join(args['cache_dir'], f'{split}.json'), 'r+') as f:
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
