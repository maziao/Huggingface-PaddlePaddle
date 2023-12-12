import yaml
import paddle
import os.path
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from config import AutoConfig
from dataset import build_dataset
from modules.model import LM_HEAD_MODEL
from utils.registry import build_from_config

import logging.config

logger = logging.getLogger(__name__)


def eval(model, eval_dataloader, tokenizer):
    criterion = paddle.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)
    bar = tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc='Validating')

    loss_list = []
    num_valid_tokens = 0
    num_correct_tokens = 0

    for batch_id, data in bar:
        """
        Forward
        """
        input_ids = data['input_ids']
        labels = data['labels']

        output = model(input_ids, attention_mask=data['attention_mask'])
        loss = criterion(output.logit.reshape([-1, output.logit.shape[-1]]), labels.reshape([-1, 1]))

        """
        Calculate accuracy
        """
        chosen_tokens = paddle.argmax(output.logit, axis=-1)
        gen_acc = (chosen_tokens.reshape([-1]) == labels.reshape([-1]))
        valid_mask = (labels != tokenizer.pad_token_id).reshape([-1])
        valid_tokens = gen_acc & valid_mask

        """
        Metrics
        """
        loss = loss.tolist()
        loss_list.extend(loss)
        num_correct_tokens += valid_tokens.sum().item()
        num_valid_tokens += valid_mask.sum().item()

        bar.update(1)
    logger.info({"ppl": np.exp(np.mean(loss_list)), "acc": num_correct_tokens / num_valid_tokens})


def test(args):
    """
    Environment Configuration
    """
    if paddle.device.cuda.device_count() >= 1:
        paddle.set_device('gpu:0')

    with open(args.log_config) as f:
        log_cfg = yaml.load(f, Loader=yaml.FullLoader)
        logging.config.dictConfig(log_cfg)

    """
    Loading Configuration
    """
    logger.info(f"[!] Loading model config from {args.pretrained_model_path} ...")
    cfg = AutoConfig.from_pretrained(args.pretrained_model_path)

    """
    Loading Tokenizer
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, use_fast=False)
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    """
    Dataset Preparation
    """
    logger.info(f"[!] Loading dataset ...")
    dataset_cfg = AutoConfig.from_yaml(os.path.join('./config/dataset_config/', f"{args.dataset.lower()}.yaml"))
    dataset_cfg.tokenizer = tokenizer
    dataset_cfg.split = 'valid'
    dataset = build_dataset(dataset_cfg)
    dataloader = paddle.io.DataLoader(dataset, batch_size=1, shuffle=False)

    """
    Model Instantiation
    """
    logger.info(f"[!] Building model ...")
    model = build_from_config(cfg, LM_HEAD_MODEL)
    logger.info(f"[!] Loading pretrained checkpoint from {args.pretrained_model_path} ...")
    model.from_pretrained(args.pretrained_model_path)

    """
    Training Preparation
    """
    logger.info(f"[!] Preparing optimizer, criterion and metrics ...")
    model.eval()

    """
    Evaluation Loop
    """
    logger.info(f"[!] Start evaluation [!]")
    eval(model, dataloader, tokenizer)

    """
    Generation
    """
    if args.generation_len is not None and args.generation_len > 0:
        for batch_id, data in enumerate(dataloader):
            past_key_values = None
            input_ids = data['input_ids']
            generated_ids = []
            for i in range(args.generation_len):
                output = model(
                    input_ids,
                    past_key_values=past_key_values,
                    output_key_values=True
                )
                past_key_values = output.past_key_values
                input_ids = paddle.argmax(output.logit[:, -1], axis=-1, keepdim=True)
                generated_ids.append(input_ids)
            generated_ids = paddle.concat(generated_ids, axis=-1).tolist()
            for j, sample in enumerate(generated_ids):
                generated_text = tokenizer.decode(sample, skip_special_tokens=True).replace('\n', ' ')
                logger.info(f"Batch {batch_id + 1} / {len(dataloader)}: {generated_text}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['wikitext-103', 'cnn_dailymail'])
    parser.add_argument('--pretrained-model-path', type=str, default=None)
    parser.add_argument('--generation-len', type=int, default=None)
    parser.add_argument('--log-config', type=str, default='./config/log_config/_base_.yaml')
    args = parser.parse_args()
    test(args)
