import yaml
import paddle
import os.path
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from config import AutoConfig
from dataset import build_dataset
from models.registry import LM_HEAD_MODEL
from utils.registry import build_from_config

import logging.config

logger = logging.getLogger(__name__)


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
    print(cfg)

    """
    Loading Tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path, use_fast=False)
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
    dataloader = paddle.io.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)

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
    criterion = paddle.nn.CrossEntropyLoss(reduction='none')

    """
    Training Loop
    """
    logger.info(f"[!] Start evaluation [!]")
    bar = tqdm(enumerate(dataloader), total=len(dataloader))

    loss_list = []
    num_valid_tokens = 0
    num_correct_tokens = 0

    for batch_id, data in bar:
        """
        Forward
        """
        input_ids = data['input_ids']
        labels = data['labels']
        with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}, level='O1'):
            output = model(input_ids, attention_mask=data['attention_mask'])
            loss = criterion(output.logit.reshape([-1, output.logit.shape[-1]]), labels.reshape([-1, 1]))

        """
        Calculate accuracy
        """
        chosen_tokens = paddle.argmax(output.logit, axis=-1)
        gen_acc = (chosen_tokens.reshape([-1]) == labels.reshape([-1]))
        valid_mask = (labels != 50256).reshape([-1])
        valid_tokens = gen_acc & valid_mask
        acc = valid_tokens.sum().item() / valid_mask.sum().item()

        """
        Metrics
        """
        loss = loss.tolist()
        loss_list.extend(loss)
        num_correct_tokens += valid_tokens.sum().item()
        num_valid_tokens += valid_mask.sum().item()

        if batch_id == len(dataloader) - 1:
            loss = np.array(loss_list)
            acc = num_correct_tokens / num_valid_tokens

        bar.set_description(f"[!] ppl {round(np.exp(np.mean(loss)), 4)} | acc {round(float(acc), 4)}")
        bar.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='wikitext-103')
    parser.add_argument('--pretrained-model-path', type=str, default='/home/mza/work-dir/paddle/gpt-neo-125m-temp')
    parser.add_argument('--log-config', type=str, default='./config/log_config/_base_.yaml')
    args = parser.parse_args()
    test(args)
