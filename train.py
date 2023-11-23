import yaml
import paddle
import os.path
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from config import AutoConfig
from dataset import build_dataset
from utils.registry import build_from_config
from models.registry import LM_HEAD_MODEL, CRITERION

import logging.config

logger = logging.getLogger(__name__)


def train(args):
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
    logger.info(f"[!] Loading model config from {args.model_config} ...")
    if args.pretrained_model_path is not None:
        cfg = AutoConfig.from_pretrained(args.pretrained_model_path)
    else:
        cfg = AutoConfig.from_yaml(args.model_config, model_name=args.model_name)
    print(cfg)

    """
    Loading Tokenizer
    """
    if args.pretrained_model_path is not None:
        tokenizer = args.pretrained_model_path
    elif args.tokenizer is not None:
        tokenizer = args.tokenizer
    else:
        raise ValueError("Please specify the tokenizer to use.")
    logger.info(f"[!] Loading tokenizer from {tokenizer} ...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
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
    if args.pretrained_model_path is not None:
        logger.info(f"[!] Loading pretrained checkpoint from {args.pretrained_model_path} ...")
        model.from_pretrained(args.pretrained_model_path)

    """
    Training Preparation
    """
    logger.info(f"[!] Preparing optimizer, criterion and metrics ...")
    model.train()
    criterion_cfg = AutoConfig.from_yaml(os.path.join('./config/criterion_config', f"{args.criterion.lower()}.yaml"))
    criterion_cfg.pad_token_id = tokenizer.pad_token_id
    if args.criterion == 'DITTO':
        criterion_cfg.end_sentence_decoded = tokenizer('.')['input_ids'][0]
    print(criterion_cfg)
    criterion = build_from_config(criterion_cfg, CRITERION)
    optimizer = paddle.optimizer.Adam(parameters=model.parameters())
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    """
    Training Loop
    """
    logger.info(f"[!] Start training [!]")
    for epoch in range(5):
        bar = tqdm(enumerate(dataloader), total=len(dataloader))
        loss_list = []
        acc_list = []
        for batch_id, data in bar:
            """
            Forward
            """
            # with paddle.amp.auto_cast(custom_white_list={'elementwise_add'}, level='O1'):
            loss, output = criterion(model, data)

            """
            Calculate accuracy
            """
            chosen_tokens = paddle.argmax(output.logit, axis=-1)
            gen_acc = (chosen_tokens.reshape([-1]) == data['labels'].reshape([-1]))
            valid_mask = (data['labels'] != 50256).reshape([-1])
            valid_tokens = gen_acc & valid_mask
            acc = valid_tokens.sum().item() / valid_mask.sum().item()

            """
            Backward
            """
            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.clear_grad(set_to_zero=False)

            loss_list.append(float(loss))
            acc_list.append(float(acc))

            if batch_id == len(dataloader) - 1:
                loss = np.mean(np.array(loss_list))
                acc = np.mean(np.array(acc_list))

            bar.set_description(f"[!] Epoch {epoch + 1} / {5} | loss {round(float(loss), 4)} | acc {round(acc, 4)}")
            bar.update(1)

    """
    Save checkpoint
    """
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    cfg.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    model.eval()
    paddle.save(obj=model.state_dict(), path=os.path.join(args.save_dir, 'paddle_model.pdparams'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config', type=str, default='./config/model_config/gpt2.yaml')
    parser.add_argument('--model-name', type=str, default='gpt2')
    parser.add_argument('--tokenizer', type=str, default='/home/mza/model-zoo/paddle/gpt2')
    parser.add_argument('--dataset', type=str, default='wikitext-103')
    parser.add_argument('--criterion', type=str, default='scalegrad')
    parser.add_argument('--pretrained-model-path', type=str, default='/home/mza/model-zoo/paddle/gpt2/')
    parser.add_argument('--save-dir', type=str, default='/home/mza/work-dir/paddle/gpt2')
    parser.add_argument('--log-config', type=str, default='./config/log_config/_base_.yaml')
    args = parser.parse_args()
    train(args)
