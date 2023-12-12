import yaml
import paddle
import os.path
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

from test import eval
from config import AutoConfig
from dataset import build_dataset
from utils.registry import build_from_config
from modules.registry import LM_HEAD_MODEL, CRITERION

import logging.config

logger = logging.getLogger(__name__)


def train(args):
    """
    Environment Configuration
    """
    if paddle.device.cuda.device_count() >= 1:
        paddle.set_device('gpu:0')

    """
    Loading Log Config
    """
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
    logger.info(cfg)

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
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
    except ValueError:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    """
    Loading Run Config
    """
    logger.info(f"[!] Loading run config from {args.run_config} ...")
    with open(args.run_config) as f:
        run_cfg = yaml.load(f, Loader=yaml.FullLoader)
    logger.info(run_cfg)

    """
    Dataset Preparation
    """
    logger.info(f"[!] Loading dataset ...")
    dataset_cfg = AutoConfig.from_yaml(os.path.join('./config/dataset_config/', f"{args.dataset.lower()}.yaml"))
    dataset_cfg.tokenizer = tokenizer
    dataset_cfg.split = 'train'
    dataset_cfg.max_sample = run_cfg['max_steps'] * run_cfg['batch_size']
    train_dataset = build_dataset(dataset_cfg)
    train_dataloader = paddle.io.DataLoader(train_dataset, batch_size=run_cfg['batch_size'], shuffle=True,
                                            collate_fn=train_dataset.collate_fn)
    dataset_cfg.split = 'valid'
    valid_dataset = build_dataset(dataset_cfg)
    valid_dataloader = paddle.io.DataLoader(valid_dataset, batch_size=1, shuffle=False)

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
    logger.info(criterion_cfg)
    criterion = build_from_config(criterion_cfg, CRITERION)
    optimizer = paddle.optimizer.Adam(learning_rate=run_cfg['learning_rate'], parameters=model.parameters())
    if run_cfg['amp_level'] == 'O2':
        model, optimizer = paddle.amp.decorate(models=model, optimizers=optimizer, level='O2')
    scaler = paddle.amp.GradScaler(init_loss_scaling=1024)

    """
    Training Loop
    """
    logger.info(f"[!] Start training [!]")
    current_step = 0
    progress_bar = tqdm(total=run_cfg['max_steps'])
    while current_step < run_cfg['max_steps']:
        loss_list = []
        acc_list = []
        for batch_id, data in enumerate(train_dataloader):
            """
            Forward
            """
            with paddle.amp.auto_cast(level=run_cfg['amp_level']):
                loss, output = criterion(model, data)

            """
            Calculate accuracy
            """
            chosen_tokens = paddle.argmax(output.logit, axis=-1)
            gen_acc = (chosen_tokens.reshape([-1]) == data['labels'].reshape([-1]))
            valid_mask = (data['labels'] != tokenizer.pad_token_id).reshape([-1])
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

            progress_bar.update(1)
            current_step += 1

            if current_step % run_cfg['log_interval'] == 0:
                # TODO: complete metrics
                log_dict = {
                    "step": current_step,
                    "loss": np.mean(loss_list),
                    "acc": np.mean(acc_list)
                }
                tqdm.write(str(log_dict))
                loss_list.clear()
                acc_list.clear()

            if current_step % run_cfg['eval_steps'] == 0:
                model.eval()
                eval(model, valid_dataloader, tokenizer)
                model.train()

            if current_step == run_cfg['max_steps']:
                break

    """
    Save checkpoint
    """
    if args.save_dir is None:
        args.save_dir = os.path.join(os.environ['HOME'], 'work-dir')
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    cfg.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    model.eval()
    paddle.save(obj=model.state_dict(), path=os.path.join(args.save_dir, 'paddle_model.pdparams'))
    logger.info(f"Checkpoint saved at {args.save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config', type=str, default=None)
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--dataset', type=str, choices=['wikitext-103', 'cnn_dailymail'])
    parser.add_argument('--criterion', type=str,
                        choices=['cross_entropy', 'ditto', 'scalegrad', 'simctg', 'candidate_penalty',
                                 'unlikelihood_seq'])
    parser.add_argument('--pretrained-model-path', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--log-config', type=str, default='./config/log_config/_base_.yaml')
    parser.add_argument('--run-config', type=str, default='./config/run_config/_base_.yaml')
    args = parser.parse_args()
    train(args)
