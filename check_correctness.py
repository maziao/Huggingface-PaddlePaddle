import yaml
import torch
import paddle
import argparse
import numpy as np
import logging.config
from config import AutoConfig
from modules.model import LM_HEAD_MODEL
from utils.registry import build_from_config
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def load_tokenizer(tokenizer_path: str):
    logger.info(f"[!] Loading tokenizer from {tokenizer_path} ...")
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return _tokenizer


def load_pd_model(pretrained_model_path: str):
    logger.info(f"[!] Loading model config from {pretrained_model_path} ...")
    cfg = AutoConfig.from_pretrained(pretrained_model_path)

    logger.info(f"[!] Building model ...")
    model = build_from_config(cfg, LM_HEAD_MODEL)

    logger.info(f"[!] Loading pretrained checkpoint from {pretrained_model_path} ...")
    model.from_pretrained(pretrained_model_path)
    model.eval()
    return model


def load_hf_model(pretrained_model_path: str):
    logger.info(f"[!] Loading pretrained model from {pretrained_model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_path)
    model.eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hf-repo', type=str, required=True)
    parser.add_argument('--pd-repo', type=str, required=True)
    parser.add_argument('--log-config', type=str, default='./config/log_config/_base_.yaml')
    args = parser.parse_args()

    with open(args.log_config) as f:
        log_cfg = yaml.load(f, Loader=yaml.FullLoader)
        logging.config.dictConfig(log_cfg)

    tokenizer = load_tokenizer(args.pd_repo)
    input_ids = tokenizer(
        'This model is being created in order to enable public research on large language models (LLMs).'
    )['input_ids']

    if paddle.device.cuda.device_count() >= 1:
        paddle.set_device('gpu:0')

    hf_model = load_hf_model(args.hf_repo).cuda()
    hf_output = hf_model(torch.tensor([input_ids]).cuda())
    del hf_model

    pd_model = load_pd_model(args.pd_repo)
    pd_output = pd_model(paddle.Tensor(np.array([input_ids])))
    del pd_model

    hf_logits = hf_output.logits.cpu().detach().numpy()
    pd_logits = pd_output.logit.numpy()
    print(f"HuggingFace logits: {hf_logits}")
    print(f"PaddlePaddle logits: {pd_logits}")
    print(f"Average error: {np.mean(np.fabs(pd_logits - hf_logits))}")
