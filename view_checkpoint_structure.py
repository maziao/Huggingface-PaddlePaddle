import yaml
import paddle
import argparse
import logging.config
from config import AutoConfig
from models.registry import LM_HEAD_MODEL
from utils.registry import build_from_config
from transformers import AutoModelForCausalLM

logger = logging.getLogger(__name__)


def load_pd_model_from_config(config_path: str, model_name: str):
    logger.info(f"[!] Loading model config from {config_path} ...")
    cfg = AutoConfig.from_yaml(config_path, model_name=model_name)

    logger.info(f"[!] Building model ...")
    model = build_from_config(cfg, LM_HEAD_MODEL)
    model.eval()
    return model


def load_pd_model_from_pretrained(pretrained_model_path: str):
    logger.info(f"[!] Loading model config from {pretrained_model_path} ...")
    cfg = AutoConfig.from_pretrained(pretrained_model_path)

    logger.info(f"[!] Building model ...")
    model = build_from_config(cfg, LM_HEAD_MODEL)
    model.eval()
    return model


def load_hf_model(pretrained_model_path: str):
    logger.info(f"[!] Loading pretrained model from {pretrained_model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(pretrained_model_path)
    model.eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, choices=['hf', 'pd'])
    parser.add_argument('--model-config', type=str, required=False, default=None)
    parser.add_argument('--model-name', type=str, required=False, default=None)
    parser.add_argument('--pretrained-model-path', type=str, required=False, default=None)
    parser.add_argument('--log-config', type=str, default='./config/log_config/_base_.yaml')
    args = parser.parse_args()

    with open(args.log_config) as f:
        log_cfg = yaml.load(f, Loader=yaml.FullLoader)
        logging.config.dictConfig(log_cfg)

    if paddle.device.cuda.device_count() >= 1:
        paddle.set_device('gpu:0')

    if args.model_type == 'hf':
        model = load_hf_model(args.pretrained_model_path).cuda()
        print(model)
        for key, value in model.state_dict().items():
            print(key, value.size())
    else:
        if args.pretrained_model_path is None:
            assert args.model_config is not None and args.model_name is not None
            model = load_pd_model_from_config(args.model_config, args.model_name)
        else:
            model = load_pd_model_from_pretrained(args.pretrained_model_path)
        print(model)
        for key, value in model.state_dict().items():
            print(key, value.shape)
