import argparse
import yaml
import paddle
from paddle.io import DistributedBatchSampler
from tqdm import tqdm, trange

import models.losses.cross_entropy
from utils.registry import build_from_config
from models.registry import LM_HEAD_MODEL
from config import Config

import logging.config

logger = logging.getLogger(__name__)


def train(args):
    paddle.device.set_device('cpu')
    # if paddle.device.cuda.device_count() >= 1:
    #     paddle.set_device('gpu:0')

    with open(args.log_config) as f:
        log_cfg = yaml.load(f, Loader=yaml.FullLoader)
        logging.config.dictConfig(log_cfg)

    logger.info(f"[!] Loading model config from {args.model_config} ...")
    cfg = Config.from_yaml(args.model_config, model_name=args.model_name)
    print(cfg)
    logger.info(f"[!] Loading dataset ...")
    with open('./config/dataset_config/cnn_dailymail.yaml') as f:
        dataset_cfg = yaml.load(f, Loader=yaml.FullLoader)
    from dataset import CNNDailyMailDataset
    dataset = CNNDailyMailDataset(args.tokenizer, 'train', dataset_cfg)
    dataloader = paddle.io.DataLoader(dataset, batch_size=1, shuffle=True)

    logger.info(f"[!] Building model ...")
    model = build_from_config(cfg, LM_HEAD_MODEL)
    print(model)
    if args.pretrained_model_path is not None:
        logger.info(f"[!] Loading pretrained checkpoint from {args.pretrained_model_path} ...")
        model.from_pretrained(args.pretrained_model_path)

    logger.info(f"[!] Preparing optimizer, criterion and metrics ...")
    # criterion = models.losses.cross_entropy.CrossEntropyCriterion(ignore_index=50256)

    logger.info(f"[!] Start training [!]")

    model.train()

    optim = paddle.optimizer.Adam(parameters=model.parameters())
    loss_fn = paddle.nn.CrossEntropyLoss()

    for epoch in range(5):
        for batch_id, data in enumerate(dataloader):
            input_ids = data[0]
            label = data[1]

            output = model(input_ids)

            loss = loss_fn(output.logit.reshape([-1, output.logit.shape[-1]]), label.reshape([-1, 1]))

            chosen_tokens = paddle.argmax(output.logit, axis=-1)
            gen_acc = (chosen_tokens.reshape([-1]) == label.reshape([-1]))
            valid_mask = (label != 50256).reshape([-1])
            valid_tokens = gen_acc & valid_mask
            acc = valid_tokens.sum().item() / valid_mask.sum().item()

            loss.backward()
            optim.step()
            optim.clear_grad()

            logger.info("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id + 1, float(loss),
                                                                                  acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-config', type=str, default='./config/model_config/temp.yaml')
    parser.add_argument('--model-name', type=str, default='gpt2')
    parser.add_argument('--tokenizer', type=str, default='gpt2')
    # parser.add_argument('--dataset-config', type=str, required=True)
    parser.add_argument('--pretrained-model-path', type=str, default=None, required=False)
    parser.add_argument('--log-config', type=str, default='./config/log_config/_base_.yaml')
    args = parser.parse_args()
    train(args)
