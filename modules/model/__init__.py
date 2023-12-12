import os
import importlib
from typing import Any
from utils.registry import Registry, build_from_config

ENCODER_ONLY_MODEL = Registry('encoder_only_model')
DECODER_ONLY_MODEL = Registry('decoder_only_model')
ENCODER_DECODER_MODEL = Registry('encoder_decoder_model')

LM_HEAD_MODEL = Registry('lm_head_model')
CLS_HEAD_MODEL = Registry('cls_head_model')
DOUBLE_HEAD_MODEL = Registry('double_head_model')


def build_encoder_only_model(config: Any):
    return build_from_config(config, ENCODER_ONLY_MODEL)


def build_decoder_only_model(config: Any):
    return build_from_config(config, DECODER_ONLY_MODEL)


def build_encoder_decoder_model(config: Any):
    return build_from_config(config, ENCODER_ONLY_MODEL)


def build_lm_head_model(config: Any):
    return build_from_config(config, LM_HEAD_MODEL)


def build_cls_head_model(config: Any):
    return build_from_config(config, CLS_HEAD_MODEL)


def build_double_head_model(config: Any):
    return build_from_config(config, DOUBLE_HEAD_MODEL)


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("modules.model." + file_name)
