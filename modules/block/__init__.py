import os
import importlib
from typing import Any
from utils.registry import Registry, build_from_config

EMBEDDING_BLOCK = Registry('embedding_block')
ENCODER_BLOCK = Registry('encoder_block')
DECODER_BLOCK = Registry('decoder_block')


def build_embedding_block(config: Any):
    return build_from_config(config, EMBEDDING_BLOCK)


def build_encoder_block(config: Any):
    return build_from_config(config, ENCODER_BLOCK)


def build_decoder_block(config: Any):
    return build_from_config(config, DECODER_BLOCK)


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("modules.block." + file_name)
