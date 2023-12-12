import os
import importlib
from typing import Any
from utils.registry import Registry, build_from_config

ATTENTION = Registry('attention')


def build_attention(config: Any):
    return build_from_config(config, ATTENTION)


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("modules.attention." + file_name)
