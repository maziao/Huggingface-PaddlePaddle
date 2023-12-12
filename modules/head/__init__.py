import os
import importlib
from typing import Any
from utils.registry import Registry, build_from_config

MODEL_HEAD = Registry('model_head')


def build_model_head(config: Any):
    return build_from_config(config, MODEL_HEAD)


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("modules.head." + file_name)
