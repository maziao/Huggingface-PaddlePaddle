import os
import importlib
from typing import Any
from utils.registry import Registry, build_from_config

MLP = Registry('mlp')


def build_mlp(config: Any):
    return build_from_config(config, MLP)


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("modules.mlp." + file_name)
