import os
import importlib
from typing import Any
from utils.registry import Registry, build_from_config

EMBEDDING = Registry('embedding')


def build_embedding(config: Any):
    return build_from_config(config, EMBEDDING)


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("modules.embedding." + file_name)
