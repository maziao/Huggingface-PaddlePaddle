import os
import importlib
from typing import Any
from utils.registry import Registry, build_from_config

CRITERION = Registry('criterion')


def build_criterion(config: Any):
    return build_from_config(config, CRITERION)


for file in sorted(os.listdir(os.path.dirname(__file__))):
    if file.endswith(".py") and not file.startswith("_"):
        file_name = file[: file.find(".py")]
        importlib.import_module("modules.criterion." + file_name)
