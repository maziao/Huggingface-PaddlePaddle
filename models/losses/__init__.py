import paddle
from typing import Any
from abc import abstractmethod
from dataclasses import dataclass
from config.base import BaseConfig


@dataclass
class CriterionConfig(BaseConfig):
    pass


class Criterion(paddle.nn.Layer):
    config_class = CriterionConfig

    def __init__(self, config: CriterionConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, **kwargs) ->Any:
        raise NotImplementedError
