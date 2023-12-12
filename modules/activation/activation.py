import paddle
from typing import Optional, Any
from abc import abstractmethod
from config.base import BaseConfig


class Activation(paddle.nn.Layer):
    config_class = BaseConfig

    def __init__(self, config: Any):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        raise NotImplementedError(f"`forward` method has not been implemented in class {self.__class__}")
