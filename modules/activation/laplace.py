import math
import paddle
from dataclasses import dataclass
from config.base import BaseConfig
from modules.activation import ACTIVATION
from modules.activation.activation import Activation


@dataclass
class LaplaceConfig(BaseConfig):
    mu: float = 0.707107
    sigma: float = 0.282095


@ACTIVATION.register_module
class LaplaceActivation(Activation):
    """
    Applies elementwise activation based on Laplace function, introduced in MEGA as an attention activation. See
    https://arxiv.org/abs/2209.10655

    Inspired by squared relu, but with bounded range and gradient for better stability
    """
    config_class = LaplaceConfig

    def __init__(self, config: LaplaceConfig):
        super().__init__(config)

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        hidden_states = (hidden_states - self.config.mu).div(self.config.sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + paddle.erf(hidden_states))
