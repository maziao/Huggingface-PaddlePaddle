import paddle
from config.base import PseudoConfig
from modules.activation import ACTIVATION
from modules.activation.activation import Activation


@ACTIVATION.register_module
class ReLU6(Activation):
    config_class = PseudoConfig

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return paddle.nn.functional.hardtanh(hidden_states, min=0.0, max=6.0)
