import paddle
from config.base import PseudoConfig
from modules.activation import ACTIVATION
from modules.activation.activation import Activation


@ACTIVATION.register_module
class ReLUSquaredActivation(Activation):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """
    config_class = PseudoConfig

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        relu_applied = paddle.nn.functional.relu(hidden_states)
        squared = paddle.square(relu_applied)
        return squared
