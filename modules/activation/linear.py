import paddle
from config.base import PseudoConfig
from modules.activation import ACTIVATION
from modules.activation.activation import Activation


@ACTIVATION.register_module
class LinearActivation(Activation):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """
    config_class = PseudoConfig

    def forward(self, hidden_states: paddle.Tensor) -> paddle.Tensor:
        return hidden_states
